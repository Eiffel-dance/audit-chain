import json
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app import (
    AuditChain,
    AuditChainConflictError,
    AuditChainStateError,
    ZERO,
)


def record_bytes(item):
    return (json.dumps(item, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def valid_row(tenant="t", seq=1, prev=ZERO, event=None):
    row = {"tenant": tenant, "seq": seq, "event": event or {}, "prev": prev}
    row["hash"] = AuditChain._hash(row)
    return row


class ImportAllRangeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.path = Path(self.tmp.name) / "audit.jsonl"
        self.chain = AuditChain(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def write_rows(self, rows, path=None):
        path = path or self.path
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(r if isinstance(r, str) else json.dumps(r, sort_keys=True))
                f.write("\n")

    def make_source(self, tenants=("a", "b"), n=6, name="source.jsonl"):
        src = AuditChain(self.path.with_name(name))
        for i in range(n):
            for tenant in tenants:
                src.append(tenant, {"i": i, "s": "审计"})
        return src

    def head_of(self, src, tenant, n):
        if n == 0:
            return 0, ZERO
        seg = src.export_tenant_range(tenant, 1, n)
        rows = [json.loads(l) for l in seg.decode().splitlines()]
        return n, rows[-1]["hash"]

    def tail_segments(self, src, prefixes):
        # prefixes: {tenant: prefix_length}; returns
        # (rows_by_tenant, heads) where rows are the open-tail segments and
        # heads are the asserted prefix tails.
        rows_by_tenant, heads = {}, []
        for tenant, n in prefixes.items():
            seg = src.export_tenant_range(tenant, n + 1)
            rows_by_tenant[tenant] = [
                json.loads(l) for l in seg.decode().splitlines()]
            _count, tail = self.head_of(src, tenant, n)
            heads.append({"tenant": tenant, "expected_count": n,
                          "expected_hash": tail})
        return rows_by_tenant, heads

    # --- boundary: validation order, empty no-op, no file/no byte touched ---

    def test_data_must_be_bytes(self):
        heads = [{"tenant": "t", "expected_count": 0, "expected_hash": ZERO}]
        for bad in ("", "not jsonl", bytearray(b""), 1, None, [], object()):
            with self.assertRaises(ValueError):
                self.chain.import_all_range(bad, heads)
        self.assertFalse(self.path.exists())

    def test_expected_heads_must_be_list(self):
        row = record_bytes(valid_row("t", 1))
        for bad in (None, {}, (), "x", 1, object()):
            with self.assertRaises(ValueError):
                self.chain.import_all_range(row, bad)
        self.assertFalse(self.path.exists())

    def test_head_members_must_have_exactly_three_keys(self):
        row = record_bytes(valid_row("t", 1))
        good = {"tenant": "t", "expected_count": 0, "expected_hash": ZERO}
        for bad in (
            None, 1, "x", [],
            {"tenant": "t", "expected_count": 0},
            {"tenant": "t", "expected_hash": ZERO},
            {"expected_count": 0, "expected_hash": ZERO},
            dict(good, extra=1),
        ):
            with self.assertRaises(ValueError):
                self.chain.import_all_range(row, [bad])
        self.assertFalse(self.path.exists())

    def test_illegal_tenant_is_value_error(self):
        bad_tenants = [
            float("nan"), float("inf"), float("-inf"),
            {"k": float("nan")}, [1, [float("-inf")]],
            {1: "x"}, {"a": {2: 3}}, object(), b"bytes", {1, 2}, ("a", 1),
        ]
        for bad in bad_tenants:
            with self.assertRaises(ValueError):
                self.chain.import_all_range(
                    b"", [{"tenant": bad, "expected_count": 0,
                           "expected_hash": ZERO}])
        self.assertFalse(self.path.exists())

    def test_expected_count_must_be_non_negative_plain_int(self):
        for bad in (True, False, -1, -100, 1.0, 2.5, "3", None, [], {}):
            with self.assertRaises(ValueError):
                self.chain.import_all_range(
                    b"", [{"tenant": "t", "expected_count": bad,
                           "expected_hash": ZERO}])
        self.assertFalse(self.path.exists())

    def test_expected_hash_must_be_64_lowercase_hex(self):
        for bad in ("", "0" * 63, "0" * 65, "A" * 64, "g" * 64,
                    0, None, []):
            with self.assertRaises(ValueError):
                self.chain.import_all_range(
                    b"", [{"tenant": "t", "expected_count": 0,
                           "expected_hash": bad}])
        self.assertFalse(self.path.exists())

    def test_duplicate_tenant_assertion_is_value_error(self):
        heads = [
            {"tenant": {"k": 1}, "expected_count": 0, "expected_hash": ZERO},
            {"tenant": {"k": 1}, "expected_count": 1, "expected_hash": "f" * 64},
        ]
        with self.assertRaises(ValueError):
            self.chain.import_all_range(b"", heads)
        # distinct JSON identities are fine at validation time
        ok = [
            {"tenant": 1, "expected_count": 0, "expected_hash": ZERO},
            {"tenant": 1.0, "expected_count": 0, "expected_hash": ZERO},
            {"tenant": True, "expected_count": 0, "expected_hash": ZERO},
            {"tenant": "1", "expected_count": 0, "expected_hash": ZERO},
        ]
        self.assertEqual(self.chain.import_all_range(b"", ok), [])
        self.assertFalse(self.path.exists())

    def test_empty_data_returns_empty_creates_and_reads_nothing(self):
        heads = [{"tenant": "t", "expected_count": 5,
                  "expected_hash": "f" * 64}]
        self.assertFalse(self.path.exists())
        self.assertEqual(self.chain.import_all_range(b"", heads), [])
        self.assertFalse(self.path.exists())
        # corrupt target: the no-op never reads history and leaves bytes
        self.path.write_bytes(b"\xff")
        corrupt = self.path.read_bytes()
        self.assertEqual(self.chain.import_all_range(b"", heads), [])
        self.assertEqual(self.path.read_bytes(), corrupt)

    def test_value_error_beats_target_state_and_keeps_bytes(self):
        tampered = valid_row("t", 1)
        tampered["event"] = {"tampered": True}
        self.write_rows([tampered])
        before = self.path.read_bytes()
        good_head = {"tenant": "t", "expected_count": 0, "expected_hash": ZERO}
        with self.assertRaises(ValueError):
            self.chain.import_all_range("not bytes", [good_head])
        with self.assertRaises(ValueError):
            self.chain.import_all_range(b"x", [
                {"tenant": "t", "expected_count": -1, "expected_hash": ZERO}])
        with self.assertRaises(ValueError):
            self.chain.import_all_range(b"x", [
                {"tenant": "t", "expected_count": 0, "expected_hash": "nope"}])
        self.assertEqual(self.path.read_bytes(), before)

    # --- success: interleaved segments continuing asserted heads ---

    def test_interleaved_segments_graft_onto_prefixes(self):
        src = self.make_source(("a", "b"), 6)
        # target gets a:1-2, b:1-3 plus an unrelated tenant
        self.chain.import_tenant("a", src.export_tenant_range("a", 1, 2))
        self.chain.import_tenant("b", src.export_tenant_range("b", 1, 3))
        self.chain.append("x", {"keep": True})
        rows, heads = self.tail_segments(src, {"a": 2, "b": 3})
        # interleave: a3, b4, a4, b5, a5, b6, a6
        order = [rows["a"][0], rows["b"][0], rows["a"][1], rows["b"][1],
                 rows["a"][2], rows["b"][2], rows["a"][3]]
        data = b"".join(record_bytes(r) for r in order)
        before = self.path.read_bytes()
        records = self.chain.import_all_range(data, heads)
        self.assertEqual(
            [(r["tenant"], r["seq"]) for r in records],
            [("a", 3), ("b", 4), ("a", 4), ("b", 5),
             ("a", 5), ("b", 6), ("a", 6)])
        self.assertEqual(self.path.read_bytes(), before + data)
        self.assertEqual(records, order)
        for r in records:
            self.assertEqual(set(r), {"tenant", "seq", "event", "prev", "hash"})
        self.assertEqual(self.chain.verify("a"), {"ok": True, "count": 6})
        self.assertEqual(self.chain.verify("b"), {"ok": True, "count": 6})
        self.assertEqual(self.chain.verify("x"), {"ok": True, "count": 1})
        self.assertEqual(self.chain.export_tenant("a"),
                         src.export_tenant("a"))
        self.assertEqual(self.chain.export_tenant("b"),
                         src.export_tenant("b"))
        self.assertTrue(self.chain.verify_all()["ok"])

    def test_zero_zero_assertions_create_missing_log(self):
        src = self.make_source(("c", "d"), 2)
        rows, heads = self.tail_segments(src, {"c": 0, "d": 0})
        order = [rows["c"][0], rows["d"][0], rows["c"][1], rows["d"][1]]
        data = b"".join(record_bytes(r) for r in order)
        self.assertFalse(self.path.exists())
        records = self.chain.import_all_range(data, heads)
        self.assertEqual(len(records), 4)
        self.assertEqual(self.path.read_bytes(), data)
        self.assertEqual(self.chain.verify_all(), {
            "ok": True,
            "tenants": [{"tenant": "c", "count": 2},
                        {"tenant": "d", "count": 2}],
        })

    def test_input_without_trailing_newline_is_normalized(self):
        src = self.make_source(("a",), 3)
        rows, heads = self.tail_segments(src, {"a": 0})
        data = b"".join(record_bytes(r) for r in rows["a"])
        records = self.chain.import_all_range(data[:-1], heads)
        self.assertEqual(len(records), 3)
        self.assertEqual(self.path.read_bytes(), data)

    def test_target_without_trailing_newline_gets_prefix(self):
        other = valid_row("x", 1)
        raw = json.dumps(other, sort_keys=True).encode()  # no \n
        self.path.write_bytes(raw)
        row = valid_row("t", 1)
        data = record_bytes(row)
        self.chain.import_all_range(
            data, [{"tenant": "t", "expected_count": 0,
                    "expected_hash": ZERO}])
        self.assertEqual(self.path.read_bytes(), raw + b"\n" + data)
        self.assertTrue(self.chain.verify_all()["ok"])

    def test_append_continues_from_grafted_tail(self):
        src = self.make_source(("a",), 4)
        self.chain.import_tenant("a", src.export_tenant_range("a", 1, 2))
        rows, heads = self.tail_segments(src, {"a": 2})
        data = b"".join(record_bytes(r) for r in rows["a"])
        records = self.chain.import_all_range(data, heads)
        nxt = self.chain.append("a", {"i": 99})
        self.assertEqual((nxt["seq"], nxt["prev"]),
                         (5, records[-1]["hash"]))
        self.assertEqual(self.chain.verify("a"), {"ok": True, "count": 5})

    def test_distinct_json_identities_are_independent_heads(self):
        self.chain.append(1, {"k": "int"})
        seg_int = record_bytes(valid_row(1, 2, self.chain.head(1)["hash"]))
        seg_str = record_bytes(valid_row("1", 1))
        data = seg_int + seg_str
        heads = [
            {"tenant": 1, "expected_count": 1,
             "expected_hash": self.chain.head(1)["hash"]},
            {"tenant": "1", "expected_count": 0, "expected_hash": ZERO},
        ]
        records = self.chain.import_all_range(data, heads)
        self.assertEqual([(r["tenant"], r["seq"]) for r in records],
                         [(1, 2), ("1", 1)])
        self.assertEqual(self.chain.verify(1), {"ok": True, "count": 2})
        self.assertEqual(self.chain.verify("1"), {"ok": True, "count": 1})

    # --- input contract: exact fields, chained from asserted heads ---

    def test_unlisted_tenant_record_is_value_error(self):
        good = valid_row("t", 1)
        foreign = valid_row("other", 1)
        heads = [{"tenant": "t", "expected_count": 0, "expected_hash": ZERO}]
        with self.assertRaises(ValueError):
            self.chain.import_all_range(
                record_bytes(good) + record_bytes(foreign), heads)
        self.assertFalse(self.path.exists())
        # distinct JSON identity is a different, unlisted tenant
        with self.assertRaises(ValueError):
            self.chain.import_all_range(
                record_bytes(valid_row("1", 1)),
                [{"tenant": 1, "expected_count": 0, "expected_hash": ZERO}])
        self.assertFalse(self.path.exists())

    def test_foreign_tenant_vs_chain_defect_decided_by_physical_line(self):
        heads = [{"tenant": "t", "expected_count": 0, "expected_hash": ZERO}]
        # line 1 digest defect (listed tenant) beats foreign tenant on line 2
        tampered = valid_row("t", 1)
        tampered["event"] = {"z": 9}
        foreign = valid_row("other", 1)
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.import_all_range(
                record_bytes(tampered) + record_bytes(foreign), heads)
        self.assertEqual((cm.exception.tenant, cm.exception.reason,
                          cm.exception.line), ("t", "digest", 1))
        # foreign tenant on line 1 beats a later chain defect
        seq_bad = valid_row("t", 2)
        with self.assertRaises(ValueError):
            self.chain.import_all_range(
                record_bytes(foreign) + record_bytes(seq_bad), heads)
        self.assertFalse(self.path.exists())

    def test_listed_tenant_without_records_is_missing_without_line(self):
        row = valid_row("a", 1)
        heads = [
            {"tenant": "a", "expected_count": 0, "expected_hash": ZERO},
            {"tenant": "b", "expected_count": 0, "expected_hash": ZERO},
        ]
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.import_all_range(record_bytes(row), heads)
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("b", 1, "missing", None))
        self.assertFalse(self.path.exists())

    def test_missing_and_extra_fields_are_missing(self):
        missing = json.dumps(
            {"tenant": "t", "seq": 1, "event": {}, "prev": ZERO}).encode() + b"\n"
        heads = [{"tenant": "t", "expected_count": 0, "expected_hash": ZERO}]
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.import_all_range(missing, heads)
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("t", 1, "missing", 1))
        extra = dict(valid_row("t", 1), surprise=1)
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.import_all_range(record_bytes(extra), heads)
        self.assertEqual((cm.exception.seq, cm.exception.reason,
                          cm.exception.line), (1, "missing", 1))
        self.assertFalse(self.path.exists())

    def test_first_seq_must_be_expected_count_plus_one(self):
        row = valid_row("t", 1)
        heads = [{"tenant": "t", "expected_count": 2,
                  "expected_hash": "f" * 64}]
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.import_all_range(record_bytes(row), heads)
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("t", 3, "sequence", 1))

    def test_float_spelled_seq_is_sequence(self):
        raw = (b'{"tenant":"t","seq":1.0,"event":{},"prev":"'
               + ZERO.encode() + b'","hash":"x"}\n')
        heads = [{"tenant": "t", "expected_count": 0, "expected_hash": ZERO}]
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.import_all_range(raw, heads)
        self.assertEqual((cm.exception.seq, cm.exception.reason,
                          cm.exception.line), (1, "sequence", 1))

    def test_sequence_gap_inside_segment_is_sequence(self):
        src = self.make_source(("t",), 4)
        prefix = [json.loads(l)
                  for l in src.export_tenant_range("t", 1, 2)
                  .decode().splitlines()]
        seg = [json.loads(l)
               for l in src.export_tenant_range("t", 3, 4)
               .decode().splitlines()]
        heads = [{"tenant": "t", "expected_count": 2,
                  "expected_hash": prefix[-1]["hash"]}]
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.import_all_range(record_bytes(seg[1]), heads)
        self.assertEqual((cm.exception.seq, cm.exception.reason,
                          cm.exception.line), (3, "sequence", 1))

    def test_first_prev_must_equal_expected_hash(self):
        row = valid_row("t", 3, prev="0" * 63 + "1")
        heads = [{"tenant": "t", "expected_count": 2,
                  "expected_hash": "f" * 64}]
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.import_all_range(record_bytes(row), heads)
        self.assertEqual((cm.exception.seq, cm.exception.reason,
                          cm.exception.line), (3, "digest", 1))

    def test_hash_mismatch_is_digest(self):
        row = valid_row("t", 1)
        row["event"] = {"tampered": True}
        heads = [{"tenant": "t", "expected_count": 0, "expected_hash": ZERO}]
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.import_all_range(record_bytes(row), heads)
        self.assertEqual((cm.exception.seq, cm.exception.reason,
                          cm.exception.line), (1, "digest", 1))

    def test_unparseable_blank_non_object_duplicate_nan_are_missing(self):
        good = record_bytes(valid_row("t", 1))
        heads = [{"tenant": "t", "expected_count": 0, "expected_hash": ZERO}]
        cases = [
            (b"\n", 1, None),
            (b"   \n", 1, None),
            (b"[1,2,3]\n", 1, None),
            (b"{not json\n", 1, None),
            (good + b"{bad\n", 2, None),
            (good + b"\n", 2, None),
        ]
        for raw, line, tenant in cases:
            with self.assertRaises(AuditChainStateError) as cm:
                self.chain.import_all_range(raw, heads)
            self.assertEqual((cm.exception.tenant, cm.exception.seq,
                              cm.exception.reason, cm.exception.line),
                             (tenant, None, "missing", line), raw)
        for raw in (
            b'{"tenant":"t","tenant":"t","seq":1,"event":{},'
            b'"prev":"' + ZERO.encode() + b'","hash":"x"}\n',
            b'{"tenant":"t","seq":1,"event":{"x":NaN},"prev":"'
            + ZERO.encode() + b'","hash":"x"}\n',
            b'{"tenant":"t","seq":1e999,"event":{},"prev":"'
            + ZERO.encode() + b'","hash":"x"}\n',
        ):
            with self.assertRaises(AuditChainStateError) as cm:
                self.chain.import_all_range(raw, heads)
            self.assertEqual((cm.exception.reason, cm.exception.line),
                             ("missing", 1), raw)

    def test_illegal_utf8_is_missing_at_physical_line(self):
        good = record_bytes(valid_row("t", 1))
        heads = [{"tenant": "t", "expected_count": 0, "expected_hash": ZERO}]
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.import_all_range(good + b'{"tenant":\xff}\n', heads)
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         (None, None, "missing", 2))

    def test_interleaved_tenant_defect_reports_that_tenant(self):
        a1 = valid_row("a", 3, prev="f" * 64)
        b1 = valid_row("b", 7, prev="e" * 64)
        a_bad = valid_row("a", 5, prev=a1["hash"])  # skips a seq 4
        heads = [
            {"tenant": "a", "expected_count": 2, "expected_hash": "f" * 64},
            {"tenant": "b", "expected_count": 6, "expected_hash": "e" * 64},
        ]
        raw = record_bytes(a1) + record_bytes(b1) + record_bytes(a_bad)
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.import_all_range(raw, heads)
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("a", 4, "sequence", 3))

    def test_first_physical_line_decides_across_tenants(self):
        # line 1: tenant a digest defect; line 2: tenant b sequence defect
        a = valid_row("a", 1)
        a["event"] = {"tampered": True}
        b = valid_row("b", 3)  # b asserts start at seq 3
        heads = [
            {"tenant": "a", "expected_count": 0, "expected_hash": ZERO},
            {"tenant": "b", "expected_count": 2, "expected_hash": ZERO},
        ]
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.import_all_range(record_bytes(a) + record_bytes(b),
                                        heads)
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("a", 1, "digest", 1))

    def test_input_error_priority_over_target_conflict_and_state(self):
        # target already has t records (would conflict) and a broken foreign
        # chain (would be a target state error for involved scans); the input
        # defect still wins
        self.chain.append("t", {})
        tampered = valid_row("z", 1)
        tampered["event"] = {"x": 1}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(tampered, sort_keys=True) + "\n")
        before = self.path.read_bytes()
        heads = [{"tenant": "t", "expected_count": 0, "expected_hash": ZERO}]
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.import_all_range(record_bytes(valid_row("t", 2)),
                                        heads)
        self.assertEqual((cm.exception.seq, cm.exception.reason),
                         (1, "sequence"))
        self.assertEqual(self.path.read_bytes(), before)

    def test_input_failure_writes_no_bytes(self):
        self.chain.append("other", {"keep": True})
        before = self.path.read_bytes()
        a1 = valid_row("a", 1)
        bad = valid_row("a", 3, prev=a1["hash"])
        heads = [{"tenant": "a", "expected_count": 0, "expected_hash": ZERO}]
        with self.assertRaises(AuditChainStateError):
            self.chain.import_all_range(record_bytes(a1) + record_bytes(bad),
                                        heads)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(self.chain.verify("a"), {"ok": True, "count": 0})

    # --- target side: state error priority, conflicts, creation rule ---

    def test_corrupt_target_state_error_beats_conflict(self):
        tampered = valid_row("t", 1)
        tampered["event"] = {"tampered": True}
        self.write_rows([tampered])
        before = self.path.read_bytes()
        src = self.make_source(("t",), 3, name="s.jsonl")
        seg = [json.loads(l)
               for l in src.export_tenant_range("t", 2, 2)
               .decode().splitlines()]
        heads = [{"tenant": "t", "expected_count": 1,
                  "expected_hash": seg[0]["prev"]}]
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.import_all_range(record_bytes(seg[0]), heads)
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("t", 1, "digest", 1))
        self.assertEqual(self.path.read_bytes(), before)

    def test_target_scan_picks_first_physical_line_across_chains(self):
        # a broken at file line 1, b broken at file line 3: line 1 wins no
        # matter the expected_heads order
        bad_a = valid_row("a", 1)
        bad_a["event"] = {"x": 1}
        good_b = valid_row("b", 1)
        bad_b = valid_row("b", 2, prev="f" * 64)
        self.write_rows([bad_a, good_b, bad_b])
        before = self.path.read_bytes()
        heads = [
            {"tenant": "b", "expected_count": 2, "expected_hash": good_b["hash"]},
            {"tenant": "a", "expected_count": 1, "expected_hash": bad_a["hash"]},
        ]
        seg_b = record_bytes(valid_row("b", 3, prev=good_b["hash"]))
        seg_a = record_bytes(valid_row("a", 2, prev=bad_a["hash"]))
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.import_all_range(seg_b + seg_a, heads)
        self.assertEqual((cm.exception.tenant, cm.exception.reason,
                          cm.exception.line), ("a", "digest", 1))
        self.assertEqual(self.path.read_bytes(), before)

    def test_uninvolved_tenant_corruption_is_ignored(self):
        bad = valid_row("z", 1)
        bad["event"] = {"x": 1}
        self.write_rows([bad])
        row = valid_row("t", 1)
        records = self.chain.import_all_range(
            record_bytes(row),
            [{"tenant": "t", "expected_count": 0, "expected_hash": ZERO}])
        self.assertEqual(len(records), 1)
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": 1})

    def test_head_mismatch_conflicts_with_actual_values(self):
        items = [self.chain.append("t", {"i": i}) for i in range(3)]
        before = self.path.read_bytes()
        # right count, wrong hash
        heads = [{"tenant": "t", "expected_count": 3,
                  "expected_hash": "f" * 64}]
        with self.assertRaises(AuditChainConflictError) as cm:
            self.chain.import_all_range(
                record_bytes(valid_row(
                    "t", 4, prev="f" * 64, event={})), heads)
        e = cm.exception
        self.assertEqual(e.reason, "conflict")
        self.assertEqual((e.tenant, e.expected_count, e.expected_hash),
                         ("t", 3, "f" * 64))
        self.assertEqual((e.actual_count, e.actual_hash),
                         (3, items[2]["hash"]))
        # wrong count: a segment starting at seq 1 anchored at the empty
        # head, while the target already holds 3 records
        seg1 = record_bytes(valid_row("t", 1, event={"i": 0}))
        with self.assertRaises(AuditChainConflictError) as cm:
            self.chain.import_all_range(seg1, [
                {"tenant": "t", "expected_count": 0, "expected_hash": ZERO}])
        self.assertEqual((cm.exception.actual_count,
                          cm.exception.actual_hash),
                         (3, items[2]["hash"]))
        self.assertEqual(self.path.read_bytes(), before)

    def test_conflicts_compared_in_expected_heads_order_not_input_order(self):
        self.chain.append("a", {})
        self.chain.append("b", {})
        # input physical order: a then b; assertions name b first
        rows = [valid_row("a", 1), valid_row("b", 1)]
        data = b"".join(record_bytes(r) for r in rows)
        heads = [
            {"tenant": "b", "expected_count": 0, "expected_hash": ZERO},
            {"tenant": "a", "expected_count": 0, "expected_hash": ZERO},
        ]
        with self.assertRaises(AuditChainConflictError) as cm:
            self.chain.import_all_range(data, heads)
        self.assertEqual(cm.exception.tenant, "b")
        self.assertEqual((cm.exception.expected_count,
                          cm.exception.expected_hash), (0, ZERO))
        self.assertEqual(cm.exception.actual_count, 1)

    def test_unknown_tenant_non_empty_assertion_conflicts_zero_head(self):
        self.chain.append("x", {})
        before = self.path.read_bytes()
        # segment continuing from a non-empty head that the target lacks
        row = valid_row("t", 2, prev="f" * 64)
        with self.assertRaises(AuditChainConflictError) as cm:
            self.chain.import_all_range(
                record_bytes(row),
                [{"tenant": "t", "expected_count": 1,
                  "expected_hash": "f" * 64}])
        self.assertEqual((cm.exception.expected_count,
                          cm.exception.expected_hash), (1, "f" * 64))
        self.assertEqual((cm.exception.actual_count,
                          cm.exception.actual_hash), (0, ZERO))
        self.assertEqual(self.path.read_bytes(), before)

    def test_missing_target_non_zero_assertion_leaves_no_file(self):
        row = valid_row("t", 3, prev="f" * 64)
        heads = [{"tenant": "t", "expected_count": 2,
                  "expected_hash": "f" * 64}]
        self.assertFalse(self.path.exists())
        with self.assertRaises(AuditChainConflictError) as cm:
            self.chain.import_all_range(record_bytes(row), heads)
        self.assertEqual((cm.exception.expected_count,
                          cm.exception.expected_hash), (2, "f" * 64))
        self.assertEqual((cm.exception.actual_count,
                          cm.exception.actual_hash), (0, ZERO))
        self.assertFalse(self.path.exists())

    def test_missing_target_requires_all_zero_assertions(self):
        # one of two assertions non-empty: first mismatch in list order, no
        # file even though the other assertion is (0, ZERO)
        rows = record_bytes(valid_row("a", 1)) + record_bytes(
            valid_row("b", 3, prev="f" * 64))
        heads = [
            {"tenant": "a", "expected_count": 0, "expected_hash": ZERO},
            {"tenant": "b", "expected_count": 2, "expected_hash": "f" * 64},
        ]
        with self.assertRaises(AuditChainConflictError) as cm:
            self.chain.import_all_range(rows, heads)
        self.assertEqual(cm.exception.tenant, "b")
        self.assertEqual((cm.exception.actual_count,
                          cm.exception.actual_hash), (0, ZERO))
        self.assertFalse(self.path.exists())

    def test_failed_import_keeps_log_appendable(self):
        self.chain.append("t", {"i": 0})
        before = self.path.read_bytes()
        with self.assertRaises(AuditChainConflictError):
            self.chain.import_all_range(
                record_bytes(valid_row("t", 1)),
                [{"tenant": "t", "expected_count": 0,
                  "expected_hash": ZERO}])
        self.assertEqual(self.path.read_bytes(), before)
        nxt = self.chain.append("t", {"i": 1})
        self.assertEqual(nxt["seq"], 2)


class ImportAllRangeConcurrencyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.path = Path(self.tmp.name) / "audit.jsonl"
        self.chain = AuditChain(self.path)
        src = AuditChain(self.path.with_name("source.jsonl"))
        for i in range(10):
            src.append("a", {"i": i})
            src.append("b", {"i": i})
        # target prefixes a:1-4, b:1-6
        self.chain.import_tenant("a", src.export_tenant_range("a", 1, 4))
        self.chain.import_tenant("b", src.export_tenant_range("b", 1, 6))
        ra = [json.loads(l)
              for l in src.export_tenant_range("a", 5).decode().splitlines()]
        rb = [json.loads(l)
              for l in src.export_tenant_range("b", 7).decode().splitlines()]
        order = []
        for i in range(max(len(ra), len(rb))):
            if i < len(ra):
                order.append(ra[i])
            if i < len(rb):
                order.append(rb[i])
        self.data = b"".join(record_bytes(r) for r in order)
        self.heads = [
            {"tenant": "a", "expected_count": 4,
             "expected_hash": ra[0]["prev"]},
            {"tenant": "b", "expected_count": 6,
             "expected_hash": rb[0]["prev"]},
        ]
        self.tail = {"a": ra[-1]["hash"], "b": rb[-1]["hash"]}

    def tearDown(self):
        self.tmp.cleanup()

    def test_parallel_commits_one_wins_rest_conflict(self):
        successes, conflicts, others = [], [], []
        box = threading.Lock()

        def importer():
            try:
                records = self.chain.import_all_range(self.data, self.heads)
                with box:
                    successes.append(len(records))
            except AuditChainConflictError as e:
                with box:
                    conflicts.append((e.tenant, e.expected_count,
                                      e.expected_hash, e.actual_count,
                                      e.actual_hash))
            except Exception as e:  # noqa: BLE001
                with box:
                    others.append(repr(e))

        threads = [threading.Thread(target=importer) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(others, [])
        self.assertEqual(successes, [10])
        self.assertEqual(len(conflicts), 7)
        expected_by_tenant = {h["tenant"]: (h["expected_count"],
                                            h["expected_hash"])
                              for h in self.heads}
        for tenant, ec, eh, ac, ah in conflicts:
            self.assertEqual((ec, eh), expected_by_tenant[tenant])
            self.assertEqual((ac, ah), (10, self.tail[tenant]))
        self.assertEqual(self.chain.verify("a"), {"ok": True, "count": 10})
        self.assertEqual(self.chain.verify("b"), {"ok": True, "count": 10})

    def test_readers_see_only_pre_or_post_snapshot(self):
        stop = threading.Event()
        problems = []
        box = threading.Lock()

        def reader():
            while not stop.is_set():
                r = self.chain.verify_all()
                if not r["ok"]:
                    with box:
                        problems.append(("verify_all", r))
                    return
                counts = {t["tenant"]: t["count"] for t in r["tenants"]}
                # a/b must be at the prefix count or the full count 10
                for tenant, prefix in (("a", 4), ("b", 6)):
                    if counts.get(tenant, 0) not in (prefix, 10):
                        with box:
                            problems.append(("partial", tenant, counts))
                        return

        readers = [threading.Thread(target=reader) for _ in range(4)]
        for t in readers:
            t.start()

        def importer():
            try:
                self.chain.import_all_range(self.data, self.heads)
            except AuditChainConflictError:
                pass

        importers = [threading.Thread(target=importer) for _ in range(6)]
        for t in importers:
            t.start()
        for t in importers:
            t.join()
        stop.set()
        for t in readers:
            t.join(timeout=2)

        self.assertEqual(problems, [])
        self.assertEqual(self.chain.verify("a"), {"ok": True, "count": 10})
        self.assertEqual(self.chain.verify("b"), {"ok": True, "count": 10})
        self.assertTrue(self.chain.verify_all()["ok"])


if __name__ == "__main__":
    unittest.main()
