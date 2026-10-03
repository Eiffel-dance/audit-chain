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


class ImportTenantRangeTest(unittest.TestCase):
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

    def make_source(self, tenant="t", n=6, name="source.jsonl"):
        src = AuditChain(self.path.with_name(name))
        for i in range(n):
            src.append(tenant, {"i": i, "s": "审计"})
        return src

    def valid_row(self, tenant="t", seq=1, prev=ZERO, event=None):
        row = {"tenant": tenant, "seq": seq, "event": event or {}, "prev": prev}
        row["hash"] = AuditChain._hash(row)
        return row

    # --- boundary: validation order, empty no-op, no file/no byte touched ---

    def test_data_must_be_bytes(self):
        for bad in ("", "not jsonl", bytearray(b""), 1, None, [], object()):
            with self.assertRaises(ValueError):
                self.chain.import_tenant_range("t", bad, 0, ZERO)
        self.assertFalse(self.path.exists())

    def test_illegal_tenant_is_value_error(self):
        bad_tenants = [
            float("nan"), float("inf"), float("-inf"),
            {"k": float("nan")}, [1, [float("-inf")]],
            {1: "x"}, {"a": {2: 3}}, object(), b"bytes", {1, 2}, ("a", 1),
        ]
        for bad in bad_tenants:
            with self.assertRaises(ValueError):
                self.chain.import_tenant_range(bad, b"", 0, ZERO)
        self.assertFalse(self.path.exists())

    def test_expected_count_must_be_non_negative_plain_int(self):
        for bad in (True, False, -1, -100, 1.0, 2.5, "3", None, [], {}):
            with self.assertRaises(ValueError):
                self.chain.import_tenant_range("t", b"", bad, ZERO)
        self.assertFalse(self.path.exists())

    def test_expected_hash_must_be_64_lowercase_hex(self):
        for bad in ("", "0" * 63, "0" * 65, "A" * 64, "g" * 64,
                    0, None, []):
            with self.assertRaises(ValueError):
                self.chain.import_tenant_range("t", b"", 0, bad)
        self.assertFalse(self.path.exists())

    def test_empty_data_returns_empty_creates_and_reads_nothing(self):
        self.assertFalse(self.path.exists())
        self.assertEqual(self.chain.import_tenant_range("t", b"", 0, ZERO), [])
        self.assertFalse(self.path.exists())
        # corrupt target: the no-op never reads history and leaves bytes
        self.path.write_bytes(b"\xff")
        corrupt = self.path.read_bytes()
        self.assertEqual(self.chain.import_tenant_range("t", b"", 5, "f" * 64),
                         [])
        self.assertEqual(self.path.read_bytes(), corrupt)

    def test_value_error_beats_target_state_and_keeps_bytes(self):
        row = self.valid_row("t", 1)
        row["event"] = {"tampered": True}
        self.write_rows([row])
        before = self.path.read_bytes()
        with self.assertRaises(ValueError):
            self.chain.import_tenant_range("t", "not bytes", 0, ZERO)
        with self.assertRaises(ValueError):
            self.chain.import_tenant_range("t", b"x", -1, ZERO)
        with self.assertRaises(ValueError):
            self.chain.import_tenant_range("t", b"x", 0, "nope")
        self.assertEqual(self.path.read_bytes(), before)

    # --- success: continuing a chain from an asserted head ---

    def test_import_segment_continues_existing_chain(self):
        src = self.make_source("t", 6)
        head_seg = src.export_tenant_range("t", 1, 3)
        tail_seg = src.export_tenant_range("t", 4, 6)
        head_rows = [json.loads(l) for l in head_seg.decode().splitlines()]
        # graft the head onto the empty log, then the tail from that head
        self.chain.import_tenant_range("t", head_seg, 0, ZERO)
        records = self.chain.import_tenant_range(
            "t", tail_seg, 3, head_rows[-1]["hash"])
        self.assertEqual([r["seq"] for r in records], [4, 5, 6])
        self.assertEqual(records[0]["prev"], head_rows[-1]["hash"])
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": 6})
        self.assertEqual(self.chain.export_tenant("t"),
                         src.export_tenant("t"))

    def test_import_whole_chain_with_zero_assertion_creates_log(self):
        src = self.make_source("t", 4)
        data = src.export_tenant_range("t", 1, 4)
        self.assertFalse(self.path.exists())
        records = self.chain.import_tenant_range("t", data, 0, ZERO)
        self.assertEqual([r["seq"] for r in records], [1, 2, 3, 4])
        self.assertEqual(self.path.read_bytes(), data)
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": 4})
        self.assertEqual(self.chain.verify_all(),
                         {"ok": True, "tenants": [{"tenant": "t", "count": 4}]})

    def test_import_open_tail_segment_and_append_continues(self):
        src = self.make_source("t", 5)
        head = src.export_tenant_range("t", 1, 2)
        rest = src.export_tenant_range("t", 3)  # open tail
        self.chain.import_tenant_range("t", head, 0, ZERO)
        h = self.chain.head("t")
        records = self.chain.import_tenant_range(
            "t", rest, h["count"], h["hash"])
        self.assertEqual([r["seq"] for r in records], [3, 4, 5])
        # a plain append continues the grafted chain without gaps
        nxt = self.chain.append("t", {"i": 99})
        self.assertEqual((nxt["seq"], nxt["prev"]),
                         (6, records[-1]["hash"]))
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": 6})

    def test_import_preserves_values_verbatim_and_coexists_with_others(self):
        for i in range(2):
            self.chain.append("a", {"i": i})
        src = self.make_source("t", 5)
        head_seg = src.export_tenant_range("t", 1, 4)
        tail_seg = src.export_tenant_range("t", 5, 5)
        self.chain.import_tenant_range("t", head_seg, 0, ZERO)
        h = self.chain.head("t")
        before = self.path.read_bytes()
        records = self.chain.import_tenant_range(
            "t", tail_seg, h["count"], h["hash"])
        src_rows = [json.loads(l) for l in tail_seg.decode().splitlines()]
        self.assertEqual(records, src_rows)
        self.assertEqual(self.path.read_bytes(), before + tail_seg)
        self.assertEqual(self.chain.verify("a"), {"ok": True, "count": 2})
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": 5})
        self.assertTrue(self.chain.verify_all()["ok"])

    def test_import_after_target_without_trailing_newline(self):
        other = self.valid_row("a", 1)
        raw = json.dumps(other, sort_keys=True).encode()  # no \n
        self.path.write_bytes(raw)
        src = self.make_source("t", 2)
        data = src.export_tenant_range("t", 1, 2)
        self.chain.import_tenant_range("t", data, 0, ZERO)
        self.assertEqual(self.path.read_bytes(), raw + b"\n" + data)
        self.assertTrue(self.chain.verify_all()["ok"])

    def test_distinct_json_identities_have_independent_heads(self):
        self.chain.append(1, {"k": "int"})
        row = self.valid_row("1", 1, event={"k": "str"})
        self.chain.import_tenant_range("1", record_bytes(row), 0, ZERO)
        self.assertEqual(self.chain.verify(1), {"ok": True, "count": 1})
        self.assertEqual(self.chain.verify("1"), {"ok": True, "count": 1})

    # --- input contract: single tenant, exactly five fields, chained ---

    def test_foreign_tenant_record_is_value_error(self):
        good = record_bytes(self.valid_row("t", 1))
        foreign = record_bytes(self.valid_row("other", 1))
        with self.assertRaises(ValueError):
            self.chain.import_tenant_range("t", good + foreign, 0, ZERO)
        self.assertFalse(self.path.exists())

    def test_extra_field_is_missing_at_physical_line(self):
        row = self.valid_row("t", 1)
        row["extra"] = True
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.import_tenant_range("t", record_bytes(row), 0, ZERO)
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("t", 1, "missing", 1))
        self.assertFalse(self.path.exists())

    def test_missing_field_is_missing(self):
        raw = json.dumps(
            {"tenant": "t", "seq": 1, "event": {}, "prev": ZERO}).encode() + b"\n"
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.import_tenant_range("t", raw, 0, ZERO)
        self.assertEqual((cm.exception.reason, cm.exception.line),
                         ("missing", 1))

    def test_seq_must_start_at_expected_count_plus_one(self):
        row = self.valid_row("t", 1)  # seq 1 where seq 3 is expected
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.import_tenant_range(
                "t", record_bytes(row), 2, "f" * 64)
        self.assertEqual((cm.exception.seq, cm.exception.reason,
                          cm.exception.line), (3, "sequence", 1))
        # gap inside the segment
        src = self.make_source("t", 4)
        seg_rows = [json.loads(l)
                    for l in src.export_tenant_range("t", 3, 4)
                    .decode().splitlines()]
        dropped = record_bytes(seg_rows[1])  # only seq 4, seq 3 missing
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.import_tenant_range(
                "t", dropped, 2, seg_rows[0]["prev"])
        self.assertEqual((cm.exception.seq, cm.exception.reason,
                          cm.exception.line), (3, "sequence", 1))

    def test_first_prev_must_equal_expected_hash(self):
        row = self.valid_row("t", 3, prev="0" * 63 + "1")
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.import_tenant_range(
                "t", record_bytes(row), 2, "f" * 64)
        self.assertEqual((cm.exception.seq, cm.exception.reason,
                          cm.exception.line), (3, "digest", 1))

    def test_hash_mismatch_is_digest(self):
        row = self.valid_row("t", 1)
        row["event"] = {"tampered": True}
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.import_tenant_range(
                "t", record_bytes(row), 0, ZERO)
        self.assertEqual((cm.exception.seq, cm.exception.reason,
                          cm.exception.line), (1, "digest", 1))

    def test_unparseable_blank_and_bad_utf8_lines_are_missing(self):
        good = record_bytes(self.valid_row("t", 1))
        for raw, line in ((b"\n", 1), (b"{not json\n", 1),
                          (good + b"{bad\n", 2), (good + b"\xff", 2)):
            with self.assertRaises(AuditChainStateError) as cm:
                self.chain.import_tenant_range("t", raw, 0, ZERO)
            self.assertEqual((cm.exception.reason, cm.exception.line),
                             ("missing", line), raw)

    def test_input_state_error_writes_no_partial_records(self):
        self.chain.append("other", {"keep": True})
        before = self.path.read_bytes()
        src = self.make_source("t", 3)
        rows = [json.loads(l)
                for l in src.export_tenant_range("t", 2, 3)
                .decode().splitlines()]
        rows[1]["event"] = {"tampered": True}  # break second record's hash
        data = record_bytes(rows[0]) + record_bytes(rows[1])
        with self.assertRaises(AuditChainStateError):
            self.chain.import_tenant_range("t", data, 1, rows[0]["prev"])
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": 0})

    # --- target side: state error priority, conflict, creation rule ---

    def test_corrupt_target_raises_state_error_before_conflict(self):
        tampered = self.valid_row("t", 1)
        tampered["event"] = {"tampered": True}
        self.write_rows([tampered])
        before = self.path.read_bytes()
        src = self.make_source("t", 2)
        src_rows = [json.loads(l)
                    for l in src.export_tenant("t").decode().splitlines()]
        data = src.export_tenant_range("t", 2, 2)
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.import_tenant_range("t", data, 1, src_rows[0]["hash"])
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("t", 1, "digest", 1))
        self.assertEqual(self.path.read_bytes(), before)

    def test_head_mismatch_conflicts_with_actual_values(self):
        # target and source both have 3-record prefixes but divergent events,
        # so the source's tail hash is a well-formed assertion that does not
        # match the target's actual head
        items = [self.chain.append("t", {"i": i}) for i in range(3)]
        before = self.path.read_bytes()
        src = self.make_source("t", 5)
        src_rows = [json.loads(l)
                    for l in src.export_tenant("t").decode().splitlines()]
        # right count, wrong hash: segment [4, 5] anchored at source head 3
        data = src.export_tenant_range("t", 4, 5)
        with self.assertRaises(AuditChainConflictError) as cm:
            self.chain.import_tenant_range(
                "t", data, 3, src_rows[2]["hash"])
        e = cm.exception
        self.assertEqual(e.reason, "conflict")
        self.assertEqual((e.expected_count, e.expected_hash),
                         (3, src_rows[2]["hash"]))
        self.assertEqual((e.actual_count, e.actual_hash),
                         (3, items[2]["hash"]))
        # wrong count: segment [3, 5] anchored at source head 2
        data = src.export_tenant_range("t", 3, 5)
        with self.assertRaises(AuditChainConflictError) as cm:
            self.chain.import_tenant_range(
                "t", data, 2, src_rows[1]["hash"])
        self.assertEqual((cm.exception.actual_count,
                          cm.exception.actual_hash),
                         (3, items[2]["hash"]))
        self.assertEqual(self.path.read_bytes(), before)

    def test_missing_target_only_zero_zero_assertion_creates(self):
        src = self.make_source("t", 4)
        src_rows = [json.loads(l)
                    for l in src.export_tenant("t").decode().splitlines()]
        data = src.export_tenant_range("t", 2, 4)
        self.assertFalse(self.path.exists())
        with self.assertRaises(AuditChainConflictError) as cm:
            self.chain.import_tenant_range(
                "t", data, 1, src_rows[0]["hash"])
        e = cm.exception
        self.assertEqual((e.expected_count, e.expected_hash),
                         (1, src_rows[0]["hash"]))
        self.assertEqual((e.actual_count, e.actual_hash), (0, ZERO))
        self.assertFalse(self.path.exists())  # losing assertion creates nothing

    def test_conflict_on_unknown_tenant_reports_zero_head(self):
        self.chain.append("a", {})
        before = self.path.read_bytes()
        src = self.make_source("t", 3)
        src_rows = [json.loads(l)
                    for l in src.export_tenant("t").decode().splitlines()]
        data = src.export_tenant_range("t", 2, 3)
        with self.assertRaises(AuditChainConflictError) as cm:
            self.chain.import_tenant_range(
                "t", data, 1, src_rows[0]["hash"])
        self.assertEqual((cm.exception.actual_count,
                          cm.exception.actual_hash), (0, ZERO))
        self.assertEqual(self.path.read_bytes(), before)

    def test_failed_import_leaves_log_appendable(self):
        self.chain.append("t", {"i": 0})
        src = self.make_source("t", 3)
        src_rows = [json.loads(l)
                    for l in src.export_tenant("t").decode().splitlines()]
        data = src.export_tenant_range("t", 2, 3)
        with self.assertRaises(AuditChainConflictError):
            self.chain.import_tenant_range(
                "t", data, 1, src_rows[0]["hash"])
        nxt = self.chain.append("t", {"i": 1})
        self.assertEqual(nxt["seq"], 2)
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": 2})


class ImportRangeConcurrencyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.path = Path(self.tmp.name) / "audit.jsonl"
        self.chain = AuditChain(self.path)
        src = AuditChain(self.path.with_name("source.jsonl"))
        for i in range(10):
            src.append("t", {"i": i})
        self.head_seg = src.export_tenant_range("t", 1, 4)
        self.tail_seg = src.export_tenant_range("t", 5, 10)
        head_rows = [json.loads(l)
                     for l in self.head_seg.decode().splitlines()]
        self.head_count = 4
        self.head_hash = head_rows[-1]["hash"]
        self.tail_hash = json.loads(
            self.tail_seg.decode().splitlines()[-1])["hash"]

    def tearDown(self):
        self.tmp.cleanup()

    def test_parallel_range_imports_one_wins_rest_conflict(self):
        self.chain.import_tenant_range("t", self.head_seg, 0, ZERO)
        successes, conflicts, others = [], [], []
        box = threading.Lock()

        def importer():
            try:
                records = self.chain.import_tenant_range(
                    "t", self.tail_seg, self.head_count, self.head_hash)
                with box:
                    successes.append(len(records))
            except AuditChainConflictError as e:
                with box:
                    conflicts.append((e.expected_count, e.expected_hash,
                                      e.actual_count, e.actual_hash))
            except Exception as e:  # noqa: BLE001
                with box:
                    others.append(repr(e))

        threads = [threading.Thread(target=importer) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(others, [])
        self.assertEqual(successes, [6])
        self.assertEqual(len(conflicts), 7)
        # every loser observed the winner's real post-commit head
        for exp_count, exp_hash, act_count, act_hash in conflicts:
            self.assertEqual((exp_count, exp_hash),
                             (self.head_count, self.head_hash))
            self.assertEqual((act_count, act_hash), (10, self.tail_hash))
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": 10})

    def test_parallel_creating_imports_one_wins(self):
        successes, conflicts, others = [], [], []
        box = threading.Lock()

        def importer():
            try:
                self.chain.import_tenant_range("t", self.head_seg, 0, ZERO)
                with box:
                    successes.append(1)
            except AuditChainConflictError as e:
                with box:
                    conflicts.append((e.actual_count, e.actual_hash))
            except Exception as e:  # noqa: BLE001
                with box:
                    others.append(repr(e))

        threads = [threading.Thread(target=importer) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(others, [])
        self.assertEqual(len(successes), 1)
        self.assertEqual(len(conflicts), 5)
        for act_count, act_hash in conflicts:
            self.assertEqual((act_count, act_hash),
                             (self.head_count, self.head_hash))
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": 4})


if __name__ == "__main__":
    unittest.main()
