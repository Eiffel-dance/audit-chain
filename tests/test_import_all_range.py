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

    def valid_row(self, tenant="t", seq=1, prev=ZERO, event=None):
        row = {"tenant": tenant, "seq": seq, "event": event or {}, "prev": prev}
        row["hash"] = AuditChain._hash(row)
        return row

    def make_source(self, spec, name="source.jsonl"):
        # spec: list of (tenant, event) appended in order to a fresh source
        src = AuditChain(self.path.with_name(name))
        for tenant, event in spec:
            src.append(tenant, event)
        return src

    def heads_of(self, chain, tenants):
        return {t["tenant"]: t for t in chain.heads()["tenants"]
                if t["tenant"] in tenants}

    # --- boundary: parameter validation before anything else ---

    def test_data_must_be_bytes(self):
        heads = [{"tenant": "t", "expected_count": 0, "expected_hash": ZERO}]
        for bad in ("", "not jsonl", bytearray(b""), 1, None, [], object()):
            with self.assertRaises(ValueError):
                self.chain.import_all_range(bad, heads)
        self.assertFalse(self.path.exists())

    def test_expected_heads_must_be_a_list(self):
        for bad in ("x", 1, None, {}, object(),
                    {"tenant": "t", "expected_count": 0,
                     "expected_hash": ZERO}):
            with self.assertRaises(ValueError):
                self.chain.import_all_range(b"", bad)
        self.assertFalse(self.path.exists())

    def test_entry_must_have_exactly_the_three_keys(self):
        good = {"tenant": "t", "expected_count": 0, "expected_hash": ZERO}
        bad_entries = [
            "x", 1, None, [],
            {},
            {"tenant": "t", "expected_count": 0},
            dict(good, extra=1),
            {"tenant": "t", "expected_count": 0, "expected_hash": ZERO,
             "events": []},
        ]
        for bad in bad_entries:
            with self.assertRaises(ValueError):
                self.chain.import_all_range(b"", [bad])
        self.assertFalse(self.path.exists())

    def test_illegal_tenant_is_value_error(self):
        bad_tenants = [
            float("nan"), float("inf"), float("-inf"),
            {"k": float("nan")}, [1, [float("-inf")]],
            {1: "x"}, object(), b"bytes", {1, 2}, ("a", 1),
        ]
        for bad in bad_tenants:
            entry = {"tenant": bad, "expected_count": 0,
                     "expected_hash": ZERO}
            with self.assertRaises(ValueError):
                self.chain.import_all_range(b"", [entry])
        self.assertFalse(self.path.exists())

    def test_expected_count_must_be_non_negative_plain_int(self):
        for bad in (True, False, -1, -100, 1.0, 2.5, "3", None, [], {}):
            entry = {"tenant": "t", "expected_count": bad,
                     "expected_hash": ZERO}
            with self.assertRaises(ValueError):
                self.chain.import_all_range(b"", [entry])
        self.assertFalse(self.path.exists())

    def test_expected_hash_must_be_64_lowercase_hex(self):
        for bad in ("", "0" * 63, "0" * 65, "A" * 64, "g" * 64,
                    0, None, []):
            entry = {"tenant": "t", "expected_count": 0,
                     "expected_hash": bad}
            with self.assertRaises(ValueError):
                self.chain.import_all_range(b"", [entry])
        self.assertFalse(self.path.exists())

    def test_duplicate_tenant_assertion_is_value_error(self):
        entry = {"tenant": "t", "expected_count": 0, "expected_hash": ZERO}
        with self.assertRaises(ValueError):
            self.chain.import_all_range(b"", [entry, dict(entry)])
        # same canonical JSON identity behind a different key order
        e1 = {"tenant": {"a": 1, "b": 2}, "expected_count": 0,
              "expected_hash": ZERO}
        e2 = {"tenant": {"b": 2, "a": 1}, "expected_count": 0,
              "expected_hash": ZERO}
        with self.assertRaises(ValueError):
            self.chain.import_all_range(b"", [e1, e2])
        # distinct JSON identities are not duplicates
        heads = [{"tenant": t, "expected_count": 0, "expected_hash": ZERO}
                 for t in (1, "1", 1.0, True)]
        self.assertEqual(self.chain.import_all_range(b"", heads), [])
        self.assertFalse(self.path.exists())

    def test_empty_data_returns_empty_creates_and_reads_nothing(self):
        heads = [{"tenant": "t", "expected_count": 0, "expected_hash": ZERO}]
        self.assertFalse(self.path.exists())
        self.assertEqual(self.chain.import_all_range(b"", heads), [])
        self.assertFalse(self.path.exists())
        # empty expected_heads list is fine too
        self.assertEqual(self.chain.import_all_range(b"", []), [])
        # corrupt target: the no-op never reads history and leaves bytes
        self.path.write_bytes(b"\xff")
        corrupt = self.path.read_bytes()
        bad_heads = [{"tenant": "t", "expected_count": 5,
                      "expected_hash": "f" * 64}]
        self.assertEqual(self.chain.import_all_range(b"", bad_heads), [])
        self.assertEqual(self.path.read_bytes(), corrupt)
        # healthy target with other tenants: nothing appended
        self.path.unlink()
        self.chain.append("other", {})
        healthy = self.path.read_bytes()
        self.assertEqual(self.chain.import_all_range(b"", heads), [])
        self.assertEqual(self.path.read_bytes(), healthy)

    def test_value_error_beats_target_state_and_keeps_bytes(self):
        row = self.valid_row("t", 1)
        row["event"] = {"tampered": True}
        self.write_rows([row])
        before = self.path.read_bytes()
        with self.assertRaises(ValueError):
            self.chain.import_all_range("not bytes", [])
        with self.assertRaises(ValueError):
            self.chain.import_all_range(b"x", "not a list")
        with self.assertRaises(ValueError):
            self.chain.import_all_range(
                b"x", [{"tenant": "t", "expected_count": -1,
                        "expected_hash": ZERO}])
        self.assertEqual(self.path.read_bytes(), before)

    # --- success: segmented multi-tenant grafting ---

    def test_segments_continue_existing_chains(self):
        spec = [("a", {"i": i}) for i in range(4)]
        spec += [("b", {"i": i}) for i in range(3)]
        src = self.make_source(spec)
        # graft the 2-record prefixes first
        self.chain.import_all_range(
            src.export_tenant_range("a", 1, 2) + src.export_tenant_range("b", 1, 2),
            [{"tenant": "a", "expected_count": 0, "expected_hash": ZERO},
             {"tenant": "b", "expected_count": 0, "expected_hash": ZERO}],
        )
        heads = self.heads_of(self.chain, ("a", "b"))
        # interleaved tail segments, b listed first
        seg = src.export_tenant_range("b", 3, 3) + src.export_tenant_range("a", 3, 4)
        records = self.chain.import_all_range(seg, [
            {"tenant": "b", "expected_count": heads["b"]["count"],
             "expected_hash": heads["b"]["hash"]},
            {"tenant": "a", "expected_count": heads["a"]["count"],
             "expected_hash": heads["a"]["hash"]},
        ])
        self.assertEqual([r["tenant"] for r in records], ["b", "a", "a"])
        self.assertEqual([r["seq"] for r in records], [3, 3, 4])
        self.assertEqual(self.chain.verify("a"), {"ok": True, "count": 4})
        self.assertEqual(self.chain.verify("b"), {"ok": True, "count": 3})
        self.assertEqual(self.chain.export_tenant("a"),
                         src.export_tenant("a"))
        self.assertEqual(self.chain.export_tenant("b"),
                         src.export_tenant("b"))
        self.assertTrue(self.chain.verify_all()["ok"])

    def test_zero_zero_assertions_create_missing_log(self):
        src = self.make_source([("a", {"i": 1}), ("b", {"i": 1}),
                                ("a", {"i": 2})])
        data = src.export_all()
        self.assertFalse(self.path.exists())
        records = self.chain.import_all_range(data, [
            {"tenant": "a", "expected_count": 0, "expected_hash": ZERO},
            {"tenant": "b", "expected_count": 0, "expected_hash": ZERO},
        ])
        self.assertEqual(len(records), 3)
        self.assertEqual(self.path.read_bytes(), data)
        self.assertEqual(self.chain.export_all(), data)
        self.assertEqual(self.chain.verify_all(), {
            "ok": True,
            "tenants": [{"tenant": "a", "count": 2},
                        {"tenant": "b", "count": 1}],
        })

    def test_returned_records_physical_order_and_original_values(self):
        src = self.make_source([("b", {"k": "审计"}), ("a", [1, 2]),
                                ("b", {"n": None})])
        data = src.export_all()
        heads = [{"tenant": "a", "expected_count": 0, "expected_hash": ZERO},
                 {"tenant": "b", "expected_count": 0, "expected_hash": ZERO}]
        records = self.chain.import_all_range(data, heads)
        src_rows = [json.loads(l) for l in data.decode("utf-8").splitlines()]
        self.assertEqual(records, src_rows)
        self.assertEqual([r["tenant"] for r in records], ["b", "a", "b"])
        for r in records:
            self.assertEqual(set(r), {"tenant", "seq", "event", "prev", "hash"})
            self.assertEqual(r["hash"], AuditChain._hash(r))

    def test_import_coexists_with_other_tenants_on_target(self):
        for i in range(3):
            self.chain.append("x", {"i": i})
        target_before = self.path.read_bytes()
        src = self.make_source([("a", {"i": 1}), ("a", {"i": 2})])
        data = src.export_tenant("a")
        records = self.chain.import_all_range(data, [
            {"tenant": "a", "expected_count": 0, "expected_hash": ZERO},
        ])
        self.assertEqual(len(records), 2)
        self.assertEqual(self.path.read_bytes(), target_before + data)
        self.assertEqual(self.chain.verify("x"), {"ok": True, "count": 3})
        self.assertEqual(self.chain.verify("a"), {"ok": True, "count": 2})

    def test_import_after_target_without_trailing_newline(self):
        other = self.valid_row("x", 1)
        raw = json.dumps(other, sort_keys=True).encode("utf-8")  # no \n
        self.path.write_bytes(raw)
        src = self.make_source([("a", {"i": 1})])
        data = src.export_tenant("a")
        self.chain.import_all_range(data, [
            {"tenant": "a", "expected_count": 0, "expected_hash": ZERO},
        ])
        self.assertEqual(self.path.read_bytes(), raw + b"\n" + data)
        self.assertTrue(self.chain.verify_all()["ok"])

    def test_import_accepts_input_without_trailing_newline(self):
        src = self.make_source([("a", {"i": 1}), ("a", {"i": 2})])
        data = src.export_tenant("a")
        records = self.chain.import_all_range(data[:-1], [
            {"tenant": "a", "expected_count": 0, "expected_hash": ZERO},
        ])
        self.assertEqual(len(records), 2)
        self.assertEqual(self.path.read_bytes(), data)  # normalized with \n
        self.assertEqual(self.chain.verify("a"), {"ok": True, "count": 2})

    def test_distinct_json_identities_migrate_independently(self):
        src = self.make_source([(1, {"k": "int"}), ("1", {"k": "str"}),
                                (1.0, {"k": "float"}), (True, {})])
        data = src.export_all()
        heads = [{"tenant": t, "expected_count": 0, "expected_hash": ZERO}
                 for t in (1, "1", 1.0, True)]
        records = self.chain.import_all_range(data, heads)
        self.assertEqual(len(records), 4)
        for tenant in (1, "1", 1.0, True):
            self.assertEqual(self.chain.verify(tenant),
                             {"ok": True, "count": 1})

    def test_imported_segments_continue_with_append(self):
        src = self.make_source([("a", {"i": 1}), ("a", {"i": 2})])
        records = self.chain.import_all_range(src.export_tenant("a"), [
            {"tenant": "a", "expected_count": 0, "expected_hash": ZERO},
        ])
        nxt = self.chain.append("a", {"i": 3})
        self.assertEqual((nxt["seq"], nxt["prev"]), (3, records[-1]["hash"]))
        self.assertEqual(self.chain.verify("a"), {"ok": True, "count": 3})

    # --- input contract: listed tenants present, no unlisted tenants ---

    def test_listed_tenant_without_records_is_value_error(self):
        src = self.make_source([("a", {"i": 1})])
        data = src.export_tenant("a")
        heads = [{"tenant": "a", "expected_count": 0, "expected_hash": ZERO},
                 {"tenant": "b", "expected_count": 0, "expected_hash": ZERO}]
        with self.assertRaises(ValueError):
            self.chain.import_all_range(data, heads)
        self.assertFalse(self.path.exists())

    def test_unlisted_tenant_record_is_value_error(self):
        a = record_bytes(self.valid_row("a", 1))
        foreign = record_bytes(self.valid_row("z", 1))
        heads = [{"tenant": "a", "expected_count": 0, "expected_hash": ZERO}]
        with self.assertRaises(ValueError):
            self.chain.import_all_range(a + foreign, heads)
        # unlisted tenant on the first line decides over a later defect
        with self.assertRaises(ValueError):
            self.chain.import_all_range(foreign + b"{bad\n", heads)
        self.assertFalse(self.path.exists())

    def test_input_state_error_beats_missing_listed_tenant(self):
        # input broken and a listed tenant absent: the chain defect wins
        heads = [{"tenant": "a", "expected_count": 0, "expected_hash": ZERO},
                 {"tenant": "b", "expected_count": 0, "expected_hash": ZERO}]
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.import_all_range(
                record_bytes(self.valid_row("a", 2)), heads)
        self.assertEqual(cm.exception.reason, "sequence")
        self.assertFalse(self.path.exists())

    # --- input chain state errors: AuditChainStateError, nothing written ---

    def test_input_not_starting_at_expected_count_plus_one_is_sequence(self):
        row = self.valid_row("t", 1)  # seq 1 where seq 3 is expected
        heads = [{"tenant": "t", "expected_count": 2,
                  "expected_hash": "f" * 64}]
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.import_all_range(record_bytes(row), heads)
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("t", 3, "sequence", 1))
        self.assertFalse(self.path.exists())

    def test_input_first_prev_not_expected_hash_is_digest(self):
        row = self.valid_row("t", 3, prev=ZERO)
        heads = [{"tenant": "t", "expected_count": 2,
                  "expected_hash": "f" * 64}]
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.import_all_range(record_bytes(row), heads)
        self.assertEqual((cm.exception.seq, cm.exception.reason,
                          cm.exception.line), (3, "digest", 1))

    def test_input_hash_mismatch_is_digest(self):
        row = self.valid_row("t", 1)
        row["event"] = {"tampered": True}
        heads = [{"tenant": "t", "expected_count": 0, "expected_hash": ZERO}]
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.import_all_range(record_bytes(row), heads)
        self.assertEqual((cm.exception.seq, cm.exception.reason,
                          cm.exception.line), (1, "digest", 1))

    def test_input_missing_and_extra_fields_are_missing(self):
        heads = [{"tenant": "t", "expected_count": 0, "expected_hash": ZERO}]
        missing = json.dumps(
            {"tenant": "t", "seq": 1, "event": {}, "prev": ZERO}).encode() + b"\n"
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.import_all_range(missing, heads)
        self.assertEqual((cm.exception.seq, cm.exception.reason,
                          cm.exception.line), (1, "missing", 1))
        extra = dict(self.valid_row("t", 1), surprise=1)
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.import_all_range(record_bytes(extra), heads)
        self.assertEqual((cm.exception.seq, cm.exception.reason,
                          cm.exception.line), (1, "missing", 1))
        self.assertFalse(self.path.exists())

    def test_input_unparseable_blank_or_non_object_lines_are_missing(self):
        heads = [{"tenant": "t", "expected_count": 0, "expected_hash": ZERO}]
        good = record_bytes(self.valid_row("t", 1))
        cases = [
            (b"\n", 1),
            (b"   \n", 1),
            (b"[1,2,3]\n", 1),
            (b"{not json\n", 1),
            (good + b"{bad\n", 2),
            (good + b"\n", 2),
        ]
        for raw, line in cases:
            with self.assertRaises(AuditChainStateError) as cm:
                self.chain.import_all_range(raw, heads)
            self.assertEqual((cm.exception.tenant, cm.exception.seq,
                              cm.exception.reason, cm.exception.line),
                             (None, None, "missing", line), raw)
        self.assertFalse(self.path.exists())

    def test_input_duplicate_keys_and_non_standard_numbers_are_missing(self):
        heads = [{"tenant": "t", "expected_count": 0, "expected_hash": ZERO}]
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

    def test_input_illegal_utf8_is_missing_at_physical_line(self):
        heads = [{"tenant": "t", "expected_count": 0, "expected_hash": ZERO}]
        raw = record_bytes(self.valid_row("t", 1)) + b'{"tenant":\xff}\n'
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.import_all_range(raw, heads)
        self.assertEqual((cm.exception.reason, cm.exception.line),
                         ("missing", 2))

    def test_input_interleaved_tenant_defect_reports_that_tenant(self):
        a1 = self.valid_row("a", 1)
        b1 = self.valid_row("b", 1)
        a3 = self.valid_row("a", 3, prev=a1["hash"])  # a skips seq 2
        raw = record_bytes(a1) + record_bytes(b1) + record_bytes(a3)
        heads = [{"tenant": "a", "expected_count": 0, "expected_hash": ZERO},
                 {"tenant": "b", "expected_count": 0, "expected_hash": ZERO}]
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.import_all_range(raw, heads)
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("a", 2, "sequence", 3))
        self.assertFalse(self.path.exists())

    def test_input_error_priority_matches_physical_order(self):
        heads = [{"tenant": "t", "expected_count": 0, "expected_hash": ZERO}]
        # digest on line 1 beats bad bytes on line 2
        tampered = self.valid_row("t", 1)
        tampered["event"] = {"z": 9}
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.import_all_range(
                record_bytes(tampered) + b"\xff\n", heads)
        self.assertEqual((cm.exception.seq, cm.exception.reason,
                          cm.exception.line), (1, "digest", 1))
        # an earlier line of an unlisted tenant (ValueError) beats a later
        # chain defect of a listed tenant
        foreign = record_bytes(self.valid_row("z", 1))
        broken = record_bytes(self.valid_row("t", 5))
        with self.assertRaises(ValueError):
            self.chain.import_all_range(foreign + broken, heads)

    def test_input_state_error_takes_priority_over_target_conflict(self):
        self.chain.append("t", {})  # would conflict, but input is broken
        heads = [{"tenant": "t", "expected_count": 1,
                  "expected_hash": self.chain.head("t")["hash"]}]
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.import_all_range(
                record_bytes(self.valid_row("t", 3)), heads)
        self.assertEqual(cm.exception.reason, "sequence")

    def test_input_state_error_writes_no_partial_records(self):
        self.chain.append("other", {"keep": True})
        before = self.path.read_bytes()
        good1 = self.valid_row("t", 1)
        bad2 = self.valid_row("t", 3, prev=good1["hash"])
        heads = [{"tenant": "t", "expected_count": 0, "expected_hash": ZERO}]
        with self.assertRaises(AuditChainStateError):
            self.chain.import_all_range(
                record_bytes(good1) + record_bytes(bad2), heads)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": 0})

    # --- target conflict: expected_heads order, actual values ---

    def test_head_mismatch_conflicts_with_actual_values(self):
        items = [self.chain.append("t", {"i": i}) for i in range(3)]
        before = self.path.read_bytes()
        # source shares the 3-record length but diverges in events, so its
        # head-3 hash is a well-formed assertion that does not match the
        # target's actual head
        src = self.make_source([("t", {"i": i, "s": "审计"})
                                for i in range(5)])
        src_rows = [json.loads(l)
                    for l in src.export_tenant("t").decode().splitlines()]
        # right count, wrong hash: segment [4, 5] anchored at source head 3
        data = src.export_tenant_range("t", 4, 5)
        heads = [{"tenant": "t", "expected_count": 3,
                  "expected_hash": src_rows[2]["hash"]}]
        with self.assertRaises(AuditChainConflictError) as cm:
            self.chain.import_all_range(data, heads)
        e = cm.exception
        self.assertEqual(e.tenant, "t")
        self.assertEqual(e.reason, "conflict")
        self.assertEqual((e.expected_count, e.expected_hash),
                         (3, src_rows[2]["hash"]))
        self.assertEqual((e.actual_count, e.actual_hash),
                         (3, items[2]["hash"]))
        # wrong count: segment [3, 5] anchored at source head 2
        data = src.export_tenant_range("t", 3, 5)
        heads = [{"tenant": "t", "expected_count": 2,
                  "expected_hash": src_rows[1]["hash"]}]
        with self.assertRaises(AuditChainConflictError) as cm:
            self.chain.import_all_range(data, heads)
        self.assertEqual((cm.exception.actual_count,
                          cm.exception.actual_hash),
                         (3, items[2]["hash"]))
        self.assertEqual(self.path.read_bytes(), before)

    def test_conflict_reports_first_mismatch_in_expected_heads_order(self):
        for tenant in ("a", "b"):
            self.chain.append(tenant, {})
        src = self.make_source([("a", {"i": 3}), ("b", {"i": 3})])
        data = src.export_all()
        # both conflict (each target chain has 1 record, assertion says 0);
        # b is listed first, so b is reported
        heads = [{"tenant": "b", "expected_count": 0, "expected_hash": ZERO},
                 {"tenant": "a", "expected_count": 0, "expected_hash": ZERO}]
        with self.assertRaises(AuditChainConflictError) as cm:
            self.chain.import_all_range(data, heads)
        self.assertEqual(cm.exception.tenant, "b")
        self.assertEqual((cm.exception.expected_count,
                          cm.exception.expected_hash), (0, ZERO))
        self.assertEqual(cm.exception.actual_count, 1)

    def test_conflict_only_for_asserted_tenants(self):
        # target has records for unlisted tenant x: no conflict
        self.chain.append("x", {"keep": True})
        src = self.make_source([("a", {"i": 1})])
        records = self.chain.import_all_range(src.export_tenant("a"), [
            {"tenant": "a", "expected_count": 0, "expected_hash": ZERO},
        ])
        self.assertEqual(len(records), 1)
        self.assertEqual(self.chain.verify("x"), {"ok": True, "count": 1})
        self.assertEqual(self.chain.verify("a"), {"ok": True, "count": 1})

    def test_missing_target_only_all_zero_assertions_create(self):
        src = self.make_source([("a", {"i": 1}), ("b", {"i": 1}),
                                ("b", {"i": 2})])
        rows = [json.loads(l)
                for l in src.export_all().decode().splitlines()]
        b1 = next(r for r in rows if r["tenant"] == "b")
        data = (record_bytes(self.valid_row("a", 1))
                + src.export_tenant_range("b", 2, 2))
        heads = [{"tenant": "a", "expected_count": 0, "expected_hash": ZERO},
                 {"tenant": "b", "expected_count": 1,
                  "expected_hash": b1["hash"]}]
        self.assertFalse(self.path.exists())
        with self.assertRaises(AuditChainConflictError) as cm:
            self.chain.import_all_range(data, heads)
        e = cm.exception
        self.assertEqual(e.tenant, "b")
        self.assertEqual((e.expected_count, e.expected_hash),
                         (1, b1["hash"]))
        self.assertEqual((e.actual_count, e.actual_hash), (0, ZERO))
        self.assertFalse(self.path.exists())  # losing assertion creates nothing

    def test_missing_target_conflict_follows_list_order(self):
        a2 = self.valid_row("a", 2, prev="f" * 64)
        b3 = self.valid_row("b", 3, prev="e" * 64)
        data = record_bytes(a2) + record_bytes(b3)
        heads = [{"tenant": "a", "expected_count": 1,
                  "expected_hash": "f" * 64},
                 {"tenant": "b", "expected_count": 2,
                  "expected_hash": "e" * 64}]
        # both assertions non-empty; the first in list order is reported
        with self.assertRaises(AuditChainConflictError) as cm:
            self.chain.import_all_range(data, heads)
        self.assertEqual(cm.exception.tenant, "a")
        self.assertEqual((cm.exception.actual_count,
                          cm.exception.actual_hash), (0, ZERO))
        self.assertFalse(self.path.exists())

    # --- target corruption: state error beats conflict ---

    def test_corrupt_target_raises_state_error_and_writes_nothing(self):
        tampered = self.valid_row("t", 1)
        tampered["event"] = {"tampered": True}
        self.write_rows([tampered])
        before = self.path.read_bytes()
        src = self.make_source([("t", {"i": 1})])
        heads = [{"tenant": "t", "expected_count": 0, "expected_hash": ZERO}]
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.import_all_range(src.export_tenant("t"), heads)
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("t", 1, "digest", 1))
        self.assertEqual(self.path.read_bytes(), before)

    def test_uninvolved_tenant_corruption_does_not_block_import(self):
        bad_other = self.valid_row("z", 1)
        bad_other["event"] = {"x": 1}  # z's digest broken; t unaffected
        self.write_rows([bad_other])
        src = self.make_source([("t", {"i": 1})])
        records = self.chain.import_all_range(src.export_tenant("t"), [
            {"tenant": "t", "expected_count": 0, "expected_hash": ZERO},
        ])
        self.assertEqual(len(records), 1)
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": 1})

    def test_failed_import_leaves_log_appendable(self):
        self.chain.append("t", {"i": 0})
        src = self.make_source([("t", {"i": 1})])
        with self.assertRaises(AuditChainConflictError):
            self.chain.import_all_range(src.export_tenant("t"), [
                {"tenant": "t", "expected_count": 0, "expected_hash": ZERO},
            ])
        nxt = self.chain.append("t", {"i": 1})
        self.assertEqual(nxt["seq"], 2)
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": 2})


class ImportAllRangeConcurrencyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.path = Path(self.tmp.name) / "audit.jsonl"
        self.chain = AuditChain(self.path)
        src = AuditChain(self.path.with_name("source.jsonl"))
        for i in range(10):
            src.append("a", {"i": i})
            src.append("b", {"i": i})
        self.data = src.export_all()
        rows = [json.loads(l) for l in self.data.decode().splitlines()]
        self.tail = {"a": [r for r in rows if r["tenant"] == "a"][-1]["hash"],
                     "b": [r for r in rows if r["tenant"] == "b"][-1]["hash"]}
        self.heads = [
            {"tenant": "a", "expected_count": 0, "expected_hash": ZERO},
            {"tenant": "b", "expected_count": 0, "expected_hash": ZERO},
        ]

    def tearDown(self):
        self.tmp.cleanup()

    def test_parallel_imports_one_wins_rest_conflict(self):
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
        self.assertEqual(successes, [20])
        self.assertEqual(len(conflicts), 7)
        for tenant, ec, eh, ac, ah in conflicts:
            self.assertEqual((ec, eh), (0, ZERO))
            self.assertEqual((ac, ah), (10, self.tail[tenant]))
        self.assertEqual(self.chain.verify("a"), {"ok": True, "count": 10})
        self.assertEqual(self.chain.verify("b"), {"ok": True, "count": 10})
        self.assertEqual(self.chain.export_all(), self.data)

    def test_import_indivisible_against_appends_and_readers(self):
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
                # a/b must be absent or present as the full 10-record graft
                for tenant in ("a", "b"):
                    if counts.get(tenant, 0) not in (0, 10):
                        with box:
                            problems.append(("partial", tenant, counts))
                        return

        readers = [threading.Thread(target=reader) for _ in range(4)]
        for t in readers:
            t.start()

        def other_writer(i):
            c = AuditChain(self.path)
            for j in range(30):
                c.append("other", {"w": i, "j": j})

        writers = [threading.Thread(target=other_writer, args=(i,))
                   for i in range(3)]
        outcomes = {"ok": 0, "conflict": 0}

        def importer():
            try:
                self.chain.import_all_range(self.data, self.heads)
                with box:
                    outcomes["ok"] += 1
            except AuditChainConflictError:
                with box:
                    outcomes["conflict"] += 1

        importers = [threading.Thread(target=importer) for _ in range(5)]
        for t in writers + importers:
            t.start()
        for t in writers + importers:
            t.join()
        stop.set()
        for t in readers:
            t.join(timeout=2)

        self.assertEqual(problems, [])
        self.assertEqual(outcomes["ok"], 1)
        self.assertEqual(outcomes["conflict"], 4)
        self.assertEqual(self.chain.verify("a"), {"ok": True, "count": 10})
        self.assertEqual(self.chain.verify("other"),
                         {"ok": True, "count": 90})
        self.assertTrue(self.chain.verify_all()["ok"])


if __name__ == "__main__":
    unittest.main()
