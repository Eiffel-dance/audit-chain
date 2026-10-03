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


def make_history(chain, tenant, events):
    return [chain.append(tenant, e) for e in events]


class ImportTenantTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.src = AuditChain(Path(self.tmp.name) / "src.jsonl")
        self.dst_path = Path(self.tmp.name) / "dst.jsonl"
        self.dst = AuditChain(self.dst_path)

    def tearDown(self):
        self.tmp.cleanup()

    def write_rows(self, path, rows):
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(r if isinstance(r, str) else json.dumps(r, sort_keys=True))
                f.write("\n")

    # --- boundary: ValueError before any history is touched ---

    def test_data_must_be_bytes(self):
        for bad in ("str", bytearray(b"x"), 1, None, [b"x"]):
            with self.assertRaises(ValueError):
                self.dst.import_tenant("t", bad)
        self.assertFalse(self.dst_path.exists())

    def test_illegal_tenant_is_value_error(self):
        data = self.src.export_tenant("t")
        for bad in (float("nan"), float("inf"), {1: "x"}, b"x", object()):
            with self.assertRaises(ValueError):
                self.dst.import_tenant(bad, data if data else b"{}")
        self.assertFalse(self.dst_path.exists())

    def test_value_error_beats_corrupt_target(self):
        row = {"tenant": "t", "seq": 1, "event": {}, "prev": ZERO}
        row["hash"] = AuditChain._hash(row)
        row["event"] = {"tampered": True}
        self.write_rows(self.dst_path, [row])
        before = self.dst_path.read_bytes()
        with self.assertRaises(ValueError):
            self.dst.import_tenant(float("nan"), b"{}")
        with self.assertRaises(ValueError):
            self.dst.import_tenant("t", "not bytes")
        self.assertEqual(self.dst_path.read_bytes(), before)

    # --- empty data: no-op ---

    def test_empty_data_returns_empty_creates_nothing(self):
        self.assertEqual(self.dst.import_tenant("t", b""), [])
        self.assertFalse(self.dst_path.exists())

    def test_empty_data_does_not_read_or_modify_corrupt_file(self):
        self.dst_path.write_bytes(b"{broken\n")
        before = self.dst_path.read_bytes()
        self.assertEqual(self.dst.import_tenant("t", b""), [])
        self.assertEqual(self.dst_path.read_bytes(), before)

    # --- success round trip ---

    def test_import_export_round_trip_into_empty_target(self):
        events = [{"i": i, "s": "审计"} for i in range(5)]
        items = make_history(self.src, "t", events)
        data = self.src.export_tenant("t")
        returned = self.dst.import_tenant("t", data)
        self.assertEqual(returned, items)
        self.assertEqual([r["seq"] for r in returned], [1, 2, 3, 4, 5])
        self.assertEqual(self.dst.verify("t"), {"ok": True, "count": 5})
        self.assertEqual(
            self.dst.verify_all(),
            {"ok": True, "tenants": [{"tenant": "t", "count": 5}]})
        self.assertEqual(self.dst.export_tenant("t"), data)
        self.assertEqual(self.dst.verify_bytes(data, "t"), {"ok": True, "count": 5})

    def test_import_preserves_original_seq_prev_hash_and_event(self):
        items = make_history(self.src, "t", [{"n": 1}, {"n": 2}, {"n": 3}])
        self.dst.import_tenant("t", self.src.export_tenant("t"))
        rows = [json.loads(l) for l in self.dst_path.read_text().splitlines()]
        for got, want in zip(rows, items):
            self.assertEqual(got, want)
        self.assertEqual(rows[0]["prev"], ZERO)
        self.assertEqual(rows[1]["prev"], items[0]["hash"])
        self.assertEqual(rows[2]["prev"], items[1]["hash"])
        # later append continues at the imported tail, no renumbering
        nxt = self.dst.append("t", {"n": 4})
        self.assertEqual((nxt["seq"], nxt["prev"]), (4, items[2]["hash"]))
        self.assertEqual(self.dst.verify("t"), {"ok": True, "count": 4})

    def test_import_creates_file_when_missing(self):
        make_history(self.src, "t", [{}])
        self.assertFalse(self.dst_path.exists())
        self.dst.import_tenant("t", self.src.export_tenant("t"))
        self.assertTrue(self.dst_path.exists())

    def test_import_into_empty_existing_file(self):
        self.dst_path.write_bytes(b"")
        make_history(self.src, "t", [{"x": 1}])
        out = self.dst.import_tenant("t", self.src.export_tenant("t"))
        self.assertEqual(len(out), 1)
        self.assertEqual(self.dst.verify("t"), {"ok": True, "count": 1})

    # --- other tenants may already exist ---

    def test_import_coexists_with_other_tenant_records(self):
        b_items = [self.dst.append("b", {"j": j}) for j in range(3)]
        make_history(self.src, "a", [{"k": 1}, {"k": 2}])
        returned = self.dst.import_tenant("a", self.src.export_tenant("a"))
        self.assertEqual([r["seq"] for r in returned], [1, 2])
        self.assertEqual(
            self.dst.verify_all(),
            {"ok": True, "tenants": [
                {"tenant": "b", "count": 3}, {"tenant": "a", "count": 2}]})
        self.assertEqual(self.dst.verify("a"), {"ok": True, "count": 2})
        self.assertEqual(self.dst.verify("b"), {"ok": True, "count": 3})
        self.assertEqual(self.dst.read_tenant("b"), b_items)
        # import bytes were appended wholesale after b's records
        self.assertEqual(
            self.dst.export_tenant("a"), self.src.export_tenant("a"))

    def test_import_into_file_without_trailing_newline_still_verifies(self):
        # A valid target whose last byte is not \n must not be corrupted by
        # the appended block.
        self.dst.append("b", {})
        raw = self.dst_path.read_bytes().rstrip(b"\n")
        self.dst_path.write_bytes(raw)
        make_history(self.src, "a", [{}])
        self.dst.import_tenant("a", self.src.export_tenant("a"))
        self.assertTrue(self.dst.verify_all()["ok"])

    def test_distinct_json_identities_are_separate_targets(self):
        # import into a file that already has tenant 1: tenants 1.0, true,
        # "1" are each still empty and may be imported independently.
        make_history(self.src, 1, [{"v": "int"}])
        self.dst.import_tenant(1, self.src.export_tenant(1))
        for t in (1.0, True, "1"):
            make_history(self.src, t, [{"v": str(t)}])
            out = self.dst.import_tenant(t, self.src.export_tenant(t))
            self.assertEqual(len(out), 1)
        res = self.dst.verify_all()
        self.assertTrue(res["ok"])
        got = {(type(s["tenant"]).__name__, AuditChain._tenant_key(s["tenant"])):
               s["count"] for s in res["tenants"]}
        self.assertEqual(
            set(got), {(type(t).__name__, AuditChain._tenant_key(t))
                       for t in (1, 1.0, True, "1")})
        self.assertTrue(all(c == 1 for c in got.values()))

    # --- conflict: target tenant already has records ---

    def test_conflict_when_target_tenant_exists(self):
        make_history(self.src, "t", [{"i": 0}, {"i": 1}])
        existing = self.dst.append("t", {"already": True})
        before = self.dst_path.read_bytes()
        with self.assertRaises(AuditChainConflictError) as cm:
            self.dst.import_tenant("t", self.src.export_tenant("t"))
        e = cm.exception
        self.assertEqual(e.tenant, "t")
        self.assertEqual(e.expected_count, 0)
        self.assertEqual(e.expected_hash, ZERO)
        self.assertEqual(e.actual_count, 1)
        self.assertEqual(e.reason, "conflict")
        self.assertEqual(e.actual_hash, existing["hash"])
        # nothing written
        self.assertEqual(self.dst_path.read_bytes(), before)
        self.assertEqual(self.dst.verify("t"), {"ok": True, "count": 1})

    def test_conflict_actual_hash_reflects_target_tail(self):
        tail = [self.dst.append("t", {"i": i}) for i in range(3)]
        make_history(self.src, "t", [{}])
        with self.assertRaises(AuditChainConflictError) as cm:
            self.dst.import_tenant("t", self.src.export_tenant("t"))
        self.assertEqual(cm.exception.actual_count, 3)
        self.assertEqual(cm.exception.actual_hash, tail[-1]["hash"])
        self.assertEqual(cm.exception.expected_hash, ZERO)

    def test_conflict_with_canonical_identity_match(self):
        # target has records under a different key order spelling: same
        # canonical identity must still conflict
        self.dst.append({"a": 1, "b": 2}, {})
        make_history(self.src, {"b": 2, "a": 1}, [{"second": True}])
        with self.assertRaises(AuditChainConflictError):
            self.dst.import_tenant(
                {"b": 2, "a": 1}, self.src.export_tenant({"b": 2, "a": 1}))

    # --- input chain validation ---

    def test_mixed_other_tenant_in_data_is_value_error(self):
        make_history(self.src, "a", [{}])
        other = {"tenant": "b", "seq": 1, "event": {}, "prev": ZERO}
        other["hash"] = AuditChain._hash(other)
        data = self.src.export_tenant("a") + record_bytes(other)
        self.assertFalse(self.dst_path.exists())
        with self.assertRaises(ValueError):
            self.dst.import_tenant("a", data)
        self.assertFalse(self.dst_path.exists())

    def test_input_not_starting_at_seq_one_is_sequence(self):
        items = make_history(self.src, "t", [{}, {}])
        data = b"".join(record_bytes(it) for it in items[1:])  # starts at seq 2
        with self.assertRaises(AuditChainStateError) as cm:
            self.dst.import_tenant("t", data)
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("t", 1, "sequence", 1))
        self.assertFalse(self.dst_path.exists())

    def test_input_first_prev_not_zero_is_digest(self):
        items = make_history(self.src, "t", [{}])
        row = dict(items[0])
        row["prev"] = "f" * 64
        row["hash"] = AuditChain._hash(row)
        with self.assertRaises(AuditChainStateError) as cm:
            self.dst.import_tenant("t", record_bytes(row))
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("t", 1, "digest", 1))

    def test_input_gap_is_sequence(self):
        good = {"tenant": "t", "seq": 1, "event": {}, "prev": ZERO}
        good["hash"] = AuditChain._hash(good)
        gap = {"tenant": "t", "seq": 3, "event": {}, "prev": good["hash"]}
        gap["hash"] = AuditChain._hash(gap)
        with self.assertRaises(AuditChainStateError) as cm:
            self.dst.import_tenant("t", record_bytes(good) + record_bytes(gap))
        self.assertEqual((cm.exception.seq, cm.exception.reason, cm.exception.line),
                         (2, "sequence", 2))

    def test_input_bad_hash_is_digest(self):
        good = {"tenant": "t", "seq": 1, "event": {}, "prev": ZERO}
        good["hash"] = AuditChain._hash(good)
        bad = dict(good)
        bad["event"] = {"tampered": True}
        with self.assertRaises(AuditChainStateError) as cm:
            self.dst.import_tenant("t", record_bytes(bad))
        self.assertEqual((cm.exception.seq, cm.exception.reason, cm.exception.line),
                         (1, "digest", 1))

    def test_input_bad_prev_is_digest(self):
        first = {"tenant": "t", "seq": 1, "event": {}, "prev": ZERO}
        first["hash"] = AuditChain._hash(first)
        second = {"tenant": "t", "seq": 2, "event": {}, "prev": "f" * 64}
        second["hash"] = AuditChain._hash(second)
        with self.assertRaises(AuditChainStateError) as cm:
            self.dst.import_tenant(
                "t", record_bytes(first) + record_bytes(second))
        self.assertEqual((cm.exception.seq, cm.exception.reason, cm.exception.line),
                         (2, "digest", 2))

    def test_input_parse_and_missing_field_errors(self):
        cases = [
            (b"{not json\n", ("t", 1, "missing", 1)),
            (b"[1,2]\n", ("t", 1, "missing", 1)),
            (b"null\n", ("t", 1, "missing", 1)),
            (json.dumps({"seq": 1, "event": {}, "prev": ZERO,
                         "hash": "x"}).encode() + b"\n",
             ("t", 1, "missing", 1)),
        ]
        valid = {"tenant": "t", "seq": 1, "event": {}, "prev": ZERO}
        valid["hash"] = AuditChain._hash(valid)
        no_hash = {"tenant": "t", "seq": 1, "event": {}, "prev": ZERO}
        cases.append((json.dumps(no_hash).encode() + b"\n",
                      ("t", 1, "missing", 1)))
        for data, want in cases:
            with self.assertRaises(AuditChainStateError) as cm:
                self.dst.import_tenant("t", data)
            self.assertEqual((cm.exception.tenant, cm.exception.seq,
                              cm.exception.reason, cm.exception.line), want)
            self.assertFalse(self.dst_path.exists())

    def test_input_non_utf8_is_missing_at_physical_line(self):
        with self.assertRaises(AuditChainStateError) as cm:
            self.dst.import_tenant("t", b"\xff")
        self.assertEqual((cm.exception.seq, cm.exception.reason, cm.exception.line),
                         (1, "missing", 1))

    def test_input_blank_line_is_missing(self):
        items = make_history(self.src, "t", [{}])
        data = record_bytes(items[0]) + b"\n"
        with self.assertRaises(AuditChainStateError) as cm:
            self.dst.import_tenant("t", data)
        self.assertEqual((cm.exception.reason, cm.exception.line), ("missing", 2))

    def test_input_duplicate_keys_and_nan_are_missing(self):
        with self.assertRaises(AuditChainStateError) as cm:
            self.dst.import_tenant(
                "t", b'{"tenant":"t","tenant":"t","seq":1,"event":null,'
                     b'"prev":"' + ZERO.encode() + b'","hash":"x"}\n')
        self.assertEqual(cm.exception.reason, "missing")

    def test_input_error_reports_physical_line_across_records(self):
        items = make_history(self.src, "t", [{}, {}])
        broken = dict(items[1])
        broken["event"] = {"x": 1}
        data = record_bytes(items[0]) + record_bytes(broken)
        with self.assertRaises(AuditChainStateError) as cm:
            self.dst.import_tenant("t", data)
        self.assertEqual((cm.exception.seq, cm.exception.reason, cm.exception.line),
                         (2, "digest", 2))

    def test_input_validation_runs_before_target_scan(self):
        # invalid input + corrupt target: input state error wins; and a
        # conflict can never mask input problems either
        self.dst.append("t", {})  # would conflict
        bad_input = b"{garbage\n"
        with self.assertRaises(AuditChainStateError) as cm:
            self.dst.import_tenant("t", bad_input)
        self.assertEqual(cm.exception.reason, "missing")

    # --- target corruption priority ---

    def test_corrupt_target_raises_state_error_before_conflict(self):
        row = {"tenant": "t", "seq": 1, "event": {}, "prev": ZERO}
        row["hash"] = AuditChain._hash(row)
        row["event"] = {"tampered": True}  # digest broken: conflict AND state
        self.write_rows(self.dst_path, [row])
        make_history(self.src, "t", [{}])
        before = self.dst_path.read_bytes()
        with self.assertRaises(AuditChainStateError) as cm:
            self.dst.import_tenant("t", self.src.export_tenant("t"))
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("t", 1, "digest", 1))
        self.assertEqual(self.dst_path.read_bytes(), before)

    def test_other_tenant_corruption_does_not_block_import(self):
        # corruption in another tenant's chain is invisible to the target scan
        rows = []
        b = {"tenant": "b", "seq": 1, "event": {}, "prev": ZERO}
        b["hash"] = AuditChain._hash(b)
        b["event"] = {"x": 1}
        self.write_rows(self.dst_path, [b])
        make_history(self.src, "t", [{"ok": True}])
        out = self.dst.import_tenant("t", self.src.export_tenant("t"))
        self.assertEqual(len(out), 1)
        self.assertEqual(self.dst.verify("t"), {"ok": True, "count": 1})

    # --- failure atomicity ---

    def test_no_partial_write_on_failure(self):
        make_history(self.src, "t", [{}, {}])
        self.dst.append("t", {})
        entries_before = sorted(p.name for p in Path(self.tmp.name).iterdir())
        bytes_before = self.dst_path.read_bytes()
        with self.assertRaises(AuditChainConflictError):
            self.dst.import_tenant("t", self.src.export_tenant("t"))
        self.assertEqual(self.dst_path.read_bytes(), bytes_before)
        self.assertEqual(sorted(p.name for p in Path(self.tmp.name).iterdir()),
                         entries_before)

    # --- concurrency ---

    def test_concurrent_imports_of_same_tenant_one_wins(self):
        make_history(self.src, "t", [{"i": i} for i in range(20)])
        data = self.src.export_tenant("t")
        results = []

        def worker():
            try:
                results.append(("ok", self.dst.import_tenant("t", data)))
            except AuditChainConflictError:
                results.append(("conflict",))
            except Exception as e:  # noqa: BLE001
                results.append(("leak", type(e).__name__))

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(sum(1 for r in results if r[0] == "ok"), 1)
        self.assertTrue(all(r[0] in ("ok", "conflict") for r in results))
        self.assertEqual(self.dst.verify("t"), {"ok": True, "count": 20})
        self.assertEqual(self.dst.export_tenant("t"), data)

    def test_concurrent_imports_of_distinct_tenants_all_commit(self):
        payloads = {}
        for name in ("a", "b", "c", "d"):
            make_history(self.src, name, [{"n": i} for i in range(5)])
            payloads[name] = self.src.export_tenant(name)

        def worker(name):
            self.dst.import_tenant(name, payloads[name])

        threads = [threading.Thread(target=worker, args=(n,))
                   for n in payloads]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        res = self.dst.verify_all()
        self.assertTrue(res["ok"])
        self.assertEqual(sorted(s["tenant"] for s in res["tenants"]),
                         ["a", "b", "c", "d"])
        for name in payloads:
            self.assertEqual(self.dst.verify(name), {"ok": True, "count": 5})
            self.assertEqual(self.dst.export_tenant(name), payloads[name])

    def test_concurrent_append_other_tenant_and_import(self):
        make_history(self.src, "t", [{"i": i} for i in range(10)])
        data = self.src.export_tenant("t")
        stop = threading.Event()
        problems = []

        def importer():
            while not stop.is_set():
                try:
                    out = self.dst.import_tenant("t", data)
                    if len(out) != 10:
                        problems.append("length")
                except AuditChainConflictError:
                    pass

        def appender():
            for i in range(100):
                self.dst.append("other", {"i": i})

        importers = [threading.Thread(target=importer) for _ in range(3)]
        appenders = [threading.Thread(target=appender) for _ in range(3)]
        for t in importers + appenders:
            t.start()
        for t in appenders:
            t.join()
        stop.set()
        for t in importers:
            t.join(timeout=5)
        self.assertEqual(problems, [])
        self.assertEqual(self.dst.verify("t"), {"ok": True, "count": 10})
        self.assertEqual(self.dst.verify("other"), {"ok": True, "count": 300})
        self.assertTrue(self.dst.verify_all()["ok"])


if __name__ == "__main__":
    unittest.main()
