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


class ImportAllTest(unittest.TestCase):
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

    def make_snapshot(self, spec, name="source.jsonl"):
        # spec: list of (tenant, event) appended in order to a fresh source
        src = AuditChain(self.path.with_name(name))
        for tenant, event in spec:
            src.append(tenant, event)
        return src, src.export_all()

    # --- boundary: type check, empty no-op ---

    def test_data_must_be_bytes(self):
        for bad in ("", "not jsonl", bytearray(b""), 1, None, [], object()):
            with self.assertRaises(ValueError):
                self.chain.import_all(bad)
        self.assertFalse(self.path.exists())

    def test_value_error_beats_target_state_and_keeps_bytes(self):
        row = self.valid_row("t", 1)
        row["event"] = {"tampered": True}
        self.write_rows([row])
        before = self.path.read_bytes()
        with self.assertRaises(ValueError):
            self.chain.import_all("not bytes")
        self.assertEqual(self.path.read_bytes(), before)

    def test_empty_data_returns_empty_creates_and_reads_nothing(self):
        self.assertFalse(self.path.exists())
        self.assertEqual(self.chain.import_all(b""), [])
        self.assertFalse(self.path.exists())
        # corrupt target: the no-op never reads history and leaves bytes
        self.path.write_bytes(b"\xff")
        corrupt = self.path.read_bytes()
        self.assertEqual(self.chain.import_all(b""), [])
        self.assertEqual(self.path.read_bytes(), corrupt)
        # healthy target with other tenants: nothing appended
        self.path.unlink()
        self.chain.append("other", {})
        healthy = self.path.read_bytes()
        self.assertEqual(self.chain.import_all(b""), [])
        self.assertEqual(self.path.read_bytes(), healthy)

    # --- success: whole-log grafting, verbatim records, offline verify ---

    def test_import_all_round_trip_into_missing_path(self):
        spec = [("a", {"i": 1}), ("b", {"i": 1}), ("a", {"i": 2}),
                ("b", {"i": 2}), ("a", {"i": 3})]
        _, data = self.make_snapshot(spec)
        records = self.chain.import_all(data)
        self.assertEqual(len(records), 5)
        self.assertEqual(self.path.read_bytes(), data)
        self.assertEqual(self.chain.export_all(), data)
        self.assertEqual(self.chain.verify("a"), {"ok": True, "count": 3})
        self.assertEqual(self.chain.verify("b"), {"ok": True, "count": 2})
        self.assertEqual(self.chain.verify_all(), {
            "ok": True,
            "tenants": [{"tenant": "a", "count": 3},
                        {"tenant": "b", "count": 2}],
        })

    def test_returned_records_physical_order_and_original_values(self):
        spec = [("b", {"k": "审计"}), ("a", [1, 2]), ("b", {"n": None})]
        _, data = self.make_snapshot(spec)
        records = self.chain.import_all(data)
        src_rows = [json.loads(l) for l in data.decode("utf-8").splitlines()]
        self.assertEqual(records, src_rows)
        self.assertEqual([r["tenant"] for r in records], ["b", "a", "b"])
        self.assertEqual([r["event"] for r in records],
                         [{"k": "审计"}, [1, 2], {"n": None}])
        for r in records:
            self.assertEqual(set(r), {"tenant", "seq", "event", "prev", "hash"})
            self.assertEqual(r["hash"], AuditChain._hash(r))

    def test_import_all_coexists_with_other_tenants_on_target(self):
        for i in range(3):
            self.chain.append("x", {"i": i})
        target_before = self.path.read_bytes()
        _, data = self.make_snapshot([("a", {"i": 1}), ("b", {"i": 1}),
                                      ("a", {"i": 2})])
        records = self.chain.import_all(data)
        self.assertEqual(len(records), 3)
        self.assertEqual(self.path.read_bytes(), target_before + data)
        self.assertEqual(self.chain.verify("x"), {"ok": True, "count": 3})
        self.assertEqual(self.chain.verify("a"), {"ok": True, "count": 2})
        self.assertEqual(self.chain.verify("b"), {"ok": True, "count": 1})
        self.assertTrue(self.chain.verify_all()["ok"])

    def test_import_all_after_target_without_trailing_newline(self):
        other = self.valid_row("x", 1)
        raw = json.dumps(other, sort_keys=True).encode("utf-8")  # no \n
        self.path.write_bytes(raw)
        _, data = self.make_snapshot([("a", {"i": 1})])
        self.chain.import_all(data)
        self.assertEqual(self.path.read_bytes(), raw + b"\n" + data)
        self.assertTrue(self.chain.verify_all()["ok"])

    def test_import_all_accepts_snapshot_without_trailing_newline(self):
        _, data = self.make_snapshot([("a", {"i": 1}), ("a", {"i": 2})])
        records = self.chain.import_all(data[:-1])  # drop final \n
        self.assertEqual(len(records), 2)
        self.assertEqual(self.path.read_bytes(), data)  # normalized with \n
        self.assertEqual(self.chain.verify("a"), {"ok": True, "count": 2})

    def test_non_canonical_input_spelling_is_normalized_on_commit(self):
        row = self.valid_row("t", 1, event={"a": 1, "b": 2})
        raw = (b'{"hash": "' + row["hash"].encode() + b'",'
               b' "tenant": "t", "seq": 1, '
               b'"event": {"b": 2, "a": 1}, "prev": "'
               + ZERO.encode() + b'"}\n')
        records = self.chain.import_all(raw)
        self.assertEqual(records[0], row)
        self.assertEqual(self.path.read_bytes(), record_bytes(row))

    def test_distinct_json_identities_import_independently(self):
        _, data = self.make_snapshot([(1, {"k": "int"}), ("1", {"k": "str"}),
                                      (1.0, {"k": "float"}), (True, {})])
        records = self.chain.import_all(data)
        self.assertEqual(len(records), 4)
        for tenant, count in ((1, 1), (1.0, 1), (True, 1), ("1", 1)):
            self.assertEqual(self.chain.verify(tenant),
                             {"ok": True, "count": count})

    def test_imported_snapshot_continues_with_append(self):
        _, data = self.make_snapshot([("a", {"i": 1}), ("a", {"i": 2})])
        records = self.chain.import_all(data)
        nxt = self.chain.append("a", {"i": 3})
        self.assertEqual((nxt["seq"], nxt["prev"]), (3, records[-1]["hash"]))
        self.assertEqual(self.chain.verify("a"), {"ok": True, "count": 3})

    # --- input chain state errors: AuditChainStateError, nothing written ---

    def test_input_not_starting_at_seq_one_is_sequence(self):
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.import_all(record_bytes(self.valid_row("t", 2)))
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("t", 1, "sequence", 1))
        self.assertFalse(self.path.exists())

    def test_input_first_prev_not_zero_is_digest(self):
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.import_all(
                record_bytes(self.valid_row("t", 1, prev="f" * 64)))
        self.assertEqual((cm.exception.seq, cm.exception.reason,
                          cm.exception.line), (1, "digest", 1))

    def test_input_hash_mismatch_is_digest(self):
        row = self.valid_row("t", 1)
        row["event"] = {"tampered": True}
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.import_all(record_bytes(row))
        self.assertEqual((cm.exception.seq, cm.exception.reason,
                          cm.exception.line), (1, "digest", 1))

    def test_input_missing_and_extra_fields_are_missing(self):
        missing = json.dumps(
            {"tenant": "t", "seq": 1, "event": {}, "prev": ZERO}).encode() + b"\n"
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.import_all(missing)
        self.assertEqual((cm.exception.seq, cm.exception.reason,
                          cm.exception.line), (1, "missing", 1))
        extra = dict(self.valid_row("t", 1), surprise=1)
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.import_all(record_bytes(extra))
        self.assertEqual((cm.exception.seq, cm.exception.reason,
                          cm.exception.line), (1, "missing", 1))
        self.assertFalse(self.path.exists())

    def test_input_unparseable_blank_or_non_object_lines_are_missing(self):
        cases = [
            (b"\n", 1),
            (b"   \n", 1),
            (b"[1,2,3]\n", 1),
            (b"{not json\n", 1),
            (record_bytes(self.valid_row("t", 1)) + b"{bad\n", 2),
            (record_bytes(self.valid_row("t", 1)) + b"\n", 2),
        ]
        for raw, line in cases:
            with self.assertRaises(AuditChainStateError) as cm:
                self.chain.import_all(raw)
            self.assertEqual((cm.exception.reason, cm.exception.line),
                             ("missing", line), raw)
        self.assertFalse(self.path.exists())

    def test_input_duplicate_keys_and_non_standard_numbers_are_missing(self):
        for raw in (
            b'{"tenant":"t","tenant":"t","seq":1,"event":{},'
            b'"prev":"' + ZERO.encode() + b'","hash":"x"}\n',
            b'{"tenant":"t","seq":1,"event":{"x":NaN},"prev":"'
            + ZERO.encode() + b'","hash":"x"}\n',
            b'{"tenant":"t","seq":1e999,"event":{},"prev":"'
            + ZERO.encode() + b'","hash":"x"}\n',
        ):
            with self.assertRaises(AuditChainStateError) as cm:
                self.chain.import_all(raw)
            self.assertEqual((cm.exception.reason, cm.exception.line),
                             ("missing", 1), raw)

    def test_input_illegal_utf8_is_missing_at_physical_line(self):
        raw = record_bytes(self.valid_row("t", 1)) + b'{"tenant":\xff}\n'
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.import_all(raw)
        self.assertEqual((cm.exception.reason, cm.exception.line),
                         ("missing", 2))

    def test_input_interleaved_tenant_defect_reports_that_tenant(self):
        a1 = self.valid_row("a", 1)
        b1 = self.valid_row("b", 1)
        a3 = self.valid_row("a", 3, prev=a1["hash"])  # a skips seq 2
        raw = record_bytes(a1) + record_bytes(b1) + record_bytes(a3)
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.import_all(raw)
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("a", 2, "sequence", 3))
        self.assertFalse(self.path.exists())

    def test_input_error_priority_matches_physical_order(self):
        # digest on line 1 beats bad bytes on line 2
        tampered = self.valid_row("t", 1)
        tampered["event"] = {"z": 9}
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.import_all(record_bytes(tampered) + b"\xff\n")
        self.assertEqual((cm.exception.seq, cm.exception.reason,
                          cm.exception.line), (1, "digest", 1))
        # missing on line 1 beats a sequence defect later; an unparseable
        # line has no determinable tenant, so tenant and seq are None
        raw = b"{oops\n" + record_bytes(self.valid_row("t", 5))
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.import_all(raw)
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         (None, None, "missing", 1))

    def test_input_state_error_takes_priority_over_target_conflict(self):
        self.chain.append("t", {})  # would conflict, but input is broken
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.import_all(record_bytes(self.valid_row("t", 2)))
        self.assertEqual(cm.exception.reason, "sequence")

    def test_input_state_error_writes_no_partial_records(self):
        self.chain.append("other", {"keep": True})
        before = self.path.read_bytes()
        good1 = self.valid_row("t", 1)
        bad2 = self.valid_row("t", 3, prev=good1["hash"])
        with self.assertRaises(AuditChainStateError):
            self.chain.import_all(
                record_bytes(good1) + record_bytes(bad2))
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": 0})

    # --- target conflict: fixed empty-head expectation, input order ---

    def test_target_with_existing_records_conflicts(self):
        first = self.chain.append("t", {"i": 0})
        second = self.chain.append("t", {"i": 1})
        before = self.path.read_bytes()
        _, data = self.make_snapshot([("t", {"i": 0})], "s.jsonl")
        with self.assertRaises(AuditChainConflictError) as cm:
            self.chain.import_all(data)
        e = cm.exception
        self.assertEqual(e.tenant, "t")
        self.assertEqual(e.reason, "conflict")
        self.assertEqual((e.expected_count, e.expected_hash), (0, ZERO))
        self.assertEqual((e.actual_count, e.actual_hash),
                         (2, second["hash"]))
        self.assertEqual(self.path.read_bytes(), before)

    def test_conflict_reports_first_involved_tenant_in_input_order(self):
        self.chain.append("a", {})
        self.chain.append("b", {})
        # input mentions b first, then a: both conflict, b wins
        _, data = self.make_snapshot([("b", {"i": 1}), ("a", {"i": 1})],
                                     "s.jsonl")
        with self.assertRaises(AuditChainConflictError) as cm:
            self.chain.import_all(data)
        self.assertEqual(cm.exception.tenant, "b")
        self.assertEqual((cm.exception.expected_count,
                          cm.exception.expected_hash), (0, ZERO))
        self.assertEqual(cm.exception.actual_count, 1)

    def test_conflict_only_for_involved_tenants(self):
        # target has records for uninvolved tenant x: no conflict
        self.chain.append("x", {"keep": True})
        _, data = self.make_snapshot([("a", {"i": 1})], "s.jsonl")
        records = self.chain.import_all(data)
        self.assertEqual(len(records), 1)
        self.assertEqual(self.chain.verify("x"), {"ok": True, "count": 1})
        self.assertEqual(self.chain.verify("a"), {"ok": True, "count": 1})

    # --- target corruption: state error beats conflict ---

    def test_corrupt_target_raises_state_error_and_writes_nothing(self):
        tampered = self.valid_row("t", 1)
        tampered["event"] = {"tampered": True}
        self.write_rows([tampered])
        before = self.path.read_bytes()
        _, data = self.make_snapshot([("t", {"i": 1})], "s.jsonl")
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.import_all(data)
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("t", 1, "digest", 1))
        self.assertEqual(self.path.read_bytes(), before)

    def test_unparseable_target_line_raises_missing(self):
        other = self.valid_row("x", 1)
        self.write_rows([other, "{not json"])
        before = self.path.read_bytes()
        _, data = self.make_snapshot([("t", {"i": 1})], "s.jsonl")
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.import_all(data)
        self.assertEqual((cm.exception.reason, cm.exception.line),
                         ("missing", 2))
        self.assertEqual(self.path.read_bytes(), before)

    def test_uninvolved_tenant_digest_corruption_does_not_block_import(self):
        bad_other = self.valid_row("z", 1)
        bad_other["event"] = {"x": 1}  # z's digest broken; t unaffected
        self.write_rows([bad_other])
        _, data = self.make_snapshot([("t", {"i": 1})], "s.jsonl")
        records = self.chain.import_all(data)
        self.assertEqual(len(records), 1)
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": 1})


class ImportAllConcurrencyTest(unittest.TestCase):
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

    def tearDown(self):
        self.tmp.cleanup()

    def test_parallel_imports_one_wins_rest_conflict(self):
        successes, conflicts, others = [], [], []
        box = threading.Lock()

        def importer():
            try:
                records = self.chain.import_all(self.data)
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

    def test_import_all_indivisible_against_appends_and_readers(self):
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
                self.chain.import_all(self.data)
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
