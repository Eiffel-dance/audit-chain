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

    def write_rows(self, rows):
        with self.path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(r if isinstance(r, str) else json.dumps(r, sort_keys=True))
                f.write("\n")

    def valid_row(self, tenant="t", seq=1, prev=ZERO, event=None):
        row = {"tenant": tenant, "seq": seq, "event": event or {}, "prev": prev}
        row["hash"] = AuditChain._hash(row)
        return row

    def make_snapshot(self, spec):
        # spec: list of (tenant, n) appended round-robin per entry
        src = AuditChain(self.path.with_name("source.jsonl"))
        for tenant, n in spec:
            for i in range(n):
                src.append(tenant, {"i": i})
        return src.export_all()

    # --- boundary: type check, empty no-op ---

    def test_data_must_be_bytes(self):
        for bad in ("", "not jsonl", bytearray(b""), 1, None, [], object()):
            with self.assertRaises(ValueError):
                self.chain.import_all(bad)
        self.assertFalse(self.path.exists())

    def test_value_error_beats_target_state_and_keeps_bytes(self):
        self.path.write_bytes(b"\xff")
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
        # healthy target: nothing appended
        self.path.unlink()
        self.chain.append("other", {})
        healthy = self.path.read_bytes()
        self.assertEqual(self.chain.import_all(b""), [])
        self.assertEqual(self.path.read_bytes(), healthy)

    # --- success: interleaved multi-tenant graft ---

    def test_import_interleaved_snapshot_into_missing_path(self):
        data = self.make_snapshot([("a", 3), ("b", 2)])
        records = self.chain.import_all(data)
        self.assertEqual(len(records), 5)
        self.assertEqual(self.path.read_bytes(), data)
        self.assertEqual(self.chain.verify("a"), {"ok": True, "count": 3})
        self.assertEqual(self.chain.verify("b"), {"ok": True, "count": 2})
        self.assertEqual(self.chain.export_all(), data)

    def test_returned_records_are_in_physical_order_with_original_values(self):
        src = AuditChain(self.path.with_name("source.jsonl"))
        src.append("a", {"i": 1})
        src.append("b", {"i": 1})
        src.append("a", {"i": 2})
        data = src.export_all()
        records = self.chain.import_all(data)
        rows = [json.loads(l) for l in data.decode("utf-8").splitlines()]
        self.assertEqual(records, rows)
        self.assertEqual([r["tenant"] for r in records], ["a", "b", "a"])
        self.assertTrue(all(set(r) == {"tenant", "seq", "event", "prev", "hash"}
                            for r in records))
        # both chains continue seamlessly from the imported tails
        nxt = self.chain.append("a", {"i": 3})
        self.assertEqual((nxt["seq"], nxt["prev"]), (3, records[2]["hash"]))

    def test_import_coexists_with_uninvolved_target_tenants(self):
        for i in range(2):
            self.chain.append("other", {"i": i})
        target_before = self.path.read_bytes()
        data = self.make_snapshot([("a", 2), ("b", 1)])
        records = self.chain.import_all(data)
        self.assertEqual(len(records), 3)
        self.assertEqual(self.path.read_bytes(), target_before + data)
        self.assertEqual(self.chain.verify("other"), {"ok": True, "count": 2})
        self.assertTrue(self.chain.verify_all()["ok"])

    def test_import_after_target_without_trailing_newline_still_valid(self):
        other = self.valid_row("x", 1)
        raw = json.dumps(other, sort_keys=True).encode("utf-8")  # no \n
        self.path.write_bytes(raw)
        data = self.make_snapshot([("t", 2)])
        self.chain.import_all(data)
        self.assertEqual(self.path.read_bytes(), raw + b"\n" + data)
        self.assertTrue(self.chain.verify_all()["ok"])

    def test_distinct_json_identities_import_as_separate_chains(self):
        src = AuditChain(self.path.with_name("source.jsonl"))
        for t in (1, 1.0, True, "1"):
            src.append(t, {})
        data = src.export_all()
        records = self.chain.import_all(data)
        self.assertEqual(len(records), 4)
        for t in (1, 1.0, True, "1"):
            self.assertEqual(self.chain.verify(t), {"ok": True, "count": 1})

    def test_non_canonical_input_spelling_is_normalized_on_write(self):
        row = self.valid_row("t", 1, event={"a": 1, "b": 2})
        raw = (b'{"hash": "' + row["hash"].encode() + b'",'
               b' "tenant": "t", "seq": 1, '
               b'"event": {"b": 2, "a": 1}, "prev": "'
               + ZERO.encode() + b'"}\n')
        records = self.chain.import_all(raw)
        self.assertEqual(records[0], row)
        self.assertEqual(self.path.read_bytes(), record_bytes(row))

    # --- input validation: AuditChainStateError with input line numbers ---

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
            self.assertIsNone(cm.exception.tenant)
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
        self.assertFalse(self.path.exists())

    def test_input_illegal_utf8_is_missing_at_physical_line(self):
        raw = record_bytes(self.valid_row("t", 1)) + b'{"tenant":\xff}\n'
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.import_all(raw)
        self.assertEqual((cm.exception.reason, cm.exception.line),
                         ("missing", 2))
        self.assertFalse(self.path.exists())

    def test_input_missing_or_extra_field_is_missing(self):
        no_hash = json.dumps(
            {"tenant": "t", "seq": 1, "event": {}, "prev": ZERO}).encode() + b"\n"
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.import_all(no_hash)
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("t", 1, "missing", 1))
        extra = dict(self.valid_row("t", 1))
        extra["extra"] = 1
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.import_all(record_bytes(extra))
        self.assertEqual((cm.exception.reason, cm.exception.line),
                         ("missing", 1))
        self.assertFalse(self.path.exists())

    def test_input_each_tenant_chain_must_start_at_one_from_zero(self):
        # second tenant's chain does not restart at seq 1
        a1 = self.valid_row("a", 1)
        b1 = self.valid_row("b", 2)
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.import_all(record_bytes(a1) + record_bytes(b1))
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("b", 1, "sequence", 2))
        # second tenant's first prev is not ZERO
        b1 = self.valid_row("b", 1, prev="f" * 64)
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.import_all(record_bytes(a1) + record_bytes(b1))
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("b", 1, "digest", 2))
        self.assertFalse(self.path.exists())

    def test_input_interleaved_chain_gap_and_hash_mismatch(self):
        a1 = self.valid_row("a", 1)
        b1 = self.valid_row("b", 1)
        a2 = self.valid_row("a", 3, prev=a1["hash"])  # gap in a's chain
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.import_all(
                record_bytes(a1) + record_bytes(b1) + record_bytes(a2))
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("a", 2, "sequence", 3))
        bad_b = self.valid_row("b", 1)
        bad_b["event"] = {"tampered": True}
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.import_all(record_bytes(a1) + record_bytes(bad_b))
        self.assertEqual((cm.exception.tenant, cm.exception.reason,
                          cm.exception.line), ("b", "digest", 2))
        self.assertFalse(self.path.exists())

    def test_input_non_integer_seq_spelling_is_sequence(self):
        row = self.valid_row("t", 1)
        raw = json.dumps(row, sort_keys=True).replace('"seq": 1', '"seq": 1.0')
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.import_all(raw.encode() + b"\n")
        self.assertEqual((cm.exception.reason, cm.exception.line),
                         ("sequence", 1))

    def test_input_state_error_writes_no_partial_records(self):
        self.chain.append("other", {"keep": True})
        before = self.path.read_bytes()
        good = self.valid_row("t", 1)
        bad = self.valid_row("t", 3, prev=good["hash"])
        with self.assertRaises(AuditChainStateError):
            self.chain.import_all(record_bytes(good) + record_bytes(bad))
        self.assertEqual(self.path.read_bytes(), before)

    def test_input_state_error_takes_priority_over_target_conflict(self):
        self.chain.append("t", {})  # would conflict, but input is broken
        broken = record_bytes(self.valid_row("t", 2))
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.import_all(broken)
        self.assertEqual(cm.exception.reason, "sequence")

    # --- target side: state errors first, then conflicts, nothing written ---

    def test_corrupt_involved_target_chain_raises_state_error(self):
        tampered = self.valid_row("t", 1)
        tampered["event"] = {"tampered": True}
        self.write_rows([tampered])
        before = self.path.read_bytes()
        data = self.make_snapshot([("t", 2)])
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.import_all(data)
        # state error takes priority over what would otherwise be a conflict
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("t", 1, "digest", 1))
        self.assertEqual(self.path.read_bytes(), before)

    def test_uninvolved_tenant_corruption_does_not_block_import(self):
        bad_other = self.valid_row("b", 1)
        bad_other["event"] = {"x": 1}
        self.write_rows([bad_other])
        data = self.make_snapshot([("t", 2)])
        records = self.chain.import_all(data)
        self.assertEqual(len(records), 2)
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": 2})

    def test_target_with_existing_records_conflicts_with_fixed_expectation(self):
        first = self.chain.append("t", {"i": 0})
        second = self.chain.append("t", {"i": 1})
        before = self.path.read_bytes()
        data = self.make_snapshot([("t", 3)])
        with self.assertRaises(AuditChainConflictError) as cm:
            self.chain.import_all(data)
        e = cm.exception
        self.assertEqual(e.tenant, "t")
        self.assertEqual(e.reason, "conflict")
        self.assertEqual((e.expected_count, e.expected_hash), (0, ZERO))
        self.assertEqual((e.actual_count, e.actual_hash), (2, second["hash"]))
        self.assertEqual(self.path.read_bytes(), before)

    def test_conflict_follows_input_first_appearance_order(self):
        a1 = self.chain.append("a", {})
        b1 = self.chain.append("b", {})
        before = self.path.read_bytes()
        # input mentions b first, then a: b's conflict is reported
        data = self.make_snapshot([("b", 1), ("a", 1)])
        with self.assertRaises(AuditChainConflictError) as cm:
            self.chain.import_all(data)
        e = cm.exception
        self.assertEqual(e.tenant, "b")
        self.assertEqual((e.expected_count, e.expected_hash), (0, ZERO))
        self.assertEqual((e.actual_count, e.actual_hash), (1, b1["hash"]))
        self.assertEqual(self.path.read_bytes(), before)

    def test_state_error_takes_priority_over_conflict_across_tenants(self):
        # a's target chain is corrupt (digest), b's target chain would conflict
        bad_a = self.valid_row("a", 1)
        bad_a["event"] = {"x": 1}
        self.write_rows([bad_a, self.valid_row("b", 1)])
        before = self.path.read_bytes()
        data = self.make_snapshot([("b", 1), ("a", 1)])
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.import_all(data)
        self.assertEqual((cm.exception.tenant, cm.exception.reason), ("a", "digest"))
        self.assertEqual(self.path.read_bytes(), before)

    def test_reimport_of_same_snapshot_conflicts(self):
        data = self.make_snapshot([("t", 2)])
        self.chain.import_all(data)
        before = self.path.read_bytes()
        with self.assertRaises(AuditChainConflictError):
            self.chain.import_all(data)
        self.assertEqual(self.path.read_bytes(), before)


class ImportAllConcurrencyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.path = Path(self.tmp.name) / "audit.jsonl"
        self.chain = AuditChain(self.path)
        src = AuditChain(self.path.with_name("source.jsonl"))
        for i in range(10):
            src.append("t", {"i": i})
            src.append("u", {"i": i})
        self.data = src.export_all()
        rows = [json.loads(l) for l in self.data.decode().splitlines()]
        self.tails = {}
        for r in rows:
            self.tails[r["tenant"]] = (r["seq"], r["hash"])

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
            self.assertEqual((ac, ah), self.tails[tenant])
        self.assertEqual(self.chain.export_all(), self.data)
        self.assertTrue(self.chain.verify_all()["ok"])

    def test_import_indivisible_against_appends_and_readers(self):
        stop = threading.Event()
        problems = []
        box = threading.Lock()

        def reader():
            while not stop.is_set():
                r = self.chain.verify_all()
                if not r["ok"]:
                    with box:
                        problems.append(r)
                    return
                counts = {t["tenant"]: t["count"] for t in r["tenants"]}
                # t/u are either both absent or both fully grafted
                if counts.get("t", 0) not in (0, 10) \
                        or counts.get("u", 0) not in (0, 10):
                    with box:
                        problems.append(counts)
                    return

        readers = [threading.Thread(target=reader) for _ in range(4)]
        for t in readers:
            t.start()

        def writer(i):
            c = AuditChain(self.path)
            for j in range(30):
                c.append("other", {"w": i, "j": j})

        writers = [threading.Thread(target=writer, args=(i,)) for i in range(3)]
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
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": 10})
        self.assertEqual(self.chain.verify("u"), {"ok": True, "count": 10})
        self.assertEqual(self.chain.verify("other"), {"ok": True, "count": 90})


if __name__ == "__main__":
    unittest.main()
