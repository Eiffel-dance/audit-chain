import hashlib
import json
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app import (
    AuditChain,
    AuditChainRangeError,
    AuditChainStateError,
    ZERO,
)


def record_bytes(item):
    return (json.dumps(item, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


class ExportTenantRangeTest(unittest.TestCase):
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

    def seed(self, tenant="t", n=6):
        return [self.chain.append(tenant, {"i": i}) for i in range(n)]

    # --- success: exact closed interval, verbatim chain fields ---

    def test_range_returns_exact_closed_interval_bytes(self):
        items = self.seed("t", 6)
        data = self.chain.export_tenant_range("t", 2, 4)
        self.assertIsInstance(data, bytes)
        self.assertEqual(data, b"".join(record_bytes(it) for it in items[1:4]))
        rows = [json.loads(l) for l in data.decode("utf-8").splitlines()]
        self.assertEqual([r["seq"] for r in rows], [2, 3, 4])
        self.assertTrue(all(set(r) == {"tenant", "seq", "event", "prev", "hash"}
                            for r in rows))
        self.assertTrue(data.endswith(b"\n"))

    def test_range_first_prev_names_omitted_predecessor(self):
        items = self.seed("t", 5)
        data = self.chain.export_tenant_range("t", 3, 4)
        rows = [json.loads(l) for l in data.decode("utf-8").splitlines()]
        # the first exported record keeps its original prev: the hash of the
        # omitted record at seq 2, not ZERO and not recomputed
        self.assertEqual(rows[0]["prev"], items[1]["hash"])
        self.assertEqual(rows[1]["prev"], items[2]["hash"])
        for r in rows:
            payload = json.dumps(
                {k: r[k] for k in ("tenant", "seq", "event", "prev")},
                sort_keys=True, separators=(",", ":"),
            ).encode("utf-8")
            self.assertEqual(hashlib.sha256(payload).hexdigest(), r["hash"])

    def test_range_open_tail_and_full_chain(self):
        items = self.seed("t", 4)
        self.assertEqual(self.chain.export_tenant_range("t", 3),
                         record_bytes(items[2]) + record_bytes(items[3]))
        # [1, None] over the whole chain is byte-identical to export_tenant
        self.assertEqual(self.chain.export_tenant_range("t", 1),
                         self.chain.export_tenant("t"))
        self.assertEqual(self.chain.export_tenant_range("t", 1, 4),
                         self.chain.export_tenant("t"))

    def test_range_single_record_and_first_record(self):
        items = self.seed("t", 3)
        self.assertEqual(self.chain.export_tenant_range("t", 2, 2),
                         record_bytes(items[1]))
        first = self.chain.export_tenant_range("t", 1, 1)
        self.assertEqual(first, record_bytes(items[0]))
        self.assertEqual(json.loads(first)["prev"], ZERO)

    def test_range_filters_interleaved_tenants(self):
        a = [self.chain.append("a", {"i": i}) for i in range(3)]
        for i in range(4):
            self.chain.append("b", {"i": i})
        a += [self.chain.append("a", {"i": 3})]
        data = self.chain.export_tenant_range("a", 2, 4)
        self.assertEqual(data, b"".join(record_bytes(x) for x in a[1:4]))
        rows = [json.loads(l) for l in data.decode("utf-8").splitlines()]
        self.assertTrue(all(r["tenant"] == "a" for r in rows))

    def test_range_does_not_create_or_modify_any_file(self):
        self.seed("t", 3)
        before = self.path.read_bytes()
        entries = sorted(p.name for p in Path(self.tmp.name).iterdir())
        self.assertTrue(self.chain.export_tenant_range("t", 1, 2))
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(sorted(p.name for p in Path(self.tmp.name).iterdir()),
                         entries)

    def test_repeated_range_exports_are_byte_identical(self):
        self.seed("t", 5)
        first = self.chain.export_tenant_range("t", 2, 4)
        for _ in range(5):
            self.assertEqual(self.chain.export_tenant_range("t", 2, 4), first)

    def test_complex_tenant_identity_range(self):
        t1 = {"a": [1, {"x": 1}], "b": None}
        for i in range(3):
            self.chain.append(t1, {"i": i})
        data = self.chain.export_tenant_range({"b": None, "a": [1, {"x": 1}]},
                                              2, 3)
        rows = [json.loads(l) for l in data.decode("utf-8").splitlines()]
        self.assertEqual([r["seq"] for r in rows], [2, 3])
        self.assertEqual(rows[0]["tenant"], t1)

    # --- boundary: ValueError before any read ---

    def test_illegal_tenant_is_value_error_before_read(self):
        bad_tenants = [
            float("nan"), float("inf"), float("-inf"),
            {"k": float("nan")}, [1, [float("-inf")]],
            {1: "x"}, {"a": {2: 3}}, object(), b"bytes", {1, 2}, ("a", 1),
        ]
        for bad in bad_tenants:
            with self.assertRaises(ValueError):
                self.chain.export_tenant_range(bad, 1)
        self.assertFalse(self.path.exists())

    def test_invalid_start_seq_is_value_error_before_read(self):
        for bad in (0, -1, -100, 1.0, 2.5, True, False, "1", None, [], {}):
            with self.assertRaises(ValueError):
                self.chain.export_tenant_range("t", bad)
        self.assertFalse(self.path.exists())

    def test_invalid_end_seq_is_value_error_before_read(self):
        for bad in (0, -1, 1.0, 2.5, True, False, "3", [], {}):
            with self.assertRaises(ValueError):
                self.chain.export_tenant_range("t", 1, bad)
        # end smaller than start is rejected too
        for start, end in ((3, 2), (5, 1), (2, 0)):
            with self.assertRaises(ValueError):
                self.chain.export_tenant_range("t", start, end)
        self.assertFalse(self.path.exists())

    def test_value_error_beats_corrupt_history_and_keeps_bytes(self):
        row = {"tenant": "t", "seq": 1, "event": {}, "prev": ZERO}
        row["hash"] = AuditChain._hash(row)
        row["event"] = {"tampered": True}
        self.write_rows([row])
        before = self.path.read_bytes()
        with self.assertRaises(ValueError):
            self.chain.export_tenant_range(float("nan"), 1)
        with self.assertRaises(ValueError):
            self.chain.export_tenant_range("t", 0)
        with self.assertRaises(ValueError):
            self.chain.export_tenant_range("t", 2, 1)
        self.assertEqual(self.path.read_bytes(), before)

    # --- range errors: AuditChainRangeError with public fields ---

    def assertRangeError(self, cm, tenant, start_seq, end_seq, count):
        e = cm.exception
        self.assertEqual(e.tenant, tenant)
        self.assertEqual(e.start_seq, start_seq)
        self.assertEqual(e.end_seq, end_seq)
        self.assertEqual(e.count, count)
        self.assertEqual(e.reason, "range")

    def test_missing_file_is_empty_chain_range_error(self):
        self.assertFalse(self.path.exists())
        with self.assertRaises(AuditChainRangeError) as cm:
            self.chain.export_tenant_range("t", 1)
        self.assertRangeError(cm, "t", 1, None, 0)
        self.assertFalse(self.path.exists())

    def test_empty_file_and_unknown_tenant_are_range_errors(self):
        self.path.write_bytes(b"")
        with self.assertRaises(AuditChainRangeError) as cm:
            self.chain.export_tenant_range("t", 1, 1)
        self.assertRangeError(cm, "t", 1, 1, 0)
        self.seed("a", 3)
        with self.assertRaises(AuditChainRangeError) as cm:
            self.chain.export_tenant_range("zzz", 1)
        self.assertRangeError(cm, "zzz", 1, None, 0)

    def test_start_beyond_tail_is_range_error(self):
        self.seed("t", 3)
        for start, end in ((4, None), (4, 5), (10, 10)):
            with self.assertRaises(AuditChainRangeError) as cm:
                self.chain.export_tenant_range("t", start, end)
            self.assertRangeError(cm, "t", start, end, 3)

    def test_end_beyond_tail_is_range_error(self):
        self.seed("t", 3)
        with self.assertRaises(AuditChainRangeError) as cm:
            self.chain.export_tenant_range("t", 2, 4)
        self.assertRangeError(cm, "t", 2, 4, 3)
        with self.assertRaises(AuditChainRangeError) as cm:
            self.chain.export_tenant_range("t", 1, 100)
        self.assertRangeError(cm, "t", 1, 100, 3)

    def test_range_error_returns_no_bytes_and_touches_nothing(self):
        self.seed("t", 2)
        before = self.path.read_bytes()
        with self.assertRaises(AuditChainRangeError):
            self.chain.export_tenant_range("t", 3)
        self.assertEqual(self.path.read_bytes(), before)

    # --- source state errors: same first broken point as export_tenant ---

    def test_corrupt_source_raises_state_error_even_before_damage(self):
        # the requested interval [1, 1] lies entirely before the broken
        # record at seq 3; the whole chain must still verify first
        self.seed("t", 3)
        rows = [json.loads(l) for l in self.path.read_text().splitlines()]
        rows[2]["event"] = {"tampered": True}
        self.write_rows(rows)
        before = self.path.read_bytes()
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.export_tenant_range("t", 1, 1)
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("t", 3, "digest", 3))
        self.assertEqual(self.path.read_bytes(), before)

    def test_sequence_error_in_source(self):
        good = {"tenant": "t", "seq": 1, "event": {}, "prev": ZERO}
        good["hash"] = AuditChain._hash(good)
        gap = {"tenant": "t", "seq": 3, "event": {}, "prev": good["hash"]}
        gap["hash"] = AuditChain._hash(gap)
        self.write_rows([good, gap])
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.export_tenant_range("t", 1, 1)
        self.assertEqual((cm.exception.seq, cm.exception.reason,
                          cm.exception.line), (2, "sequence", 2))

    def test_unparseable_line_and_bad_utf8_raise_missing(self):
        other = {"tenant": "x", "seq": 1, "event": {}, "prev": ZERO}
        other["hash"] = AuditChain._hash(other)
        self.write_rows([other, "{not json"])
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.export_tenant_range("t", 1)
        self.assertEqual((cm.exception.reason, cm.exception.line),
                         ("missing", 2))
        valid = {"tenant": "t", "seq": 1, "event": {}, "prev": ZERO}
        valid["hash"] = AuditChain._hash(valid)
        self.path.write_bytes(record_bytes(valid) + b"\xff")
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.export_tenant_range("t", 1, 1)
        self.assertEqual((cm.exception.reason, cm.exception.line),
                         ("missing", 2))

    def test_other_tenant_corruption_does_not_block_range_export(self):
        bad_other = {"tenant": "b", "seq": 1, "event": {}, "prev": ZERO}
        bad_other["hash"] = AuditChain._hash(bad_other)
        bad_other["event"] = {"x": 1}  # b's digest broken; t unaffected
        self.write_rows([bad_other])
        items = self.seed("t", 3)
        self.assertEqual(self.chain.export_tenant_range("t", 2, 3),
                         record_bytes(items[1]) + record_bytes(items[2]))


class ExportRangeConcurrencyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.path = Path(self.tmp.name) / "audit.jsonl"
        self.chain = AuditChain(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_range_export_sees_only_complete_snapshots(self):
        for i in range(10):
            self.chain.append("t", {"i": i})
        problems = []
        box = threading.Lock()
        stop = threading.Event()

        def exporter():
            while not stop.is_set():
                try:
                    data = self.chain.export_tenant_range("t", 5, 8)
                except AuditChainRangeError:
                    continue  # tail not at 8 yet in this snapshot
                rows = [json.loads(l) for l in data.decode("utf-8").splitlines()]
                if [r["seq"] for r in rows] != [5, 6, 7, 8]:
                    with box:
                        problems.append([r["seq"] for r in rows])
                    return
                for a, b in zip(rows, rows[1:]):
                    if b["prev"] != a["hash"] or b["hash"] != AuditChain._hash(b):
                        with box:
                            problems.append(("link", b["seq"]))
                        return

        exporters = [threading.Thread(target=exporter) for _ in range(4)]
        for t in exporters:
            t.start()

        def writer(i):
            for j in range(30):
                self.chain.append("t", {"w": i, "j": j})

        writers = [threading.Thread(target=writer, args=(i,)) for i in range(4)]
        for t in writers:
            t.start()
        for t in writers:
            t.join()
        stop.set()
        for t in exporters:
            t.join(timeout=2)

        self.assertEqual(problems, [])
        total = 10 + 4 * 30
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": total})
        # final range export is exact and re-importable onto the prefix
        seg = self.chain.export_tenant_range("t", 11, total)
        rows = [json.loads(l) for l in seg.decode("utf-8").splitlines()]
        self.assertEqual([r["seq"] for r in rows], list(range(11, total + 1)))


if __name__ == "__main__":
    unittest.main()
