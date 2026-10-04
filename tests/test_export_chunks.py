import json
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app import AuditChain, AuditChainStateError

ZERO = "0" * 64


def record_bytes(item):
    return (json.dumps(item, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


class ExportChunksTest(unittest.TestCase):
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

    def seed(self):
        for i in range(6):
            self.chain.append("a", {"i": i, "s": "审计-世界"})
            self.chain.append("b", {"i": i})

    # --- result shape ---

    def test_returns_an_iterator_of_bytes_chunks(self):
        self.seed()
        it = self.chain.export_tenant_chunks("a", 10)
        self.assertEqual(iter(it), it)
        chunks = list(it)
        self.assertTrue(chunks)
        self.assertTrue(all(isinstance(c, bytes) for c in chunks))
        self.assertTrue(all(len(c) <= 10 for c in chunks))
        it = self.chain.export_all_chunks(10)
        self.assertEqual(iter(it), it)
        self.assertTrue(all(isinstance(c, bytes) for c in it))

    def test_chunk_sizes_are_at_most_chunk_size_last_short(self):
        self.seed()
        data = self.chain.export_tenant("a")
        for size in (1, 2, 5, 13, 64, 1000):
            chunks = list(self.chain.export_tenant_chunks("a", size))
            self.assertTrue(all(len(c) <= size for c in chunks), size)
            self.assertEqual(b"".join(chunks), data)
            if size < len(data):
                self.assertTrue(len(chunks) > 1)
                self.assertTrue(len(chunks[-1]) <= size)
            else:
                self.assertEqual(chunks, [data])

    # --- byte-for-byte equivalence with the non-chunked entries ---

    def test_concatenation_equals_export_tenant_for_many_sizes(self):
        self.seed()
        expected = self.chain.export_tenant("a")
        for size in range(1, 40):
            chunks = list(self.chain.export_tenant_chunks("a", size))
            self.assertEqual(b"".join(chunks), expected, size)
        # size larger than the whole history yields exactly one chunk
        self.assertEqual(
            list(self.chain.export_tenant_chunks("a", len(expected) + 50)),
            [expected])

    def test_concatenation_equals_export_all_for_many_sizes(self):
        self.seed()
        expected = self.chain.export_all()
        for size in (1, 2, 3, 7, 16, 100, len(expected), len(expected) + 1):
            chunks = list(self.chain.export_all_chunks(size))
            self.assertEqual(b"".join(chunks), expected, size)
            self.assertTrue(all(len(c) <= size for c in chunks))

    def test_chunks_match_non_chunked_for_unknown_tenant(self):
        self.seed()
        self.assertEqual(self.chain.export_tenant("zzz"), b"")
        self.assertEqual(list(self.chain.export_tenant_chunks("zzz", 1)), [])

    def test_chunks_match_at_every_single_byte_cut_with_multibyte_utf8(self):
        self.chain.append("t", {"msg": "héllo→世界"})
        self.chain.append("t", {"msg": "✓" * 40})
        self.chain.append("u", {"msg": "αβγ"})
        tenant_data = self.chain.export_tenant("t")
        all_data = self.chain.export_all()
        # size=1 cuts after every byte, including inside UTF-8 sequences:
        # the iterator yields one single-byte chunk per source byte and the
        # concatenation is still byte-for-byte the unchunked export.
        one_by_one = list(self.chain.export_all_chunks(1))
        self.assertEqual(one_by_one, [bytes([b]) for b in all_data])
        self.assertEqual(b"".join(one_by_one), all_data)
        one_tenant = list(self.chain.export_tenant_chunks("t", 1))
        self.assertEqual(b"".join(one_tenant), tenant_data)

    def test_single_byte_chunks_round_trip_through_import_verification(self):
        self.seed()
        for tenant in ("a", "b"):
            data = self.chain.export_tenant(tenant)
            chunks = list(self.chain.export_tenant_chunks(tenant, 1))
            self.assertEqual(b"".join(chunks), data)
            out = Path(self.tmp.name) / f"out-{tenant}.jsonl"
            out.write_bytes(b"".join(chunks))
            self.assertEqual(AuditChain(out).verify(tenant),
                             {"ok": True, "count": 6})
        out = Path(self.tmp.name) / "out-all.jsonl"
        out.write_bytes(b"".join(self.chain.export_all_chunks(3)))
        fresh = AuditChain(out)
        self.assertEqual(fresh.verify("a"), {"ok": True, "count": 6})
        self.assertEqual(fresh.verify("b"), {"ok": True, "count": 6})

    # --- empty / missing history: empty iterator, no chunks ---

    def test_missing_file_yields_no_chunks_creates_nothing(self):
        self.assertFalse(self.path.exists())
        self.assertEqual(list(self.chain.export_tenant_chunks("t", 10)), [])
        self.assertEqual(list(self.chain.export_all_chunks(10)), [])
        self.assertFalse(self.path.exists())

    def test_empty_file_yields_no_chunks(self):
        self.path.write_bytes(b"")
        self.assertEqual(list(self.chain.export_tenant_chunks("t", 1)), [])
        self.assertEqual(list(self.chain.export_all_chunks(1)), [])

    def test_tenant_with_no_records_yields_no_chunks_but_other_tenants_export(self):
        self.seed()
        self.assertEqual(list(self.chain.export_tenant_chunks("zzz", 4)), [])
        all_chunks = b"".join(self.chain.export_all_chunks(4))
        self.assertEqual(all_chunks, self.path.read_bytes())

    # --- chunk_size boundary ---

    def test_chunk_size_must_be_positive_plain_int(self):
        self.seed()
        for bad in (0, -1, -100, 1.0, 0.0, -1.0, 1e3, True, False,
                    "1", None, [], object(), 1 + 0j):
            with self.assertRaises(ValueError):
                self.chain.export_tenant_chunks("a", bad)
            with self.assertRaises(ValueError):
                self.chain.export_all_chunks(bad)

    def test_tenant_boundary_is_value_error(self):
        self.seed()
        for bad in (float("nan"), float("inf"), {1: "x"}, object(), b"x"):
            with self.assertRaises(ValueError):
                self.chain.export_tenant_chunks(bad, 10)

    def test_boundary_errors_raise_before_any_chunk_and_keep_bytes(self):
        self.seed()
        before = self.path.read_bytes()
        for bad in (0, True, 1.0):
            with self.assertRaises(ValueError):
                list(self.chain.export_tenant_chunks("a", bad))
            with self.assertRaises(ValueError):
                list(self.chain.export_all_chunks(bad))
        with self.assertRaises(ValueError):
            list(self.chain.export_tenant_chunks(float("nan"), 10))
        self.assertEqual(self.path.read_bytes(), before)

    def test_value_error_beats_corrupt_history(self):
        row = {"tenant": "t", "seq": 1, "event": {}, "prev": ZERO}
        row["hash"] = AuditChain._hash(row)
        row["event"] = {"tampered": True}  # digest broken at line 1
        self.write_rows([row])
        before = self.path.read_bytes()
        with self.assertRaises(ValueError):
            self.chain.export_tenant_chunks(float("nan"), 10)
        with self.assertRaises(ValueError):
            self.chain.export_tenant_chunks(float("nan"), 10)
        self.assertEqual(self.path.read_bytes(), before)

    # --- corruption: same fields as the non-chunked entry, no partial export ---

    def test_corrupt_history_raises_state_error_before_first_chunk(self):
        good = {"tenant": "a", "seq": 1, "event": {}, "prev": ZERO}
        good["hash"] = AuditChain._hash(good)
        gap = {"tenant": "a", "seq": 3, "event": {}, "prev": good["hash"]}
        gap["hash"] = AuditChain._hash(gap)
        self.write_rows([good, gap])
        before = self.path.read_bytes()

        def expect_state_error(fn, t, seq, reason, line):
            with self.assertRaises(AuditChainStateError) as cm:
                fn()
            self.assertEqual((cm.exception.tenant, cm.exception.seq,
                              cm.exception.reason, cm.exception.line),
                             (t, seq, reason, line))

        # the error is raised when the entry is called, before iteration
        expect_state_error(lambda: self.chain.export_tenant_chunks("a", 1),
                           "a", 2, "sequence", 2)
        expect_state_error(lambda: self.chain.export_all_chunks(1),
                           "a", 2, "sequence", 2)
        # nothing could have been yielded and no byte changed
        self.assertEqual(self.path.read_bytes(), before)

    def test_state_error_matches_non_chunked_entry_for_many_damages(self):
        cases = [
            ("t", b"{not json\n", ("t", 1, "missing", 1)),
            ("t", b"\xff", ("t", 1, "missing", 1)),
        ]
        for tenant, raw, fields in cases:
            self.path.write_bytes(raw)
            with self.assertRaises(AuditChainStateError) as cm:
                self.chain.export_tenant_chunks(tenant, 5)
            self.assertEqual((cm.exception.tenant, cm.exception.seq,
                              cm.exception.reason, cm.exception.line), fields)
            with self.assertRaises(AuditChainStateError) as cm2:
                self.chain.export_tenant(tenant)
            self.assertEqual((cm2.exception.tenant, cm2.exception.seq,
                              cm2.exception.reason, cm2.exception.line), fields)
            self.path.unlink()

    def test_export_all_corruption_fields_match_export_all(self):
        good = {"tenant": "a", "seq": 1, "event": {}, "prev": ZERO}
        good["hash"] = AuditChain._hash(good)
        bad = {"tenant": "b", "seq": 1, "event": {}, "prev": "f" * 64}
        bad["hash"] = AuditChain._hash(bad)
        self.write_rows([good, bad])
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.export_all_chunks(1)
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("b", 1, "digest", 2))
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.export_all_chunks(100)
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("b", 1, "digest", 2))

    def test_digest_error_yields_no_partial_prefix_even_with_tiny_chunks(self):
        for i in range(4):
            self.chain.append("t", {"i": i})
        rows = [json.loads(l) for l in self.path.read_text().splitlines()]
        rows[0]["event"] = {"i": 99}  # break the first hash
        self.write_rows(rows)
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.export_tenant_chunks("t", 1)
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("t", 1, "digest", 1))

    # --- read-only purity ---

    def test_never_creates_or_modifies_any_file(self):
        self.seed()
        before = self.path.read_bytes()
        entries = sorted(p.name for p in Path(self.tmp.name).iterdir())
        list(self.chain.export_tenant_chunks("a", 7))
        list(self.chain.export_tenant_chunks("b", 7))
        list(self.chain.export_all_chunks(7))
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(sorted(p.name for p in Path(self.tmp.name).iterdir()),
                         entries)

    # --- fixed snapshot under concurrent appends ---

    def test_concurrent_appends_each_export_is_one_complete_snapshot(self):
        for i in range(8):
            self.chain.append("t", {"i": i})

        problems = []
        seen = set()
        counts = set()
        box = threading.Lock()
        stop = threading.Event()

        def exporter():
            while not stop.is_set():
                chunks = list(self.chain.export_tenant_chunks("t", 37))
                data = b"".join(chunks)
                # must be exactly the bytes export_tenant would cut for one
                # complete chain state: seqs 1..k, links and hashes intact
                rows = [json.loads(l) for l in data.decode("utf-8").splitlines()]
                if [r["seq"] for r in rows] != list(range(1, len(rows) + 1)):
                    with box:
                        problems.append("seqs")
                    return
                prev = ZERO
                for r in rows:
                    if r["prev"] != prev or r["hash"] != AuditChain._hash(r):
                        with box:
                            problems.append("chain")
                        return
                    prev = r["hash"]
                with box:
                    seen.add(data)
                    counts.add(len(rows))

        threads = [threading.Thread(target=exporter) for _ in range(4)]
        for t in threads:
            t.start()

        def writer(i):
            for j in range(40):
                self.chain.append("t", {"w": i, "j": j})

        writers = [threading.Thread(target=writer, args=(i,)) for i in range(5)]
        for w in writers:
            w.start()
        for w in writers:
            w.join()
        stop.set()
        for t in threads:
            t.join(timeout=2)

        self.assertEqual(problems, [])
        total = 8 + 5 * 40
        # every observed byte stream verifies offline on a fresh chain
        for data in seen:
            out = Path(self.tmp.name) / "obs.jsonl"
            out.write_bytes(data)
            r = AuditChain(out).verify("t")
            self.assertTrue(r["ok"])
            out.unlink()
        # spinning exporters must observe a staircase from the baseline to
        # the committed total, never a torn prefix
        self.assertTrue(all(8 <= k <= total for k in counts))
        self.assertEqual(max(counts), total)
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": total})


if __name__ == "__main__":
    unittest.main()
