import json
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app import AuditChain, AuditChainStateError

ZERO = "0" * 64


class ReadTenantTest(unittest.TestCase):
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

    def seed(self, tenant, n, other=()):
        # append n tenant records, interleaving one record of each tenant in
        # `other` between successive tenant records
        items = []
        for i in range(n):
            for o in other:
                self.chain.append(o, {"between": i})
            items.append(self.chain.append(tenant, {"i": i}))
        return items

    # --- success: shape, ordering and field preservation ---

    def test_default_read_returns_all_records_ascending_with_five_fields(self):
        items = self.seed("t", 5)
        rows = self.chain.read_tenant("t")
        self.assertEqual(len(rows), 5)
        self.assertEqual([r["seq"] for r in rows], [1, 2, 3, 4, 5])
        for r in rows:
            self.assertEqual(set(r), {"tenant", "seq", "event", "prev", "hash"})
        self.assertEqual(rows, items)

    def test_read_page_start_and_size(self):
        self.seed("t", 10)
        self.assertEqual([r["seq"] for r in self.chain.read_tenant("t", 1, 3)],
                         [1, 2, 3])
        self.assertEqual([r["seq"] for r in self.chain.read_tenant("t", 4, 3)],
                         [4, 5, 6])
        self.assertEqual([r["seq"] for r in self.chain.read_tenant("t", 8, 10)],
                         [8, 9, 10])

    def test_start_default_is_first_record(self):
        self.seed("t", 3)
        self.assertEqual([r["seq"] for r in self.chain.read_tenant("t", page_size=2)],
                         [1, 2])

    def test_page_size_none_reads_through_tail(self):
        self.seed("t", 6)
        self.assertEqual(
            [r["seq"] for r in self.chain.read_tenant("t", 3, None)],
            [3, 4, 5, 6])
        self.assertEqual(
            [r["seq"] for r in self.chain.read_tenant("t", 3)],
            [3, 4, 5, 6])

    def test_start_past_count_returns_empty_without_error(self):
        self.seed("t", 3)
        self.assertEqual(self.chain.read_tenant("t", 4), [])
        self.assertEqual(self.chain.read_tenant("t", 100, 10), [])

    def test_short_tail_page_returns_suffix_not_error(self):
        self.seed("t", 5)
        rows = self.chain.read_tenant("t", 4, 10)
        self.assertEqual([r["seq"] for r in rows], [4, 5])

    def test_page_size_one_boundaries(self):
        self.seed("t", 3)
        for i in (1, 2, 3):
            rows = self.chain.read_tenant("t", i, 1)
            self.assertEqual([row["seq"] for row in rows], [i])

    # --- interleaving ---

    def test_interleaved_tenants_do_not_enter_or_move_pages(self):
        items = self.seed("a", 6, other=("b", "c"))
        rows = self.chain.read_tenant("a", 2, 3)
        self.assertEqual([r["seq"] for r in rows], [2, 3, 4])
        self.assertTrue(all(r["tenant"] == "a" for r in rows))
        self.assertEqual(rows, items[1:4])
        # chain linkage inside the page is against tenant-a's own history
        self.assertEqual(rows[0]["prev"], items[0]["hash"])
        self.assertEqual(rows[1]["prev"], rows[0]["hash"])

    def test_unknown_tenant_is_empty_chain(self):
        self.seed("t", 3)
        self.assertEqual(self.chain.read_tenant("zzz"), [])
        self.assertEqual(self.chain.read_tenant("zzz", 1, 10), [])

    def test_distinct_json_identities_are_separate_pages(self):
        for t in (1, 1.0, True, "1"):
            self.chain.append(t, {})
            self.chain.append(t, {})
        for t in (1, 1.0, True, "1"):
            rows = self.chain.read_tenant(t)
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(
                AuditChain._tenant_key(r["tenant"]) == AuditChain._tenant_key(t)
                for r in rows))
            self.assertEqual([r["seq"] for r in rows], [1, 2])

    # --- empty / missing log ---

    def test_missing_file_returns_empty_creates_nothing(self):
        self.assertFalse(self.path.exists())
        self.assertEqual(self.chain.read_tenant("t"), [])
        self.assertEqual(self.chain.read_tenant("t", 1, 5), [])
        self.assertFalse(self.path.exists())

    def test_empty_file_returns_empty(self):
        self.path.write_bytes(b"")
        self.assertEqual(self.chain.read_tenant("t"), [])

    # --- read-only ---

    def test_read_creates_or_modifies_nothing_and_writes_no_cache(self):
        self.seed("t", 4, other=("u",))
        before = self.path.read_bytes()
        entries = sorted(p.name for p in Path(self.tmp.name).iterdir())
        self.chain.read_tenant("t")
        self.chain.read_tenant("t", 2, 2)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(sorted(p.name for p in Path(self.tmp.name).iterdir()),
                         entries)

    # --- parameter validation: ValueError before any read ---

    def test_bad_start_seq_raises_value_error(self):
        for bad in (0, -1, 1.0, 1.5, True, False, "1", None, [1], object()):
            with self.assertRaises(ValueError):
                self.chain.read_tenant("t", bad)
        self.assertFalse(self.path.exists())

    def test_bad_page_size_raises_value_error(self):
        for bad in (0, -1, 1.0, True, False, "2", [], {}):
            with self.assertRaises(ValueError):
                self.chain.read_tenant("t", 1, bad)
        self.assertFalse(self.path.exists())

    def test_bad_tenant_raises_value_error(self):
        bad_tenants = [
            float("nan"), float("inf"), float("-inf"),
            {"k": float("nan")}, [1, [float("-inf")]],
            {1: "x"}, {"a": {2: 3}}, object(), b"bytes", {1, 2}, ("a", 1),
        ]
        for bad in bad_tenants:
            with self.assertRaises(ValueError):
                self.chain.read_tenant(bad)
        self.assertFalse(self.path.exists())

    def test_cyclic_tenant_raises_value_error(self):
        cyc = []
        cyc.append(cyc)
        with self.assertRaises(ValueError):
            self.chain.read_tenant(cyc)
        d = {}
        d["self"] = d
        with self.assertRaises(ValueError):
            self.chain.read_tenant(d)

    def test_value_error_beats_corrupt_history_and_keeps_bytes(self):
        row = {"tenant": "t", "seq": 1, "event": {}, "prev": ZERO}
        row["hash"] = AuditChain._hash(row)
        row["event"] = {"tampered": True}
        self.write_rows([row])
        before = self.path.read_bytes()
        for bad_tenant in (float("nan"), {1: "x"}):
            with self.assertRaises(ValueError):
                self.chain.read_tenant(bad_tenant)
        for bad_start in (0, True, 1.0):
            with self.assertRaises(ValueError):
                self.chain.read_tenant("t", bad_start)
        for bad_size in (0, False, -2):
            with self.assertRaises(ValueError):
                self.chain.read_tenant("t", 1, bad_size)
        self.assertEqual(self.path.read_bytes(), before)

    # --- state errors: identical location semantics to verify/export ---

    def test_missing_field_raises_missing_at_own_seq(self):
        self.chain.append("t", {})
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(
                {"tenant": "t", "seq": 2, "event": {}, "prev": "x"}) + "\n")
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.read_tenant("t")
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("t", 2, "missing", 2))

    def test_unparseable_line_raises_missing_at_line(self):
        other = {"tenant": "x", "seq": 1, "event": {}, "prev": ZERO}
        other["hash"] = AuditChain._hash(other)
        self.write_rows([other, "{not json"])
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.read_tenant("t")
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("t", 1, "missing", 2))

    def test_sequence_gap_raises_sequence_at_expected(self):
        good = {"tenant": "t", "seq": 1, "event": {}, "prev": ZERO}
        good["hash"] = AuditChain._hash(good)
        gap = {"tenant": "t", "seq": 3, "event": {}, "prev": good["hash"]}
        gap["hash"] = AuditChain._hash(gap)
        self.write_rows([good, gap])
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.read_tenant("t")
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("t", 2, "sequence", 2))

    def test_digest_mismatch_raises_digest(self):
        self.chain.append("t", {"i": 1})
        rows = [json.loads(l) for l in self.path.read_text().splitlines()]
        rows[0]["event"] = {"i": 99}
        self.write_rows(rows)
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.read_tenant("t")
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("t", 1, "digest", 1))

    def test_illegal_utf8_raises_missing_with_physical_line(self):
        valid = {"tenant": "t", "seq": 1, "event": {}, "prev": ZERO}
        valid["hash"] = AuditChain._hash(valid)
        before = (json.dumps(valid, sort_keys=True) + "\n").encode("utf-8") + b"\xff"
        self.path.write_bytes(before)
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.read_tenant("t")
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("t", 2, "missing", 2))

    def test_corruption_after_requested_page_still_raises_no_partial_page(self):
        # records 1..3 valid, record 4 has a broken hash; reading only page
        # [1..2] must still fail and never return the verified prefix
        items = [self.chain.append("t", {"i": i}) for i in range(3)]
        bad = {"tenant": "t", "seq": 4, "event": {"i": 3},
               "prev": items[-1]["hash"], "hash": "0" * 64}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(bad, sort_keys=True) + "\n")
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.read_tenant("t", 1, 2)
        self.assertEqual((cm.exception.seq, cm.exception.reason, cm.exception.line),
                         (4, "digest", 4))

    def test_error_priority_digest_before_later_missing(self):
        tampered = {"tenant": "t", "seq": 1, "event": {"z": 9}, "prev": ZERO}
        tampered["hash"] = AuditChain._hash(
            {"tenant": "t", "seq": 1, "event": {}, "prev": ZERO})
        self.path.write_bytes(
            (json.dumps(tampered, sort_keys=True) + "\n").encode("utf-8")
            + b"{oops\n")
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.read_tenant("t")
        self.assertEqual((cm.exception.seq, cm.exception.reason, cm.exception.line),
                         (1, "digest", 1))

    def test_other_tenant_corruption_does_not_block_read(self):
        self.chain.append("b", {})
        rows = [json.loads(l) for l in self.path.read_text().splitlines()]
        rows[0]["event"] = {"x": 1}  # b's digest broken; t unaffected
        self.write_rows(rows)
        self.chain.append("t", {"ok": True})
        rows = self.chain.read_tenant("t")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["tenant"], "t")

    def test_target_corruption_after_interleaving_reports_target(self):
        good = {"tenant": "a", "seq": 1, "event": {}, "prev": ZERO}
        good["hash"] = AuditChain._hash(good)
        b = {"tenant": "b", "seq": 1, "event": {}, "prev": ZERO}
        b["hash"] = AuditChain._hash(b)
        bad = {"tenant": "a", "seq": 3, "event": {}, "prev": good["hash"]}
        bad["hash"] = AuditChain._hash(bad)
        self.write_rows([good, b, bad])
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.read_tenant("a")
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("a", 2, "sequence", 3))


class ReadTenantConcurrencyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.path = Path(self.tmp.name) / "audit.jsonl"
        self.chain = AuditChain(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_pages_are_always_complete_pre_or_post_append_snapshots(self):
        for i in range(10):
            self.chain.append("t", {"i": i})
        problems = []
        box = threading.Lock()
        stop = threading.Event()

        def reader():
            while not stop.is_set():
                start = 1 + (id(threading.current_thread()) % 5)
                rows = self.chain.read_tenant("t", start, 7)
                if not rows:
                    # only legitimate if start is past the snapshot count
                    continue
                seqs = [r["seq"] for r in rows]
                if seqs != list(range(seqs[0], seqs[0] + len(seqs))):
                    with box:
                        problems.append(("non-contiguous", seqs))
                    return
                if any(r["tenant"] != "t" for r in rows):
                    with box:
                        problems.append(("foreign",))
                    return
                if len(rows) > 7 or seqs[0] != start:
                    with box:
                        problems.append(("bounds", seqs))
                    return
                for r in rows:
                    if set(r) != {"tenant", "seq", "event", "prev", "hash"}:
                        with box:
                            problems.append(("fields",))
                        return
                    if r["hash"] != AuditChain._hash(r):
                        with box:
                            problems.append(("hash", r["seq"]))
                        return

        readers = [threading.Thread(target=reader) for _ in range(6)]
        for t in readers:
            t.start()

        def writer(i):
            for j in range(40):
                self.chain.append("t", {"w": i, "j": j})

        writers = [threading.Thread(target=writer, args=(i,)) for i in range(6)]
        for t in writers:
            t.start()
        for t in writers:
            t.join()
        stop.set()
        for t in readers:
            t.join(timeout=2)

        self.assertEqual(problems, [])
        total = 10 + 6 * 40
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": total})
        self.assertEqual(
            [r["seq"] for r in self.chain.read_tenant("t")],
            list(range(1, total + 1)))

    def test_appends_to_other_tenants_keep_read_stable(self):
        for i in range(5):
            self.chain.append("a", {"i": i})
        stop = threading.Event()
        captured = []
        box = threading.Lock()

        def reader():
            while not stop.is_set():
                rows = self.chain.read_tenant("a", 2, 3)
                with box:
                    captured.append([(r["seq"], r["event"]) for r in rows])

        t = threading.Thread(target=reader)
        t.start()

        def writer():
            for i in range(100):
                self.chain.append("b", {"i": i})

        ws = [threading.Thread(target=writer) for _ in range(4)]
        for w in ws:
            w.start()
        for w in ws:
            w.join()
        stop.set()
        t.join(timeout=2)

        self.assertTrue(captured)
        for page in captured:
            self.assertEqual(page, [(2, {"i": 1}), (3, {"i": 2}), (4, {"i": 3})])


if __name__ == "__main__":
    unittest.main()
