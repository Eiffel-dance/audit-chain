import json
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app import AuditChain, AuditChainStateError

ZERO = "0" * 64
FIELDS = ("tenant", "seq", "event", "prev", "hash")


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

    # --- empty chains ---

    def test_missing_file_is_empty_chain_creates_nothing(self):
        self.assertFalse(self.path.exists())
        self.assertEqual(self.chain.read_tenant("t"), [])
        self.assertFalse(self.path.exists())

    def test_empty_file_is_empty_chain(self):
        self.path.write_bytes(b"")
        self.assertEqual(self.chain.read_tenant("t"), [])
        self.assertEqual(self.chain.read_tenant("t", 1, 100), [])

    def test_unknown_tenant_is_empty(self):
        for i in range(3):
            self.chain.append("other", {"i": i})
        self.assertEqual(self.chain.read_tenant("t"), [])

    # --- basic shape and ordering ---

    def test_read_returns_records_ascending_with_five_fields(self):
        items = [self.chain.append("t", {"i": i}) for i in range(5)]
        rows = self.chain.read_tenant("t")
        self.assertEqual([r["seq"] for r in rows], [1, 2, 3, 4, 5])
        for r in rows:
            self.assertEqual(set(r), set(FIELDS))
        self.assertEqual(rows, items)

    def test_read_records_match_stored_values_and_chain_links(self):
        items = [self.chain.append("t", {"s": "审计", "n": i}) for i in range(3)]
        rows = self.chain.read_tenant("t")
        self.assertEqual([r["event"] for r in rows],
                         [{"s": "审计", "n": 0}, {"s": "审计", "n": 1},
                          {"s": "审计", "n": 2}])
        self.assertEqual(rows[0]["prev"], ZERO)
        self.assertEqual(rows[1]["prev"], items[0]["hash"])
        self.assertEqual(rows[2]["prev"], items[1]["hash"])
        for r in rows:
            self.assertEqual(r["hash"], AuditChain._hash(r))

    def test_interleaved_tenants_do_not_change_page_order(self):
        a_items, b_items = [], []
        for i in range(6):
            a_items.append(self.chain.append("a", {"i": i}))
            b_items.append(self.chain.append("b", {"i": i}))
        rows = self.chain.read_tenant("a")
        self.assertEqual(rows, a_items)
        self.assertTrue(all(r["tenant"] == "a" for r in rows))
        self.assertEqual(self.chain.read_tenant("b"), b_items)

    def test_distinct_json_identities_are_partitioned(self):
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

    def test_object_tenant_key_order_normalized(self):
        t1 = {"a": [1, {"x": 1, "y": 2}], "b": {"p": True, "q": None}}
        t2 = {"b": {"q": None, "p": True}, "a": [1, {"y": 2, "x": 1}]}
        self.chain.append(t1, {"i": 1})
        self.chain.append(t2, {"i": 2})
        rows = self.chain.read_tenant({"a": [1, {"x": 1, "y": 2}],
                                       "b": {"p": True, "q": None}})
        self.assertEqual([r["seq"] for r in rows], [1, 2])
        self.assertEqual(rows[1]["prev"], rows[0]["hash"])

    # --- pagination ---

    def test_default_reads_from_first_to_tail(self):
        items = [self.chain.append("t", {"i": i}) for i in range(4)]
        self.assertEqual(self.chain.read_tenant("t"), items)

    def test_start_seq_reads_from_given_seq_to_tail(self):
        items = [self.chain.append("t", {"i": i}) for i in range(5)]
        self.assertEqual([r["seq"] for r in self.chain.read_tenant("t", 3)],
                         [3, 4, 5])
        self.assertEqual(self.chain.read_tenant("t", 3), items[2:])
        self.assertEqual([r["seq"] for r in self.chain.read_tenant("t", 5)], [5])

    def test_page_size_limits_the_page(self):
        items = [self.chain.append("t", {"i": i}) for i in range(5)]
        self.assertEqual(self.chain.read_tenant("t", 1, 2), items[0:2])
        self.assertEqual(self.chain.read_tenant("t", 2, 2), items[1:3])
        self.assertEqual(self.chain.read_tenant("t", 5, 2), items[4:5])

    def test_walk_all_pages_and_one_past_tail(self):
        items = [self.chain.append("t", {"i": i}) for i in range(25)]
        pages, start = [], 1
        while True:
            page = self.chain.read_tenant("t", start, 10)
            pages.extend(page)
            if not page:
                break
            start += len(page)
        self.assertEqual(pages, items)
        self.assertEqual(self.chain.read_tenant("t", 26), [])
        self.assertEqual(self.chain.read_tenant("t", 100, 10), [])

    def test_short_tail_does_not_error(self):
        for i in range(3):
            self.chain.append("t", {"i": i})
        self.assertEqual([r["seq"] for r in self.chain.read_tenant("t", 2, 50)],
                         [2, 3])

    def test_explicit_page_size_none_reads_to_tail(self):
        items = [self.chain.append("t", {"i": i}) for i in range(4)]
        self.assertEqual(self.chain.read_tenant("t", 2, None), items[1:])

    def test_pagination_is_independent_of_physical_interleaving(self):
        a_items = []
        for i in range(10):
            a_items.append(self.chain.append("a", {"i": i}))
            for _ in range(3):
                self.chain.append("b", {})
        page = self.chain.read_tenant("a", 4, 3)
        self.assertEqual([r["seq"] for r in page], [4, 5, 6])
        self.assertEqual(page, a_items[3:6])

    # --- read-only ---

    def test_read_does_not_create_or_modify_any_file(self):
        for i in range(3):
            self.chain.append("t", {"i": i})
            self.chain.append("u", {"i": i})
        before = self.path.read_bytes()
        entries = sorted(p.name for p in Path(self.tmp.name).iterdir())
        for kwargs in ({}, {"start_seq": 2}, {"start_seq": 2, "page_size": 1}):
            self.chain.read_tenant("t", **kwargs)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(sorted(p.name for p in Path(self.tmp.name).iterdir()),
                         entries)

    # --- parameter boundary: ValueError before any read ---

    def test_rejects_bad_start_seq(self):
        bad = [0, -1, 1.0, 1.5, True, False, "1", None, [], 1 + 0j]
        for v in bad:
            with self.assertRaises(ValueError):
                self.chain.read_tenant("t", v)
        self.assertFalse(self.path.exists())

    def test_rejects_bad_page_size(self):
        bad = [0, -1, 1.0, 1.5, True, False, "2", [], {}]
        for v in bad:
            with self.assertRaises(ValueError):
                self.chain.read_tenant("t", 1, v)
        self.assertFalse(self.path.exists())

    def test_rejects_illegal_tenant_with_value_error(self):
        bad_tenants = [
            float("nan"), float("inf"), float("-inf"),
            {"k": float("nan")}, [1, [float("-inf")]],
            {1: "x"}, {"a": {2: 3}}, object(), b"bytes", {1, 2}, ("a", 1),
        ]
        for bad in bad_tenants:
            with self.assertRaises(ValueError):
                self.chain.read_tenant(bad)
            with self.assertRaises(ValueError):
                self.chain.read_tenant(bad, 1, 10)
        self.assertFalse(self.path.exists())

    def test_rejects_cyclic_tenant_with_value_error(self):
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
        for bad in (float("nan"), {1: "x"}):
            with self.assertRaises(ValueError):
                self.chain.read_tenant(bad)
        for kwargs in ({"start_seq": 0}, {"start_seq": 1, "page_size": 0},
                       {"start_seq": True}):
            with self.assertRaises(ValueError):
                self.chain.read_tenant("t", **kwargs)
        self.assertEqual(self.path.read_bytes(), before)

    # --- state errors: never partial results, same location as verify ---

    def test_digest_error_raises_with_same_location_as_verify(self):
        self.chain.append("t", {"i": 1})
        self.chain.append("t", {"i": 2})
        rows = [json.loads(l) for l in self.path.read_text().splitlines()]
        rows[0]["event"] = {"i": 99}
        self.write_rows(rows)
        before = self.path.read_bytes()
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.read_tenant("t")
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("t", 1, "digest", 1))
        # a page sitting inside the verified prefix still must not return it
        with self.assertRaises(AuditChainStateError):
            self.chain.read_tenant("t", 1, 1)
        self.assertEqual(self.path.read_bytes(), before)

    def test_sequence_error_after_interleaving_reports_expected(self):
        good = {"tenant": "a", "seq": 1, "event": {}, "prev": ZERO}
        good["hash"] = AuditChain._hash(good)
        b = {"tenant": "b", "seq": 1, "event": {}, "prev": ZERO}
        b["hash"] = AuditChain._hash(b)
        bad = {"tenant": "a", "seq": 3, "event": {}, "prev": good["hash"]}
        bad["hash"] = AuditChain._hash(bad)
        self.write_rows([good, b, bad])
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.read_tenant("a", 1, 1)
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("a", 2, "sequence", 3))

    def test_missing_field_and_bad_json_raise_missing(self):
        self.chain.append("t", {})
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(
                {"tenant": "t", "seq": 2, "event": {}, "prev": "x"}) + "\n")
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.read_tenant("t")
        self.assertEqual((cm.exception.seq, cm.exception.reason, cm.exception.line),
                         (2, "missing", 2))

    def test_unparseable_other_tenant_line_still_fatal(self):
        other = {"tenant": "x", "seq": 1, "event": {}, "prev": ZERO}
        other["hash"] = AuditChain._hash(other)
        self.write_rows([other, "{not json"])
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.read_tenant("t")
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("t", 1, "missing", 2))

    def test_illegal_utf8_raises_missing_at_physical_line(self):
        valid = {"tenant": "t", "seq": 1, "event": {}, "prev": ZERO}
        valid["hash"] = AuditChain._hash(valid)
        before = (json.dumps(valid, sort_keys=True) + "\n").encode("utf-8") + b"\xff"
        self.path.write_bytes(before)
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.read_tenant("t")
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("t", 2, "missing", 2))
        self.assertEqual(self.path.read_bytes(), before)

    def test_other_tenant_corruption_does_not_block_read(self):
        self.chain.append("b", {})
        rows = [json.loads(l) for l in self.path.read_text().splitlines()]
        rows[0]["event"] = {"x": 1}  # b's digest broken; t unaffected
        self.write_rows(rows)
        self.chain.append("t", {"ok": True})
        rows = self.chain.read_tenant("t")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["tenant"], "t")

    def test_error_priority_first_position(self):
        # digest error on line 1 beats a later unparseable line
        good = {"tenant": "t", "seq": 1, "event": {}, "prev": ZERO}
        good["hash"] = AuditChain._hash(good)
        tampered = dict(good)
        tampered["event"] = {"z": 9}
        self.path.write_bytes(
            (json.dumps(tampered, sort_keys=True) + "\n").encode("utf-8")
            + b"{oops\n")
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.read_tenant("t")
        self.assertEqual((cm.exception.seq, cm.exception.reason, cm.exception.line),
                         (1, "digest", 1))

    def test_start_beyond_corrupt_chain_still_raises(self):
        # full history is scanned first; a page past the prefix is no escape
        good = {"tenant": "t", "seq": 1, "event": {}, "prev": ZERO}
        good["hash"] = AuditChain._hash(good)
        gap = {"tenant": "t", "seq": 3, "event": {}, "prev": good["hash"]}
        gap["hash"] = AuditChain._hash(gap)
        self.write_rows([good, gap])
        with self.assertRaises(AuditChainStateError):
            self.chain.read_tenant("t", 50, 10)

    # --- determinism ---

    def test_repeated_reads_are_equal(self):
        for i in range(6):
            self.chain.append("t", {"i": i})
            self.chain.append("other", {"i": i})
        first = self.chain.read_tenant("t", 2, 3)
        for _ in range(5):
            self.assertEqual(self.chain.read_tenant("t", 2, 3), first)


class ReadTenantConcurrencyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.path = Path(self.tmp.name) / "audit.jsonl"
        self.chain = AuditChain(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_pages_always_match_one_complete_snapshot(self):
        for i in range(10):
            self.chain.append("t", {"i": i})
        problems = []
        seen_counts = set()
        box = threading.Lock()
        stop = threading.Event()

        def check_rows(rows):
            seqs = [r["seq"] for r in rows]
            if seqs != list(range(1, len(rows) + 1)):
                return ("seqs", seqs)
            prev = ZERO
            for r in rows:
                if r["prev"] != prev:
                    return ("link", r["seq"])
                if r["hash"] != AuditChain._hash(r):
                    return ("hash", r["seq"])
                if r["tenant"] != "t":
                    return ("tenant", r["seq"])
                prev = r["hash"]
            return None

        def full_reader():
            while not stop.is_set():
                rows = self.chain.read_tenant("t")
                problem = check_rows(rows)
                with box:
                    if problem:
                        problems.append(problem)
                        return
                    seen_counts.add(len(rows))

        def window_reader():
            # fixed window over the head: it must always be exactly seqs
            # 3..7 once the snapshot has reached 7, and links must still
            # chain across the window boundary
            while not stop.is_set():
                page = self.chain.read_tenant("t", 3, 5)
                seqs = [r["seq"] for r in page]
                if len(seqs) >= 1 and seqs[0] != 3:
                    with box:
                        problems.append(("window-start", seqs))
                    return
                if len(seqs) > 5 or seqs != list(range(3, 3 + len(seqs))):
                    with box:
                        problems.append(("window-seqs", seqs))
                    return
                for r in page:
                    if r["hash"] != AuditChain._hash(r) or r["tenant"] != "t":
                        with box:
                            problems.append(("window-row", r["seq"]))
                        return

        def foreign_reader():
            while not stop.is_set():
                if self.chain.read_tenant("never-used"):
                    with box:
                        problems.append(("foreign",))
                    return

        readers = [threading.Thread(target=full_reader) for _ in range(4)]
        readers += [threading.Thread(target=window_reader) for _ in range(2)]
        readers += [threading.Thread(target=foreign_reader)]
        for t in readers:
            t.start()

        def writer(tenant, i):
            for j in range(40):
                self.chain.append(tenant, {"w": i, "j": j})

        threads = [threading.Thread(target=writer, args=("t", i)) for i in range(4)]
        threads += [threading.Thread(target=writer, args=("u", i)) for i in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        stop.set()
        for t in readers:
            t.join(timeout=2)

        self.assertEqual(problems, [])
        total = 10 + 4 * 40
        # final snapshot: full read equals 1..total with no torn prefix
        rows = self.chain.read_tenant("t")
        self.assertEqual([r["seq"] for r in rows], list(range(1, total + 1)))
        self.assertIn(total, seen_counts)
        lo, hi = min(seen_counts), max(seen_counts)
        self.assertTrue(all(10 <= k <= total for k in seen_counts))
        self.assertEqual(hi, total)

    def test_other_tenant_appends_keep_read_stable(self):
        for i in range(5):
            self.chain.append("a", {"i": i})
        stop = threading.Event()
        captured = set()
        box = threading.Lock()

        def reader():
            while not stop.is_set():
                page = self.chain.read_tenant("a", 1, 10)
                with box:
                    captured.add(json.dumps(page, sort_keys=True))

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

        self.assertEqual(len(captured), 1)
        rows = json.loads(next(iter(captured)))
        self.assertEqual([r["seq"] for r in rows], [1, 2, 3, 4, 5])
        self.assertTrue(all(r["tenant"] == "a" for r in rows))


if __name__ == "__main__":
    unittest.main()
