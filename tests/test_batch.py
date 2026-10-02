import json
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app import AuditChain, AuditChainStateError

ZERO = "0" * 64


class AppendBatchTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.path = Path(self.tmp.name) / "audit.jsonl"
        self.chain = AuditChain(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def read_rows(self):
        return [json.loads(l) for l in self.path.read_text().splitlines()]

    def test_empty_list_is_a_noop_and_creates_nothing(self):
        self.assertEqual(self.chain.append_batch("t", []), [])
        self.assertFalse(self.path.exists())
        # also a no-op against an existing file: bytes untouched
        self.chain.append("u", {"v": 1})
        before = self.path.read_bytes()
        self.assertEqual(self.chain.append_batch("t", []), [])
        self.assertEqual(self.path.read_bytes(), before)

    def test_events_must_be_a_list(self):
        for bad in (None, True, 1, 1.5, "x", {"a": 1}, ("x",), {"a", "b"}):
            with self.assertRaises(ValueError):
                self.chain.append_batch("t", bad)
        # rejection happens before any file is created
        self.assertFalse(self.path.exists())

    def test_tenant_and_each_event_cross_the_json_boundary(self):
        with self.assertRaises(ValueError):
            self.chain.append_batch(float("nan"), [{}])
        with self.assertRaises(ValueError):
            self.chain.append_batch("t", [{"ok": 1}, {"bad": float("inf")}])
        cyc = []
        cyc.append(cyc)
        with self.assertRaises(ValueError):
            self.chain.append_batch("t", [cyc])
        with self.assertRaises(ValueError):
            self.chain.append_batch("t", [{1: "x"}])  # non-string key
        self.assertFalse(self.path.exists())

    def test_batch_assigns_contiguous_seqs_and_links_inside(self):
        items = self.chain.append_batch("t", [{"i": 1}, {"i": 2}, {"i": 3}])
        self.assertEqual([it["seq"] for it in items], [1, 2, 3])
        self.assertEqual(items[0]["prev"], ZERO)
        self.assertEqual(items[1]["prev"], items[0]["hash"])
        self.assertEqual(items[2]["prev"], items[1]["hash"])
        # returned records match the on-disk fields and values exactly
        self.assertEqual(items, self.read_rows())
        for raw in self.path.read_bytes().splitlines():
            json.loads(raw)  # one JSON object per physical line

    def test_batch_continues_existing_chain(self):
        first = self.chain.append("t", {"i": 0})
        items = self.chain.append_batch("t", [{"i": 1}, {"i": 2}])
        self.assertEqual([it["seq"] for it in items], [2, 3])
        self.assertEqual(items[0]["prev"], first["hash"])
        self.assertEqual(items[1]["prev"], items[0]["hash"])
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": 3})

    def test_first_seq_and_prev_match_state_before_commit(self):
        self.chain.append("a", {})
        anchor = self.chain.append("t", {})
        self.chain.append("a", {})
        items = self.chain.append_batch("t", [{}, {}])
        self.assertEqual(items[0]["seq"], anchor["seq"] + 1)
        self.assertEqual(items[0]["prev"], anchor["hash"])

    def test_byte_equivalent_to_sequential_appends(self):
        # build one history with a batch, another with one-by-one appends
        events = [{"i": i, "s": "审计"} for i in range(5)]
        self.chain.append("a", {})
        batch_items = self.chain.append_batch("t", events)
        self.chain.append("z", {})
        batched = self.path.read_bytes()

        other = AuditChain(self.path.with_name("other.jsonl"))
        other.append("a", {})
        sequential = [other.append("t", e) for e in events]
        other.append("z", {})
        self.assertEqual(other.path.read_bytes(), batched)
        self.assertEqual(sequential, batch_items)
        self.assertEqual(other.verify("t"), self.chain.verify("t"))
        r1, r2 = other.verify_all(), self.chain.verify_all()
        self.assertEqual(r1, r2)

    def test_distinct_tenants_keep_independent_chains(self):
        a = self.chain.append_batch("a", [{}, {}])
        b = self.chain.append_batch("b", [{}, {}, {}])
        a2 = self.chain.append_batch("a", [{}])
        self.assertEqual([it["seq"] for it in a], [1, 2])
        self.assertEqual([it["seq"] for it in b], [1, 2, 3])
        self.assertEqual([it["seq"] for it in a2], [3])
        self.assertEqual(self.chain.verify("a"), {"ok": True, "count": 3})
        self.assertEqual(self.chain.verify("b"), {"ok": True, "count": 3})
        self.assertTrue(self.chain.verify_all()["ok"])

    def test_corrupt_history_raises_without_partial_records(self):
        self.chain.append("t", {})
        rows = self.read_rows()
        rows[0]["event"] = {"x": 9}
        with self.path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, sort_keys=True) + "\n")
        before = self.path.read_bytes()
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.append_batch("t", [{"i": 1}, {"i": 2}, {"i": 3}])
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("t", 1, "digest", 1))
        self.assertEqual(self.path.read_bytes(), before)

    def test_corrupt_history_other_tenant_does_not_block_batch(self):
        self.chain.append("b", {})
        rows = self.read_rows()
        rows[0]["event"] = {"x": 1}
        with self.path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, sort_keys=True) + "\n")
        items = self.chain.append_batch("t", [{}, {}])
        self.assertEqual([it["seq"] for it in items], [1, 2])
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": 2})

    def test_illegal_utf8_reports_same_first_broken_point_as_append(self):
        valid = {"tenant": "t", "seq": 1, "event": {}, "prev": ZERO}
        valid["hash"] = AuditChain._hash(valid)
        before = (json.dumps(valid, sort_keys=True) + "\n").encode("utf-8") + b"\xff"
        self.path.write_bytes(before)
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.append_batch("t", [{}, {}])
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("t", 2, "missing", 2))
        self.assertEqual(self.path.read_bytes(), before)

    def test_legacy_file_without_trailing_newline_still_one_line_each(self):
        self.chain.append_batch("t", [{"v": 1}])
        self.path.write_bytes(self.path.read_bytes().rstrip(b"\n"))
        items = self.chain.append_batch("t", [{"v": 2}, {"v": 3}])
        lines = self.path.read_text().splitlines()
        self.assertEqual(len(lines), 3)
        for raw in lines:
            json.loads(raw)  # records must never be glued together
        self.assertEqual([it["seq"] for it in items], [2, 3])
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": 3})


class AppendBatchConcurrencyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.path = Path(self.tmp.name) / "audit.jsonl"
        self.chain = AuditChain(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def record_bytes(self, item):
        return (json.dumps(item, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")

    def test_batches_are_indivisible_intervals(self):
        n_thread, size = 12, 10
        results, errors = [], []
        box = threading.Lock()

        def worker(i):
            local = []
            try:
                local = self.chain.append_batch(
                    "t", [{"w": i, "j": j} for j in range(size)])
            except Exception as e:  # noqa: BLE001
                with box:
                    errors.append(repr(e))
            with box:
                results.extend((i, it) for it in local)

        threads = [threading.Thread(target=worker, args=(i,))
                   for i in range(n_thread)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [])
        total = n_thread * size
        # every seq 1..total appears exactly once
        self.assertEqual(sorted(it["seq"] for _, it in results),
                         list(range(1, total + 1)))
        # on disk the records of each winning batch form a contiguous block
        # with consecutive seqs and internal prev links
        rows = [json.loads(l) for l in self.path.read_text().splitlines()]
        self.assertEqual(len(rows), total)
        by_worker = {}
        for row in rows:
            by_worker.setdefault(row["event"]["w"], []).append(row)
        for i, group in by_worker.items():
            seqs = [r["seq"] for r in group]
            self.assertEqual(seqs, list(range(seqs[0], seqs[0] + size)), i)
            for prev_row, row in zip(group, group[1:]):
                self.assertEqual(row["prev"], prev_row["hash"])
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": total})
        self.assertTrue(self.chain.verify_all()["ok"])

    def test_batches_interleave_with_single_appends_of_other_tenants(self):
        done = []
        box = threading.Lock()

        def batch_worker():
            items = self.chain.append_batch(
                "a", [{"i": j} for j in range(20)])
            with box:
                done.extend(("a", it) for it in items)

        def single_worker():
            local = [self.chain.append("b", {"i": j}) for j in range(20)]
            with box:
                done.extend(("b", it) for it in local)

        threads = [threading.Thread(target=batch_worker) for _ in range(4)]
        threads += [threading.Thread(target=single_worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        counts = {"a": 0, "b": 0}
        for tenant, it in done:
            counts[tenant] += 1
        self.assertEqual(counts, {"a": 80, "b": 80})
        self.assertEqual(self.chain.verify("a"), {"ok": True, "count": 80})
        self.assertEqual(self.chain.verify("b"), {"ok": True, "count": 80})
        self.assertTrue(self.chain.verify_all()["ok"])

    def test_value_error_under_contention_leaves_no_trace(self):
        outcomes = []
        box = threading.Lock()

        def good(i):
            self.chain.append_batch("t", [{"i": i}, {"i": i + 1}])
            with box:
                outcomes.append("ok")

        def bad(i):
            try:
                self.chain.append_batch("t", [{"i": i}, float("nan")])
            except ValueError:
                with box:
                    outcomes.append("value")

        threads = []
        for i in range(20):
            threads.append(threading.Thread(target=good, args=(i,)))
            threads.append(threading.Thread(target=bad, args=(i,)))
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(outcomes.count("ok"), 20)
        self.assertEqual(outcomes.count("value"), 20)
        self.assertEqual(self.chain.verify("t")["count"], 40)


if __name__ == "__main__":
    unittest.main()
