import json
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app import AuditChain, AuditChainStateError

ZERO = "0" * 64


class AppendBatchStreamTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.path = Path(self.tmp.name) / "audit.jsonl"
        self.chain = AuditChain(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def read_rows(self):
        return [json.loads(l) for l in self.path.read_text().splitlines()]

    def test_empty_stream_is_a_noop_and_creates_nothing(self):
        self.assertEqual(self.chain.append_batch_stream("t", iter([])), [])
        self.assertFalse(self.path.exists())
        # also a no-op against an existing file: bytes untouched
        self.chain.append("u", {"v": 1})
        before = self.path.read_bytes()
        self.assertEqual(self.chain.append_batch_stream("t", iter(())), [])
        self.assertEqual(self.path.read_bytes(), before)

    def test_events_must_be_an_iterable_container(self):
        for bad in (None, True, 1, 1.5, b"x", bytearray(b"x"), "x",
                    {"a": 1}, object()):
            with self.assertRaises(ValueError):
                self.chain.append_batch_stream("t", bad)
        # rejection happens before any file is created
        self.assertFalse(self.path.exists())

    def test_any_iterable_is_accepted(self):
        items = self.chain.append_batch_stream("t", ({"i": i} for i in range(3)))
        self.assertEqual([it["seq"] for it in items], [1, 2, 3])
        more = self.chain.append_batch_stream("t", [{"i": 3}, {"i": 4}])
        self.assertEqual([it["seq"] for it in more], [4, 5])
        even_more = self.chain.append_batch_stream("t", ({"i": 5},))
        self.assertEqual([it["seq"] for it in even_more], [6])
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": 6})

    def test_tenant_and_each_event_cross_the_json_boundary(self):
        with self.assertRaises(ValueError):
            self.chain.append_batch_stream(float("nan"), iter([{}]))
        with self.assertRaises(ValueError):
            self.chain.append_batch_stream(
                "t", iter([{"ok": 1}, {"bad": float("inf")}]))
        cyc = []
        cyc.append(cyc)
        with self.assertRaises(ValueError):
            self.chain.append_batch_stream("t", iter([cyc]))
        with self.assertRaises(ValueError):
            self.chain.append_batch_stream("t", iter([{1: "x"}]))
        self.assertFalse(self.path.exists())

    def test_invalid_tenant_is_rejected_before_consuming_the_stream(self):
        pulled = []

        def source():
            pulled.append(1)
            yield {}

        with self.assertRaises(ValueError):
            self.chain.append_batch_stream(float("inf"), source())
        self.assertEqual(pulled, [])
        self.assertFalse(self.path.exists())

    def test_first_bad_event_stops_consumption_at_once(self):
        produced = []

        def source():
            for event in ({"ok": 1}, {"bad": float("nan")}, {"later": 2}):
                produced.append(event)
                yield event

        with self.assertRaises(ValueError):
            self.chain.append_batch_stream("t", source())
        # the event after the bad one is never pulled
        self.assertEqual(len(produced), 2)
        self.assertEqual(produced[0], {"ok": 1})
        self.assertEqual(list(produced[1]), ["bad"])
        self.assertFalse(self.path.exists())

    def test_iterator_error_propagates_unchanged_without_writing(self):
        class Boom(Exception):
            pass

        def source():
            yield {"i": 1}
            raise Boom("stream failed")

        with self.assertRaises(Boom):
            self.chain.append_batch_stream("t", source())
        self.assertFalse(self.path.exists())
        # same against an existing log: not a single byte changes
        self.chain.append("t", {"i": 0})
        before = self.path.read_bytes()
        with self.assertRaises(Boom):
            self.chain.append_batch_stream("t", source())
        self.assertEqual(self.path.read_bytes(), before)

    def test_stream_is_consumed_exactly_once_in_order(self):
        class OneShot:
            def __init__(self):
                self.iters = 0

            def __iter__(self):
                self.iters += 1
                return iter([{"i": 1}, {"i": 2}])

        events = OneShot()
        items = self.chain.append_batch_stream("t", events)
        self.assertEqual(events.iters, 1)
        self.assertEqual([it["event"] for it in items], [{"i": 1}, {"i": 2}])

    def test_stream_assigns_contiguous_seqs_and_links_inside(self):
        items = self.chain.append_batch_stream(
            "t", ({"i": i} for i in (1, 2, 3)))
        self.assertEqual([it["seq"] for it in items], [1, 2, 3])
        self.assertEqual(items[0]["prev"], ZERO)
        self.assertEqual(items[1]["prev"], items[0]["hash"])
        self.assertEqual(items[2]["prev"], items[1]["hash"])
        # returned records match the on-disk fields and values exactly
        self.assertEqual(items, self.read_rows())
        for raw in self.path.read_bytes().splitlines():
            json.loads(raw)  # one JSON object per physical line

    def test_stream_continues_existing_chain(self):
        first = self.chain.append("t", {"i": 0})
        items = self.chain.append_batch_stream("t", iter([{"i": 1}, {"i": 2}]))
        self.assertEqual([it["seq"] for it in items], [2, 3])
        self.assertEqual(items[0]["prev"], first["hash"])
        self.assertEqual(items[1]["prev"], items[0]["hash"])
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": 3})

    def test_byte_equivalent_to_append_batch_on_same_history(self):
        events = [{"i": i, "s": "审计"} for i in range(5)]
        self.chain.append("a", {})
        stream_items = self.chain.append_batch_stream("t", iter(events))
        self.chain.append("z", {})
        streamed = self.path.read_bytes()

        other = AuditChain(self.path.with_name("other.jsonl"))
        other.append("a", {})
        batch_items = other.append_batch("t", list(events))
        other.append("z", {})
        self.assertEqual(other.path.read_bytes(), streamed)
        self.assertEqual(batch_items, stream_items)
        self.assertEqual(other.verify("t"), self.chain.verify("t"))
        self.assertEqual(other.verify_all(), self.chain.verify_all())

    def test_corrupt_history_raises_without_partial_records(self):
        self.chain.append("t", {})
        rows = self.read_rows()
        rows[0]["event"] = {"x": 9}
        with self.path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, sort_keys=True) + "\n")
        before = self.path.read_bytes()
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.append_batch_stream("t", iter([{"i": 1}, {"i": 2}]))
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("t", 1, "digest", 1))
        self.assertEqual(self.path.read_bytes(), before)

    def test_corrupt_history_other_tenant_does_not_block_stream(self):
        self.chain.append("b", {})
        rows = self.read_rows()
        rows[0]["event"] = {"x": 1}
        with self.path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, sort_keys=True) + "\n")
        items = self.chain.append_batch_stream("t", iter([{}, {}]))
        self.assertEqual([it["seq"] for it in items], [1, 2])
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": 2})

    def test_legacy_file_without_trailing_newline_still_one_line_each(self):
        self.chain.append_batch_stream("t", iter([{"v": 1}]))
        self.path.write_bytes(self.path.read_bytes().rstrip(b"\n"))
        items = self.chain.append_batch_stream("t", iter([{"v": 2}, {"v": 3}]))
        lines = self.path.read_text().splitlines()
        self.assertEqual(len(lines), 3)
        for raw in lines:
            json.loads(raw)  # records must never be glued together
        self.assertEqual([it["seq"] for it in items], [2, 3])
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": 3})


class AppendBatchStreamConcurrencyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.path = Path(self.tmp.name) / "audit.jsonl"
        self.chain = AuditChain(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_stream_batches_are_indivisible_intervals(self):
        n_thread, size = 12, 10
        results, errors = [], []
        box = threading.Lock()

        def worker(i):
            local = []
            try:
                local = self.chain.append_batch_stream(
                    "t", ({"w": i, "j": j} for j in range(size)))
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


if __name__ == "__main__":
    unittest.main()
