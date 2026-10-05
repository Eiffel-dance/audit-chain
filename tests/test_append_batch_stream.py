import json
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app import AuditChain, AuditChainStateError

ZERO = "0" * 64


class CountingIterable:
    # A one-shot iterable that records whether its container __iter__ ran
    # and how far its iterator was pulled, so a test can prove the entry
    # consumes the stream exactly once and stops pulling at the first bad
    # produced event.
    def __init__(self, values):
        self._values = list(values)
        self.iter_calls = 0
        self.pulled = 0

    def __iter__(self):
        self.iter_calls += 1
        for value in self._values:
            self.pulled += 1
            yield value


class Boom(Exception):
    pass


def raising_stream(values, raise_after):
    for value in values[:raise_after]:
        yield value
    raise Boom("iterator blew up")


class AppendBatchStreamTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.path = Path(self.tmp.name) / "audit.jsonl"
        self.chain = AuditChain(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def read_rows(self):
        return [json.loads(l) for l in self.path.read_text().splitlines()]

    # -- container boundary -------------------------------------------------

    def test_bare_scalars_and_containers_are_value_errors(self):
        for bad in (b"x", bytearray(b"x"), "x", {"a": 1}, {},
                    None, True, 1, 1.5, object()):
            with self.subTest(bad=type(bad).__name__):
                with self.assertRaises(ValueError):
                    self.chain.append_batch_stream("t", bad)
        # rejection happens before any file is created
        self.assertFalse(self.path.exists())

    def test_lists_tuples_sets_generators_and_iterators_are_accepted(self):
        events = [1, "two", None, [True, 3.5], {"k": "v"}]
        self.assertEqual(
            [it["event"] for it in self.chain.append_batch_stream("l", events)],
            events,
        )
        self.assertEqual(
            [it["event"] for it in
             self.chain.append_batch_stream("u", tuple(events))],
            events,
        )
        gen_items = self.chain.append_batch_stream(
            "g", (e for e in events))
        self.assertEqual([it["event"] for it in gen_items], events)
        # a plain iterator object is itself a valid one-shot input
        flag_items = self.chain.append_batch_stream("i", iter(events))
        self.assertEqual([it["event"] for it in flag_items], events)
        # a set is an iterable: accepted and handled in production order
        set_items = self.chain.append_batch_stream("s", {3, 1, 2})
        self.assertEqual([it["seq"] for it in set_items], [1, 2, 3])
        self.assertEqual(self.chain.verify("s"), {"ok": True, "count": 3})
        self.assertTrue(self.chain.verify_all()["ok"])

    # -- single consumption -------------------------------------------------

    def test_iterator_is_consumed_exactly_once(self):
        source = CountingIterable([{"i": 1}, {"i": 2}, {"i": 3}])
        items = self.chain.append_batch_stream("t", source)
        self.assertEqual(source.iter_calls, 1)
        self.assertEqual(source.pulled, 3)
        self.assertEqual([it["event"] for it in items],
                         [{"i": 1}, {"i": 2}, {"i": 3}])
        # a generator object is exhausted by the call
        gen = (e for e in [{"i": 9}])
        self.chain.append_batch_stream("g", gen)
        with self.assertRaises(StopIteration):
            next(gen)

    def test_production_order_is_preserved(self):
        def stream():
            for i in range(7):
                yield {"i": i}

        items = self.chain.append_batch_stream("t", stream())
        self.assertEqual([it["event"]["i"] for it in items], list(range(7)))
        self.assertEqual([it["seq"] for it in items], list(range(1, 8)))

    def test_pulling_stops_at_the_first_bad_produced_event(self):
        source = CountingIterable(
            [{"ok": 1}, float("nan"), {"never": 1}])
        with self.assertRaises(ValueError):
            self.chain.append_batch_stream("t", source)
        # the container was entered once and only the two values preceding
        # the boundary failure were ever pulled
        self.assertEqual(source.iter_calls, 1)
        self.assertEqual(source.pulled, 2)
        self.assertFalse(self.path.exists())

    # -- empty stream -------------------------------------------------------

    def test_empty_iterator_is_a_noop_and_creates_nothing(self):
        self.assertEqual(self.chain.append_batch_stream("t", iter([])), [])
        self.assertFalse(self.path.exists())
        self.assertEqual(self.chain.append_batch_stream("t", (e for e in ())), [])
        self.assertFalse(self.path.exists())
        # also a no-op against an existing file: bytes untouched
        self.chain.append("u", {"v": 1})
        before = self.path.read_bytes()
        self.assertEqual(self.chain.append_batch_stream("t", iter([])), [])
        self.assertEqual(self.path.read_bytes(), before)

    def test_empty_iterator_does_not_read_a_corrupt_history(self):
        self.chain.append("t", {})
        rows = self.read_rows()
        rows[0]["event"] = {"x": 9}
        with self.path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, sort_keys=True) + "\n")
        before = self.path.read_bytes()
        # an empty stream returns [] without ever scanning the broken chain
        self.assertEqual(self.chain.append_batch_stream("t", iter([])), [])
        self.assertEqual(self.path.read_bytes(), before)

    # -- JSON boundary ------------------------------------------------------

    def test_tenant_boundary_does_not_enter_the_iterable(self):
        source = CountingIterable([{}])
        with self.assertRaises(ValueError):
            self.chain.append_batch_stream(float("nan"), source)
        self.assertEqual(source.iter_calls, 0)
        self.assertEqual(source.pulled, 0)
        self.assertFalse(self.path.exists())

    def test_each_produced_event_crosses_the_json_boundary(self):
        def inf_stream():
            yield {"ok": 1}
            yield float("inf")

        with self.assertRaises(ValueError):
            self.chain.append_batch_stream("t", inf_stream())

        cyc = []
        cyc.append(cyc)
        with self.assertRaises(ValueError):
            self.chain.append_batch_stream("t", iter([cyc]))
        with self.assertRaises(ValueError):
            self.chain.append_batch_stream("t", iter([{1: "x"}]))
        self.assertFalse(self.path.exists())

    def test_iterator_exception_propagates_verbatim_and_writes_nothing(self):
        with self.assertRaises(Boom):
            self.chain.append_batch_stream(
                "t", raising_stream([{"i": 1}, {"i": 2}, {"i": 3}], 2))
        self.assertFalse(self.path.exists())
        # even against an existing log the original error wins and no byte
        # from the already-produced prefix reaches disk
        self.chain.append("other", {})
        before = self.path.read_bytes()
        with self.assertRaises(Boom):
            self.chain.append_batch_stream(
                "t", raising_stream([{"i": 1}, {"i": 2}], 1))
        self.assertEqual(self.path.read_bytes(), before)

    # -- records, links and byte-equivalence with append_batch --------------

    def test_records_have_five_fields_contiguous_seqs_and_internal_links(self):
        items = self.chain.append_batch_stream(
            "t", ({"i": i} for i in range(3)))
        self.assertEqual([set(it) for it in items], [set(
            ("tenant", "seq", "event", "prev", "hash"))] * 3)
        self.assertEqual([it["seq"] for it in items], [1, 2, 3])
        self.assertEqual(items[0]["prev"], ZERO)
        self.assertEqual(items[1]["prev"], items[0]["hash"])
        self.assertEqual(items[2]["prev"], items[1]["hash"])
        self.assertEqual(items, self.read_rows())

    def test_stream_continues_the_existing_chain_tail(self):
        first = self.chain.append("t", {"i": 0})
        self.chain.append("a", {})
        items = self.chain.append_batch_stream("t", iter([{"i": 1}, {"i": 2}]))
        self.assertEqual([it["seq"] for it in items], [2, 3])
        self.assertEqual(items[0]["prev"], first["hash"])
        self.assertEqual(items[1]["prev"], items[0]["hash"])
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": 3})

    def test_committed_bytes_identical_to_append_batch(self):
        events = [{"i": i, "s": "审计"} for i in range(5)]

        def build(seed_path, use_stream):
            chain = AuditChain(seed_path)
            chain.append("a", {})
            if use_stream:
                chain.append_batch_stream("t", (e for e in events))
            else:
                chain.append_batch("t", events)
            chain.append("z", {})
            return chain

        streamed = build(self.path, True)
        batched = build(self.path.with_name("batched.jsonl"), False)
        self.assertEqual(streamed.path.read_bytes(), batched.path.read_bytes())

    def test_bytes_identical_over_a_legacy_file_without_trailing_newline(self):
        events = [{"v": 1}, {"v": 2}, {"v": 3}]
        for use_stream, name in ((True, "stream.jsonl"),
                                 (False, "batch.jsonl")):
            chain = AuditChain(self.path.with_name(name))
            chain.append_batch("t", [{"v": 0}])
            chain.path.write_bytes(chain.path.read_bytes().rstrip(b"\n"))
            if use_stream:
                chain.append_batch_stream("t", iter(events))
            else:
                chain.append_batch("t", events)
        stream_bytes = (self.path.with_name("stream.jsonl")).read_bytes()
        batch_bytes = (self.path.with_name("batch.jsonl")).read_bytes()
        self.assertEqual(stream_bytes, batch_bytes)
        rows = [json.loads(l) for l in stream_bytes.splitlines()]
        self.assertEqual(len(rows), 4)

    # -- corrupt history ----------------------------------------------------

    def test_corrupt_history_raises_without_partial_records(self):
        self.chain.append("t", {})
        rows = self.read_rows()
        rows[0]["event"] = {"x": 9}
        with self.path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, sort_keys=True) + "\n")
        before = self.path.read_bytes()
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.append_batch_stream(
                "t", ({"i": i} for i in range(3)))
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("t", 1, "digest", 1))
        self.assertEqual(self.path.read_bytes(), before)

    def test_corrupt_other_tenant_does_not_block_the_stream(self):
        self.chain.append("b", {})
        rows = self.read_rows()
        rows[0]["event"] = {"x": 1}
        with self.path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, sort_keys=True) + "\n")
        items = self.chain.append_batch_stream("t", iter([{}, {}]))
        self.assertEqual([it["seq"] for it in items], [1, 2])
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": 2})

    def test_illegal_utf8_reports_same_first_broken_point_as_append(self):
        valid = {"tenant": "t", "seq": 1, "event": {}, "prev": ZERO}
        valid["hash"] = AuditChain._hash(valid)
        before = (json.dumps(valid, sort_keys=True) + "\n").encode("utf-8") \
            + b"\xff"
        self.path.write_bytes(before)
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.append_batch_stream("t", iter([{}, {}]))
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("t", 2, "missing", 2))
        self.assertEqual(self.path.read_bytes(), before)


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
            def stream():
                for j in range(size):
                    yield {"w": i, "j": j}

            try:
                local = self.chain.append_batch_stream("t", stream())
            except Exception as e:  # noqa: BLE001
                with box:
                    errors.append(repr(e))
                return
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
        self.assertEqual(sorted(it["seq"] for _, it in results),
                         list(range(1, total + 1)))
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

    def test_stream_batches_interleave_with_single_appends(self):
        done = []
        box = threading.Lock()

        def batch_worker():
            def stream():
                for j in range(20):
                    yield {"i": j}

            items = self.chain.append_batch_stream("a", stream())
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
        for tenant, _it in done:
            counts[tenant] += 1
        self.assertEqual(counts, {"a": 80, "b": 80})
        self.assertEqual(self.chain.verify("a"), {"ok": True, "count": 80})
        self.assertEqual(self.chain.verify("b"), {"ok": True, "count": 80})
        self.assertTrue(self.chain.verify_all()["ok"])


if __name__ == "__main__":
    unittest.main()
