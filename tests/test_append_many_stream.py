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
    # produced member.
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


class AppendManyStreamTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.path = Path(self.tmp.name) / "audit.jsonl"
        self.chain = AuditChain(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def read_rows(self):
        return [json.loads(l) for l in self.path.read_text().splitlines()]

    def entry(self, tenant="t", event=None):
        return {"tenant": tenant, "event": {} if event is None else event}

    # -- container boundary -------------------------------------------------

    def test_bare_scalars_and_containers_are_value_errors(self):
        for bad in (b"x", bytearray(b"x"), "x", {"a": 1}, {},
                    None, True, 1, 1.5, object()):
            with self.subTest(bad=type(bad).__name__):
                with self.assertRaises(ValueError):
                    self.chain.append_many_stream(bad)
        # rejection happens before any file is created
        self.assertFalse(self.path.exists())

    def test_lists_tuples_sets_generators_and_iterators_are_accepted(self):
        entries = [self.entry("a", {"i": 1}), self.entry("b", {"i": 2})]
        self.assertEqual(
            [it["event"] for it in self.chain.append_many_stream(entries)],
            [{"i": 1}, {"i": 2}],
        )
        tuple_items = self.chain.append_many_stream(tuple(entries))
        self.assertEqual([it["tenant"] for it in tuple_items], ["a", "b"])
        gen_items = self.chain.append_many_stream(e for e in entries)
        self.assertEqual([it["seq"] for it in gen_items], [3, 3])
        # a plain iterator object is itself a valid one-shot input
        iter_items = self.chain.append_many_stream(iter(entries))
        self.assertEqual([it["seq"] for it in iter_items], [4, 4])
        # a set is an iterable container too: accepted (entries are dicts
        # and therefore unhashable, so only the empty set is a valid set of
        # entries -- the container shape itself is what is accepted here)
        self.assertEqual(self.chain.append_many_stream(set()), [])
        self.assertTrue(self.chain.verify_all()["ok"])

    # -- single consumption -------------------------------------------------

    def test_iterator_is_consumed_exactly_once(self):
        source = CountingIterable(
            [self.entry("a", {"i": i}) for i in range(3)])
        items = self.chain.append_many_stream(source)
        self.assertEqual(source.iter_calls, 1)
        self.assertEqual(source.pulled, 3)
        self.assertEqual([it["event"] for it in items],
                         [{"i": 0}, {"i": 1}, {"i": 2}])
        # a generator object is exhausted by the call
        gen = (e for e in [self.entry("g")])
        self.chain.append_many_stream(gen)
        with self.assertRaises(StopIteration):
            next(gen)

    def test_production_order_is_preserved(self):
        plan = ["a", "b", "c", "a", "c", "b", "a"]

        def stream():
            for i, t in enumerate(plan):
                yield self.entry(t, {"i": i})

        items = self.chain.append_many_stream(stream())
        self.assertEqual([it["event"]["i"] for it in items], list(range(7)))
        self.assertEqual([it["tenant"] for it in items], plan)
        self.assertEqual([(r["tenant"], r["seq"]) for r in self.read_rows()],
                         [(it["tenant"], it["seq"]) for it in items])

    def test_pulling_stops_at_the_first_bad_produced_member(self):
        source = CountingIterable(
            [self.entry("a"), {"tenant": "b"}, self.entry("never")])
        with self.assertRaises(ValueError):
            self.chain.append_many_stream(source)
        # the container was entered once and only the two members preceding
        # the boundary failure were ever pulled
        self.assertEqual(source.iter_calls, 1)
        self.assertEqual(source.pulled, 2)
        self.assertFalse(self.path.exists())

    # -- empty stream -------------------------------------------------------

    def test_empty_iterator_is_a_noop_and_creates_nothing(self):
        self.assertEqual(self.chain.append_many_stream(iter([])), [])
        self.assertFalse(self.path.exists())
        self.assertEqual(self.chain.append_many_stream(e for e in ()), [])
        self.assertFalse(self.path.exists())
        # also a no-op against an existing file: bytes untouched
        self.chain.append("u", {"v": 1})
        before = self.path.read_bytes()
        self.assertEqual(self.chain.append_many_stream(iter([])), [])
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
        self.assertEqual(self.chain.append_many_stream(iter([])), [])
        self.assertEqual(self.path.read_bytes(), before)

    # -- member boundary ----------------------------------------------------

    def test_members_must_be_objects_with_exactly_tenant_and_event(self):
        bad_members = [
            None, True, 1, 1.5, "x", [], [1], (1,),
            {"tenant": "t"},                           # missing event
            {"event": {}},                             # missing tenant
            {},                                        # both missing
            {"tenant": "t", "event": {}, "extra": 1},  # extra key
            {"tenant": "t", "event": {}, "seq": 1},    # reserved key
            {1: "x", "tenant": "t", "event": {}},      # non-string key
        ]
        for bad in bad_members:
            with self.subTest(bad=repr(bad)):
                with self.assertRaises(ValueError):
                    self.chain.append_many_stream(
                        iter([self.entry("a"), bad, self.entry("b")]))
        self.assertFalse(self.path.exists())

    def test_tenants_and_events_cross_the_json_boundary(self):
        with self.assertRaises(ValueError):
            self.chain.append_many_stream(
                iter([{"tenant": float("nan"), "event": {}}]))
        with self.assertRaises(ValueError):
            self.chain.append_many_stream(
                iter([self.entry("t"),
                      {"tenant": "u", "event": float("inf")}]))
        with self.assertRaises(ValueError):
            self.chain.append_many_stream(iter([self.entry("t", {1: "x"})]))
        cyc = []
        cyc.append(cyc)
        with self.assertRaises(ValueError):
            self.chain.append_many_stream(iter([self.entry("t", cyc)]))
        d = {}
        d["self"] = d
        with self.assertRaises(ValueError):
            self.chain.append_many_stream(iter([{"tenant": d, "event": {}}]))
        self.assertFalse(self.path.exists())

    def test_iterator_exception_propagates_verbatim_and_writes_nothing(self):
        entries = [self.entry("a"), self.entry("b"), self.entry("c")]
        with self.assertRaises(Boom):
            self.chain.append_many_stream(raising_stream(entries, 2))
        self.assertFalse(self.path.exists())
        # even against an existing log the original error wins and no byte
        # from the already-produced prefix reaches disk
        self.chain.append("other", {})
        before = self.path.read_bytes()
        with self.assertRaises(Boom):
            self.chain.append_many_stream(raising_stream(entries, 1))
        self.assertEqual(self.path.read_bytes(), before)

    # -- records, links and byte-equivalence with append_many ---------------

    def test_records_have_five_fields_and_cross_tenant_numbering(self):
        items = self.chain.append_many_stream(iter([
            self.entry("a", {"i": 1}),
            self.entry("b", {"i": 1}),
            self.entry("a", {"i": 2}),
            self.entry("b", {"i": 2}),
            self.entry("b", {"i": 3}),
        ]))
        self.assertEqual([set(it) for it in items],
                         [set(("tenant", "seq", "event", "prev", "hash"))] * 5)
        self.assertEqual(
            [(it["tenant"], it["seq"]) for it in items],
            [("a", 1), ("b", 1), ("a", 2), ("b", 2), ("b", 3)],
        )
        self.assertEqual(items[0]["prev"], ZERO)
        self.assertEqual(items[1]["prev"], ZERO)
        self.assertEqual(items[2]["prev"], items[0]["hash"])
        self.assertEqual(items[3]["prev"], items[1]["hash"])
        self.assertEqual(items[4]["prev"], items[3]["hash"])
        self.assertEqual(items, self.read_rows())
        self.assertTrue(self.chain.verify_all()["ok"])

    def test_stream_continues_each_existing_chain_tail(self):
        a0 = self.chain.append("a", {"i": 0})
        b0 = self.chain.append("b", {"i": 0})
        a1 = self.chain.append("a", {"i": 1})
        items = self.chain.append_many_stream(iter([
            self.entry("a", {"i": 2}),
            self.entry("b", {"i": 1}),
            self.entry("c", {"i": 1}),
            self.entry("a", {"i": 3}),
        ]))
        self.assertEqual([it["seq"] for it in items], [3, 2, 1, 4])
        self.assertEqual(items[0]["prev"], a1["hash"])
        self.assertEqual(items[1]["prev"], b0["hash"])
        self.assertEqual(items[2]["prev"], ZERO)
        self.assertEqual(items[3]["prev"], items[0]["hash"])
        self.assertEqual(self.chain.verify("a"), {"ok": True, "count": 4})
        self.assertEqual(self.chain.verify("b"), {"ok": True, "count": 2})
        self.assertEqual(self.chain.verify("c"), {"ok": True, "count": 1})

    def test_distinct_json_identities_are_independent_chains(self):
        tenants = [1, 1.0, True, "1", {"k": "v"}]
        entries = []
        for t in tenants:
            entries += [self.entry(t), self.entry(t)]
        items = self.chain.append_many_stream(e for e in entries)
        seqs = {}
        for it in items:
            seqs.setdefault(
                json.dumps(it["tenant"], sort_keys=True), []).append(it["seq"])
        self.assertTrue(all(s == [1, 2] for s in seqs.values()))
        self.assertEqual(len(seqs), 5)
        self.assertTrue(self.chain.verify_all()["ok"])

    def test_committed_bytes_identical_to_append_many(self):
        entries = [self.entry(t, {"i": i, "s": "审计"})
                   for i, t in enumerate(["a", "b", "a", "c", "b", "a"])]

        def build(seed_path, use_stream):
            chain = AuditChain(seed_path)
            chain.append("a", {"i": 0})
            chain.append("z", {"i": 0})
            if use_stream:
                streamed = chain.append_many_stream(e for e in entries)
            else:
                streamed = chain.append_many(entries)
            chain.append("z", {"i": 1})
            return chain, streamed

        streamed, stream_items = build(self.path, True)
        batched, batch_items = build(self.path.with_name("batched.jsonl"),
                                     False)
        self.assertEqual(stream_items, batch_items)
        self.assertEqual(streamed.path.read_bytes(),
                         batched.path.read_bytes())

    def test_bytes_identical_over_a_legacy_file_without_trailing_newline(self):
        entries = [self.entry("t", {"v": 1}), self.entry("u", {"v": 2})]
        for use_stream, name in ((True, "stream.jsonl"),
                                 (False, "many.jsonl")):
            chain = AuditChain(self.path.with_name(name))
            chain.append("t", {"v": 0})
            chain.path.write_bytes(chain.path.read_bytes().rstrip(b"\n"))
            if use_stream:
                chain.append_many_stream(iter(entries))
            else:
                chain.append_many(entries)
        stream_bytes = (self.path.with_name("stream.jsonl")).read_bytes()
        many_bytes = (self.path.with_name("many.jsonl")).read_bytes()
        self.assertEqual(stream_bytes, many_bytes)
        rows = [json.loads(l) for l in stream_bytes.splitlines()]
        self.assertEqual(len(rows), 3)

    # -- corrupt history ----------------------------------------------------

    def tamper(self, index, **changes):
        rows = self.read_rows()
        rows[index].update(changes)
        with self.path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, sort_keys=True) + "\n")

    def test_corrupt_history_raises_without_partial_records(self):
        self.chain.append("t", {})
        self.tamper(0, event={"x": 9})  # breaks digest at line 1
        before = self.path.read_bytes()
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.append_many_stream(
                iter([self.entry("t"), self.entry("u")]))
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("t", 1, "digest", 1))
        self.assertEqual(self.path.read_bytes(), before)

    def test_earliest_physical_line_wins_across_broken_chains(self):
        # line 1: tenant b with a broken digest
        # line 2: valid a seq 1
        # line 3: tenant a with a broken digest
        self.chain.append("b", {})
        self.chain.append("a", {})
        self.chain.append("a", {})
        self.tamper(0, event={"x": 1})
        self.tamper(2, event={"x": 2})
        before = self.path.read_bytes()
        # a listed first in input, but b's defect sits on an earlier line
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.append_many_stream(
                iter([self.entry("a"), self.entry("b")]))
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("b", 1, "digest", 1))
        # reversing input order cannot move the later defect ahead
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.append_many_stream(
                iter([self.entry("b"), self.entry("a")]))
        self.assertEqual((cm.exception.tenant, cm.exception.line), ("b", 1))
        self.assertEqual(self.path.read_bytes(), before)

    def test_equal_physical_line_broken_by_input_order(self):
        # an unparseable first line is a missing defect for every chain scan
        self.path.write_text("not json\n", encoding="utf-8")
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.append_many_stream(
                iter([self.entry("a"), self.entry("b")]))
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("a", 1, "missing", 1))
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.append_many_stream(
                iter([self.entry("b"), self.entry("a")]))
        self.assertEqual((cm.exception.tenant, cm.exception.line), ("b", 1))

    def test_sequence_error_location(self):
        self.chain.append("t", {})
        self.chain.append("t", {})
        rows = self.read_rows()
        rows[1]["seq"] = 3  # gap: expected 2
        with self.path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, sort_keys=True) + "\n")
        before = self.path.read_bytes()
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.append_many_stream(iter([self.entry("t")]))
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("t", 2, "sequence", 2))
        self.assertEqual(self.path.read_bytes(), before)

    def test_corrupt_other_tenant_does_not_block_the_stream(self):
        self.chain.append("b", {})
        self.tamper(0, event={"x": 1})
        items = self.chain.append_many_stream(
            iter([self.entry("t"), self.entry("u")]))
        self.assertEqual([it["seq"] for it in items], [1, 1])
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": 1})
        self.assertEqual(self.chain.verify("u"), {"ok": True, "count": 1})

    def test_illegal_utf8_reports_same_first_broken_point_as_append_many(self):
        valid = {"tenant": "t", "seq": 1, "event": {}, "prev": ZERO}
        valid["hash"] = AuditChain._hash(valid)
        before = (json.dumps(valid, sort_keys=True) + "\n").encode("utf-8") \
            + b"\xff"
        self.path.write_bytes(before)
        with self.assertRaises(AuditChainStateError) as cm:
            # t expects seq 2 on the bad line; a fresh tenant u expects 1,
            # and u appears first in the input
            self.chain.append_many_stream(
                iter([self.entry("u"), self.entry("t")]))
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("u", 1, "missing", 2))
        self.assertEqual(self.path.read_bytes(), before)


class AppendManyStreamConcurrencyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.path = Path(self.tmp.name) / "audit.jsonl"
        self.chain = AuditChain(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def entry(self, tenant="t", event=None):
        return {"tenant": tenant, "event": {} if event is None else event}

    def test_stream_groups_are_indivisible_intervals(self):
        n_thread, size = 10, 6
        tenants = ["a", "b", "c"]
        results, errors = [], []
        box = threading.Lock()

        def worker(w):
            def stream():
                for j in range(size):
                    yield {"tenant": tenants[j % len(tenants)],
                           "event": {"w": w, "j": j}}

            try:
                local = self.chain.append_many_stream(stream())
            except Exception as e:  # noqa: BLE001
                with box:
                    errors.append(repr(e))
                return
            with box:
                results.extend((w, it) for it in local)

        threads = [threading.Thread(target=worker, args=(w,))
                   for w in range(n_thread)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [])
        total = n_thread * size
        self.assertEqual(len(results), total)
        rows = [json.loads(l) for l in self.path.read_text().splitlines()]
        self.assertEqual(len(rows), total)
        # each winning call occupies one contiguous physical interval; within
        # it, each tenant's occurrences chain internally
        by_worker = {}
        for row in rows:
            by_worker.setdefault(row["event"]["w"], []).append(row)
        for w, group in by_worker.items():
            self.assertEqual([r["event"]["j"] for r in group],
                             list(range(size)), w)
            last = {}
            prev_hash = {}
            for r in group:
                if r["tenant"] in last:
                    self.assertEqual(r["seq"], last[r["tenant"]] + 1, w)
                    self.assertEqual(r["prev"], prev_hash[r["tenant"]], w)
                last[r["tenant"]] = r["seq"]
                prev_hash[r["tenant"]] = r["hash"]
        r = self.chain.verify_all()
        self.assertTrue(r["ok"], r)
        per_tenant = {x["tenant"]: x["count"] for x in r["tenants"]}
        self.assertEqual(sum(per_tenant.values()), total)

    def test_stream_groups_interleave_with_single_appends(self):
        done = []
        box = threading.Lock()

        def many_worker(w):
            def stream():
                for j in range(10):
                    yield {"tenant": "a" if j % 2 else "m",
                           "event": {"w": w, "j": j}}

            items = self.chain.append_many_stream(stream())
            with box:
                done.extend((it["tenant"], it) for it in items)

        def single_worker(w):
            local = [self.chain.append("s", {"w": w, "j": j})
                     for j in range(10)]
            with box:
                done.extend(("s", it) for it in local)

        threads = [threading.Thread(target=many_worker, args=(w,))
                   for w in range(6)]
        threads += [threading.Thread(target=single_worker, args=(w,))
                    for w in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        counts = {}
        for tenant, _it in done:
            counts[tenant] = counts.get(tenant, 0) + 1
        self.assertEqual(counts, {"m": 30, "a": 30, "s": 40})
        self.assertTrue(self.chain.verify_all()["ok"])


if __name__ == "__main__":
    unittest.main()
