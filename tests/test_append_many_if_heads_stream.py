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

KEYS = {"tenant", "events", "expected_count", "expected_hash"}


def entry(tenant, events, expected_count=0, expected_hash=ZERO):
    return {"tenant": tenant, "events": events,
            "expected_count": expected_count, "expected_hash": expected_hash}


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


class AppendManyIfHeadsStreamTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.path = Path(self.tmp.name) / "audit.jsonl"
        self.chain = AuditChain(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def read_rows(self):
        return [json.loads(l) for l in self.path.read_text().splitlines()]

    def tamper(self, index, **changes):
        rows = self.read_rows()
        rows[index].update(changes)
        with self.path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, sort_keys=True) + "\n")

    # -- container boundary -------------------------------------------------

    def test_bare_scalars_and_containers_are_value_errors(self):
        for bad in (b"x", bytearray(b"x"), "x", {"a": 1}, {},
                    None, True, 1, 1.5, object()):
            with self.subTest(bad=type(bad).__name__):
                with self.assertRaises(ValueError):
                    self.chain.append_many_if_heads_stream(bad)
        # rejection happens before any file is created
        self.assertFalse(self.path.exists())

    def test_lists_tuples_generators_and_iterators_are_accepted(self):
        a = self.chain.append("a", {})
        entries = [
            entry("a", [{"i": 1}], 1, a["hash"]),
            entry("b", [{"i": 2}]),
        ]
        self.assertEqual(
            [(it["tenant"], it["event"]) for it in
             self.chain.append_many_if_heads_stream(entries)],
            [("a", {"i": 1}), ("b", {"i": 2})],
        )
        heads = {h["tenant"]: h for h in self.chain.heads()["tenants"]}
        tuple_items = self.chain.append_many_if_heads_stream(tuple([
            entry("a", [{"i": 2}], heads["a"]["count"], heads["a"]["hash"]),
            entry("b", [{"i": 2}], heads["b"]["count"], heads["b"]["hash"]),
        ]))
        self.assertEqual([it["seq"] for it in tuple_items], [3, 2])
        gen_items = self.chain.append_many_if_heads_stream(
            e for e in [entry("c", [{}])])
        self.assertEqual([it["seq"] for it in gen_items], [1])
        # a plain iterator object is itself a valid one-shot input
        iter_items = self.chain.append_many_if_heads_stream(
            iter([entry("d", [{}])]))
        self.assertEqual([it["seq"] for it in iter_items], [1])
        # a set is an iterable container too: accepted (entries are dicts
        # and therefore unhashable, so only the empty set is a valid set of
        # entries -- the container shape itself is what is accepted here)
        self.assertEqual(self.chain.append_many_if_heads_stream(set()), [])
        self.assertTrue(self.chain.verify_all()["ok"])

    # -- single consumption -------------------------------------------------

    def test_iterator_is_consumed_exactly_once(self):
        source = CountingIterable(
            [entry(f"a{i}", [{"i": i}]) for i in range(3)])
        items = self.chain.append_many_if_heads_stream(source)
        self.assertEqual(source.iter_calls, 1)
        self.assertEqual(source.pulled, 3)
        self.assertEqual([it["event"] for it in items],
                         [{"i": 0}, {"i": 1}, {"i": 2}])
        # a generator object is exhausted by the call
        gen = (e for e in [entry("g", [{}])])
        self.chain.append_many_if_heads_stream(gen)
        with self.assertRaises(StopIteration):
            next(gen)

    def test_production_order_is_preserved(self):
        plan = ["a", "b", "c", "a", "c", "b", "a"]

        def stream():
            # each entry asserts a unique fresh tenant, so the physical
            # JSONL order is exactly the stream production order
            for i, t in enumerate(plan):
                yield entry(f"{t}{i}", [{"i": i, "t": t}])

        items = self.chain.append_many_if_heads_stream(stream())
        self.assertEqual([it["event"]["i"] for it in items], list(range(7)))
        self.assertEqual([it["event"]["t"] for it in items], plan)
        self.assertEqual([(r["tenant"], r["seq"]) for r in self.read_rows()],
                         [(it["tenant"], it["seq"]) for it in items])

    def test_pulling_stops_at_the_first_bad_produced_member(self):
        source = CountingIterable(
            [entry("a", [{}]), {"tenant": "b"}, entry("never", [{}])])
        with self.assertRaises(ValueError):
            self.chain.append_many_if_heads_stream(source)
        # the container was entered once and only the two members preceding
        # the boundary failure were ever pulled
        self.assertEqual(source.iter_calls, 1)
        self.assertEqual(source.pulled, 2)
        self.assertFalse(self.path.exists())

    # -- empty stream -------------------------------------------------------

    def test_empty_iterator_is_a_noop_and_creates_nothing(self):
        self.assertEqual(self.chain.append_many_if_heads_stream(iter([])), [])
        self.assertFalse(self.path.exists())
        self.assertEqual(
            self.chain.append_many_if_heads_stream(e for e in ()), [])
        self.assertFalse(self.path.exists())
        # also a no-op against an existing file: bytes untouched
        self.chain.append("u", {"v": 1})
        before = self.path.read_bytes()
        self.assertEqual(self.chain.append_many_if_heads_stream(iter([])), [])
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
        self.assertEqual(self.chain.append_many_if_heads_stream(iter([])), [])
        self.assertEqual(self.path.read_bytes(), before)

    # -- member boundary ----------------------------------------------------

    def test_members_must_be_objects_with_exact_four_keys(self):
        good = entry("g", [{}])
        bad_members = [
            None, True, 1, 1.5, "x", [], [1], (1,),
            {"tenant": "t", "events": [{}], "expected_count": 0},  # no hash
            {"tenant": "t", "events": [{}], "expected_hash": ZERO},  # no count
            {"tenant": "t", "expected_count": 0,
             "expected_hash": ZERO},                                # no events
            {"events": [{}], "expected_count": 0,
             "expected_hash": ZERO},                                # no tenant
            {},                                                    # empty
            dict(entry("t", [{}]), extra=1),                       # extra key
            {"tenant": "t", "event": [{}], "expected_count": 0,
             "expected_hash": ZERO},                               # singular key
        ]
        for bad in bad_members:
            with self.subTest(bad=repr(bad)):
                with self.assertRaises(ValueError):
                    self.chain.append_many_if_heads_stream(
                        iter([good, bad, entry("u", [{}])]))
        self.assertFalse(self.path.exists())

    def test_non_string_entry_key_is_value_error(self):
        with self.assertRaises(ValueError):
            self.chain.append_many_if_heads_stream(iter([
                {1: "x", "tenant": "t", "events": [{}],
                 "expected_count": 0, "expected_hash": ZERO}]))
        self.assertFalse(self.path.exists())

    def test_events_must_be_a_non_empty_list(self):
        for bad_events in (None, True, 1, 1.5, "x", {"a": 1}, ({}), {0}, []):
            with self.subTest(bad=repr(bad_events)):
                with self.assertRaises(ValueError):
                    self.chain.append_many_if_heads_stream(
                        iter([entry("t", bad_events)]))
        self.assertFalse(self.path.exists())

    def test_count_and_hash_boundaries(self):
        for bad_count in (True, False, -1, 1.0, 0.0, "0", None, [0]):
            with self.assertRaises(ValueError):
                self.chain.append_many_if_heads_stream(
                    iter([entry("t", [{}], bad_count, ZERO)]))
        for bad_hash in ("", "a" * 63, "a" * 65, "A" * 64, "g" * 64,
                         ZERO[:-1] + "A", 0, bytes(64), None):
            with self.assertRaises(ValueError):
                self.chain.append_many_if_heads_stream(
                    iter([entry("t", [{}], 0, bad_hash)]))
        self.assertFalse(self.path.exists())

    def test_duplicate_canonical_tenant_rejected(self):
        def ve(values):
            with self.assertRaises(ValueError):
                self.chain.append_many_if_heads_stream(iter(values))
            self.assertFalse(self.path.exists())

        ve([entry("t", [{}]), entry("t", [{}], 3, "f" * 64)])
        # object key order does not create a second identity
        ve([entry({"k": 1, "j": 2}, [{}]),
            entry({"j": 2, "k": 1}, [{}])])
        # distinct JSON identities are not duplicates
        items = self.chain.append_many_if_heads_stream(
            iter([entry(1, [{}]), entry(1.0, [{}]), entry(True, [{}]),
                  entry("1", [{}])]))
        self.assertEqual([it["seq"] for it in items], [1, 1, 1, 1])
        self.assertTrue(self.chain.verify_all()["ok"])

    def test_tenants_and_events_cross_the_json_boundary(self):
        def ve(values):
            with self.assertRaises(ValueError):
                self.chain.append_many_if_heads_stream(iter(values))
            self.assertFalse(self.path.exists())

        ve([entry(float("nan"), [{}])])
        ve([entry("t", [{}]), entry("u", [float("inf")])])
        ve([entry("t", [{1: "x"}])])  # non-string event key
        cyc = []
        cyc.append(cyc)
        ve([entry("t", [cyc])])
        d = {}
        d["self"] = d
        ve([entry(d, [{}])])

    def test_value_error_takes_priority_over_corrupt_history(self):
        self.chain.append("t", {"v": 1})
        row = json.loads(self.path.read_text())
        row["event"] = {"v": 2}  # digest corruption at line 1
        self.path.write_text(json.dumps(row, sort_keys=True) + "\n")
        before = self.path.read_bytes()
        for bad_entries in (
            b"raw",
            [entry("t", [float("nan")], 0, ZERO)],
            [entry("t", [{}], True, ZERO)],
            [entry("t", [{}], 0, "X" * 64)],
            [entry(float("nan"), [{}], 0, ZERO)],
            [entry("t", [])],
            [entry("t", [{}]), entry("t", [{}])],
            [None],
        ):
            with self.assertRaises(ValueError):
                self.chain.append_many_if_heads_stream(
                    bad_entries if isinstance(bad_entries, (bytes, bytearray))
                    else iter(bad_entries))
        self.assertEqual(self.path.read_bytes(), before)

    def test_iterator_exception_propagates_verbatim_and_writes_nothing(self):
        entries = [entry("a", [{}]), entry("b", [{}]), entry("c", [{}])]
        with self.assertRaises(Boom):
            self.chain.append_many_if_heads_stream(
                raising_stream(entries, 2))
        self.assertFalse(self.path.exists())
        # even against an existing log the original error wins and no byte
        # from the already-produced prefix reaches disk
        self.chain.append("other", {})
        before = self.path.read_bytes()
        with self.assertRaises(Boom):
            self.chain.append_many_if_heads_stream(
                raising_stream(entries, 1))
        self.assertEqual(self.path.read_bytes(), before)

    # -- success: numbering, linking, physical order ------------------------

    def test_empty_heads_create_and_link_multi_tenant_batches(self):
        items = self.chain.append_many_if_heads_stream(iter([
            entry("a", [{"i": 1}, {"i": 2}, {"i": 3}]),
            entry("b", [{"i": 1}]),
        ]))
        self.assertEqual(
            [(it["tenant"], it["seq"]) for it in items],
            [("a", 1), ("a", 2), ("a", 3), ("b", 1)],
        )
        self.assertEqual(items[0]["prev"], ZERO)
        self.assertEqual(items[1]["prev"], items[0]["hash"])
        self.assertEqual(items[2]["prev"], items[1]["hash"])
        self.assertEqual(items[3]["prev"], ZERO)
        for it in items:
            self.assertEqual(set(it),
                             set(("tenant", "seq", "event", "prev", "hash")))
            self.assertEqual(it["hash"], AuditChain._hash(it))
        self.assertEqual(items, self.read_rows())
        self.assertEqual(self.chain.verify("a"), {"ok": True, "count": 3})
        self.assertEqual(self.chain.verify("b"), {"ok": True, "count": 1})

    def test_each_tenant_continues_its_asserted_tail(self):
        self.chain.append("a", {"i": 0})
        self.chain.append("a", {"i": 1})
        a2 = self.read_rows()[-1]
        b1 = self.chain.append("b", {"i": 0})
        items = self.chain.append_many_if_heads_stream(iter([
            entry("b", [{"i": 1}, {"i": 2}], 1, b1["hash"]),
            entry("a", [{"i": 2}], 2, a2["hash"]),
            entry("c", [{"i": 1}], 0, ZERO),
        ]))
        self.assertEqual(
            [(it["tenant"], it["seq"]) for it in items],
            [("b", 2), ("b", 3), ("a", 3), ("c", 1)],
        )
        self.assertEqual(items[0]["prev"], b1["hash"])
        self.assertEqual(items[1]["prev"], items[0]["hash"])
        self.assertEqual(items[2]["prev"], a2["hash"])
        self.assertEqual(items[3]["prev"], ZERO)
        self.assertTrue(self.chain.verify_all()["ok"])

    def test_assertions_built_from_heads_succeed(self):
        self.chain.append_many([
            {"tenant": "a", "event": {"i": 0}},
            {"tenant": "b", "event": {"i": 0}},
        ])
        heads = {h["tenant"]: h for h in self.chain.heads()["tenants"]}
        items = self.chain.append_many_if_heads_stream(iter([
            entry("a", [{"i": 1}], heads["a"]["count"], heads["a"]["hash"]),
            entry("b", [{"i": 1}, {"i": 2}],
                  heads["b"]["count"], heads["b"]["hash"]),
        ]))
        self.assertEqual([(it["tenant"], it["seq"]) for it in items],
                         [("a", 2), ("b", 2), ("b", 3)])
        self.assertTrue(self.chain.verify_all()["ok"])

    def test_distinct_json_tenants_are_independent_heads(self):
        self.chain.append(1, {})
        items = self.chain.append_many_if_heads_stream(iter([
            entry(1, [{}], 1, self.chain.head(1)["hash"]),
            entry("1", [{}, {}], 0, ZERO),
        ]))
        self.assertEqual([(it["tenant"], it["seq"]) for it in items],
                         [(1, 2), ("1", 1), ("1", 2)])
        self.assertTrue(self.chain.verify_all()["ok"])

    def test_committed_bytes_identical_to_append_many_if_heads(self):
        def seed(chain):
            chain.append("a", {"i": 0})
            chain.append("z", {"i": 0})
            chain.append("a", {"i": 1})

        def planned(chain):
            a_head = chain.head("a")
            return [
                entry("a", [{"i": 2}, {"i": 3, "s": "审计"}],
                      a_head["count"], a_head["hash"]),
                entry("b", [{"i": 1}]),
            ]

        streamed = AuditChain(self.path.with_name("stream.jsonl"))
        seed(streamed)
        stream_items = streamed.append_many_if_heads_stream(
            e for e in planned(streamed))

        batched = AuditChain(self.path.with_name("batch.jsonl"))
        seed(batched)
        batch_items = batched.append_many_if_heads(planned(batched))

        self.assertEqual(stream_items, batch_items)
        self.assertEqual(streamed.path.read_bytes(),
                         batched.path.read_bytes())
        streamed.append("z", {"i": 1})
        batched.append("z", {"i": 1})
        self.assertEqual(streamed.path.read_bytes(),
                         batched.path.read_bytes())

    def test_bytes_identical_over_a_legacy_file_without_trailing_newline(self):
        for use_stream, name in ((True, "stream.jsonl"),
                                 (False, "heads.jsonl")):
            chain = AuditChain(self.path.with_name(name))
            chain.append("t", {"v": 0})
            tail = chain.read_tenant("t")[-1]
            chain.path.write_bytes(chain.path.read_bytes().rstrip(b"\n"))
            args = [entry("t", [{"v": 1}], 1, tail["hash"]),
                    entry("u", [{"v": 2}])]
            if use_stream:
                chain.append_many_if_heads_stream(iter(args))
            else:
                chain.append_many_if_heads(args)
        stream_bytes = (self.path.with_name("stream.jsonl")).read_bytes()
        heads_bytes = (self.path.with_name("heads.jsonl")).read_bytes()
        self.assertEqual(stream_bytes, heads_bytes)
        rows = [json.loads(l) for l in stream_bytes.splitlines()]
        self.assertEqual(len(rows), 3)

    # -- missing log: only all-(0, ZERO) may create -------------------------

    def test_missing_log_nonempty_count_conflicts_and_creates_nothing(self):
        with self.assertRaises(AuditChainConflictError) as cm:
            self.chain.append_many_if_heads_stream(iter([
                entry("a", [{}], 0, ZERO),
                entry("b", [{}], 1, ZERO),
            ]))
        e = cm.exception
        self.assertEqual((e.tenant, e.reason), ("b", "conflict"))
        self.assertEqual((e.expected_count, e.expected_hash), (1, ZERO))
        self.assertEqual((e.actual_count, e.actual_hash), (0, ZERO))
        self.assertFalse(self.path.exists())

    def test_missing_log_wrong_hash_conflicts_and_creates_nothing(self):
        with self.assertRaises(AuditChainConflictError) as cm:
            self.chain.append_many_if_heads_stream(iter([
                entry("a", [{}], 0, ZERO),
                entry("b", [{}], 0, "f" * 64),
                entry("c", [{}], 0, "f" * 64),  # never reached: b is first
            ]))
        e = cm.exception
        self.assertEqual(e.tenant, "b")
        self.assertEqual((e.actual_count, e.actual_hash), (0, ZERO))
        self.assertFalse(self.path.exists())

    def test_missing_log_reports_first_mismatching_entry_in_order(self):
        with self.assertRaises(AuditChainConflictError) as cm:
            self.chain.append_many_if_heads_stream(iter([
                entry("a", [{}], 0, "f" * 64),
                entry("b", [{}], 1, ZERO),
            ]))
        self.assertEqual(cm.exception.tenant, "a")
        self.assertFalse(self.path.exists())

    # -- conflicts on an existing log ---------------------------------------

    def test_stale_head_conflicts_with_actual_tail_and_writes_nothing(self):
        first = self.chain.append("t", {"i": 1})
        second = self.chain.append("t", {"i": 2})
        u1 = self.chain.append("u", {})
        before = self.path.read_bytes()
        with self.assertRaises(AuditChainConflictError) as cm:
            self.chain.append_many_if_heads_stream(iter([
                entry("u", [{}], 1, u1["hash"]),
                entry("t", [{"i": 3}], 1, first["hash"]),
            ]))
        e = cm.exception
        self.assertEqual(e.tenant, "t")
        self.assertEqual((e.expected_count, e.expected_hash),
                         (1, first["hash"]))
        self.assertEqual((e.actual_count, e.actual_hash),
                         (2, second["hash"]))
        self.assertEqual(self.path.read_bytes(), before)

    def test_first_mismatching_entry_in_entries_order_conflicts(self):
        self.chain.append("a", {})
        self.chain.append("b", {})
        before = self.path.read_bytes()
        ha = self.chain.head("a")
        with self.assertRaises(AuditChainConflictError) as cm:
            self.chain.append_many_if_heads_stream(iter([
                entry("a", [{}], ha["count"], ha["hash"]),
                entry("b", [{}], 9, ZERO),
            ]))
        self.assertEqual(cm.exception.tenant, "b")
        self.assertEqual(self.path.read_bytes(), before)
        with self.assertRaises(AuditChainConflictError) as cm:
            self.chain.append_many_if_heads_stream(iter([
                entry("a", [{}], 9, ZERO),
                entry("b", [{}], 9, ZERO),
            ]))
        self.assertEqual(cm.exception.tenant, "a")
        self.assertEqual(self.path.read_bytes(), before)

    def test_conflict_for_interleaved_other_tenant_writes_nothing(self):
        self.chain.append("a", {})
        before = self.path.read_bytes()
        with self.assertRaises(AuditChainConflictError):
            self.chain.append_many_if_heads_stream(iter([
                entry("a", [{}], 1, self.chain.head("a")["hash"]),
                entry("b", [{}], 1, ZERO),
            ]))
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(self.chain.verify("b"), {"ok": True, "count": 0})

    # -- corrupt history: state errors, priority, tie-breaks ----------------

    def test_corrupt_history_raises_state_error_not_conflict(self):
        self.chain.append("t", {})
        self.chain.append("u", {})
        self.tamper(0, event={"x": 9})  # t digest broken at line 1
        before = self.path.read_bytes()
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.append_many_if_heads_stream(iter([
                entry("u", [{}], 9, ZERO),  # also a conflict...
                entry("t", [{}], 1, "f" * 64),
            ]))
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("t", 1, "digest", 1))
        self.assertEqual(self.path.read_bytes(), before)

    def test_earliest_physical_line_wins_across_broken_chains(self):
        self.chain.append("b", {})
        self.chain.append("a", {})
        self.chain.append("a", {})
        self.tamper(0, event={"x": 1})  # b broken at line 1
        self.tamper(2, event={"x": 2})  # a broken at line 3
        before = self.path.read_bytes()
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.append_many_if_heads_stream(iter([
                entry("a", [{}], 2, ZERO), entry("b", [{}], 1, ZERO)]))
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("b", 1, "digest", 1))
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.append_many_if_heads_stream(iter([
                entry("b", [{}], 1, ZERO), entry("a", [{}], 2, ZERO)]))
        self.assertEqual((cm.exception.tenant, cm.exception.line), ("b", 1))
        self.assertEqual(self.path.read_bytes(), before)

    def test_equal_physical_line_broken_by_input_order(self):
        self.path.write_text("not json\n", encoding="utf-8")
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.append_many_if_heads_stream(
                iter([entry("a", [{}]), entry("b", [{}])]))
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("a", 1, "missing", 1))
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.append_many_if_heads_stream(
                iter([entry("b", [{}]), entry("a", [{}])]))
        self.assertEqual((cm.exception.tenant, cm.exception.line), ("b", 1))

    def test_bad_utf8_line_is_missing_for_every_affected_chain(self):
        valid = {"tenant": "t", "seq": 1, "event": {}, "prev": ZERO}
        valid["hash"] = AuditChain._hash(valid)
        before = (json.dumps(valid, sort_keys=True) + "\n").encode() + b"\xff"
        self.path.write_bytes(before)
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.append_many_if_heads_stream(iter([
                entry("u", [{}]), entry("t", [{}], 1, valid["hash"])]))
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("u", 1, "missing", 2))
        self.assertEqual(self.path.read_bytes(), before)

    def test_sequence_error_location(self):
        self.chain.append("t", {})
        self.chain.append("t", {})
        rows = self.read_rows()
        rows[1]["seq"] = 3
        with self.path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, sort_keys=True) + "\n")
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.append_many_if_heads_stream(
                iter([entry("t", [{}], 2, ZERO)]))
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("t", 2, "sequence", 2))

    def test_corruption_of_an_uninvolved_tenant_does_not_block(self):
        self.chain.append("b", {})
        self.tamper(0, event={"x": 1})
        items = self.chain.append_many_if_heads_stream(
            iter([entry("t", [{}]), entry("u", [{}])]))
        self.assertEqual([(it["tenant"], it["seq"]) for it in items],
                         [("t", 1), ("u", 1)])
        self.assertTrue(self.chain.verify("t")["ok"])
        self.assertTrue(self.chain.verify("u")["ok"])

    # -- compatibility with every existing entry point ----------------------

    def test_result_verifiable_by_all_read_and_offline_entries(self):
        self.chain.append("a", {"i": 0})
        ha = self.chain.head("a")
        items = self.chain.append_many_if_heads_stream(iter([
            entry("a", [{"i": 1}, {"i": 2}], ha["count"], ha["hash"]),
            entry("b", [{"i": 1}, {"i": 2}], 0, ZERO),
        ]))
        self.assertEqual(self.chain.verify("a"), {"ok": True, "count": 3})
        self.assertEqual(self.chain.verify("b"), {"ok": True, "count": 2})
        self.assertTrue(self.chain.verify_all()["ok"])
        data = self.path.read_bytes()
        self.assertTrue(self.chain.verify_bytes(data, "a")["ok"])
        self.assertTrue(self.chain.verify_all_bytes(data)["ok"])
        self.chain.verify_heads(self.chain.heads()["tenants"])
        self.assertEqual([r["seq"] for r in self.chain.read_tenant("a")],
                         [1, 2, 3])
        export_b = self.chain.export_tenant("b")
        self.assertEqual(self.chain.verify_bytes(export_b, "b")["count"], 2)
        fresh = AuditChain(self.path.with_name("fresh.jsonl"))
        fresh.import_tenant("b", export_b)
        self.assertEqual(fresh.verify("b"), {"ok": True, "count": 2})
        for it in items:
            self.assertEqual(
                set(it), {"tenant", "seq", "event", "prev", "hash"})

    def test_append_if_head_continues_from_a_committed_tail(self):
        items = self.chain.append_many_if_heads_stream(iter([
            entry("t", [{"i": 1}, {"i": 2}]), entry("u", [{}])]))
        t_tail = next(it for it in items
                      if it["tenant"] == "t" and it["seq"] == 2)
        nxt = self.chain.append_if_head("t", {"i": 3}, 2, t_tail["hash"])
        self.assertEqual((nxt["seq"], nxt["prev"]), (3, t_tail["hash"]))
        with self.assertRaises(AuditChainConflictError):
            self.chain.append_batch_if_head("t", [{}], 2, t_tail["hash"])


class AppendManyIfHeadsStreamConcurrencyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.path = Path(self.tmp.name) / "audit.jsonl"
        self.chain = AuditChain(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_stream_groups_are_indivisible_intervals(self):
        # every entry asserts a worker-private fresh tenant, so every group
        # commits; the assertion then checks each winning group occupies one
        # contiguous physical interval whose records chain internally
        n_thread, size = 10, 6
        results, errors = [], []
        box = threading.Lock()

        def worker(w):
            def stream():
                for j in range(size):
                    yield {
                        "tenant": f"t{w}_{j}",
                        "events": [{"w": w, "j": j}],
                        "expected_count": 0,
                        "expected_hash": ZERO,
                    }

            try:
                local = self.chain.append_many_if_heads_stream(stream())
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
        # each winning call occupies one contiguous physical interval
        by_worker = {}
        for row in rows:
            by_worker.setdefault(row["event"]["w"], []).append(row)
        for w, group in by_worker.items():
            self.assertEqual([r["event"]["j"] for r in group],
                             list(range(size)), w)
            for r in group:
                self.assertEqual(r["seq"], 1, w)
                self.assertEqual(r["prev"], ZERO, w)
        self.assertTrue(self.chain.verify_all()["ok"])

    def test_concurrent_cas_loops_commit_every_batch_exactly_once(self):
        n, size = 8, 3
        committed = []
        box = threading.Lock()

        def worker(w):
            for _ in range(500):
                heads = {h["tenant"]: h
                         for h in self.chain.heads()["tenants"]}
                a = heads.get("a", {"count": 0, "hash": ZERO})
                b = heads.get("b", {"count": 0, "hash": ZERO})

                def stream():
                    yield entry("a", [{"w": w, "j": j} for j in range(size)],
                                a["count"], a["hash"])
                    yield entry("b", [{"w": w, "j": j} for j in range(size)],
                                b["count"], b["hash"])

                try:
                    items = self.chain.append_many_if_heads_stream(stream())
                    with box:
                        committed.extend(items)
                    return
                except AuditChainConflictError:
                    continue
            self.fail("worker never won the CAS race")

        threads = [threading.Thread(target=worker, args=(w,))
                   for w in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        total = n * size * 2
        self.assertEqual(len(committed), total)
        r = self.chain.verify_all()
        self.assertTrue(r["ok"], r)
        per = {h["tenant"]: h["count"] for h in r["tenants"]}
        self.assertEqual(per, {"a": n * size, "b": n * size})
        for w in range(n):
            evs = sorted((it["seq"], it["event"]["j"])
                         for it in committed
                         if it["event"]["w"] == w and it["tenant"] == "a")
            self.assertEqual([j for _, j in evs], list(range(size)))

    def test_concurrent_readers_never_observe_a_half_group(self):
        stop = threading.Event()
        problems = []
        box = threading.Lock()

        def reader():
            while not stop.is_set():
                r = self.chain.verify_all()
                if not r["ok"]:
                    with box:
                        problems.append(r)

        readers = [threading.Thread(target=reader, daemon=True)
                   for _ in range(5)]
        for t in readers:
            t.start()

        def writer(w):
            for _ in range(20):
                for _try in range(500):
                    heads = {h["tenant"]: h
                             for h in self.chain.heads()["tenants"]}
                    a = heads.get("a", {"count": 0, "hash": ZERO})
                    b = heads.get("b", {"count": 0, "hash": ZERO})

                    def stream():
                        yield entry("a", [{"w": w}, {"w": w}],
                                    a["count"], a["hash"])
                        yield entry("b", [{"w": w}], b["count"], b["hash"])

                    try:
                        self.chain.append_many_if_heads_stream(stream())
                        break
                    except AuditChainConflictError:
                        continue

        threads = [threading.Thread(target=writer, args=(w,)) for w in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        stop.set()
        for t in readers:
            t.join(timeout=2)

        self.assertEqual(problems, [])
        r = self.chain.verify_all()
        self.assertTrue(r["ok"], r)
        counts = {x["tenant"]: x["count"] for x in r["tenants"]}
        self.assertEqual((counts["a"], counts["b"]), (320, 160))


if __name__ == "__main__":
    unittest.main()
