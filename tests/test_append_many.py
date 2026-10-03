import json
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app import AuditChain, AuditChainConflictError, AuditChainStateError

ZERO = "0" * 64


class AppendManyTest(unittest.TestCase):
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

    # --- boundary: shape, key set, JSON boundary all ValueError first ------

    def test_entries_must_be_a_list(self):
        for bad in (None, True, 1, 1.5, "x", {"a": 1}, ("x",), {1, 2}):
            with self.assertRaises(ValueError):
                self.chain.append_many(bad)
        self.assertFalse(self.path.exists())

    def test_members_must_be_objects(self):
        for bad in (None, True, 1, 1.5, "x", [], [1], (1,)):
            with self.assertRaises(ValueError):
                self.chain.append_many([self.entry(), bad, self.entry("u")])
        self.assertFalse(self.path.exists())

    def test_key_set_must_be_exactly_tenant_and_event(self):
        good = self.entry()
        bad_shapes = [
            {"tenant": "t"},                                  # missing event
            {"event": {}},                                    # missing tenant
            {},                                               # both missing
            {"tenant": "t", "event": {}, "extra": 1},         # extra key
            {"tenant": "t", "event": {}, "seq": 1},           # reserved key
            {"t": "t", "event": {}},                          # misspelled tenant
            {"tenant": "t", "ev": {}},                        # misspelled event
        ]
        for shape in bad_shapes:
            with self.assertRaises(ValueError):
                self.chain.append_many([good, shape])
        self.assertFalse(self.path.exists())

    def test_non_string_entry_key_is_value_error(self):
        with self.assertRaises(ValueError):
            self.chain.append_many([{1: "x", "tenant": "t", "event": {}}])

    def test_tenants_and_events_cross_the_json_boundary(self):
        with self.assertRaises(ValueError):
            self.chain.append_many([{"tenant": float("nan"), "event": {}}])
        with self.assertRaises(ValueError):
            self.chain.append_many([self.entry("t"),
                                    {"tenant": "u", "event": float("inf")}])
        with self.assertRaises(ValueError):
            self.chain.append_many([self.entry("t", {1: "x"})])  # non-string key
        cyc = []
        cyc.append(cyc)
        with self.assertRaises(ValueError):
            self.chain.append_many([self.entry("t", cyc)])
        d = {}
        d["self"] = d
        with self.assertRaises(ValueError):
            self.chain.append_many([{"tenant": d, "event": {}}])
        self.assertFalse(self.path.exists())

    def test_validation_runs_in_input_order_and_stops_at_first_bad(self):
        # the first malformed entry is the one reported, regardless of later
        with self.assertRaises(ValueError):
            self.chain.append_many([
                self.entry("a"),
                {"tenant": "b"},  # bad: missing event
                {"tenant": float("nan"), "event": {}},
            ])
        self.assertFalse(self.path.exists())

    # --- empty list: validated no-op that never reads or creates ----------

    def test_empty_list_is_a_noop_and_creates_nothing(self):
        self.assertEqual(self.chain.append_many([]), [])
        self.assertFalse(self.path.exists())
        # against an existing file: bytes untouched
        self.chain.append("u", {"v": 1})
        before = self.path.read_bytes()
        self.assertEqual(self.chain.append_many([]), [])
        self.assertEqual(self.path.read_bytes(), before)

    def test_empty_list_does_not_read_a_corrupt_log(self):
        valid = {"tenant": "t", "seq": 1, "event": {}, "prev": ZERO}
        valid["hash"] = AuditChain._hash(valid)
        raw = (json.dumps(valid, sort_keys=True) + "\n").encode() + b"not json\n"
        self.path.write_bytes(raw)
        before = self.path.read_bytes()
        # the no-op returns before any lease/scan, so corruption is invisible
        self.assertEqual(self.chain.append_many([]), [])
        self.assertEqual(self.path.read_bytes(), before)

    # --- numbering, linking, physical order --------------------------------

    def test_distinct_tenants_each_start_from_one(self):
        items = self.chain.append_many([
            self.entry("a", {"i": 1}),
            self.entry("b", {"i": 1}),
            self.entry("a", {"i": 2}),
            self.entry("b", {"i": 2}),
            self.entry("b", {"i": 3}),
        ])
        self.assertEqual(
            [(it["tenant"], it["seq"]) for it in items],
            [("a", 1), ("b", 1), ("a", 2), ("b", 2), ("b", 3)],
        )
        self.assertEqual(items[0]["prev"], ZERO)
        self.assertEqual(items[1]["prev"], ZERO)
        self.assertEqual(items[2]["prev"], items[0]["hash"])
        self.assertEqual(items[3]["prev"], items[1]["hash"])
        self.assertEqual(items[4]["prev"], items[3]["hash"])
        # returned records match the on-disk rows 1:1, five fields each
        self.assertEqual(items, self.read_rows())
        for it in items:
            self.assertEqual(set(it), {"tenant", "seq", "event", "prev", "hash"})
        self.assertTrue(self.chain.verify_all()["ok"])

    def test_each_tenant_continues_its_current_chain_tail(self):
        a0 = self.chain.append("a", {"i": 0})
        b0 = self.chain.append("b", {"i": 0})
        a1 = self.chain.append("a", {"i": 1})
        items = self.chain.append_many([
            self.entry("a", {"i": 2}),
            self.entry("b", {"i": 1}),
            self.entry("c", {"i": 1}),
            self.entry("a", {"i": 3}),
        ])
        self.assertEqual([it["seq"] for it in items], [3, 2, 1, 4])
        self.assertEqual(items[0]["prev"], a1["hash"])
        self.assertEqual(items[1]["prev"], b0["hash"])
        self.assertEqual(items[2]["prev"], ZERO)
        self.assertEqual(items[3]["prev"], items[0]["hash"])
        self.assertEqual(self.chain.verify("a")["count"], 4)
        self.assertEqual(self.chain.verify("b")["count"], 2)
        self.assertEqual(self.chain.verify("c")["count"], 1)

    def test_physical_jsonl_order_is_input_order(self):
        items = self.chain.append_many([
            self.entry(t, {"i": i})
            for i, t in enumerate(["a", "b", "c", "a", "c", "b", "a"])
        ])
        on_disk = self.read_rows()
        self.assertEqual([(r["tenant"], r["seq"]) for r in on_disk],
                         [(r["tenant"], r["seq"]) for r in items])

    def test_repeated_tenant_records_are_contiguous_in_numbering(self):
        items = self.chain.append_many([self.entry("t", {"i": i}) for i in range(5)])
        self.assertEqual([it["seq"] for it in items], [1, 2, 3, 4, 5])
        for prev, cur in zip(items, items[1:]):
            self.assertEqual(cur["prev"], prev["hash"])

    def test_distinct_json_identities_are_independent_chains(self):
        tenants = [1, 1.0, True, "1", {"k": "v"}]
        entries = []
        for t in tenants:
            entries += [self.entry(t), self.entry(t)]
        items = self.chain.append_many(entries)
        seqs = {}
        for it in items:
            seqs.setdefault(json.dumps(it["tenant"], sort_keys=True), []).append(it["seq"])
        self.assertTrue(all(s == [1, 2] for s in seqs.values()))
        self.assertEqual(len(seqs), 5)
        self.assertTrue(self.chain.verify_all()["ok"])

    def test_byte_equivalent_to_sequential_appends(self):
        self.chain.append("a", {"i": 0})
        self.chain.append("c", {"i": 0})
        plan = [("a", 1), ("b", 1), ("a", 2), ("b", 2), ("c", 1), ("a", 3)]
        items = self.chain.append_many(
            [self.entry(t, {"i": i}) for t, i in plan])
        batched = self.path.read_bytes()

        other = AuditChain(self.path.with_name("other.jsonl"))
        other.append("a", {"i": 0})
        other.append("c", {"i": 0})
        sequential = [other.append(t, {"i": i}) for t, i in plan]
        self.assertEqual(sequential, items)
        self.assertEqual(other.path.read_bytes(), batched)
        self.assertEqual(other.verify_all(), self.chain.verify_all())

    def test_creates_file_only_after_validation_passes(self):
        self.assertFalse(self.path.exists())
        self.chain.append_many([self.entry("t")])
        self.assertTrue(self.path.exists())

    # --- chain-state validation and error priority -------------------------

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
            self.chain.append_many([self.entry("t"), self.entry("u")])
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
            self.chain.append_many([self.entry("a"), self.entry("b")])
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("b", 1, "digest", 1))
        # reversing input order cannot move the later defect ahead
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.append_many([self.entry("b"), self.entry("a")])
        self.assertEqual((cm.exception.tenant, cm.exception.line), ("b", 1))
        self.assertEqual(self.path.read_bytes(), before)

    def test_equal_physical_line_broken_by_input_order(self):
        # an unparseable first line is a missing defect for every chain scan
        self.path.write_text(b"not json\n".decode(), encoding="utf-8")
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.append_many([self.entry("a"), self.entry("b")])
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("a", 1, "missing", 1))
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.append_many([self.entry("b"), self.entry("a")])
        self.assertEqual((cm.exception.tenant, cm.exception.line), ("b", 1))

    def test_bad_utf8_line_is_missing_for_every_affected_chain(self):
        valid = {"tenant": "t", "seq": 1, "event": {}, "prev": ZERO}
        valid["hash"] = AuditChain._hash(valid)
        before = (json.dumps(valid, sort_keys=True) + "\n").encode() + b"\xff"
        self.path.write_bytes(before)
        with self.assertRaises(AuditChainStateError) as cm:
            # t expects seq 2 on the bad line; a fresh tenant u expects 1
            self.chain.append_many([self.entry("u"), self.entry("t")])
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("u", 1, "missing", 2))
        self.assertEqual(self.path.read_bytes(), before)

    def test_sequence_error_location(self):
        self.chain.append("t", {})
        self.chain.append("t", {})
        rows = self.read_rows()
        rows[1]["seq"] = 3  # gap: expected 2
        with self.path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, sort_keys=True) + "\n")
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.append_many([self.entry("t")])
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("t", 2, "sequence", 2))

    def test_corruption_of_an_uninvolved_tenant_does_not_block(self):
        self.chain.append("b", {})
        self.tamper(0, event={"x": 1})
        items = self.chain.append_many([self.entry("t"), self.entry("u")])
        self.assertEqual([it["seq"] for it in items], [1, 1])
        self.assertTrue(self.chain.verify("t")["ok"])
        self.assertTrue(self.chain.verify("u")["ok"])

    def test_other_tenants_interleaved_records_are_preserved(self):
        self.chain.append("x", {"i": 0})
        items = self.chain.append_many([self.entry("a"), self.entry("b")])
        rows = self.read_rows()
        self.assertEqual([r["tenant"] for r in rows], ["x", "a", "b"])
        self.assertEqual(items[0]["prev"], ZERO)
        self.assertTrue(self.chain.verify_all()["ok"])

    def test_legacy_file_without_trailing_newline_still_one_line_each(self):
        self.chain.append("t", {"v": 1})
        self.path.write_bytes(self.path.read_bytes().rstrip(b"\n"))
        items = self.chain.append_many([self.entry("t"), self.entry("u")])
        lines = self.path.read_text().splitlines()
        self.assertEqual(len(lines), 3)
        for raw in lines:
            json.loads(raw)
        self.assertEqual([it["seq"] for it in items], [2, 1])
        self.assertTrue(self.chain.verify_all()["ok"])

    # --- compatibility with every existing entry point ---------------------

    def test_result_verifiable_by_all_read_and_offline_entries(self):
        self.chain.append("a", {"i": 0})
        items = self.chain.append_many([
            self.entry("a", {"i": 1}),
            self.entry("b", {"i": 1}),
            self.entry("a", {"i": 2}),
            self.entry("b", {"i": 2}),
        ])
        self.assertEqual(self.chain.verify("a"), {"ok": True, "count": 3})
        self.assertEqual(self.chain.verify("b"), {"ok": True, "count": 2})
        self.assertTrue(self.chain.verify_all()["ok"])
        data = self.path.read_bytes()
        self.assertTrue(self.chain.verify_bytes(data, "a")["ok"])
        self.assertTrue(self.chain.verify_all_bytes(data)["ok"])
        self.assertEqual([r["seq"] for r in self.chain.read_tenant("a")], [1, 2, 3])
        self.assertEqual([r["seq"] for r in self.chain.read_tenant("b")], [1, 2])
        export_a = self.chain.export_tenant("a")
        self.assertEqual(self.chain.verify_bytes(export_a, "a")["count"], 3)
        # export bytes graft onto a fresh log and verify identically there
        fresh = AuditChain(self.path.with_name("fresh.jsonl"))
        fresh.import_tenant("a", export_a)
        self.assertEqual(fresh.verify("a"), {"ok": True, "count": 3})
        self.assertEqual(fresh.export_tenant("a"), export_a)
        for it in items:
            self.assertEqual(set(it), {"tenant", "seq", "event", "prev", "hash"})

    def test_import_conflict_semantics_unchanged(self):
        self.chain.append_many([self.entry("t"), self.entry("u")])
        source = AuditChain(self.path.with_name("source.jsonl"))
        source.append("t", {"i": 0})
        exported = source.export_tenant("t")
        # t already has a record here: import keeps its empty-chain conflict
        with self.assertRaises(AuditChainConflictError) as cm:
            self.chain.import_tenant("t", exported)
        self.assertEqual(cm.exception.reason, "conflict")
        self.assertEqual(cm.exception.actual_count, 1)
        # u also exists, so importing a u history conflicts too; a brand-new
        # tenant imports cleanly alongside t/u records already interleaved
        u_source = AuditChain(self.path.with_name("u_source.jsonl"))
        u_source.append("u", {"i": 0})
        with self.assertRaises(AuditChainConflictError):
            self.chain.import_tenant("u", u_source.export_tenant("u"))
        v_source = AuditChain(self.path.with_name("v_source.jsonl"))
        v_source.append("v", {"i": 0})
        self.chain.import_tenant("v", v_source.export_tenant("v"))
        self.assertTrue(self.chain.verify_all()["ok"])

    def test_append_if_head_continues_from_a_many_tail(self):
        items = self.chain.append_many([self.entry("t"), self.entry("t")])
        head = items[-1]
        nxt = self.chain.append_if_head("t", {"i": 9}, 2, head["hash"])
        self.assertEqual((nxt["seq"], nxt["prev"]), (3, head["hash"]))
        with self.assertRaises(AuditChainConflictError):
            self.chain.append_if_head("t", {}, 2, head["hash"])

    # --- concurrency: indivisible group, serial writers, stable readers ----

    def test_concurrent_many_calls_keep_every_chain_contiguous(self):
        n_thread, size = 10, 6
        tenants = ["a", "b", "c"]
        results, errors = [], []
        box = threading.Lock()

        def worker(w):
            local = []
            try:
                local = self.chain.append_many([
                    self.entry(tenants[j % len(tenants)], {"w": w, "j": j})
                    for j in range(size)
                ])
            except Exception as e:  # noqa: BLE001
                with box:
                    errors.append(repr(e))
            with box:
                results.extend((w, it) for it in local)

        threads = [threading.Thread(target=worker, args=(w,)) for w in range(n_thread)]
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
        # it, each tenant's occurrences are consecutive continuations of the
        # global chain head the winning call observed, with internal links
        by_worker = {}
        for row in rows:
            by_worker.setdefault(row["event"]["w"], []).append(row)
        for w, group in by_worker.items():
            self.assertEqual([r["event"]["j"] for r in group], list(range(size)))
            last = {}
            for r in group:
                seq = r["seq"]
                if r["tenant"] in last:
                    self.assertEqual(seq, last[r["tenant"]] + 1, w)
                last[r["tenant"]] = seq
            # internal prev links of same-tenant neighbors must chain
            prev_hash = {}
            for r in group:
                if r["tenant"] in prev_hash:
                    self.assertEqual(r["prev"], prev_hash[r["tenant"]], w)
                prev_hash[r["tenant"]] = r["hash"]
        # globally every tenant chain is 1..count exactly once
        r = self.chain.verify_all()
        self.assertTrue(r["ok"], r)
        per_tenant = {x["tenant"]: x["count"] for x in r["tenants"]}
        for t in tenants:
            self.assertGreaterEqual(per_tenant[t], 1)
        self.assertEqual(sum(per_tenant.values()), total)

    def test_concurrent_many_and_single_appends_serialize(self):
        done = []
        box = threading.Lock()

        def many_worker(w):
            items = self.chain.append_many([
                self.entry("a" if j % 2 else "m", {"w": w, "j": j})
                for j in range(10)
            ])
            with box:
                done.extend((it["tenant"], it) for it in items)

        def single_worker(w):
            local = [self.chain.append("s", {"w": w, "j": j}) for j in range(10)]
            with box:
                done.extend(("s", it) for it in local)

        threads = [threading.Thread(target=many_worker, args=(w,)) for w in range(6)]
        threads += [threading.Thread(target=single_worker, args=(w,)) for w in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        counts = {}
        for tenant, it in done:
            counts[tenant] = counts.get(tenant, 0) + 1
        self.assertEqual(counts, {"m": 30, "a": 30, "s": 40})
        self.assertTrue(self.chain.verify_all()["ok"])

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

        readers = [threading.Thread(target=reader, daemon=True) for _ in range(5)]
        for t in readers:
            t.start()

        def writer(w):
            for _ in range(20):
                self.chain.append_many([
                    self.entry(t, {"w": w}) for t in ("a", "b", "a", "c")
                ])

        writers = [threading.Thread(target=writer, args=(w,)) for w in range(8)]
        for t in writers:
            t.start()
        for t in writers:
            t.join()
        stop.set()
        for t in readers:
            t.join(timeout=2)

        self.assertEqual(problems, [])
        r = self.chain.verify_all()
        self.assertTrue(r["ok"], r)
        counts = {x["tenant"]: x["count"] for x in r["tenants"]}
        self.assertEqual((counts["a"], counts["b"], counts["c"]), (320, 160, 160))

    def test_corrupt_history_under_contention_leaves_bytes_untouched(self):
        self.chain.append("t", {})
        self.tamper(0, event={"x": 9})
        before = self.path.read_bytes()
        details = []
        box = threading.Lock()

        def attempt(i):
            try:
                self.chain.append_many([self.entry("t"), self.entry("u", {"i": i})])
                with box:
                    details.append(("success",))
            except AuditChainStateError as e:
                with box:
                    details.append((e.tenant, e.seq, e.reason, e.line))

        threads = [threading.Thread(target=attempt, args=(i,)) for i in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(details, [("t", 1, "digest", 1)] * 16)
        self.assertEqual(self.path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
