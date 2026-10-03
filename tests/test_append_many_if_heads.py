import json
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app import AuditChain, AuditChainConflictError, AuditChainStateError, ZERO

KEYS = {"tenant", "events", "expected_count", "expected_hash"}


def entry(tenant, events, expected_count=0, expected_hash=ZERO):
    return {"tenant": tenant, "events": events,
            "expected_count": expected_count, "expected_hash": expected_hash}


class AppendManyIfHeadsTest(unittest.TestCase):
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

    # --- boundary: shape, key set, JSON boundary all ValueError first ------

    def test_entries_must_be_a_list(self):
        for bad in (None, True, 1, 1.5, "x", {"a": 1}, ("x",), {1, 2}):
            with self.assertRaises(ValueError):
                self.chain.append_many_if_heads(bad)
        self.assertFalse(self.path.exists())

    def test_members_must_be_objects(self):
        good = entry("t", [{}])
        for bad in (None, True, 1, 1.5, "x", [], [1], (1,)):
            with self.assertRaises(ValueError):
                self.chain.append_many_if_heads([good, bad, entry("u", [{}])])
        self.assertFalse(self.path.exists())

    def test_key_set_must_be_exact(self):
        bad_shapes = [
            {"tenant": "t", "events": [{}], "expected_count": 0},          # missing hash
            {"tenant": "t", "events": [{}], "expected_hash": ZERO},        # missing count
            {"tenant": "t", "expected_count": 0, "expected_hash": ZERO},   # missing events
            {"events": [{}], "expected_count": 0, "expected_hash": ZERO},  # missing tenant
            {},                                                                        # empty
            dict(entry("t", [{}]), extra=1),                                           # extra key
            dict(entry("t", [{}]), seq=1),                                             # reserved key
            {"tenant": "t", "event": [{}], "expected_count": 0,
             "expected_hash": ZERO},                                                  # singular event key
        ]
        for shape in bad_shapes:
            with self.assertRaises(ValueError):
                self.chain.append_many_if_heads([entry("g", [{}]), shape])
        self.assertFalse(self.path.exists())

    def test_non_string_entry_key_is_value_error(self):
        with self.assertRaises(ValueError):
            self.chain.append_many_if_heads(
                [{1: "x", "tenant": "t", "events": [{}],
                  "expected_count": 0, "expected_hash": ZERO}])
        self.assertFalse(self.path.exists())

    def test_events_must_be_a_non_empty_list(self):
        # a bare non-list value is not a batch
        for bad_events in (None, True, 1, 1.5, "x", {"a": 1}, ({}), {0}):
            with self.assertRaises(ValueError):
                self.chain.append_many_if_heads([entry("t", bad_events)])
        # unlike append_batch_if_head, an empty events list is invalid here:
        # the empty *entries* list is the only validated no-op
        with self.assertRaises(ValueError):
            self.chain.append_many_if_heads([entry("t", [])])
        with self.assertRaises(ValueError):
            self.chain.append_many_if_heads(
                [entry("a", [{}]), entry("b", [])])
        self.assertFalse(self.path.exists())

    def test_count_must_be_non_negative_plain_int(self):
        for bad in (True, False, -1, 1.0, 0.0, "0", None, [0]):
            with self.assertRaises(ValueError):
                self.chain.append_many_if_heads([entry("t", [{}], bad, ZERO)])
        self.assertFalse(self.path.exists())

    def test_hash_must_be_64_lowercase_hex(self):
        for bad in ("", "a" * 63, "a" * 65, "A" * 64, "g" * 64,
                    ZERO[:-1] + "A", 0, bytes(64), None):
            with self.assertRaises(ValueError):
                self.chain.append_many_if_heads([entry("t", [{}], 0, bad)])
        self.assertFalse(self.path.exists())

    def test_duplicate_canonical_tenant_rejected(self):
        def ve(entries):
            with self.assertRaises(ValueError):
                self.chain.append_many_if_heads(entries)
            self.assertFalse(self.path.exists())

        ve([entry("t", [{}]), entry("t", [{}], 3, "f" * 64)])
        # object key order does not create a second identity
        ve([entry({"k": 1, "j": 2}, [{}]),
            entry({"j": 2, "k": 1}, [{}])])
        # distinct JSON identities are not duplicates
        items = self.chain.append_many_if_heads([
            entry(1, [{}]), entry(1.0, [{}]), entry(True, [{}]),
            entry("1", [{}]),
        ])
        self.assertEqual([it["seq"] for it in items], [1, 1, 1, 1])
        self.assertTrue(self.chain.verify_all()["ok"])

    def test_tenants_and_events_cross_the_json_boundary(self):
        def ve(entries):
            with self.assertRaises(ValueError):
                self.chain.append_many_if_heads(entries)
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

    def test_validation_runs_in_input_order_and_stops_at_first_bad(self):
        with self.assertRaises(ValueError):
            self.chain.append_many_if_heads([
                entry("a", [{}]),
                {"tenant": "b"},  # bad shape
                entry(float("nan"), [{}]),
            ])
        self.assertFalse(self.path.exists())

    def test_value_error_takes_priority_over_corrupt_history(self):
        self.chain.append("t", {"v": 1})
        row = json.loads(self.path.read_text())
        row["event"] = {"v": 2}  # digest corruption at line 1
        self.path.write_text(json.dumps(row, sort_keys=True) + "\n")
        before = self.path.read_bytes()
        for bad_entries in (
            "not-a-list",
            [entry("t", [float("nan")], 0, ZERO)],
            [entry("t", [{}], True, ZERO)],
            [entry("t", [{}], 0, "X" * 64)],
            [entry(float("nan"), [{}], 0, ZERO)],
            [entry("t", [])],
            [entry("t", [{}]), entry("t", [{}])],
            [None],
        ):
            with self.assertRaises(ValueError):
                self.chain.append_many_if_heads(bad_entries)
        self.assertEqual(self.path.read_bytes(), before)

    # --- empty list: validated no-op that never reads or creates ----------

    def test_empty_list_is_a_noop_and_creates_nothing(self):
        self.assertEqual(self.chain.append_many_if_heads([]), [])
        self.assertFalse(self.path.exists())
        self.chain.append("u", {"v": 1})
        before = self.path.read_bytes()
        self.assertEqual(self.chain.append_many_if_heads([]), [])
        self.assertEqual(self.path.read_bytes(), before)

    def test_empty_list_does_not_read_a_corrupt_log(self):
        valid = {"tenant": "t", "seq": 1, "event": {}, "prev": ZERO}
        valid["hash"] = AuditChain._hash(valid)
        raw = (json.dumps(valid, sort_keys=True) + "\n").encode() + b"not json\n"
        self.path.write_bytes(raw)
        before = self.path.read_bytes()
        self.assertEqual(self.chain.append_many_if_heads([]), [])
        self.assertEqual(self.path.read_bytes(), before)

    # --- success: numbering, linking, physical order -----------------------

    def test_empty_heads_create_and_link_multi_tenant_batches(self):
        items = self.chain.append_many_if_heads([
            entry("a", [{"i": 1}, {"i": 2}, {"i": 3}]),
            entry("b", [{"i": 1}]),
        ])
        self.assertEqual(
            [(it["tenant"], it["seq"]) for it in items],
            [("a", 1), ("a", 2), ("a", 3), ("b", 1)],
        )
        self.assertEqual(items[0]["prev"], ZERO)
        self.assertEqual(items[1]["prev"], items[0]["hash"])
        self.assertEqual(items[2]["prev"], items[1]["hash"])
        self.assertEqual(items[3]["prev"], ZERO)
        for it in items:
            self.assertEqual(set(it), set(("tenant", "seq", "event", "prev", "hash")))
            self.assertEqual(it["hash"], AuditChain._hash(it))
        self.assertEqual(items, self.read_rows())
        self.assertEqual(self.chain.verify("a"), {"ok": True, "count": 3})
        self.assertEqual(self.chain.verify("b"), {"ok": True, "count": 1})

    def test_each_tenant_continues_its_asserted_tail(self):
        a2 = self.chain.append("a", {"i": 0})
        self.chain.append("a", {"i": 1})
        a2 = self.read_rows()[-1]
        b1 = self.chain.append("b", {"i": 0})
        items = self.chain.append_many_if_heads([
            entry("b", [{"i": 1}, {"i": 2}], 1, b1["hash"]),
            entry("a", [{"i": 2}], 2, a2["hash"]),
            entry("c", [{"i": 1}], 0, ZERO),
        ])
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
        items = self.chain.append_many_if_heads([
            entry("a", [{"i": 1}], heads["a"]["count"], heads["a"]["hash"]),
            entry("b", [{"i": 1}, {"i": 2}],
                  heads["b"]["count"], heads["b"]["hash"]),
        ])
        self.assertEqual([(it["tenant"], it["seq"]) for it in items],
                         [("a", 2), ("b", 2), ("b", 3)])
        self.assertTrue(self.chain.verify_all()["ok"])

    def test_physical_jsonl_order_is_entry_then_event_order(self):
        self.chain.append("x", {"i": 0})
        items = self.chain.append_many_if_heads([
            entry("a", [{"i": 1}, {"i": 2}], 0, ZERO),
            entry("b", [{"i": 1}], 0, ZERO),
        ])
        rows = self.read_rows()
        self.assertEqual([r["tenant"] for r in rows], ["x", "a", "a", "b"])
        self.assertEqual(
            [(r["tenant"], r["seq"]) for r in rows[1:]],
            [(r["tenant"], r["seq"]) for r in items],
        )
        self.assertTrue(self.chain.verify_all()["ok"])

    def test_distinct_json_tenants_are_independent_heads(self):
        self.chain.append(1, {})
        items = self.chain.append_many_if_heads([
            entry(1, [{}], 1, self.chain.head(1)["hash"]),
            entry("1", [{}, {}], 0, ZERO),
        ])
        self.assertEqual([(it["tenant"], it["seq"]) for it in items],
                         [(1, 2), ("1", 1), ("1", 2)])
        self.assertTrue(self.chain.verify_all()["ok"])

    def test_byte_equivalent_to_sequential_appends_from_same_heads(self):
        self.chain.append("a", {"i": 0})
        self.chain.append("c", {"i": 0})
        self.chain.append("a", {"i": 1})
        heads = {h["tenant"]: h for h in self.chain.heads()["tenants"]}
        # records are emitted entry by entry, events of one entry before the
        # next entry, so the equivalent sequential order is a,a,b,b,c
        plan = [("a", {"i": 2}), ("a", {"i": 3}), ("b", {"i": 1}),
                ("b", {"i": 2}), ("c", {"i": 1})]
        items = self.chain.append_many_if_heads([
            entry("a", [{"i": 2}, {"i": 3}], heads["a"]["count"], heads["a"]["hash"]),
            entry("b", [{"i": 1}, {"i": 2}], 0, ZERO),
            entry("c", [{"i": 1}], heads["c"]["count"], heads["c"]["hash"]),
        ])
        batched = self.path.read_bytes()

        other = AuditChain(self.path.with_name("other.jsonl"))
        other.append("a", {"i": 0})
        other.append("c", {"i": 0})
        other.append("a", {"i": 1})
        sequential = [other.append(t, e) for t, e in plan]
        self.assertEqual(sequential, items)
        self.assertEqual(other.path.read_bytes(), batched)
        self.assertEqual(other.verify_all(), self.chain.verify_all())

    def test_byte_equivalent_to_append_many_when_assertions_match(self):
        self.chain.append("a", {"i": 0})
        heads = {h["tenant"]: h for h in self.chain.heads()["tenants"]}
        items = self.chain.append_many_if_heads([
            entry("a", [{"i": 1}, {"i": 2}], heads["a"]["count"], heads["a"]["hash"]),
            entry("b", [{"i": 1}, {"i": 2}], 0, ZERO),
        ])

        plain = AuditChain(self.path.with_name("plain.jsonl"))
        plain.append("a", {"i": 0})
        # append_many emits in the same entry/event physical order
        many_items = plain.append_many([
            {"tenant": "a", "event": {"i": 1}},
            {"tenant": "a", "event": {"i": 2}},
            {"tenant": "b", "event": {"i": 1}},
            {"tenant": "b", "event": {"i": 2}},
        ])
        self.assertEqual(many_items, items)
        self.assertEqual(plain.path.read_bytes(), self.path.read_bytes())

    def test_legacy_file_without_trailing_newline_still_one_line_each(self):
        self.chain.append("t", {"v": 1})
        self.path.write_bytes(self.path.read_bytes().rstrip(b"\n"))
        tail = json.loads(self.path.read_text())
        items = self.chain.append_many_if_heads([
            entry("t", [{"v": 2}], 1, tail["hash"]),
            entry("u", [{}], 0, ZERO),
        ])
        lines = self.path.read_text().splitlines()
        self.assertEqual(len(lines), 3)
        for raw in lines:
            json.loads(raw)
        self.assertEqual([(it["tenant"], it["seq"]) for it in items],
                         [("t", 2), ("u", 1)])
        self.assertTrue(self.chain.verify_all()["ok"])

    # --- missing log: only all-(0, ZERO) may create ------------------------

    def test_missing_log_nonempty_count_conflicts_and_creates_nothing(self):
        with self.assertRaises(AuditChainConflictError) as cm:
            self.chain.append_many_if_heads([
                entry("a", [{}], 0, ZERO),
                entry("b", [{}], 1, ZERO),
            ])
        e = cm.exception
        self.assertEqual((e.tenant, e.reason), ("b", "conflict"))
        self.assertEqual((e.expected_count, e.expected_hash), (1, ZERO))
        self.assertEqual((e.actual_count, e.actual_hash), (0, ZERO))
        self.assertFalse(self.path.exists())

    def test_missing_log_wrong_hash_conflicts_and_creates_nothing(self):
        with self.assertRaises(AuditChainConflictError) as cm:
            self.chain.append_many_if_heads([
                entry("a", [{}], 0, ZERO),
                entry("b", [{}], 0, "f" * 64),
                entry("c", [{}], 0, "f" * 64),  # never reached: b is first
            ])
        e = cm.exception
        self.assertEqual(e.tenant, "b")
        self.assertEqual((e.actual_count, e.actual_hash), (0, ZERO))
        self.assertFalse(self.path.exists())

    def test_missing_log_reports_first_mismatching_entry_in_order(self):
        # reorder: a's bad assertion is now first even though b is also wrong
        with self.assertRaises(AuditChainConflictError) as cm:
            self.chain.append_many_if_heads([
                entry("a", [{}], 0, "f" * 64),
                entry("b", [{}], 1, ZERO),
            ])
        self.assertEqual(cm.exception.tenant, "a")
        self.assertFalse(self.path.exists())

    # --- conflicts on an existing log --------------------------------------

    def test_stale_head_conflicts_with_actual_tail_and_writes_nothing(self):
        first = self.chain.append("t", {"i": 1})
        second = self.chain.append("t", {"i": 2})
        u1 = self.chain.append("u", {})
        before = self.path.read_bytes()
        # u's assertion is correct; t's is stale and is the first mismatch
        with self.assertRaises(AuditChainConflictError) as cm:
            self.chain.append_many_if_heads([
                entry("u", [{}], 1, u1["hash"]),
                entry("t", [{"i": 3}], 1, first["hash"]),
            ])
        e = cm.exception
        self.assertEqual(e.tenant, "t")
        self.assertEqual((e.expected_count, e.expected_hash), (1, first["hash"]))
        self.assertEqual((e.actual_count, e.actual_hash), (2, second["hash"]))
        self.assertEqual(self.path.read_bytes(), before)

    def test_first_mismatching_entry_in_entries_order_conflicts(self):
        self.chain.append("a", {})
        self.chain.append("b", {})
        before = self.path.read_bytes()
        # a is correct, b asserts a wrong count: b is reported; a not written
        ha = self.chain.head("a")
        with self.assertRaises(AuditChainConflictError) as cm:
            self.chain.append_many_if_heads([
                entry("a", [{}], ha["count"], ha["hash"]),
                entry("b", [{}], 9, ZERO),
            ])
        self.assertEqual(cm.exception.tenant, "b")
        self.assertEqual(self.path.read_bytes(), before)
        # swapping order: an earlier wrong a wins the report over wrong b
        with self.assertRaises(AuditChainConflictError) as cm:
            self.chain.append_many_if_heads([
                entry("a", [{}], 9, ZERO),
                entry("b", [{}], 9, ZERO),
            ])
        self.assertEqual(cm.exception.tenant, "a")
        self.assertEqual(self.path.read_bytes(), before)

    def test_conflict_for_interleaved_other_tenant_writes_nothing(self):
        self.chain.append("a", {})
        before = self.path.read_bytes()
        with self.assertRaises(AuditChainConflictError):
            self.chain.append_many_if_heads([
                entry("a", [{}], 1, self.chain.head("a")["hash"]),
                entry("b", [{}], 1, ZERO),
            ])
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(self.chain.verify("b"), {"ok": True, "count": 0})

    # --- corrupt history: state errors, priority, tie-breaks ---------------

    def test_corrupt_history_raises_state_error_not_conflict(self):
        self.chain.append("t", {})
        self.chain.append("u", {})
        self.tamper(0, event={"x": 9})  # t digest broken at line 1
        before = self.path.read_bytes()
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.append_many_if_heads([
                entry("u", [{}], 9, ZERO),  # also a conflict...
                entry("t", [{}], 1, "f" * 64),
            ])
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
        # a listed first, but b's defect sits on an earlier physical line
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.append_many_if_heads([
                entry("a", [{}], 2, ZERO), entry("b", [{}], 1, ZERO)])
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("b", 1, "digest", 1))
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.append_many_if_heads([
                entry("b", [{}], 1, ZERO), entry("a", [{}], 2, ZERO)])
        self.assertEqual((cm.exception.tenant, cm.exception.line), ("b", 1))
        self.assertEqual(self.path.read_bytes(), before)

    def test_equal_physical_line_broken_by_input_order(self):
        self.path.write_text("not json\n", encoding="utf-8")
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.append_many_if_heads([
                entry("a", [{}]), entry("b", [{}])])
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("a", 1, "missing", 1))
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.append_many_if_heads([
                entry("b", [{}]), entry("a", [{}])])
        self.assertEqual((cm.exception.tenant, cm.exception.line), ("b", 1))

    def test_bad_utf8_line_is_missing_for_every_affected_chain(self):
        valid = {"tenant": "t", "seq": 1, "event": {}, "prev": ZERO}
        valid["hash"] = AuditChain._hash(valid)
        before = (json.dumps(valid, sort_keys=True) + "\n").encode() + b"\xff"
        self.path.write_bytes(before)
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.append_many_if_heads([
                entry("u", [{}]), entry("t", [{}], 1, valid["hash"])])
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
            self.chain.append_many_if_heads([entry("t", [{}], 2, ZERO)])
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("t", 2, "sequence", 2))

    def test_corruption_of_an_uninvolved_tenant_does_not_block(self):
        self.chain.append("b", {})
        self.tamper(0, event={"x": 1})
        items = self.chain.append_many_if_heads([
            entry("t", [{}]), entry("u", [{}])])
        self.assertEqual([(it["tenant"], it["seq"]) for it in items],
                         [("t", 1), ("u", 1)])
        self.assertTrue(self.chain.verify("t")["ok"])
        self.assertTrue(self.chain.verify("u")["ok"])

    # --- compatibility with every existing entry point ---------------------

    def test_result_verifiable_by_all_read_and_offline_entries(self):
        self.chain.append("a", {"i": 0})
        ha = self.chain.head("a")
        items = self.chain.append_many_if_heads([
            entry("a", [{"i": 1}, {"i": 2}], ha["count"], ha["hash"]),
            entry("b", [{"i": 1}, {"i": 2}], 0, ZERO),
        ])
        self.assertEqual(self.chain.verify("a"), {"ok": True, "count": 3})
        self.assertEqual(self.chain.verify("b"), {"ok": True, "count": 2})
        self.assertTrue(self.chain.verify_all()["ok"])
        data = self.path.read_bytes()
        self.assertTrue(self.chain.verify_bytes(data, "a")["ok"])
        self.assertTrue(self.chain.verify_all_bytes(data)["ok"])
        self.assertEqual([r["seq"] for r in self.chain.read_tenant("a")],
                         [1, 2, 3])
        self.assertEqual(
            [r["seq"] for r in self.chain.read_tenant("b", page_size=1)], [1])
        export_b = self.chain.export_tenant("b")
        self.assertEqual(self.chain.verify_bytes(export_b, "b")["count"], 2)
        fresh = AuditChain(self.path.with_name("fresh.jsonl"))
        fresh.import_tenant("b", export_b)
        self.assertEqual(fresh.verify("b"), {"ok": True, "count": 2})
        for it in items:
            self.assertEqual(
                set(it), {"tenant", "seq", "event", "prev", "hash"})

    def test_append_if_head_continues_from_a_committed_tail(self):
        items = self.chain.append_many_if_heads([
            entry("t", [{"i": 1}, {"i": 2}]), entry("u", [{}])])
        t_tail = next(it for it in items if it["tenant"] == "t" and it["seq"] == 2)
        nxt = self.chain.append_if_head("t", {"i": 3}, 2, t_tail["hash"])
        self.assertEqual((nxt["seq"], nxt["prev"]), (3, t_tail["hash"]))
        with self.assertRaises(AuditChainConflictError):
            self.chain.append_batch_if_head("t", [{}], 2, t_tail["hash"])

    # --- concurrency: one winner, losers see winner heads, stable reads ----

    def test_concurrent_same_heads_exactly_one_winner(self):
        n, size = 12, 5
        outcomes = []
        box = threading.Lock()

        def worker(w):
            try:
                items = self.chain.append_many_if_heads([
                    entry("t1", [{"w": w, "j": j} for j in range(size)]),
                    entry("t2", [{"w": w, "j": j} for j in range(size)]),
                ])
                with box:
                    outcomes.append(("ok", items))
            except AuditChainConflictError as e:
                with box:
                    outcomes.append(("conflict", e.tenant,
                                     e.actual_count, e.actual_hash))
            except Exception as e:  # noqa: BLE001
                with box:
                    outcomes.append(("error", repr(e)))

        threads = [threading.Thread(target=worker, args=(w,)) for w in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        oks = [o for o in outcomes if o[0] == "ok"]
        conflicts = [o for o in outcomes if o[0] == "conflict"]
        self.assertEqual(len(oks), 1, outcomes)
        self.assertEqual(len(conflicts), n - 1, outcomes)
        items = oks[0][1]
        self.assertEqual([(it["tenant"], it["seq"]) for it in items],
                         [("t1", j) for j in range(1, size + 1)]
                         + [("t2", j) for j in range(1, size + 1)])
        self.assertEqual(items[0]["prev"], ZERO)
        t1_tail = self.chain.read_tenant("t1")[-1]["hash"]
        # every loser deterministically observed the winner's post-win head at
        # the first mismatching entry (t1, asserted empty)
        for _, tenant, actual_count, actual_hash in conflicts:
            self.assertEqual(tenant, "t1")
            self.assertEqual((actual_count, actual_hash), (size, t1_tail))
        self.assertEqual(self.chain.verify("t1"), {"ok": True, "count": size})
        self.assertEqual(self.chain.verify("t2"), {"ok": True, "count": size})

    def test_concurrent_cas_loops_commit_every_batch_exactly_once(self):
        n, size = 8, 3
        committed = []
        box = threading.Lock()

        def worker(w):
            for _ in range(500):
                heads = {h["tenant"]: h for h in self.chain.heads()["tenants"]}
                a = heads.get("a", {"count": 0, "hash": ZERO})
                b = heads.get("b", {"count": 0, "hash": ZERO})
                try:
                    items = self.chain.append_many_if_heads([
                        entry("a", [{"w": w, "j": j} for j in range(size)],
                              a["count"], a["hash"]),
                        entry("b", [{"w": w, "j": j} for j in range(size)],
                              b["count"], b["hash"]),
                    ])
                    with box:
                        committed.extend(items)
                    return
                except AuditChainConflictError:
                    continue
            self.fail("worker never won the CAS race")

        threads = [threading.Thread(target=worker, args=(w,)) for w in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        total = n * size * 2
        self.assertEqual(len(committed), total)
        self.assertTrue(self.chain.verify_all()["ok"])
        r = self.chain.verify_all()
        per = {h["tenant"]: h["count"] for h in r["tenants"]}
        self.assertEqual(per, {"a": n * size, "b": n * size})
        # one contiguous winning block per worker on each tenant
        for w in range(n):
            evs = sorted((it["seq"], it["event"]["j"])
                         for it in committed
                         if it["event"]["w"] == w and it["tenant"] == "a")
            self.assertEqual([j for _, j in evs], list(range(size)))

    def test_concurrent_writers_and_plain_appends_serialize(self):
        errors = []
        box = threading.Lock()

        def cond_worker(w):
            for _ in range(50):
                heads = {h["tenant"]: h for h in self.chain.heads()["tenants"]}
                a = heads.get("a", {"count": 0, "hash": ZERO})
                m = heads.get("m", {"count": 0, "hash": ZERO})
                try:
                    self.chain.append_many_if_heads([
                        entry("a", [{"w": w}], a["count"], a["hash"]),
                        entry("m", [{"w": w}, {"w": w}], m["count"], m["hash"]),
                    ])
                    return
                except AuditChainConflictError:
                    continue
            with box:
                errors.append(("cond-stuck", w))

        def plain_worker(w):
            for j in range(50):
                self.chain.append("s", {"w": w, "j": j})

        threads = [threading.Thread(target=cond_worker, args=(w,))
                   for w in range(8)]
        threads += [threading.Thread(target=plain_worker, args=(w,))
                    for w in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [])
        r = self.chain.verify_all()
        self.assertTrue(r["ok"], r)
        per = {h["tenant"]: h["count"] for h in r["tenants"]}
        self.assertEqual(per["a"], 8)
        self.assertEqual(per["m"], 16)
        self.assertEqual(per["s"], 200)

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
                for _try in range(500):
                    heads = {h["tenant"]: h
                             for h in self.chain.heads()["tenants"]}
                    a = heads.get("a", {"count": 0, "hash": ZERO})
                    b = heads.get("b", {"count": 0, "hash": ZERO})
                    try:
                        self.chain.append_many_if_heads([
                            entry("a", [{"w": w}, {"w": w}],
                                  a["count"], a["hash"]),
                            entry("b", [{"w": w}], b["count"], b["hash"]),
                        ])
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

    def test_corrupt_history_under_contention_leaves_bytes_untouched(self):
        self.chain.append("t", {})
        self.tamper(0, event={"x": 9})
        before = self.path.read_bytes()
        details = []
        box = threading.Lock()

        def attempt(i):
            try:
                self.chain.append_many_if_heads([
                    entry("t", [{"i": i}], 1, ZERO),
                    entry("u", [{"i": i}], 0, ZERO),
                ])
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
