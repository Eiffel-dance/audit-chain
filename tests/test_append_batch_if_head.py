import json
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app import AuditChain, AuditChainConflictError, AuditChainStateError, ZERO


class AppendBatchIfHeadTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.path = Path(self.tmp.name) / "audit.jsonl"
        self.chain = AuditChain(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def read_rows(self):
        return [json.loads(l) for l in self.path.read_text().splitlines()]

    # --- success cases ---

    def test_empty_chain_asserted_with_zero_creates_and_links_batch(self):
        items = self.chain.append_batch_if_head(
            "t", [{"i": 1}, {"i": 2}, {"i": 3}], 0, ZERO)
        self.assertEqual([it["seq"] for it in items], [1, 2, 3])
        self.assertEqual(items[0]["prev"], ZERO)
        self.assertEqual(items[1]["prev"], items[0]["hash"])
        self.assertEqual(items[2]["prev"], items[1]["hash"])
        for it in items:
            self.assertEqual(set(it), {"tenant", "seq", "event", "prev", "hash"})
            self.assertEqual(it["hash"], AuditChain._hash(it))
        self.assertEqual(items, self.read_rows())
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": 3})

    def test_matching_nonempty_head_appends_after_tail(self):
        first = self.chain.append("t", {"i": 0})
        items = self.chain.append_batch_if_head(
            "t", [{"i": 1}, {"i": 2}], 1, first["hash"])
        self.assertEqual([it["seq"] for it in items], [2, 3])
        self.assertEqual(items[0]["prev"], first["hash"])
        self.assertEqual(items[1]["prev"], items[0]["hash"])
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": 3})

    def test_byte_equivalent_to_sequential_appends_from_same_head(self):
        events = [{"i": i, "s": "审计"} for i in range(4)]
        self.chain.append("a", {})
        anchor = self.chain.append("t", {"i": -1})
        self.chain.append("z", {})
        batch_items = self.chain.append_batch_if_head(
            "t", events, 1, anchor["hash"])
        batched = self.path.read_bytes()

        other = AuditChain(self.path.with_name("other.jsonl"))
        other.append("a", {})
        other.append("t", {"i": -1})
        other.append("z", {})
        sequential = [other.append("t", e) for e in events]
        self.assertEqual(other.path.read_bytes(), batched)
        self.assertEqual(sequential, batch_items)
        self.assertEqual(other.verify_all(), self.chain.verify_all())

    def test_byte_equivalent_to_append_batch_from_same_head(self):
        events = [{"i": i} for i in range(3)]
        items = self.chain.append_batch_if_head("t", events, 0, ZERO)
        other = AuditChain(self.path.with_name("other.jsonl"))
        self.assertEqual(other.append_batch("t", events), items)
        self.assertEqual(other.path.read_bytes(), self.path.read_bytes())

    def test_interleaved_tenant_head_is_independent(self):
        a1 = self.chain.append("a", {})
        self.chain.append("b", {})
        items = self.chain.append_batch_if_head("a", [{}, {}], 1, a1["hash"])
        self.assertEqual([it["seq"] for it in items], [2, 3])
        self.assertEqual(items[0]["prev"], a1["hash"])
        self.assertEqual(self.chain.verify("a"), {"ok": True, "count": 3})
        self.assertEqual(self.chain.verify("b"), {"ok": True, "count": 1})

    def test_distinct_json_tenants_are_distinct_heads(self):
        self.chain.append(1, {})
        items = self.chain.append_batch_if_head("1", [{}, {}], 0, ZERO)
        self.assertEqual([it["seq"] for it in items], [1, 2])
        with self.assertRaises(AuditChainConflictError) as cm:
            self.chain.append_batch_if_head(1, [{}], 0, ZERO)
        self.assertEqual(cm.exception.actual_count, 1)
        self.assertNotEqual(cm.exception.actual_hash, ZERO)

    def test_events_use_standard_json_boundary(self):
        items = self.chain.append_batch_if_head(
            "t", [{"n": None, "s": "x", "f": 1.5}, [1, "a"]], 0, ZERO)
        self.assertEqual(items[0]["event"], {"n": None, "s": "x", "f": 1.5})
        self.assertEqual(items[1]["event"], [1, "a"])

    # --- empty batch ---

    def test_empty_list_is_a_noop_and_creates_nothing(self):
        self.assertEqual(
            self.chain.append_batch_if_head("t", [], 0, ZERO), [])
        self.assertFalse(self.path.exists())
        # also a no-op against an existing file: bytes untouched, and the
        # head assertion is not evaluated (a mismatching one changes nothing)
        self.chain.append("u", {"v": 1})
        before = self.path.read_bytes()
        self.assertEqual(
            self.chain.append_batch_if_head("t", [], 5, "f" * 64), [])
        self.assertEqual(self.path.read_bytes(), before)

    def test_empty_list_still_validates_all_parameters_first(self):
        for bad_args in (
            ("t", [], True, ZERO),
            ("t", [], -1, ZERO),
            ("t", [], 0, "A" * 64),
            (float("nan"), [], 0, ZERO),
        ):
            with self.assertRaises(ValueError):
                self.chain.append_batch_if_head(*bad_args)
        self.assertFalse(self.path.exists())

    # --- input validation: ValueError before any read or byte ---

    def assertValueErrorBeforeIo(self, *args):
        self.assertFalse(self.path.exists())
        with self.assertRaises(ValueError):
            self.chain.append_batch_if_head(*args)
        self.assertFalse(self.path.exists())

    def test_events_must_be_a_list(self):
        for bad in (None, True, 1, 1.5, "x", {"a": 1}, ("x",), {"a", "b"}):
            self.assertValueErrorBeforeIo("t", bad, 0, ZERO)

    def test_tenant_and_each_event_cross_the_json_boundary(self):
        self.assertValueErrorBeforeIo(float("nan"), [{}], 0, ZERO)
        self.assertValueErrorBeforeIo("t", [{"ok": 1}, float("inf")], 0, ZERO)
        self.assertValueErrorBeforeIo("t", [{1: "x"}], 0, ZERO)
        cyc = []
        cyc.append(cyc)
        self.assertValueErrorBeforeIo("t", [cyc], 0, ZERO)

    def test_bool_count_rejected(self):
        self.assertValueErrorBeforeIo("t", [{}], True, ZERO)
        self.assertValueErrorBeforeIo("t", [{}], False, ZERO)

    def test_negative_float_string_none_count_rejected(self):
        for bad in (-1, 1.0, "0", None, 0.0, [0]):
            self.assertValueErrorBeforeIo("t", [{}], bad, ZERO)

    def test_malformed_hash_rejected(self):
        for bad in ("", "a" * 63, "a" * 65, "A" * 64, "g" * 64,
                    "0" * 63 + "G", ZERO[:-1] + "A", 0, bytes(64), None):
            self.assertValueErrorBeforeIo("t", [{}], 0, bad)

    def test_value_error_takes_priority_over_corrupt_history(self):
        self.chain.append("t", {"v": 1})
        row = json.loads(self.path.read_text())
        row["event"] = {"v": 2}  # digest corruption at line 1
        self.path.write_text(json.dumps(row, sort_keys=True) + "\n")
        for bad_args in (
            ("t", [float("nan")], 1, "x" * 64),
            ("t", [{}], True, ZERO),
            ("t", [{}], 1, "X" * 64),
            (float("nan"), [{}], 1, ZERO),
            ("t", "not-a-list", 1, ZERO),
        ):
            with self.assertRaises(ValueError):
                self.chain.append_batch_if_head(*bad_args)
        self.assertEqual(
            self.path.read_bytes(),
            (json.dumps(row, sort_keys=True) + "\n").encode("utf-8"),
        )

    # --- conflict cases ---

    def test_missing_log_with_nonempty_assertion_conflicts_and_creates_nothing(self):
        with self.assertRaises(AuditChainConflictError) as cm:
            self.chain.append_batch_if_head("t", [{}], 1, ZERO)
        e = cm.exception
        self.assertEqual((e.tenant, e.reason), ("t", "conflict"))
        self.assertEqual((e.expected_count, e.expected_hash), (1, ZERO))
        self.assertEqual((e.actual_count, e.actual_hash), (0, ZERO))
        self.assertFalse(self.path.exists())

    def test_missing_log_with_wrong_hash_conflicts_and_creates_nothing(self):
        with self.assertRaises(AuditChainConflictError) as cm:
            self.chain.append_batch_if_head("t", [{}], 0, "f" * 64)
        e = cm.exception
        self.assertEqual((e.expected_count, e.expected_hash), (0, "f" * 64))
        self.assertEqual((e.actual_count, e.actual_hash), (0, ZERO))
        self.assertFalse(self.path.exists())

    def test_stale_head_conflicts_with_actual_tail_and_writes_nothing(self):
        first = self.chain.append("t", {"i": 1})
        second = self.chain.append("t", {"i": 2})
        before = self.path.read_bytes()
        with self.assertRaises(AuditChainConflictError) as cm:
            self.chain.append_batch_if_head(
                "t", [{"i": 3}, {"i": 4}], 1, first["hash"])
        e = cm.exception
        self.assertEqual((e.expected_count, e.expected_hash), (1, first["hash"]))
        self.assertEqual((e.actual_count, e.actual_hash), (2, second["hash"]))
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": 2})

    def test_stale_count_but_right_hash_conflicts(self):
        first = self.chain.append("t", {})
        with self.assertRaises(AuditChainConflictError) as cm:
            self.chain.append_batch_if_head("t", [{}], 2, first["hash"])
        e = cm.exception
        self.assertEqual((e.expected_count, e.actual_count), (2, 1))
        self.assertEqual((e.expected_hash, e.actual_hash),
                         (first["hash"], first["hash"]))
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": 1})

    def test_conflict_for_other_tenant_writes_nothing(self):
        self.chain.append("a", {})
        before = self.path.read_bytes()
        with self.assertRaises(AuditChainConflictError):
            self.chain.append_batch_if_head("b", [{}], 1, ZERO)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(self.chain.verify("b"), {"ok": True, "count": 0})

    # --- corrupt history: same AuditChainStateError semantics as append ---

    def test_corrupt_history_raises_state_error_not_conflict(self):
        self.chain.append("t", {"v": 1})
        good = json.loads(self.path.read_text())
        row = dict(good)
        row["event"] = {"v": 2}  # breaks digest at line 1
        self.path.write_text(json.dumps(row, sort_keys=True) + "\n")
        before = self.path.read_bytes()
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.append_batch_if_head("t", [{}, {}], 1, good["hash"])
        e = cm.exception
        self.assertEqual((e.tenant, e.seq, e.reason, e.line),
                         ("t", 1, "digest", 1))
        self.assertEqual(self.path.read_bytes(), before)

    def test_corrupt_history_sequence_and_missing_classification(self):
        gap = {"tenant": "t", "seq": 2, "event": {}, "prev": ZERO}
        gap["hash"] = AuditChain._hash(gap)
        self.path.write_text(json.dumps(gap, sort_keys=True) + "\n")
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.append_batch_if_head("t", [{}], 2, gap["hash"])
        self.assertEqual((cm.exception.seq, cm.exception.reason,
                          cm.exception.line), (1, "sequence", 1))
        self.path.write_bytes(b"{not json\n")
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.append_batch_if_head("t", [{}], 0, ZERO)
        self.assertEqual((cm.exception.seq, cm.exception.reason,
                          cm.exception.line), (1, "missing", 1))

    def test_state_error_takes_priority_over_conflict(self):
        # corrupt chain AND mismatching assertion: state error wins
        self.chain.append("t", {"v": 1})
        row = json.loads(self.path.read_text())
        row["event"] = {"v": 2}
        self.path.write_text(json.dumps(row, sort_keys=True) + "\n")
        with self.assertRaises(AuditChainStateError):
            self.chain.append_batch_if_head("t", [{}], 0, ZERO)

    # --- concurrency: exactly one winner per head ---

    def test_concurrent_same_head_exactly_one_batch_wins(self):
        n, size = 12, 5
        outcomes = []
        box = threading.Lock()

        def worker(i):
            try:
                items = self.chain.append_batch_if_head(
                    "t", [{"w": i, "j": j} for j in range(size)], 0, ZERO)
                with box:
                    outcomes.append(("ok", items))
            except AuditChainConflictError as e:
                with box:
                    outcomes.append(("conflict", e.actual_count, e.actual_hash))
            except Exception as e:  # noqa: BLE001
                with box:
                    outcomes.append(("error", repr(e)))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        oks = [o for o in outcomes if o[0] == "ok"]
        conflicts = [o for o in outcomes if o[0] == "conflict"]
        self.assertEqual(len(oks), 1, outcomes)
        self.assertEqual(len(conflicts), n - 1, outcomes)
        # the winner's batch is one contiguous block built on the empty head
        items = oks[0][1]
        self.assertEqual([it["seq"] for it in items], list(range(1, size + 1)))
        self.assertEqual(items[0]["prev"], ZERO)
        tail = self.chain.read_tenant("t")[-1]
        # every loser deterministically observed the post-win tail
        for _, actual_count, actual_hash in conflicts:
            self.assertEqual((actual_count, actual_hash),
                             (size, tail["hash"]))
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": size})
        self.assertEqual(len(self.path.read_bytes().splitlines()), size)

    def test_concurrent_chained_batch_assertions_serialize(self):
        # CAS loop: every batch lands exactly once, in one verifiable chain
        n, size = 8, 3
        committed = []
        box = threading.Lock()

        def worker(i):
            for _ in range(200):
                r = self.chain.verify("t")
                count = r["count"]
                head = ZERO if count == 0 else self.chain.read_tenant(
                    "t", start_seq=count)[0]["hash"]
                try:
                    items = self.chain.append_batch_if_head(
                        "t", [{"w": i, "j": j} for j in range(size)],
                        count, head)
                    with box:
                        committed.extend(items)
                    return
                except AuditChainConflictError:
                    continue

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        total = n * size
        self.assertEqual(len(committed), total)
        self.assertEqual(sorted(it["seq"] for it in committed),
                         list(range(1, total + 1)))
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": total})
        self.assertTrue(self.chain.verify_all()["ok"])

    def test_concurrent_conflict_and_corruption_share_state_semantics(self):
        self.chain.append("t", {"v": 1})
        row = json.loads(self.path.read_text())
        row["event"] = {"v": 2}
        self.path.write_text(json.dumps(row, sort_keys=True) + "\n")
        details = []
        box = threading.Lock()

        def worker():
            try:
                self.chain.append_batch_if_head("t", [{}], 1, "f" * 64)
            except AuditChainStateError as e:
                with box:
                    details.append((e.tenant, e.seq, e.reason, e.line))
            except AuditChainConflictError:
                with box:
                    details.append(("conflict",))

        threads = [threading.Thread(target=worker) for _ in range(12)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(details, [("t", 1, "digest", 1)] * 12)


if __name__ == "__main__":
    unittest.main()
