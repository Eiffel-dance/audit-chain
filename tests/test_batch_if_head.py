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

    # --- empty batch ---

    def test_empty_list_is_a_noop_and_creates_nothing(self):
        self.assertEqual(self.chain.append_batch_if_head("t", [], 0, ZERO), [])
        self.assertFalse(self.path.exists())
        # a no-op against an existing file leaves every byte untouched
        self.chain.append("u", {"v": 1})
        before = self.path.read_bytes()
        self.assertEqual(
            self.chain.append_batch_if_head("t", [], 5, "f" * 64), []
        )
        self.assertEqual(self.path.read_bytes(), before)

    def test_empty_list_still_validates_parameters_first(self):
        for args in (
            (float("nan"), [], 0, ZERO),
            ("t", [], True, ZERO),
            ("t", [], -1, ZERO),
            ("t", [], 1.0, ZERO),
            ("t", [], 0, "x" * 64),
            ("t", [], 0, "A" * 64),
        ):
            self.assertFalse(self.path.exists())
            with self.assertRaises(ValueError):
                self.chain.append_batch_if_head(*args)
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

    def test_illegal_tenant_event_rejected_like_append(self):
        self.assertValueErrorBeforeIo(float("nan"), [{}], 0, ZERO)
        self.assertValueErrorBeforeIo("t", [float("inf")], 0, ZERO)
        self.assertValueErrorBeforeIo("t", [{}, {"bad": float("nan")}], 0, ZERO)
        self.assertValueErrorBeforeIo({1: "x"}, [{}], 0, ZERO)
        cyc = {}
        cyc["self"] = cyc
        self.assertValueErrorBeforeIo("t", [cyc], 0, ZERO)

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

    # --- success cases ---

    def test_empty_chain_asserted_with_zero_creates_chain(self):
        items = self.chain.append_batch_if_head(
            "t", [{"i": 1}, {"i": 2}, {"i": 3}], 0, ZERO
        )
        self.assertEqual([it["seq"] for it in items], [1, 2, 3])
        self.assertEqual(items[0]["prev"], ZERO)
        self.assertEqual(items[1]["prev"], items[0]["hash"])
        self.assertEqual(items[2]["prev"], items[1]["hash"])
        for it in items:
            self.assertEqual(set(it), {"tenant", "seq", "event", "prev", "hash"})
            self.assertEqual(it["hash"], AuditChain._hash(it))
        self.assertEqual(items, self.read_rows())
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": 3})

    def test_matching_nonempty_head_appends_contiguous_batch(self):
        first = self.chain.append("t", {"i": 0})
        items = self.chain.append_batch_if_head(
            "t", [{"i": 1}, {"i": 2}], 1, first["hash"]
        )
        self.assertEqual([it["seq"] for it in items], [2, 3])
        self.assertEqual(items[0]["prev"], first["hash"])
        self.assertEqual(items[1]["prev"], items[0]["hash"])
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": 3})

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
        # tenant "1" is a different chain: empty head (0, ZERO) must match
        items = self.chain.append_batch_if_head("1", [{}, {}], 0, ZERO)
        self.assertEqual([it["seq"] for it in items], [1, 2])
        # tenant 1 now has one record; asserting an empty head there conflicts
        with self.assertRaises(AuditChainConflictError) as cm:
            self.chain.append_batch_if_head(1, [{}], 0, ZERO)
        self.assertEqual(cm.exception.actual_count, 1)
        self.assertNotEqual(cm.exception.actual_hash, ZERO)

    def test_byte_equivalent_to_sequential_appends_against_same_head(self):
        events = [{"i": i, "s": "审计"} for i in range(5)]
        self.chain.append("a", {})
        anchor = self.chain.append("t", {"i": -1})
        self.chain.append("z", {})
        batch_items = self.chain.append_batch_if_head(
            "t", events, 1, anchor["hash"]
        )
        self.chain.append("z", {})
        batched = self.path.read_bytes()

        other = AuditChain(self.path.with_name("other.jsonl"))
        other.append("a", {})
        o_anchor = other.append("t", {"i": -1})
        other.append("z", {})
        # build the sequential history with one append_if_head per event
        # against the same head the batch was asserted against
        sequential = []
        prev, count = o_anchor["hash"], 1
        for e in events:
            it = other.append_if_head("t", e, count, prev)
            sequential.append(it)
            count += 1
            prev = it["hash"]
        other.append("z", {})
        self.assertEqual(other.path.read_bytes(), batched)
        self.assertEqual(sequential, batch_items)
        self.assertEqual(other.verify_all(), self.chain.verify_all())

    def test_single_event_batch_matches_one_append_if_head(self):
        item = self.chain.append_batch_if_head("t", [{"a": 1}], 0, ZERO)[0]
        other_path = self.path.with_name("other.jsonl")
        other = AuditChain(other_path).append_if_head(
            "t", {"a": 1}, 0, ZERO
        )
        self.assertEqual(other, item)
        self.assertEqual(other_path.read_bytes(), self.path.read_bytes())

    # --- conflict cases ---

    def test_wrong_hash_on_missing_log_conflicts_and_creates_nothing(self):
        with self.assertRaises(AuditChainConflictError) as cm:
            self.chain.append_batch_if_head("t", [{}], 0, "f" * 64)
        e = cm.exception
        self.assertEqual((e.tenant, e.reason), ("t", "conflict"))
        self.assertEqual(e.expected_count, 0)
        self.assertEqual(e.expected_hash, "f" * 64)
        self.assertEqual((e.actual_count, e.actual_hash), (0, ZERO))
        self.assertFalse(self.path.exists())

    def test_wrong_count_on_missing_log_conflicts_and_creates_nothing(self):
        with self.assertRaises(AuditChainConflictError) as cm:
            self.chain.append_batch_if_head("t", [{}], 1, ZERO)
        e = cm.exception
        self.assertEqual((e.expected_count, e.expected_hash), (1, ZERO))
        self.assertEqual((e.actual_count, e.actual_hash), (0, ZERO))
        self.assertFalse(self.path.exists())

    def test_stale_head_after_append_conflicts_and_reports_tail(self):
        first = self.chain.append("t", {"i": 1})
        second = self.chain.append("t", {"i": 2})
        before = self.path.read_bytes()
        with self.assertRaises(AuditChainConflictError) as cm:
            self.chain.append_batch_if_head(
                "t", [{"i": 3}, {"i": 4}], 1, first["hash"]
            )
        e = cm.exception
        self.assertEqual((e.expected_count, e.expected_hash), (1, first["hash"]))
        self.assertEqual((e.actual_count, e.actual_hash), (2, second["hash"]))
        # losing batch writes nothing at all
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": 2})

    def test_conflict_creates_no_record_for_other_tenant(self):
        self.chain.append("a", {})
        before = self.path.read_bytes()
        with self.assertRaises(AuditChainConflictError):
            self.chain.append_batch_if_head("b", [{}, {}], 1, ZERO)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(self.chain.verify("b"), {"ok": True, "count": 0})
        self.assertEqual(self.chain.verify("a"), {"ok": True, "count": 1})

    # --- corrupt history: same AuditChainStateError semantics ---

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

    def test_sequence_and_missing_classification_with_line_numbers(self):
        other = {"tenant": "x", "seq": 1, "event": {}, "prev": ZERO}
        other["hash"] = AuditChain._hash(other)
        gap = {"tenant": "t", "seq": 2, "event": {}, "prev": ZERO}
        gap["hash"] = AuditChain._hash(gap)
        self.path.write_text(
            json.dumps(other, sort_keys=True) + "\n"
            + json.dumps(gap, sort_keys=True) + "\n"
        )
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.append_batch_if_head("t", [{}], 2, gap["hash"])
        self.assertEqual((cm.exception.seq, cm.exception.reason, cm.exception.line),
                         (1, "sequence", 2))
        self.path.write_bytes(b"{not json\n")
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.append_batch_if_head("t", [{}], 0, ZERO)
        self.assertEqual((cm.exception.seq, cm.exception.reason, cm.exception.line),
                         (1, "missing", 1))

    def test_illegal_utf8_reports_same_first_broken_point(self):
        valid = {"tenant": "t", "seq": 1, "event": {}, "prev": ZERO}
        valid["hash"] = AuditChain._hash(valid)
        before = (json.dumps(valid, sort_keys=True) + "\n").encode("utf-8") + b"\xff"
        self.path.write_bytes(before)
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.append_batch_if_head("t", [{}, {}], 1, valid["hash"])
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("t", 2, "missing", 2))
        self.assertEqual(self.path.read_bytes(), before)

    def test_corrupt_history_other_tenant_does_not_block_batch(self):
        self.chain.append("b", {})
        rows = self.read_rows()
        rows[0]["event"] = {"x": 1}
        with self.path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, sort_keys=True) + "\n")
        items = self.chain.append_batch_if_head("t", [{}, {}], 0, ZERO)
        self.assertEqual([it["seq"] for it in items], [1, 2])
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": 2})


class AppendBatchIfHeadConcurrencyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.path = Path(self.tmp.name) / "audit.jsonl"
        self.chain = AuditChain(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_concurrent_same_head_exactly_one_wins_rest_conflict(self):
        n, size = 24, 3
        outcomes = []
        box = threading.Lock()

        def worker(i):
            try:
                items = self.chain.append_batch_if_head(
                    "t", [{"i": i, "j": j} for j in range(size)], 0, ZERO
                )
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
        items = oks[0][1]
        self.assertEqual([it["seq"] for it in items], list(range(1, size + 1)))
        self.assertEqual(items[0]["prev"], ZERO)
        rows = self.chain.read_tenant("t")
        self.assertEqual(rows, items)
        winning_hash = rows[-1]["hash"]
        # every loser observed the winner's post-commit head, deterministically
        for _, actual_count, actual_hash in conflicts:
            self.assertEqual((actual_count, actual_hash), (size, winning_hash))
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": size})
        self.assertEqual(len(self.path.read_bytes().splitlines()), size)

    def test_concurrent_chained_assertions_serialize_batches(self):
        # CAS-style retries with batches: every event lands exactly once and
        # the final chain verifies as one history
        n, size = 16, 2
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
                        "t", [{"i": i, "j": j} for j in range(size)],
                        count, head,
                    )
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

    def test_winning_batch_is_indivisible_against_other_tenant_appends(self):
        done = []
        box = threading.Lock()

        def batch_worker():
            try:
                items = self.chain.append_batch_if_head(
                    "a", [{"i": j} for j in range(25)], 0, ZERO
                )
                with box:
                    done.extend(("a", it) for it in items)
            except AuditChainConflictError:
                # Several identical empty-head batches race: at most one wins.
                pass

        def single_worker():
            local = [self.chain.append("b", {"i": j}) for j in range(25)]
            with box:
                done.extend(("b", it) for it in local)

        threads = [threading.Thread(target=batch_worker) for _ in range(8)]
        threads += [threading.Thread(target=single_worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        counts = {"a": 0, "b": 0}
        for tenant, _ in done:
            counts[tenant] += 1
        self.assertEqual(counts["a"], 25)  # exactly one batch won
        self.assertEqual(counts["b"], 100)
        self.assertEqual(self.chain.verify("a"), {"ok": True, "count": 25})
        self.assertTrue(self.chain.verify_all()["ok"])
        # the winning batch records are contiguous on disk with internal links
        rows = [json.loads(l) for l in self.path.read_text().splitlines()]
        a_rows = [r for r in rows if r["tenant"] == "a"]
        self.assertEqual([r["seq"] for r in a_rows], list(range(1, 26)))
        for prev_row, row in zip(a_rows, a_rows[1:]):
            self.assertEqual(row["prev"], prev_row["hash"])

    def test_concurrent_corruption_shares_state_semantics(self):
        self.chain.append("t", {"v": 1})
        row = json.loads(self.path.read_text())
        row["event"] = {"v": 2}
        self.path.write_text(json.dumps(row, sort_keys=True) + "\n")
        details = []
        box = threading.Lock()

        def worker():
            try:
                self.chain.append_batch_if_head(
                    "t", [{}, {}], 1, "f" * 64
                )
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
