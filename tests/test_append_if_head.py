import json
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app import AuditChain, AuditChainConflictError, AuditChainStateError, ZERO


class AppendIfHeadTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.path = Path(self.tmp.name) / "audit.jsonl"
        self.chain = AuditChain(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    # --- success cases ---

    def test_empty_chain_asserted_with_zero_succeeds_and_matches_append(self):
        item = self.chain.append_if_head("t", {"a": 1}, 0, ZERO)
        self.assertEqual(item["seq"], 1)
        self.assertEqual(item["prev"], ZERO)
        self.assertEqual(set(item), {"tenant", "seq", "event", "prev", "hash"})
        self.assertEqual(item["hash"], AuditChain._hash(item))
        # byte-for-byte identical to a plain append on a fresh chain
        other_path = Path(self.tmp.name) / "other.jsonl"
        other = AuditChain(other_path).append("t", {"a": 1})
        self.assertEqual(other, item)
        self.assertEqual(other_path.read_bytes(), self.path.read_bytes())
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": 1})

    def test_matching_nonempty_head_appends_next_record(self):
        first = self.chain.append("t", {"i": 1})
        item = self.chain.append_if_head("t", {"i": 2}, 1, first["hash"])
        self.assertEqual((item["seq"], item["prev"]), (2, first["hash"]))
        self.assertEqual(item["hash"], AuditChain._hash(item))
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": 2})

    def test_interleaved_tenant_head_is_independent(self):
        a1 = self.chain.append("a", {})
        self.chain.append("b", {})
        # "a" sees its own tail even though "b" appended later on disk
        a2 = self.chain.append_if_head("a", {}, 1, a1["hash"])
        self.assertEqual((a2["seq"], a2["prev"]), (2, a1["hash"]))
        self.assertEqual(self.chain.verify("a"), {"ok": True, "count": 2})
        self.assertEqual(self.chain.verify("b"), {"ok": True, "count": 1})

    def test_distinct_json_tenants_are_distinct_heads(self):
        self.chain.append(1, {})
        # tenant "1" is a different chain: empty head (0, ZERO) must match
        item = self.chain.append_if_head("1", {}, 0, ZERO)
        self.assertEqual((item["seq"], item["prev"]), (1, ZERO))
        # tenant 1 now has one record; asserting an empty head there conflicts
        with self.assertRaises(AuditChainConflictError) as cm:
            self.chain.append_if_head(1, {}, 0, ZERO)
        self.assertEqual(cm.exception.actual_count, 1)
        self.assertNotEqual(cm.exception.actual_hash, ZERO)

    def test_event_uses_standard_json_boundary(self):
        item = self.chain.append_if_head("t", {"n": None, "s": "x", "f": 1.5},
                                         0, ZERO)
        self.assertEqual(item["event"], {"n": None, "s": "x", "f": 1.5})

    # --- conflict cases ---

    def test_wrong_hash_on_empty_chain_conflicts_with_zero_actual(self):
        with self.assertRaises(AuditChainConflictError) as cm:
            self.chain.append_if_head("t", {}, 0, "f" * 64)
        e = cm.exception
        self.assertEqual((e.tenant, e.reason), ("t", "conflict"))
        self.assertEqual(e.expected_count, 0)
        self.assertEqual(e.expected_hash, "f" * 64)
        self.assertEqual((e.actual_count, e.actual_hash), (0, ZERO))
        self.assertFalse(self.path.exists())

    def test_wrong_count_on_empty_chain_conflicts(self):
        with self.assertRaises(AuditChainConflictError) as cm:
            self.chain.append_if_head("t", {}, 1, ZERO)
        e = cm.exception
        self.assertEqual((e.expected_count, e.expected_hash), (1, ZERO))
        self.assertEqual((e.actual_count, e.actual_hash), (0, ZERO))
        self.assertFalse(self.path.exists())

    def test_stale_hash_after_append_conflicts_and_reports_tail(self):
        first = self.chain.append("t", {"i": 1})
        second = self.chain.append("t", {"i": 2})
        before = self.path.read_bytes()
        with self.assertRaises(AuditChainConflictError) as cm:
            self.chain.append_if_head("t", {"i": 3}, 1, first["hash"])
        e = cm.exception
        self.assertEqual((e.expected_count, e.expected_hash), (1, first["hash"]))
        self.assertEqual((e.actual_count, e.actual_hash), (2, second["hash"]))
        # losing call writes nothing
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": 2})

    def test_stale_count_but_right_hash_conflicts(self):
        first = self.chain.append("t", {})
        with self.assertRaises(AuditChainConflictError) as cm:
            self.chain.append_if_head("t", {}, 2, first["hash"])
        e = cm.exception
        self.assertEqual((e.expected_count, e.actual_count), (2, 1))
        self.assertEqual((e.expected_hash, e.actual_hash),
                         (first["hash"], first["hash"]))
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": 1})

    def test_conflict_creates_no_record_for_other_tenant(self):
        self.chain.append("a", {})
        before = self.path.read_bytes()
        with self.assertRaises(AuditChainConflictError):
            self.chain.append_if_head("b", {}, 1, ZERO)
        # file already exists from "a"; "b" must still have no records and
        # no byte may have been appended
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(self.chain.verify("b"), {"ok": True, "count": 0})
        self.assertEqual(self.chain.verify("a"), {"ok": True, "count": 1})

    # --- input validation: ValueError before any read or byte ---

    def assertValueErrorBeforeIo(self, *args):
        self.assertFalse(self.path.exists())
        with self.assertRaises(ValueError):
            self.chain.append_if_head(*args)
        self.assertFalse(self.path.exists())

    def test_bool_count_rejected(self):
        self.assertValueErrorBeforeIo("t", {}, True, ZERO)
        self.assertValueErrorBeforeIo("t", {}, False, ZERO)

    def test_negative_float_string_none_count_rejected(self):
        for bad in (-1, 1.0, "0", None, 0.0, [0]):
            self.assertValueErrorBeforeIo("t", {}, bad, ZERO)

    def test_malformed_hash_rejected(self):
        for bad in ("", "a" * 63, "a" * 65, "A" * 64, "g" * 64,
                    "0" * 63 + "G", ZERO[:-1] + "A", 0, bytes(64), None):
            self.assertValueErrorBeforeIo("t", {}, 0, bad)

    def test_illegal_tenant_event_rejected_like_append(self):
        self.assertValueErrorBeforeIo(float("nan"), {}, 0, ZERO)
        self.assertValueErrorBeforeIo("t", float("inf"), 0, ZERO)
        self.assertValueErrorBeforeIo({1: "x"}, {}, 0, ZERO)
        cyc = {}
        cyc["self"] = cyc
        self.assertValueErrorBeforeIo("t", cyc, 0, ZERO)

    def test_value_error_takes_priority_over_corrupt_history(self):
        self.chain.append("t", {"v": 1})
        row = json.loads(self.path.read_text())
        row["event"] = {"v": 2}  # digest corruption at line 1
        self.path.write_text(json.dumps(row, sort_keys=True) + "\n")
        for bad_args in (
            ("t", float("nan"), 1, "x" * 64),
            ("t", {}, True, ZERO),
            ("t", {}, 1, "X" * 64),
            (float("nan"), {}, 1, ZERO),
        ):
            with self.assertRaises(ValueError):
                self.chain.append_if_head(*bad_args)
        # bytes untouched by all rejected calls
        self.assertEqual(
            self.path.read_bytes(),
            (json.dumps(row, sort_keys=True) + "\n").encode("utf-8"),
        )

    # --- corrupt history: same AuditChainStateError semantics as append ---

    def test_corrupt_history_raises_state_error_not_conflict(self):
        self.chain.append("t", {"v": 1})
        good = json.loads(self.path.read_text())
        row = dict(good)
        row["event"] = {"v": 2}  # breaks digest at line 1
        self.path.write_text(json.dumps(row, sort_keys=True) + "\n")
        before = self.path.read_bytes()
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.append_if_head("t", {}, 1, good["hash"])
        e = cm.exception
        self.assertEqual((e.tenant, e.seq, e.reason, e.line),
                         ("t", 1, "digest", 1))
        self.assertEqual(self.path.read_bytes(), before)

    def test_corrupt_history_sequence_and_missing_classification(self):
        other = {"tenant": "x", "seq": 1, "event": {}, "prev": ZERO}
        other["hash"] = AuditChain._hash(other)
        # gap: seq 2 without seq 1 for tenant t -> sequence at line 2
        gap = {"tenant": "t", "seq": 2, "event": {}, "prev": ZERO}
        gap["hash"] = AuditChain._hash(gap)
        self.path.write_text(
            json.dumps(other, sort_keys=True) + "\n"
            + json.dumps(gap, sort_keys=True) + "\n"
        )
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.append_if_head("t", {}, 2, gap["hash"])
        self.assertEqual((cm.exception.seq, cm.exception.reason, cm.exception.line),
                         (1, "sequence", 2))
        # unparseable line -> missing
        self.path.write_bytes(b"{not json\n")
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.append_if_head("t", {}, 0, ZERO)
        self.assertEqual((cm.exception.seq, cm.exception.reason, cm.exception.line),
                         (1, "missing", 1))

    # --- concurrency: exactly one winner per head ---

    def test_concurrent_same_head_exactly_one_wins_rest_conflict(self):
        n = 24
        outcomes = []
        box = threading.Lock()

        def worker(i):
            try:
                item = self.chain.append_if_head("t", {"i": i}, 0, ZERO)
                with box:
                    outcomes.append(("ok", item["seq"], item["prev"]))
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
        # winner built its record on the asserted empty head
        self.assertEqual(oks[0][1:], (1, ZERO))
        winning_hash = self.chain.read_tenant("t")[0]["hash"]
        # every loser deterministically observed the post-win tail
        for _, actual_count, actual_hash in conflicts:
            self.assertEqual((actual_count, actual_hash), (1, winning_hash))
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": 1})
        self.assertEqual(len(self.path.read_bytes().splitlines()), 1)

    def test_concurrent_chained_assertions_serialize(self):
        # each worker retries against the newly observed head: like a CAS
        # loop, every event must land exactly once in one verifiable chain
        n = 20
        committed = []
        box = threading.Lock()

        def worker(i):
            for _ in range(100):
                r = self.chain.verify("t")
                count = r["count"]
                head = ZERO if count == 0 else self.chain.read_tenant(
                    "t", start_seq=count)[0]["hash"]
                try:
                    item = self.chain.append_if_head(
                        "t", {"i": i}, count, head)
                    with box:
                        committed.append(item)
                    return
                except AuditChainConflictError:
                    continue

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(committed), n)
        self.assertEqual(sorted(it["seq"] for it in committed),
                         list(range(1, n + 1)))
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": n})

    def test_concurrent_conflict_and_corruption_share_state_semantics(self):
        # A corrupted tail: every racing call gets the same StateError,
        # never a conflict against a partially-read head.
        self.chain.append("t", {"v": 1})
        row = json.loads(self.path.read_text())
        row["event"] = {"v": 2}
        self.path.write_text(json.dumps(row, sort_keys=True) + "\n")
        details = []
        box = threading.Lock()

        def worker():
            try:
                self.chain.append_if_head("t", {}, 1, "f" * 64)
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
