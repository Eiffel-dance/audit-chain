import json
import multiprocessing as mp
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


class AppendIfHeadTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.path = Path(self.tmp.name) / "audit.jsonl"
        self.chain = AuditChain(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def read(self):
        return self.path.read_text(encoding="utf-8")

    def write(self, rows):
        with self.path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(r if isinstance(r, str) else json.dumps(r, sort_keys=True))
                f.write("\n")

    # --- success path: same fields/serialization as append ---

    def test_append_on_empty_chain_with_zero_assertion(self):
        item = self.chain.append_if_head("t", {"a": 1}, 0, ZERO)
        self.assertEqual(item["seq"], 1)
        self.assertEqual(item["prev"], ZERO)
        self.assertEqual(item["hash"], AuditChain._hash(item))
        self.assertEqual(set(item), {"tenant", "seq", "event", "prev", "hash"})
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": 1})

    def test_success_writes_byte_identical_record_as_append(self):
        item = self.chain.append_if_head("t", {"v": 1}, 0, ZERO)
        other = Path(self.tmp.name) / "plain.jsonl"
        plain = AuditChain(other).append("t", {"v": 1})
        self.assertEqual(item, plain)
        self.assertEqual(self.path.read_bytes(), other.read_bytes())

    def test_chaining_after_first_record(self):
        a = self.chain.append_if_head("t", {"i": 1}, 0, ZERO)
        b = self.chain.append_if_head("t", {"i": 2}, 1, a["hash"])
        self.assertEqual((b["seq"], b["prev"]), (2, a["hash"]))
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": 2})

    def test_other_tenant_interleaving_does_not_move_count_or_head(self):
        a1 = self.chain.append("a", {})
        self.chain.append("b", {})
        item = self.chain.append_if_head("a", {}, 1, a1["hash"])
        self.assertEqual((item["seq"], item["prev"]), (2, a1["hash"]))
        self.assertEqual(self.chain.verify("a"), {"ok": True, "count": 2})
        self.assertEqual(self.chain.verify("b"), {"ok": True, "count": 1})

    # --- conflict: actual head exposed, no bytes written ---

    def test_wrong_count_conflicts_with_actual_head(self):
        a = self.chain.append("t", {})
        b = self.chain.append("t", {})
        before = self.path.read_bytes()
        with self.assertRaises(AuditChainConflictError) as cm:
            self.chain.append_if_head("t", {}, 1, a["hash"])
        e = cm.exception
        self.assertEqual(e.reason, "conflict")
        self.assertEqual(e.tenant, "t")
        self.assertEqual((e.expected_count, e.expected_hash), (1, a["hash"]))
        self.assertEqual((e.actual_count, e.actual_hash), (2, b["hash"]))
        self.assertEqual(self.path.read_bytes(), before)

    def test_wrong_hash_conflicts(self):
        self.chain.append("t", {})
        before = self.path.read_bytes()
        with self.assertRaises(AuditChainConflictError) as cm:
            self.chain.append_if_head("t", {}, 1, "f" * 64)
        e = cm.exception
        self.assertEqual(e.reason, "conflict")
        head_hash = json.loads(self.read())["hash"]
        self.assertEqual((e.actual_count, e.actual_hash), (1, head_hash))
        self.assertEqual(e.expected_hash, "f" * 64)
        self.assertEqual(self.path.read_bytes(), before)

    def test_unseen_tenant_actual_head_is_zero(self):
        self.chain.append("other", {})
        before = self.path.read_bytes()
        with self.assertRaises(AuditChainConflictError) as cm:
            self.chain.append_if_head("t", {}, 1, ZERO)
        e = cm.exception
        self.assertEqual((e.actual_count, e.actual_hash), (0, ZERO))
        self.assertEqual(self.path.read_bytes(), before)
        # the same empty assertion succeeds and starts that tenant at seq 1
        item = self.chain.append_if_head("t", {}, 0, ZERO)
        self.assertEqual((item["seq"], item["prev"]), (1, ZERO))

    def test_nonexistent_file_conflict_reports_zero_head_without_record(self):
        with self.assertRaises(AuditChainConflictError) as cm:
            self.chain.append_if_head("t", {}, 0, "f" * 64)
        e = cm.exception
        self.assertEqual((e.actual_count, e.actual_hash), (0, ZERO))
        # no JSONL record landed; the existing path (if any) holds no newline
        if self.path.exists():
            self.assertNotIn(b"\n", self.path.read_bytes())

    def test_conflict_on_legacy_file_without_trailing_newline_writes_nothing(self):
        self.chain.append("t", {"v": 1})
        self.path.write_bytes(self.path.read_bytes().rstrip(b"\n"))
        before = self.path.read_bytes()
        with self.assertRaises(AuditChainConflictError):
            self.chain.append_if_head("t", {}, 0, ZERO)
        self.assertEqual(self.path.read_bytes(), before)

    # --- malformed assertions: ValueError before reading, nothing created ---

    def test_rejects_bool_negative_float_and_other_counts(self):
        fresh = Path(self.tmp.name) / "fresh.jsonl"
        chain = AuditChain(fresh)
        for bad in (True, False, -1, 0.0, 1.0, "0", None, [], (0,)):
            with self.assertRaises(ValueError):
                chain.append_if_head("t", {}, bad, ZERO)
        self.assertFalse(fresh.exists())

    def test_rejects_malformed_hash(self):
        fresh = Path(self.tmp.name) / "fresh2.jsonl"
        chain = AuditChain(fresh)
        for bad in ("", "a" * 63, "A" * 64, "g" * 64, ZERO + "a",
                    0, None, bytes(ZERO, "ascii"), bytearray()):
            with self.assertRaises(ValueError):
                chain.append_if_head("t", {}, 0, bad)
        self.assertFalse(fresh.exists())

    def test_rejects_illegal_tenant_and_event_before_reading(self):
        # corrupt history sits on disk: ValueError must still win, as for append
        row = {"tenant": "t", "seq": 1, "event": {}, "prev": ZERO}
        row["hash"] = AuditChain._hash(row)
        row["event"] = {"tampered": True}
        self.write([row])
        before = self.path.read_bytes()
        for bad_tenant in (float("nan"), {1: "x"}):
            with self.assertRaises(ValueError):
                self.chain.append_if_head(bad_tenant, {}, 0, ZERO)
        with self.assertRaises(ValueError):
            self.chain.append_if_head("t", float("inf"), 0, ZERO)
        self.assertEqual(self.path.read_bytes(), before)

    # --- corrupt history: same error semantics as append, beats conflict ---

    def test_corrupt_history_raises_state_error_with_same_location(self):
        self.chain.append("t", {"v": 1})
        rows = [json.loads(l) for l in self.read().splitlines()]
        rows[0]["event"] = {"v": 2}
        self.write(rows)
        before = self.path.read_bytes()
        with self.assertRaises(AuditChainStateError) as cm:
            # hash that would otherwise be a perfectly plausible assertion
            self.chain.append_if_head("t", {}, 1, "f" * 64)
        e = cm.exception
        self.assertEqual((e.tenant, e.seq, e.reason, e.line),
                         ("t", 1, "digest", 1))
        self.assertEqual(self.path.read_bytes(), before)

    def test_missing_corruption_reports_expected_seq_and_line(self):
        valid = {"tenant": "x", "seq": 1, "event": {}, "prev": ZERO}
        valid["hash"] = AuditChain._hash(valid)
        with self.path.open("w", encoding="utf-8") as f:
            f.write(json.dumps(valid, sort_keys=True) + "\n")
            f.write(b"{oops".decode())
            f.write("\n")
        before = self.path.read_bytes()
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.append_if_head("t", {}, 0, ZERO)
        e = cm.exception
        self.assertEqual((e.tenant, e.seq, e.reason, e.line),
                         ("t", 1, "missing", 2))
        self.assertEqual(self.path.read_bytes(), before)

    # --- concurrency: one winner, deterministic conflicts for the rest ---

    def test_threads_asserting_same_head_have_one_winner(self):
        n = 24
        outcomes = []
        box = threading.Lock()

        def worker(i):
            try:
                item = self.chain.append_if_head("t", {"i": i}, 0, ZERO)
                with box:
                    outcomes.append(("ok", item))
            except AuditChainConflictError as e:
                with box:
                    outcomes.append(("conflict", e.actual_count, e.actual_hash))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        winners = [o for o in outcomes if o[0] == "ok"]
        conflicts = [o for o in outcomes if o[0] == "conflict"]
        self.assertEqual(len(winners), 1)
        winner_hash = winners[0][1]["hash"]
        self.assertTrue(all(o[1] == 1 and o[2] == winner_hash for o in conflicts))
        self.assertEqual(len(self.path.read_bytes().splitlines()), 1)
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": 1})

    def test_processes_asserting_same_head_have_one_winner(self):
        ctx = mp.get_context("fork")
        n = 16
        with ctx.Pool(8) as pool:
            results = pool.map(_mp_append_if_head,
                               [(str(self.path), i) for i in range(n)])
        self.assertEqual(results.count("ok"), 1, results)
        self.assertEqual(results.count("conflict"), n - 1)
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": 1})


def _mp_append_if_head(args):
    path, i = args
    try:
        AuditChain(path).append_if_head("t", {"i": i}, 0, ZERO)
        return "ok"
    except AuditChainConflictError:
        return "conflict"


if __name__ == "__main__":
    unittest.main()
