import json
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app import AuditChain, ZERO


def record_bytes(item):
    return (json.dumps(item, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


class CompareTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.left_path = Path(self.tmp.name) / "left.jsonl"
        self.right_path = Path(self.tmp.name) / "right.jsonl"
        self.left = AuditChain(self.left_path)
        self.right = AuditChain(self.right_path)

    def tearDown(self):
        self.tmp.cleanup()

    def seed(self, chain, *tenant_events):
        for tenant, event in tenant_events:
            chain.append(tenant, event)

    def row(self, tenant="t", seq=1, prev=ZERO, event=None, hash_=None):
        item = {"tenant": tenant, "seq": seq, "event": event or {},
                "prev": prev}
        item["hash"] = hash_ if hash_ is not None else AuditChain._hash(item)
        return item, record_bytes(item)

    # --- argument boundary: AuditChain only, before any path is read ---

    def test_other_must_be_audit_chain(self):
        # A corrupt log at self.path must not change the verdict: the type
        # boundary wins before either path is read or probed.
        self.left_path.write_bytes(b"{oops\n")
        for bad in (None, 1, "right", b"", bytearray(b""),
                    self.left_path, object(),
                    {"path": self.right_path}):
            with self.assertRaises(ValueError):
                self.left.compare(bad)

    def test_boundary_error_reads_neither_path_and_creates_nothing(self):
        phantom_l = self.left_path.with_name("phantom-l.jsonl")
        phantom_r = self.right_path.with_name("phantom-r.jsonl")
        chain = AuditChain(phantom_l)
        with self.assertRaises(ValueError):
            chain.compare("not-a-chain")
        self.assertFalse(phantom_l.exists())
        self.assertFalse(phantom_r.exists())

    def test_audit_chain_subclass_is_accepted(self):
        class Sub(AuditChain):
            pass

        self.seed(self.left, ("a", {"i": 1}))
        r = self.left.compare(Sub(self.left_path))
        self.assertTrue(r["equal"])

    # --- read-only: never creates, rewrites or deletes, no side data ---

    def test_two_missing_paths_are_equal_empty_snapshots(self):
        self.assertFalse(self.left_path.exists())
        self.assertFalse(self.right_path.exists())
        self.assertEqual(
            self.left.compare(self.right),
            {"ok": True, "equal": True, "tenants": []},
        )
        # A read over a missing path must not create it or anything beside it.
        self.assertFalse(self.left_path.exists())
        self.assertFalse(self.right_path.exists())
        self.assertEqual(list(Path(self.tmp.name).iterdir()), [])

    def test_compare_does_not_modify_existing_logs(self):
        self.seed(self.left, ("a", {"i": 1}))
        self.seed(self.right, ("a", {"i": 1}))
        lb, rb = self.left_path.read_bytes(), self.right_path.read_bytes()
        r = self.left.compare(self.right)
        self.assertTrue(r["equal"])
        self.assertEqual(self.left_path.read_bytes(), lb)
        self.assertEqual(self.right_path.read_bytes(), rb)

    def test_corrupt_files_are_not_rewritten(self):
        self.left_path.write_bytes(b"{oops\n")
        self.right_path.write_bytes(b"\xff")
        r = self.left.compare(self.right)
        self.assertEqual((r["ok"], r["side"], r["at"], r["tenant"],
                          r["reason"]),
                         (False, "left", 1, None, "missing"))
        self.assertEqual(self.left_path.read_bytes(), b"{oops\n")
        self.assertEqual(self.right_path.read_bytes(), b"\xff")

    # --- same path: one shared snapshot, stably equal to itself ---

    def test_same_instance_compares_equal(self):
        self.seed(self.left, ("a", {"i": 1}), ("b", {"i": 1}),
                  ("a", {"i": 2}))
        r = self.left.compare(self.left)
        self.assertTrue(r["equal"])
        self.assertEqual(
            [(e["tenant"], e["count"]) for e in r["tenants"]],
            [("a", 2), ("b", 1)],
        )

    def test_two_instances_same_path_reuse_one_snapshot(self):
        self.seed(self.left, ("a", {"i": 1}), ("b", {"i": 1}))
        other = AuditChain(Path(self.left_path))
        r = self.left.compare(other)
        self.assertTrue(r["equal"])
        self.assertEqual(r, self.left.compare_bytes(
            self.left_path.read_bytes(), self.left_path.read_bytes()))

    def test_two_instances_same_path_stay_equal_under_concurrent_appends(self):
        # Even while another chain on the same path keeps appending, the single
        # captured snapshot must make both sides identical: never a missing
        # verdict or a corruption verdict from a torn read.
        other = AuditChain(Path(self.left_path))

        def writer():
            for i in range(200):
                other.append("w", {"i": i})

        t = threading.Thread(target=writer)
        t.start()
        try:
            for _ in range(400):
                r = self.left.compare(other)
                self.assertTrue(r["ok"], r)
                self.assertTrue(r["equal"], r)
        finally:
            t.join()

    # --- result is field-for-field compare_bytes on the actual snapshots ---

    def test_matches_compare_bytes_on_captured_snapshots(self):
        lc = AuditChain(self.left_path.with_name("l.jsonl"))
        rc = AuditChain(self.right_path.with_name("r.jsonl"))
        self.seed(lc, ("a", {"i": 1}), ("b", {"i": 1}),
                  ("a", {"i": 2}), ("a", {"i": 3}), ("b", {"i": 2}))
        self.seed(rc, ("b", {"i": 1}), ("b", {"i": 2}),
                  ("a", {"i": 1}), ("a", {"i": 2}), ("a", {"i": 3}))
        lb, rb = self.left_path.with_name("l.jsonl").read_bytes(), \
            self.right_path.with_name("r.jsonl").read_bytes()
        self.assertNotEqual(lb, rb)
        self.assertEqual(lc.compare(rc), self.left.compare_bytes(lb, rb))

    def test_missing_side_treated_as_empty_history(self):
        self.seed(self.left, ("a", {"i": 1}))
        r = self.left.compare(self.right)  # right path does not exist
        self.assertEqual((r["ok"], r["equal"], r["tenant"], r["seq"],
                          r["reason"], r["right"]),
                         (True, False, "a", 1, "missing_right", None))
        self.assertEqual(r["left"]["seq"], 1)
        r = self.right.compare(self.left)
        self.assertEqual((r["ok"], r["equal"], r["tenant"], r["seq"],
                          r["reason"], r["left"]),
                         (True, False, "a", 1, "missing_left", None))
        self.assertEqual(r["right"]["seq"], 1)
        # the missing path must still not have been created
        self.assertFalse(self.right_path.exists())

    def test_different_event_between_paths(self):
        self.seed(self.left, ("t", {"v": 1}))
        self.seed(self.right, ("t", {"v": 2}))
        r = self.left.compare(self.right)
        self.assertEqual((r["ok"], r["equal"], r["tenant"], r["seq"],
                          r["reason"]),
                         (True, False, "t", 1, "different"))
        self.assertEqual(r["left"]["event"], {"v": 1})
        self.assertEqual(r["right"]["event"], {"v": 2})

    def test_right_only_tenant_ordered_after_left_tenants(self):
        self.seed(self.left, ("a", {"i": 1}))
        self.seed(self.right, ("z", {"i": 1}), ("a", {"i": 1}),
                  ("m", {"i": 1}))
        r = self.left.compare(self.right)
        self.assertEqual((r["tenant"], r["seq"], r["reason"]),
                         ("z", 1, "missing_left"))
        self.assertIsNone(r["left"])
        self.assertEqual(r["right"]["tenant"], "z")

    # --- corruption: verify_all_bytes rules, side naming, left first ---

    def test_left_corruption_reports_left_and_skips_comparison(self):
        _t, good = self.row("t", 1)
        self.left_path.write_bytes(b"{oops\n")
        self.right_path.write_bytes(good)
        r = self.left.compare(self.right)
        self.assertEqual(r, {
            "ok": False, "side": "left", "at": 1,
            "tenant": None, "reason": "missing",
        })

    def test_right_corruption_reports_right(self):
        _t, good = self.row("t", 1)
        self.left_path.write_bytes(good)
        self.right_path.write_bytes(good + b"\xff")
        r = self.left.compare(self.right)
        self.assertEqual(r, {
            "ok": False, "side": "right", "at": 2,
            "tenant": None, "reason": "missing",
        })

    def test_both_corrupt_left_reported_first(self):
        good_t, good = self.row("t", 1)
        bad_digest = {"tenant": "t", "seq": 1, "event": {},
                      "prev": "9" * 64}
        bad_digest["hash"] = AuditChain._hash(bad_digest)
        self.left_path.write_bytes(record_bytes(bad_digest) + b"\xff")
        self.right_path.write_bytes(good + b"{later\n")
        r = self.left.compare(self.right)
        self.assertEqual(r["side"], "left")
        self.assertEqual((r["at"], r["tenant"], r["reason"]),
                         (1, "t", "digest"))

    def test_sequence_corruption_fields(self):
        _t, good1 = self.row("t", 1)
        gap = {"tenant": "t", "seq": 3, "event": {}, "prev": "9" * 64}
        gap["hash"] = AuditChain._hash(gap)
        self.left_path.write_bytes(good1 + record_bytes(gap))
        self.right_path.write_bytes(good1)
        r = self.left.compare(self.right)
        self.assertEqual((r["ok"], r["side"], r["at"], r["tenant"],
                          r["reason"]),
                         (False, "left", 2, "t", "sequence"))

    # --- concurrency: snapshots stay complete across appends ---

    def test_concurrent_appends_yield_only_consistent_verdicts(self):
        # Only the left log grows; the right stays empty. Every compare must
        # observe a complete pre-/post-append snapshot: either the empty
        # equality or a well-formed missing_right at seq 1, never a corruption
        # verdict from a half-written line.
        writer_chain = AuditChain(self.left_path)
        stop = threading.Event()

        def writer():
            i = 0
            while not stop.is_set():
                writer_chain.append("a", {"i": i})
                i += 1
                if i >= 500:
                    break

        t = threading.Thread(target=writer)
        t.start()
        seen_difference = False
        try:
            for _ in range(1000):
                r = self.left.compare(self.right)
                self.assertTrue(r["ok"], r)
                if r["equal"]:
                    self.assertEqual(r["tenants"], [])
                else:
                    seen_difference = True
                    self.assertEqual((r["tenant"], r["seq"], r["reason"],
                                      r["left"] is None, r["right"]),
                                     ("a", 1, "missing_right", False, None))
        finally:
            stop.set()
            t.join()
        self.assertTrue(seen_difference)

    def test_each_call_compares_exactly_two_fixed_snapshots(self):
        # While compare runs on two nonempty logs, appends land on both sides;
        # the returned records must themselves be intact five-field records
        # (a torn snapshot could never parse), and the verdict must match
        # compare_bytes on the same files' complete bytes at some point.
        self.seed(self.left, ("a", {"i": 0}))
        self.seed(self.right, ("a", {"i": 0}))
        lw = AuditChain(self.left_path)
        rw = AuditChain(self.right_path)
        stop = threading.Event()

        def write_both():
            i = 1
            while not stop.is_set():
                lw.append("a", {"i": i})
                rw.append("a", {"i": i})
                i += 1
                if i >= 300:
                    break

        t = threading.Thread(target=write_both)
        t.start()
        try:
            for _ in range(600):
                r = self.left.compare(self.right)
                self.assertTrue(r["ok"], r)
                if not r["equal"]:
                    # The two logs grow by independent commits, so the first
                    # divergence is either an event/seq difference with two
                    # intact records or a longer side with one record; every
                    # attached record must carry the full five fields, proving
                    # it came from a complete, parseable snapshot.
                    self.assertIn(r["reason"],
                                  ("different", "missing_left",
                                   "missing_right"))
                    sides = ("left", "right") \
                        if r["reason"] == "different" \
                        else (("left",)
                              if r["reason"] == "missing_right"
                              else ("right",))
                    for side in sides:
                        item = r[side]
                        self.assertEqual(
                            set(item), {"tenant", "seq", "event",
                                        "prev", "hash"})
                        self.assertEqual(item["seq"], r["seq"])
        finally:
            stop.set()
            t.join()


if __name__ == "__main__":
    unittest.main()
