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

    # --- type boundary: exactly an AuditChain, before any path is read ---

    def test_other_must_be_audit_chain(self):
        for bad in (None, 1, 1.0, True, "path", b"path", object(),
                    str(self.left_path), Path(self.left_path),
                    {"path": self.left_path}, ["x"]):
            with self.assertRaises(ValueError):
                self.left.compare(bad)

    def test_type_error_raises_before_either_path_is_read(self):
        calls = []
        original = AuditChain._read_snapshot

        def spy(inst):
            calls.append(inst.path)
            return original(inst)

        self.left._read_snapshot = spy
        self.right._read_snapshot = spy
        with self.assertRaises(ValueError):
            self.left.compare(str(self.left_path))
        self.assertEqual(calls, [])

    def test_type_error_creates_nothing_even_when_neither_path_exists(self):
        phantom_left = Path(self.tmp.name) / "phantom-l.jsonl"
        chain = AuditChain(phantom_left)
        with self.assertRaises(ValueError):
            chain.compare(object())
        self.assertFalse(phantom_left.exists())
        self.assertEqual(list(Path(self.tmp.name).iterdir()), [])

    # --- missing paths are empty histories ---

    def test_two_missing_paths_are_equal_empty_histories(self):
        self.assertEqual(
            self.left.compare(self.right),
            {"ok": True, "equal": True, "tenants": []},
        )
        # nothing was created to reach that verdict
        self.assertFalse(self.left_path.exists())
        self.assertFalse(self.right_path.exists())

    def test_missing_side_is_the_empty_chain(self):
        self.seed(self.right, ("a", {"i": 1}))
        r = self.left.compare(self.right)
        self.assertEqual((r["ok"], r["equal"], r["tenant"], r["seq"],
                          r["reason"], r["left"]),
                         (True, False, "a", 1, "missing_left", None))
        self.assertEqual(r["right"]["seq"], 1)
        # reverse orientation: the other side is the missing one
        r = self.right.compare(self.left)
        self.assertEqual((r["ok"], r["equal"], r["tenant"], r["seq"],
                          r["reason"], r["right"]),
                         (True, False, "a", 1, "missing_right", None))
        self.assertEqual(r["left"]["seq"], 1)

    # --- self / same path: one shared snapshot, stable equality ---

    def test_compare_self_is_equal_when_empty(self):
        r = self.left.compare(self.left)
        self.assertEqual(r, {"ok": True, "equal": True, "tenants": []})

    def test_compare_self_is_equal_with_records(self):
        self.seed(self.left, ("a", {"i": 1}), ("b", {"i": 1}),
                  ("a", {"i": 2}))
        r = self.left.compare(self.left)
        self.assertTrue(r["ok"])
        self.assertTrue(r["equal"])
        self.assertEqual(
            [(e["tenant"], e["count"]) for e in r["tenants"]],
            [("a", 2), ("b", 1)])

    def test_two_instances_on_one_path_share_one_snapshot(self):
        self.seed(self.left, ("a", {"i": 1}), ("a", {"i": 2}))
        alias = AuditChain(self.left_path)  # same path, second instance
        calls = []
        original = AuditChain._read_snapshot

        def spy(inst):
            calls.append(inst.path)
            return original(inst)

        self.left._read_snapshot = lambda: spy(self.left)
        alias._read_snapshot = lambda: spy(alias)
        r = self.left.compare(alias)
        self.assertEqual(len(calls), 1)
        self.assertTrue(r["equal"])
        # path equality is on resolved Path values, so str/Path construction
        # mixes still name the same log
        alias_str = AuditChain(str(self.left_path))
        self.assertEqual(alias_str.path, self.left.path)
        calls.clear()
        alias_str._read_snapshot = lambda: spy(alias_str)
        r = self.left.compare(alias_str)
        self.assertEqual(len(calls), 1)
        self.assertTrue(r["equal"])

    # --- field-for-field parity with compare_bytes ---

    def parity_cases(self):
        cases = []
        # equal content, different physical interleavings
        la = AuditChain(self.left_path)
        rb = AuditChain(self.right_path)
        self.seed(la, ("a", {"i": 1}), ("b", {"i": 1}),
                  ("a", {"i": 2}), ("a", {"i": 3}), ("b", {"i": 2}))
        self.seed(rb, ("b", {"i": 1}), ("b", {"i": 2}),
                  ("a", {"i": 1}), ("a", {"i": 2}), ("a", {"i": 3}))
        cases.append(("interleaved-equal", la, rb))
        return cases

    def test_equal_interleaved_paths_match_compare_bytes(self):
        for _name, la, rb in self.parity_cases():
            lb = la.path.read_bytes()
            rb_bytes = rb.path.read_bytes()
            self.assertNotEqual(lb, rb_bytes)
            r = la.compare(rb)
            self.assertEqual(r, AuditChain(self.left_path.with_name("x"))
                             .compare_bytes(lb, rb_bytes))
            self.assertTrue(r["equal"])
            self.assertEqual([e["tenant"] for e in r["tenants"]], ["a", "b"])

    def test_different_event_matches_compare_bytes(self):
        self.seed(self.left, ("t", {"v": 1}), ("t", {"v": 2}))
        self.seed(self.right, ("t", {"v": 1}), ("t", {"v": 9}))
        r = self.left.compare(self.right)
        expected = AuditChain(self.left_path.with_name("x")).compare_bytes(
            self.left_path.read_bytes(), self.right_path.read_bytes())
        self.assertEqual(r, expected)
        self.assertEqual((r["tenant"], r["seq"], r["reason"]),
                         ("t", 2, "different"))
        self.assertEqual(r["left"]["event"], {"v": 2})
        self.assertEqual(r["right"]["event"], {"v": 9})

    def test_missing_left_right_match_compare_bytes(self):
        self.seed(self.left, ("t", {"v": 1}), ("t", {"v": 2}))
        self.seed(self.right, ("t", {"v": 1}))
        self.assertEqual(self.left.compare(self.right),
                         AuditChain(self.left_path.with_name("x")).compare_bytes(
                             self.left_path.read_bytes(),
                             self.right_path.read_bytes()))
        r = self.left.compare(self.right)
        self.assertEqual((r["tenant"], r["seq"], r["reason"], r["right"]),
                         ("t", 2, "missing_right", None))
        r = self.right.compare(self.left)
        self.assertEqual(r["reason"], "missing_left")
        self.assertIsNone(r["left"])

    def test_right_only_tenant_order_matches_compare_bytes(self):
        self.seed(self.left, ("a", {"i": 1}))
        self.seed(self.right, ("z", {"i": 1}), ("a", {"i": 1}),
                  ("m", {"i": 1}))
        r = self.left.compare(self.right)
        self.assertEqual(r, AuditChain(self.left_path.with_name("x"))
                         .compare_bytes(self.left_path.read_bytes(),
                                        self.right_path.read_bytes()))
        self.assertEqual((r["tenant"], r["seq"], r["reason"]),
                         ("z", 1, "missing_left"))

    def test_equality_summary_uses_original_tenant_values(self):
        la = AuditChain(self.left_path)
        rb = AuditChain(self.right_path)
        self.seed(la, ({"a": 1, "b": 2}, {"i": 1}), ("s", {"i": 1}))
        self.seed(rb, ("s", {"i": 1}), ({"b": 2, "a": 1}, {"i": 1}))
        r = la.compare(rb)
        self.assertTrue(r["equal"])
        self.assertEqual([e["tenant"] for e in r["tenants"]],
                         [{"a": 1, "b": 2}, "s"])

    # --- corruption: verify_all_bytes shape, side naming, left first ---

    def corrupt_variants(self):
        # (bytes, expected_at, expected_tenant, expected_reason)
        good_item = {"tenant": "t", "seq": 1, "event": {}, "prev": ZERO}
        good_item["hash"] = AuditChain._hash(good_item)
        good = record_bytes(good_item)
        gap = {"tenant": "t", "seq": 3, "event": {}, "prev": "9" * 64}
        gap["hash"] = AuditChain._hash(gap)
        bad_prev = {"tenant": "t", "seq": 1, "event": {}, "prev": "9" * 64}
        bad_prev["hash"] = AuditChain._hash(bad_prev)
        return [
            (b"{oops\n", 1, None, "missing"),
            (b"\xff", 1, None, "missing"),
            (good + b"\xff", 2, None, "missing"),
            (good + record_bytes(gap), 2, "t", "sequence"),
            (record_bytes(bad_prev), 1, "t", "digest"),
        ]

    def test_corrupt_side_reports_side_and_is_not_compared(self):
        good_item = {"tenant": "t", "seq": 1, "event": {}, "prev": ZERO}
        good_item["hash"] = AuditChain._hash(good_item)
        good = record_bytes(good_item)
        probe = AuditChain(self.left_path.with_name("x"))
        for bad, at, tenant, reason in self.corrupt_variants():
            # corrupt left (self), valid right
            self.left_path.write_bytes(bad)
            self.right_path.write_bytes(good)
            r = self.left.compare(self.right)
            self.assertEqual(r, probe.compare_bytes(bad, good))
            self.assertEqual(r, {"ok": False, "side": "left", "at": at,
                                 "tenant": tenant, "reason": reason})
            # corrupt right (other), valid left
            r = self.right.compare(self.left)
            self.assertEqual(r, probe.compare_bytes(good, bad))
            self.assertEqual(r, {"ok": False, "side": "right", "at": at,
                                 "tenant": tenant, "reason": reason})

    def test_both_sides_corrupt_reports_left_first(self):
        item = {"tenant": "t", "seq": 1, "event": {}, "prev": ZERO}
        item["hash"] = AuditChain._hash(item)
        good = record_bytes(item)
        bad_digest = {"tenant": "t", "seq": 1, "event": {},
                      "prev": "9" * 64}
        bad_digest["hash"] = AuditChain._hash(bad_digest)
        self.left_path.write_bytes(record_bytes(bad_digest) + b"\xff")
        self.right_path.write_bytes(good + b"{later\n")
        r = self.left.compare(self.right)
        self.assertEqual(r, {"ok": False, "side": "left", "at": 1,
                             "tenant": "t", "reason": "digest"})

    def test_corrupt_log_is_not_modified(self):
        bad = b"{oops\n"
        self.left_path.write_bytes(bad)
        self.right_path.write_bytes(bad)
        before = {p: p.read_bytes()
                  for p in (self.left_path, self.right_path)}
        entries_before = sorted(p.name for p in Path(self.tmp.name).iterdir())
        r = self.left.compare(self.right)
        self.assertFalse(r["ok"])
        self.assertEqual(r["side"], "left")
        for p in (self.left_path, self.right_path):
            self.assertEqual(p.read_bytes(), before[p])
        # no manifest, cache or other side data appeared
        self.assertEqual(sorted(p.name for p in Path(self.tmp.name).iterdir()),
                         entries_before)

    # --- read-only guarantees ---

    def test_valid_compare_writes_nothing(self):
        self.seed(self.left, ("a", {"i": 1}), ("b", {"i": 1}))
        self.seed(self.right, ("a", {"i": 1}))
        before = {p: p.read_bytes()
                  for p in (self.left_path, self.right_path)}
        entries = sorted(p.name for p in Path(self.tmp.name).iterdir())
        self.left.compare(self.right)
        for p in (self.left_path, self.right_path):
            self.assertEqual(p.read_bytes(), before[p])
        self.assertEqual(sorted(p.name for p in Path(self.tmp.name).iterdir()),
                         entries)

    # --- concurrency: only complete pre/post-append snapshots are compared ---

    def test_concurrent_appends_never_yield_torn_comparison(self):
        self.seed(self.left, ("t", {"i": 0}))
        self.seed(self.right, ("t", {"i": 0}))
        stop = threading.Event()
        failures = []

        def writer(chain, base):
            for j in range(1, 81):
                if stop.is_set():
                    break
                chain.append("t", {"i": base + j})
                chain.append("u", {"i": base + j})

        def comparer():
            while not stop.is_set():
                r = self.left.compare(self.right)
                # A torn/half-line snapshot would surface as a verify failure
                # (ok False). Both files are produced solely by appends under
                # exclusive leases, so every complete snapshot must validate.
                if not r["ok"]:
                    failures.append(r)
                    return

        threads = [threading.Thread(target=writer, args=(self.left, 0)),
                   threading.Thread(target=writer, args=(self.right, 1000)),
                   threading.Thread(target=comparer)]
        for t in threads:
            t.start()
        threads[0].join()
        threads[1].join()
        stop.set()
        threads[2].join()
        self.assertEqual(failures, [])
        # after quiescence the chains may legitimately differ in counts, but
        # both are internally valid
        r = self.left.compare(self.right)
        self.assertTrue(r["ok"])

    def test_self_compare_under_concurrent_appends_is_always_equal(self):
        alias = AuditChain(self.left_path)
        stop = threading.Event()
        failures = []

        def writer():
            for j in range(120):
                if stop.is_set():
                    return
                self.left.append("t", {"i": j})

        def comparer():
            while not stop.is_set():
                # two instances on the same path reuse one shared snapshot, so
                # a concurrent append can never make the two sides differ
                r = self.left.compare(alias)
                if not r.get("equal"):
                    failures.append(r)
                    return

        threads = [threading.Thread(target=writer),
                   threading.Thread(target=comparer)]
        for t in threads:
            t.start()
        threads[0].join()
        stop.set()
        threads[1].join()
        self.assertEqual(failures, [])
        self.assertTrue(self.left.compare(alias)["equal"])


if __name__ == "__main__":
    unittest.main()
