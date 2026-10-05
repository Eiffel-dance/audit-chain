import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app import AuditChain, ZERO


def record_bytes(item):
    return (json.dumps(item, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


class CompareBytesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.path = Path(self.tmp.name) / "audit.jsonl"
        self.chain = AuditChain(self.path)
        # Every compare runs through an AuditChain pointed at a path that must
        # never be read, created or modified by the pure in-memory entry.
        self.other = AuditChain(self.path.with_name("never-touched.jsonl"))

    def tearDown(self):
        self.tmp.cleanup()

    def seed(self, chain, *tenant_events):
        for tenant, event in tenant_events:
            chain.append(tenant, event)

    def snapshot(self, chain=None):
        chain = chain or self.chain
        return Path(chain.path).read_bytes()

    def row(self, tenant="t", seq=1, prev=ZERO, event=None, hash_=None):
        item = {"tenant": tenant, "seq": seq, "event": event or {}, "prev": prev}
        item["hash"] = hash_ if hash_ is not None else AuditChain._hash(item)
        return item, record_bytes(item)

    # --- argument boundary: exact bytes, ValueError before any parsing ---

    def test_arguments_must_be_bytes(self):
        for bad in ("", bytearray(b""), bytearray(b"{}"), 1, 1.0, None,
                    True, [b""], {b"": 1}):
            with self.assertRaises(ValueError):
                self.other.compare_bytes(bad, b"")
            with self.assertRaises(ValueError):
                self.other.compare_bytes(b"", bad)

    def test_left_type_checked_first(self):
        # A non-bytes left raises even though the right is non-bytes too or a
        # corrupt buffer: no snapshot may be parsed to reach this verdict.
        with self.assertRaises(ValueError):
            self.other.compare_bytes("left", b"\xff")
        with self.assertRaises(ValueError):
            self.other.compare_bytes("left", "right")

    def test_never_reads_creates_or_modifies_constructor_path(self):
        phantom = self.path.with_name("phantom.jsonl")
        chain = AuditChain(phantom)
        self.seed(self.chain, ("a", {"i": 1}), ("b", {"i": 1}),
                  ("a", {"i": 2}))
        data = self.snapshot()
        self.assertEqual(chain.compare_bytes(b"", b"")["equal"], True)
        self.assertFalse(phantom.exists())
        self.assertTrue(chain.compare_bytes(data, data)["equal"])
        self.assertFalse(phantom.exists())

    def test_does_not_mutate_buffers(self):
        self.seed(self.chain, ("a", {"i": 1}))
        left = self.snapshot()
        right = bytearray(left)  # mutable copy handed over elsewhere
        before_left = bytes(left)
        with self.assertRaises(ValueError):
            self.other.compare_bytes(left, right)  # bytearray rejected
        self.assertEqual(bytes(left), before_left)

    # --- empty snapshots ---

    def test_empty_snapshots_are_equal(self):
        self.assertEqual(
            self.other.compare_bytes(b"", b""),
            {"ok": True, "equal": True, "tenants": []},
        )

    def test_empty_against_nonempty_is_missing_side(self):
        self.seed(self.chain, ("a", {"i": 1}))
        data = self.snapshot()
        r = self.other.compare_bytes(b"", data)
        self.assertEqual(
            (r["ok"], r["equal"], r["tenant"], r["seq"], r["reason"],
             r["left"]),
            (True, False, "a", 1, "missing_left", None),
        )
        self.assertEqual(r["right"]["seq"], 1)
        r = self.other.compare_bytes(data, b"")
        self.assertEqual(
            (r["ok"], r["equal"], r["tenant"], r["seq"], r["reason"],
             r["right"]),
            (True, False, "a", 1, "missing_right", None),
        )
        self.assertEqual(r["left"]["seq"], 1)

    # --- corruption: verify_all_bytes rules, side naming, left first ---

    def test_corrupt_snapshot_is_not_compared(self):
        item, good = self.row("t", 1)
        # Right holds a perfectly equal chain, yet the left's defect alone
        # decides and is reported with verify_all_bytes' exact fields.
        r = self.other.compare_bytes(b"{oops\n", good)
        self.assertEqual(r, {
            "ok": False, "side": "left", "at": 1,
            "tenant": None, "reason": "missing",
        })
        r = self.other.compare_bytes(good, b"\xff")
        self.assertEqual(r, {
            "ok": False, "side": "right", "at": 1,
            "tenant": None, "reason": "missing",
        })

    def test_left_corruption_reported_when_both_sides_corrupt(self):
        # Both corrupt at physical line 1: fixed left-first, regardless of the
        # kind of corruption on the right.
        r = self.other.compare_bytes(b"{oops\n", b"\xff")
        self.assertEqual(r["side"], "left")
        self.assertEqual((r["at"], r["tenant"], r["reason"]),
                         (1, None, "missing"))
        _t, good = self.row("t", 1)
        bad_digest_left, _ = self.row("t", 1, prev="9" * 64)
        r = self.other.compare_bytes(record_bytes(bad_digest_left) + b"\xff",
                                     good + b"{later\n")
        self.assertEqual(r["side"], "left")
        self.assertEqual((r["at"], r["tenant"], r["reason"]),
                         (1, "t", "digest"))

    def test_corruption_reasons_and_tenant_fields(self):
        _t, good1 = self.row("t", 1)
        gap = {"tenant": "t", "seq": 3, "event": {}, "prev": "9" * 64}
        gap["hash"] = AuditChain._hash(gap)
        r = self.other.compare_bytes(good1 + record_bytes(gap), good1)
        self.assertEqual((r["ok"], r["side"], r["at"], r["tenant"],
                          r["reason"]),
                         (False, "left", 2, "t", "sequence"))
        # illegal UTF-8: tenant cannot be determined
        r = self.other.compare_bytes(good1, good1 + b"\xff")
        self.assertEqual((r["ok"], r["side"], r["at"], r["tenant"],
                          r["reason"]),
                         (False, "right", 2, None, "missing"))

    # --- equality across interleaved physical orders ---

    def test_interleaved_orders_are_equal(self):
        left_chain = AuditChain(self.path.with_name("l.jsonl"))
        self.seed(left_chain, ("a", {"i": 1}), ("b", {"i": 1}),
                  ("a", {"i": 2}), ("a", {"i": 3}), ("b", {"i": 2}))
        right_chain = AuditChain(self.path.with_name("r.jsonl"))
        self.seed(right_chain, ("b", {"i": 1}), ("b", {"i": 2}),
                  ("a", {"i": 1}), ("a", {"i": 2}), ("a", {"i": 3}))
        left, right = self.snapshot(left_chain), self.snapshot(right_chain)
        self.assertNotEqual(left, right)  # bytes differ, logical content equal
        r = self.other.compare_bytes(left, right)
        self.assertTrue(r["equal"])
        # order is the left side's first-appearance order
        self.assertEqual([e["tenant"] for e in r["tenants"]], ["a", "b"])
        self.assertEqual(
            {(e["tenant"], e["count"]) for e in r["tenants"]},
            {("a", 3), ("b", 2)},
        )
        # the tail hashes agree with each side's heads directory
        self.assertEqual(
            r["tenants"], self.other.heads_bytes(left)["tenants"])

    def test_summary_matches_heads_directory_and_includes_right_only(self):
        left_chain = AuditChain(self.path.with_name("l.jsonl"))
        right_chain = AuditChain(self.path.with_name("r.jsonl"))
        # Same logical content; on the right the tenants the left names later
        # appear interleaved before a, exercising the right-only append order.
        self.seed(left_chain, ("a", {"i": 1}), ("z", {"i": 1}),
                  ("m", {"i": 1}))
        self.seed(right_chain, ("z", {"i": 1}), ("a", {"i": 1}),
                  ("m", {"i": 1}))
        r = self.other.compare_bytes(self.snapshot(left_chain),
                                     self.snapshot(right_chain))
        self.assertTrue(r["equal"])
        # left first-appearance order (a, then z, m); no right-only tenant here
        self.assertEqual([e["tenant"] for e in r["tenants"]],
                         ["a", "z", "m"])
        self.assertEqual(r["tenants"],
                         self.other.heads_bytes(
                             self.snapshot(left_chain))["tenants"])

    def test_right_only_tenant_is_visited_after_left_tenants(self):
        # A tenant only the right names is an empty chain on the left, so the
        # snapshots cannot be equal: the difference surfaces after every left
        # tenant has been compared, in the right side's first-appearance order.
        left_chain = AuditChain(self.path.with_name("l.jsonl"))
        right_chain = AuditChain(self.path.with_name("r.jsonl"))
        self.seed(left_chain, ("a", {"i": 1}))
        # right's unique tenants appear before a physically, yet a (a left
        # tenant) is compared first and matches; only then comes the first
        # right-only tenant in the right's first-appearance order (z, not m).
        self.seed(right_chain, ("z", {"i": 1}), ("a", {"i": 1}),
                  ("m", {"i": 1}))
        r = self.other.compare_bytes(self.snapshot(left_chain),
                                     self.snapshot(right_chain))
        self.assertEqual((r["ok"], r["equal"], r["tenant"], r["seq"],
                          r["reason"], r["left"]),
                         (True, False, "z", 1, "missing_left", None))
        self.assertEqual(r["right"]["tenant"], "z")

    def test_canonical_tenant_identity_aligns_records(self):
        # Same canonical object tenant spelled with different key orders.
        _i, left = self.row({"a": 1, "b": 2}, 1)
        _i, right = self.row({"b": 2, "a": 1}, 1)
        r = self.other.compare_bytes(left, right)
        self.assertTrue(r["equal"])
        # the first-appearance original value of each side is preserved, and
        # the left side supplies the reported tenant value
        self.assertEqual(r["tenants"][0]["tenant"], {"a": 1, "b": 2})

    def test_distinct_json_identities_are_different_chains(self):
        # 1 and 1.0 are unrelated tenants: left has tenant 1, right has 1.0,
        # so each side is missing the other's chain.
        _i, left = self.row(1, 1)
        _i, right = self.row(1.0, 1)
        r = self.other.compare_bytes(left, right)
        self.assertFalse(r["equal"])
        # left's tenant 1 comes first in the ordering: right misses it
        self.assertEqual((r["tenant"], r["seq"], r["reason"]),
                         (1, 1, "missing_right"))
        self.assertEqual(r["left"]["tenant"], 1)
        self.assertIsNone(r["right"])

    # --- first-difference reporting ---

    def test_different_event_is_reported_with_both_records(self):
        _i, left = self.row("t", 1, event={"v": 1})
        _i, right = self.row("t", 1, event={"v": 2})
        r = self.other.compare_bytes(left, right)
        self.assertEqual((r["ok"], r["equal"], r["tenant"], r["seq"],
                          r["reason"]),
                         (True, False, "t", 1, "different"))
        self.assertEqual(r["left"]["event"], {"v": 1})
        self.assertEqual(r["right"]["event"], {"v": 2})

    def test_prev_or_hash_tampering_is_a_corrupt_snapshot_not_a_difference(self):
        # Between two independently valid chains an equal prefix forces equal
        # prev, and the hash is deterministic, so a prev/hash-only mismatch at
        # the first differing position cannot exist: such a buffer fails the
        # verify_all_bytes pass and is reported as corruption, never compared.
        _i, good = self.row("t", 1)
        bad_prev = {"tenant": "t", "seq": 1, "event": {}, "prev": "9" * 64}
        bad_prev["hash"] = AuditChain._hash(bad_prev)
        r = self.other.compare_bytes(good, record_bytes(bad_prev))
        self.assertEqual((r["ok"], r["side"], r["at"], r["tenant"],
                          r["reason"]),
                         (False, "right", 1, "t", "digest"))
        bad_hash = {"tenant": "t", "seq": 1, "event": {}, "prev": ZERO,
                    "hash": "a" * 64}
        r = self.other.compare_bytes(record_bytes(bad_hash), good)
        self.assertEqual((r["ok"], r["side"], r["at"], r["tenant"],
                          r["reason"]),
                         (False, "left", 1, "t", "digest"))

    def test_different_at_later_seq_and_only_one_position(self):
        left_items = []
        right_items = []
        prev_l = prev_r = ZERO
        for seq in range(1, 4):
            il, bl = self.row("t", seq, prev=prev_l, event={"i": seq})
            event = {"i": seq} if seq != 3 else {"i": 30}
            ir, br = self.row("t", seq, prev=prev_r, event=event)
            left_items.append(bl)
            right_items.append(br)
            prev_l = il["hash"]
            prev_r = ir["hash"]
        # The fourth record chains off each (different) third hash; even so the
        # first difference -- seq 3's event -- is the only one reported, with no
        # cascade from the now-divergent prev/hash of later records.
        i4l, b4l = self.row("t", 4, prev=prev_l, event={"i": 4})
        i4r, b4r = self.row("t", 4, prev=prev_r, event={"i": 4})
        r = self.other.compare_bytes(b"".join(left_items) + b4l,
                                     b"".join(right_items) + b4r)
        self.assertEqual((r["tenant"], r["seq"], r["reason"]),
                         ("t", 3, "different"))

    def test_missing_sides_attach_existing_record(self):
        _i1, one = self.row("t", 1)
        i2, two = self.row("t", 2, prev=_i1["hash"], event={"i": 2})
        r = self.other.compare_bytes(one + two, one)
        self.assertEqual((r["tenant"], r["seq"], r["reason"]),
                         ("t", 2, "missing_right"))
        self.assertIsNone(r["right"])
        self.assertEqual(r["left"], i2)
        r = self.other.compare_bytes(one, one + two)
        self.assertEqual((r["tenant"], r["seq"], r["reason"]),
                         ("t", 2, "missing_left"))
        self.assertIsNone(r["left"])
        self.assertEqual(r["right"], i2)

    def test_tenant_order_decides_first_difference(self):
        # Both sides differ in two chains; the left's first-appearance order
        # picks which difference is reported.
        lc = AuditChain(self.path.with_name("l.jsonl"))
        rc = AuditChain(self.path.with_name("r.jsonl"))
        self.seed(lc, ("a", {"v": 1}), ("b", {"v": 1}))
        self.seed(rc, ("b", {"v": 9}), ("a", {"v": 9}))
        r = self.other.compare_bytes(self.snapshot(lc), self.snapshot(rc))
        self.assertEqual((r["tenant"], r["seq"], r["reason"]),
                         ("a", 1, "different"))
        # and when a appears only second on the left, b (left-first) still wins
        lc2 = AuditChain(self.path.with_name("l2.jsonl"))
        self.seed(lc2, ("b", {"v": 1}), ("a", {"v": 1}))
        r = self.other.compare_bytes(self.snapshot(lc2), self.snapshot(rc))
        self.assertEqual((r["tenant"], r["seq"], r["reason"]),
                         ("b", 1, "different"))

    def test_difference_tenant_value_uses_first_appearance(self):
        # Left first names the canonical tenant with one spelling; a differing
        # chain on that identity must echo that spelling, not a later one.
        _ia, left1 = self.row({"k": 1}, 1, event={"v": 1})
        _ib, left2 = self.row({"k": 1}, 2, prev=_ia["hash"], event={"v": 2})
        _i1, right1 = self.row({"k": 1}, 1, event={"v": 1})
        _i2, right2 = self.row({"k": 1}, 2, prev=_i1["hash"], event={"v": 9})
        r = self.other.compare_bytes(left1 + left2, right1 + right2)
        self.assertEqual((r["seq"], r["reason"]), (2, "different"))
        self.assertEqual(r["tenant"], {"k": 1})

    def test_identical_chains_compare_equal_record_for_record(self):
        chain = AuditChain(self.path.with_name("c.jsonl"))
        for i in range(5):
            chain.append("t", {"i": i})
            chain.append("u", {"j": i})
        data = self.snapshot(chain)
        r = self.other.compare_bytes(data, bytes(data))
        self.assertEqual(r["ok"], True)
        self.assertEqual(r["equal"], True)
        self.assertEqual(
            [(e["tenant"], e["count"]) for e in r["tenants"]],
            [("t", 5), ("u", 5)])

    def test_result_records_keep_original_values(self):
        _i, left = self.row("t", 1, event={"nested": [1, 2, {"x": True}]})
        _i, right = self.row("t", 1, event={"nested": [1, 2, {"x": False}]})
        r = self.other.compare_bytes(left, right)
        self.assertEqual(r["reason"], "different")
        self.assertEqual(r["left"]["event"], {"nested": [1, 2, {"x": True}]})
        self.assertEqual(r["right"]["event"], {"nested": [1, 2, {"x": False}]})


if __name__ == "__main__":
    unittest.main()
