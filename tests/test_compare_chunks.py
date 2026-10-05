import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app import AuditChain, ZERO


def record_bytes(item):
    return (json.dumps(item, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


class CompareChunksTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.path = Path(self.tmp.name) / "audit.jsonl"
        self.chain = AuditChain(self.path)
        # A compare entry point pointed at a path it must never read, create
        # or modify.
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
        item = {"tenant": tenant, "seq": seq, "event": event or {},
                "prev": prev}
        item["hash"] = hash_ if hash_ is not None else AuditChain._hash(item)
        return item, record_bytes(item)

    def chunkings(self, data):
        if not data:
            return [[], (), [b""], [b"", b""], (b"",)]
        return [
            [data],
            [data[:1], data[1:]],
            [data[i:i + 1] for i in range(len(data))],
            [b"", data[:10], b"", data[10:37], data[37:], b""],
            list(data[i:i + 7] for i in range(0, len(data), 7)),
            tuple(data[i:i + 3] for i in range(0, len(data), 3)),
        ]

    # --- equivalence with compare_bytes over many chunkings ---

    def test_matches_compare_bytes_for_many_chunkings(self):
        left_chain = AuditChain(self.path.with_name("l.jsonl"))
        right_chain = AuditChain(self.path.with_name("r.jsonl"))
        self.seed(left_chain, ("a", {"i": 1}), ("b", {"i": 1}),
                  ("a", {"i": 2}), ("a", {"i": 3}), ("b", {"i": 2}))
        self.seed(right_chain, ("b", {"i": 1}), ("b", {"i": 2}),
                  ("a", {"i": 1}), ("a", {"i": 2}), ("a", {"i": 3}))
        left, right = self.snapshot(left_chain), self.snapshot(right_chain)
        for lc in self.chunkings(left):
            for rc in self.chunkings(right):
                self.assertEqual(
                    self.other.compare_chunks(lc, rc),
                    self.other.compare_bytes(left, right), (lc, rc))

    def test_accepts_any_iterable_container(self):
        left_chain = AuditChain(self.path.with_name("l.jsonl"))
        right_chain = AuditChain(self.path.with_name("r.jsonl"))
        self.seed(left_chain, ("a", {"i": 1}))
        self.seed(right_chain, ("a", {"i": 1}), ("b", {"i": 1}))
        left, right = self.snapshot(left_chain), self.snapshot(right_chain)
        lh, rh = [left[:len(left) // 2], left[len(left) // 2:]], \
            [right[:3], right[3:]]
        expected = self.other.compare_bytes(left, right)
        for lc in (iter(lh), (c for c in lh)):
            self.assertEqual(self.other.compare_chunks(lc, rh), expected)
        for rc in (iter(rh), (c for c in rh)):
            self.assertEqual(self.other.compare_chunks(lh, rc), expected)
        # one-shot empty iterables are the empty snapshot
        self.assertEqual(
            self.other.compare_chunks(iter([]), (c for c in ())),
            {"ok": True, "equal": True, "tenants": []})

    def test_empty_snapshots(self):
        for lc in self.chunkings(b""):
            for rc in self.chunkings(b""):
                self.assertEqual(
                    self.other.compare_chunks(lc, rc),
                    {"ok": True, "equal": True, "tenants": []})

    def test_empty_against_nonempty(self):
        _i, one = self.row("a", 1)
        for empty in self.chunkings(b""):
            r = self.other.compare_chunks(empty, [one])
            self.assertEqual((r["ok"], r["equal"], r["tenant"], r["seq"],
                              r["reason"], r["left"]),
                             (True, False, "a", 1, "missing_left", None))
            self.assertEqual(r["right"]["seq"], 1)
            r = self.other.compare_chunks([one], empty)
            self.assertEqual((r["tenant"], r["seq"], r["reason"],
                              r["right"]),
                             ("a", 1, "missing_right", None))
            self.assertEqual(r["left"]["seq"], 1)

    def test_chunk_may_split_multibyte_utf8_and_records(self):
        lc_chain = AuditChain(self.path.with_name("lc.jsonl"))
        rc_chain = AuditChain(self.path.with_name("rc.jsonl"))
        lc_chain.append("t", {"msg": "héllo→世界"})
        rc_chain.append("t", {"msg": "héllo→世界", "n": 1})
        left, right = self.snapshot(lc_chain), self.snapshot(rc_chain)
        for cut in range(len(left) + 1):
            r = self.other.compare_chunks([left[:cut], left[cut:]],
                                          [right])
            self.assertEqual(r, self.other.compare_bytes(left, right), cut)

    def test_first_difference_matches_bytes_entry(self):
        _i1, one = self.row("t", 1)
        i2, two = self.row("t", 2, prev=_i1["hash"], event={"i": 2})
        il, dl = self.row("t", 1, event={"v": 1})
        _ir, dr = self.row("t", 1, event={"v": 2})
        cases = [
            (one + two, one, ("t", 2, "missing_right")),
            (one, one + two, ("t", 2, "missing_left")),
            (dl, dr, ("t", 1, "different")),
        ]
        for left, right, expect in cases:
            r = self.other.compare_chunks(
                [left[:9], b"", left[9:]], [right[:5], right[5:]])
            self.assertEqual((r["ok"], r["equal"], r["tenant"], r["seq"],
                              r["reason"]), (True, False, *expect))
            self.assertEqual(r, self.other.compare_bytes(left, right))

    def test_summary_order_and_right_only_tenant(self):
        left_chain = AuditChain(self.path.with_name("l.jsonl"))
        right_chain = AuditChain(self.path.with_name("r.jsonl"))
        self.seed(left_chain, ("a", {"i": 1}))
        self.seed(right_chain, ("z", {"i": 1}), ("a", {"i": 1}),
                  ("m", {"i": 1}))
        left, right = self.snapshot(left_chain), self.snapshot(right_chain)
        r = self.other.compare_chunks(iter([left[:4], left[4:]]),
                                      (right[:1], right[1:]))
        self.assertEqual((r["tenant"], r["seq"], r["reason"]),
                         ("z", 1, "missing_left"))
        self.assertIsNone(r["left"])
        self.assertEqual(r["right"]["tenant"], "z")

    # --- corruption reported through compare_bytes, left first ---

    def test_corrupt_side_is_reported_not_compared(self):
        _t, good = self.row("t", 1)
        r = self.other.compare_chunks([b"{oops\n"], [good])
        self.assertEqual(r, {"ok": False, "side": "left", "at": 1,
                             "tenant": None, "reason": "missing"})
        r = self.other.compare_chunks([good], [b"\xff"])
        self.assertEqual(r["side"], "right")
        self.assertEqual((r["at"], r["tenant"], r["reason"]),
                         (1, None, "missing"))
        # both corrupt: left always wins
        r = self.other.compare_chunks([b"{oops\n"], [b"\xff"])
        self.assertEqual(r["side"], "left")

    def test_sequence_and_digest_classification(self):
        _t, good1 = self.row("t", 1)
        gap = {"tenant": "t", "seq": 3, "event": {}, "prev": "9" * 64}
        gap["hash"] = AuditChain._hash(gap)
        raw = good1 + record_bytes(gap)
        r = self.other.compare_chunks([raw[:7], raw[7:]], [good1])
        self.assertEqual((r["ok"], r["side"], r["at"], r["tenant"],
                          r["reason"]),
                         (False, "left", 2, "t", "sequence"))
        bad_prev = {"tenant": "t", "seq": 2, "event": {}, "prev": "9" * 64}
        bad_prev["hash"] = AuditChain._hash(bad_prev)
        raw = good1 + record_bytes(bad_prev)
        r = self.other.compare_chunks([good1], [raw[:3], raw[3:]])
        self.assertEqual((r["side"], r["at"], r["reason"]),
                         ("right", 2, "digest"))

    # --- container / element boundary ---

    def test_bare_bytes_container_is_value_error(self):
        for bad in (b"", b"{}", bytearray(b""), bytearray(b"abc")):
            with self.assertRaises(ValueError):
                self.other.compare_chunks(bad, [])
            with self.assertRaises(ValueError):
                self.other.compare_chunks([], bad)

    def test_non_iterable_container_is_value_error(self):
        for bad in (1, 1.0, None, True, object()):
            with self.assertRaises(ValueError):
                self.other.compare_chunks(bad, [])
            with self.assertRaises(ValueError):
                self.other.compare_chunks([], bad)

    def test_non_bytes_element_is_value_error(self):
        for bad in (["x"], [1], [None], [bytearray(b"")], [b"", "x"],
                    [[b""]], [{}]):
            with self.assertRaises(ValueError):
                self.other.compare_chunks(bad, [])
            with self.assertRaises(ValueError):
                self.other.compare_chunks([], bad)

    def test_left_boundary_checked_before_right_boundary(self):
        # Both sides non-iterable: left raises first.
        with self.assertRaises(ValueError):
            self.other.compare_chunks(1, 2)
        # Left bare bytes even though right is a corrupt-snapshot container.
        with self.assertRaises(ValueError):
            self.other.compare_chunks(b"x", [b"\xff"])
        # A bad left element wins over a bad right container: the right
        # container must never even be probed.
        class ProbeWouldFail:
            def __iter__(self):
                raise AssertionError("right container must not be touched")

        with self.assertRaises(ValueError):
            self.other.compare_chunks([b"", object()], ProbeWouldFail())

    def test_boundary_error_wins_over_corrupt_content(self):
        # The left concatenation is corrupt, but the right has a bad element:
        # boundary errors beat content validation on either side.
        with self.assertRaises(ValueError):
            self.other.compare_chunks([b"{oops\n"], [b"", "not bytes"])
        # Left container bad: raises even though the right is corrupt too.
        with self.assertRaises(ValueError):
            self.other.compare_chunks([b"", 7], [b"\xff"])

    def test_bad_element_stops_consumption_of_that_side(self):
        pulled = []

        def gen():
            pulled.append(1)
            yield b""
            pulled.append(2)
            yield "not bytes"
            pulled.append(3)  # must never be reached
            yield b""

        with self.assertRaises(ValueError):
            self.other.compare_chunks(gen(), [])
        self.assertEqual(pulled, [1, 2])

        pulled.clear()
        with self.assertRaises(ValueError):
            self.other.compare_chunks([], gen())
        self.assertEqual(pulled, [1, 2])

    def test_left_fully_consumed_before_right_probed(self):
        pulled_left = []
        pulled_right = []

        def left_gen():
            for i, c in enumerate((b'{"tenant":', b'"t",', b"\xff"), 1):
                pulled_left.append(i)
                yield c

        def right_gen():
            pulled_right.append(1)
            yield b""

        # Left content is corrupt, but its boundary is valid: every left
        # chunk is pulled before the right iterator is created/consumed.
        r = self.other.compare_chunks(left_gen(), right_gen())
        self.assertEqual((r["ok"], r["side"], r["reason"]),
                         (False, "left", "missing"))
        self.assertEqual(pulled_left, [1, 2, 3])
        # Right is still consumed after the left boundary crossed (its join
        # happens before content validation), but never during the left join.
        self.assertEqual(pulled_right, [1])

    # --- offline purity ---

    def test_never_reads_creates_or_modifies_constructor_path(self):
        phantom = self.path.with_name("phantom.jsonl")
        chain = AuditChain(phantom)
        self.assertTrue(chain.compare_chunks([], [])["equal"])
        self.assertFalse(phantom.exists())
        self.seed(self.chain, ("a", {"i": 1}), ("b", {"i": 1}),
                  ("a", {"i": 2}))
        data = self.snapshot()
        self.assertTrue(chain.compare_chunks([data[:9], data[9:]],
                                             [bytes(data)])["equal"])
        self.assertFalse(phantom.exists())

    def test_does_not_mutate_chunks(self):
        self.seed(self.chain, ("a", {"i": 1}))
        data = self.snapshot()
        left = [bytearray(data[:5]), bytearray(data[5:])]
        # bytearray elements are rejected ...
        with self.assertRaises(ValueError):
            self.other.compare_chunks(left, [data])
        snapshots = [bytes(c) for c in left]
        # ... and nothing about the inputs is changed by the rejection.
        self.assertEqual([bytes(c) for c in left], snapshots)
        right = [data[:len(data) // 2], data[len(data) // 2:]]
        before = [bytes(c) for c in right]
        self.other.compare_chunks([data], right)
        self.assertEqual([bytes(c) for c in right], before)


if __name__ == "__main__":
    unittest.main()
