import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app import AuditChain, ZERO


def record_bytes(item):
    return (json.dumps(item, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


class MergeChunksTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.path = Path(self.tmp.name) / "audit.jsonl"
        self.chain = AuditChain(self.path)
        # A merge entry pointed at a path it must never read, create or
        # modify.
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

    # --- equivalence with merge_bytes over many chunkings ---

    def test_matches_merge_bytes_for_many_chunkings(self):
        left_chain = AuditChain(self.path.with_name("l.jsonl"))
        right_chain = AuditChain(self.path.with_name("r.jsonl"))
        self.seed(left_chain, ("a", {"i": 1}), ("b", {"i": 1}),
                  ("a", {"i": 2}), ("a", {"i": 3}), ("b", {"i": 2}))
        self.seed(right_chain, ("b", {"i": 1}), ("b", {"i": 2}),
                  ("a", {"i": 1}), ("a", {"i": 2}), ("a", {"i": 3}),
                  ("c", {"i": 1}), ("c", {"i": 2}))
        left, right = self.snapshot(left_chain), self.snapshot(right_chain)
        expected = self.other.merge_bytes(left, right)
        for lc in self.chunkings(left):
            for rc in self.chunkings(right):
                self.assertEqual(
                    self.other.merge_chunks(lc, rc), expected, (lc, rc))

    def test_matches_bytes_entry_for_conflicts_and_corruption(self):
        _i, good = self.row("t", 1)
        _l, le = self.row("t", 1, event={"v": 1})
        _r, re = self.row("t", 1, event={"v": 2})
        cases = [
            (le, re),                                   # conflict
            (b"{oops\n", good),                         # left corrupt
            (good, b"\xff"),                           # right corrupt
            (b"", b""),                                # both empty
        ]
        for left, right in cases:
            for lc in self.chunkings(left):
                for rc in self.chunkings(right):
                    self.assertEqual(
                        self.other.merge_chunks(lc, rc),
                        self.other.merge_bytes(left, right),
                        (left, right, lc, rc))

    def test_accepts_any_iterable_container(self):
        left_chain = AuditChain(self.path.with_name("l.jsonl"))
        right_chain = AuditChain(self.path.with_name("r.jsonl"))
        self.seed(left_chain, ("a", {"i": 1}))
        self.seed(right_chain, ("a", {"i": 1}), ("b", {"i": 1}))
        left, right = self.snapshot(left_chain), self.snapshot(right_chain)
        lh, rh = [left[:len(left) // 2], left[len(left) // 2:]], \
            [right[:3], right[3:]]
        expected = self.other.merge_bytes(left, right)
        self.assertEqual(
            self.other.merge_chunks(iter(lh), (c for c in rh)), expected)
        self.assertEqual(
            self.other.merge_chunks((c for c in lh), iter(rh)), expected)
        empty = self.other.merge_chunks(iter([]), (c for c in ()))
        self.assertTrue(empty["ok"])
        self.assertEqual(empty["data"], b"")

    def test_empty_snapshots(self):
        for lc in self.chunkings(b""):
            for rc in self.chunkings(b""):
                r = self.other.merge_chunks(lc, rc)
                self.assertTrue(r["ok"])
                self.assertEqual(r["data"], b"")

    def test_chunk_may_split_multibyte_utf8_and_records(self):
        lc_chain = AuditChain(self.path.with_name("lc.jsonl"))
        rc_chain = AuditChain(self.path.with_name("rc.jsonl"))
        lc_chain.append("t", {"msg": "héllo→世界"})
        rc_chain.append("t", {"msg": "héllo→世界"})
        rc_chain.append("t", {"msg": "next"})
        left, right = self.snapshot(lc_chain), self.snapshot(rc_chain)
        expected = self.other.merge_bytes(left, right)
        for cut in range(len(left) + 1):
            r = self.other.merge_chunks([left[:cut], left[cut:]], [right])
            self.assertEqual(r, expected, cut)

    # --- container boundary: ValueError before content, left first ---

    def test_bare_bytes_or_bytearray_is_not_a_container(self):
        with self.assertRaises(ValueError):
            self.other.merge_chunks(b"abc", [b""])
        with self.assertRaises(ValueError):
            self.other.merge_chunks(bytearray(b"abc"), [b""])
        with self.assertRaises(ValueError):
            self.other.merge_chunks([b""], b"abc")
        with self.assertRaises(ValueError):
            self.other.merge_chunks([b""], bytearray(b"abc"))

    def test_non_iterable_container_is_value_error(self):
        for bad in (None, 1, 1.0, True):
            with self.assertRaises(ValueError):
                self.other.merge_chunks(bad, [b""])
            with self.assertRaises(ValueError):
                self.other.merge_chunks([b""], bad)

    def test_non_bytes_element_is_value_error(self):
        for bad in ("x", bytearray(b"x"), 1, None, True, [b""]):
            with self.assertRaises(ValueError):
                self.other.merge_chunks([bad], [b""])
            with self.assertRaises(ValueError):
                self.other.merge_chunks([b""], [bad])

    def test_left_boundary_checked_and_consumed_first(self):
        # Left container error wins even though the right side is also bad
        # or its concatenation corrupt.
        with self.assertRaises(ValueError):
            self.other.merge_chunks(b"left", b"\xff")
        with self.assertRaises(ValueError):
            self.other.merge_chunks("left", ["x"])
        # A bad right element raises even though the left concatenation is a
        # corrupt snapshot: every boundary error wins over content.
        with self.assertRaises(ValueError):
            self.other.merge_chunks([b"{oops\n"], ["x"])

    def test_consumption_stops_at_first_bad_element(self):
        def gen():
            yield b"{}"
            yield "not-bytes"
            raise AssertionError("iteration must stop at the bad element")
            yield b"{}"  # pragma: no cover

        with self.assertRaises(ValueError):
            self.other.merge_chunks(gen(), [b""])
        with self.assertRaises(ValueError):
            self.other.merge_chunks([b""], gen())

    def test_iterator_consumed_once(self):
        left_chain = AuditChain(self.path.with_name("l.jsonl"))
        right_chain = AuditChain(self.path.with_name("r.jsonl"))
        self.seed(left_chain, ("a", {"i": 1}))
        self.seed(right_chain, ("a", {"i": 1}), ("a", {"i": 2}))
        left, right = self.snapshot(left_chain), self.snapshot(right_chain)
        expected = self.other.merge_bytes(left, right)
        left_once = iter([left[:5], left[5:]])
        right_once = iter([right])
        r = self.other.merge_chunks(left_once, right_once)
        self.assertEqual(r, expected)
        # both one-shot iterators are exhausted
        self.assertEqual(list(left_once), [])
        self.assertEqual(list(right_once), [])

    def test_iterator_exception_propagates(self):
        def boom():
            yield b"{}"
            raise RuntimeError("from the iterable")

        with self.assertRaisesRegex(RuntimeError, "from the iterable"):
            self.other.merge_chunks(boom(), [b""])
        with self.assertRaisesRegex(RuntimeError, "from the iterable"):
            self.other.merge_chunks([b""], boom())

    def test_never_touches_constructor_path(self):
        phantom = self.path.with_name("phantom.jsonl")
        chain = AuditChain(phantom)
        _i, one = self.row("a", 1)
        r = chain.merge_chunks([one], [b""])
        self.assertTrue(r["ok"])
        self.assertFalse(phantom.exists())


if __name__ == "__main__":
    unittest.main()
