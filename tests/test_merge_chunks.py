import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app import AuditChain, ZERO


def record_bytes(item):
    return (json.dumps(item, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def chunked(data, size):
    return [data[i:i + size] for i in range(0, len(data), size)]


class MergeChunksTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.path = Path(self.tmp.name) / "audit.jsonl"
        self.chain = AuditChain(self.path)
        # The pure in-memory entry must never touch this (or any) path.
        self.other = AuditChain(self.path.with_name("never-touched.jsonl"))

    def tearDown(self):
        self.tmp.cleanup()

    def seed(self, chain, *tenant_events):
        for tenant, event in tenant_events:
            chain.append(tenant, event)

    def snapshot(self, chain=None):
        chain = chain or self.chain
        return Path(chain.path).read_bytes()

    # --- container boundary: identical to compare_chunks, left first ---

    def test_bare_bytes_is_not_a_chunk_container(self):
        for bad in (b"", b"{}", bytearray(b"")):
            with self.assertRaises(ValueError):
                self.other.merge_chunks(bad, [])
            with self.assertRaises(ValueError):
                self.other.merge_chunks([], bad)

    def test_non_iterable_container_is_value_error(self):
        for bad in (1, 1.0, None, True):
            with self.assertRaises(ValueError):
                self.other.merge_chunks(bad, [])
            with self.assertRaises(ValueError):
                self.other.merge_chunks([], bad)

    def test_non_bytes_element_is_value_error(self):
        with self.assertRaises(ValueError):
            self.other.merge_chunks([b"{}", "x"], [])
        with self.assertRaises(ValueError):
            self.other.merge_chunks([], [bytearray(b"")])

    def test_left_boundary_error_wins_and_stops_consumption(self):
        consumed = []

        def right_side():
            consumed.append(True)
            yield b""

        with self.assertRaises(ValueError):
            self.other.merge_chunks([1], right_side())
        # the right side is never even probed
        self.assertEqual(consumed, [])

    def test_first_bad_element_ends_consumption_of_that_side(self):
        pulled = []

        def left_side():
            pulled.append(1)
            yield b""
            pulled.append(2)
            yield "bad"
            pulled.append(3)
            yield b""

        with self.assertRaises(ValueError):
            self.other.merge_chunks(left_side(), [])
        self.assertEqual(pulled, [1, 2])

    def test_each_side_is_consumed_exactly_once(self):
        self.seed(self.chain, ("a", {"i": 1}))
        data = self.snapshot()
        counts = {"left": 0, "right": 0}

        def side(name):
            counts[name] += 1
            yield data

        r = self.other.merge_chunks(side("left"), side("right"))
        self.assertTrue(r["ok"])
        self.assertEqual(counts, {"left": 1, "right": 1})

    # --- result identity with merge_bytes ---

    def test_result_matches_merge_bytes_on_concatenation(self):
        lc = AuditChain(self.path.with_name("l.jsonl"))
        rc = AuditChain(self.path.with_name("r.jsonl"))
        self.seed(lc, ("a", {"i": 1}), ("b", {"j": 1}))
        self.seed(rc, ("b", {"j": 1}), ("a", {"i": 1}), ("a", {"i": 2}),
                  ("c", {"k": 1}))
        left, right = self.snapshot(lc), self.snapshot(rc)
        # cuts may split UTF-8 sequences, JSON objects and lines
        r = self.other.merge_chunks(chunked(left, 7), chunked(right, 5))
        expected = self.other.merge_bytes(left, right)
        self.assertEqual(r, expected)
        self.assertTrue(r["ok"])

    def test_empty_iterables_and_empty_chunks_are_empty_snapshots(self):
        r = self.other.merge_chunks([], [])
        self.assertEqual(r, self.other.merge_bytes(b"", b""))
        r = self.other.merge_chunks([b"", b""], iter([b""]))
        self.assertEqual(r, self.other.merge_bytes(b"", b""))

    def test_corruption_and_conflict_results_match_merge_bytes(self):
        item = {"tenant": "t", "seq": 1, "event": {}, "prev": ZERO}
        item["hash"] = AuditChain._hash(item)
        good = record_bytes(item)
        r = self.other.merge_chunks([b"{oops\n"], [good])
        self.assertEqual(r, self.other.merge_bytes(b"{oops\n", good))
        left = self.other.merge_bytes(good, good)
        divergent = dict(item, event={"v": 1})
        divergent["hash"] = AuditChain._hash(divergent)
        r = self.other.merge_chunks([good], [record_bytes(divergent)])
        self.assertEqual(r["ok"], False)
        self.assertEqual(r["reason"], "conflict")
        self.assertNotIn("data", r)

    def test_never_touches_the_path(self):
        phantom = self.path.with_name("phantom.jsonl")
        chain = AuditChain(phantom)
        self.seed(self.chain, ("a", {"i": 1}))
        data = self.snapshot()
        r = chain.merge_chunks([data], [])
        self.assertTrue(r["ok"])
        self.assertFalse(phantom.exists())


if __name__ == "__main__":
    unittest.main()
