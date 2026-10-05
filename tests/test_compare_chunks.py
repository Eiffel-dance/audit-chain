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
        # Every compare runs through an AuditChain pointed at a path that must
        # never be read, created or modified by the pure offline entry.
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

    # --- equivalence with compare_bytes over arbitrary chunkings ---

    def test_matches_compare_bytes_for_many_chunkings(self):
        left_chain = AuditChain(self.path.with_name("l.jsonl"))
        self.seed(left_chain, ("a", {"i": 1}), ("b", {"i": 1}),
                  ("a", {"i": 2}), ("a", {"i": 3}), ("b", {"i": 2}))
        right_chain = AuditChain(self.path.with_name("r.jsonl"))
        self.seed(right_chain, ("b", {"i": 1}), ("b", {"i": 2}),
                  ("a", {"i": 1}), ("a", {"i": 2}), ("a", {"i": 9}))
        left, right = self.snapshot(left_chain), self.snapshot(right_chain)
        chunkings = [
            ([left], [right]),
            ([left[:1], left[1:]], [right[:1], right[1:]]),
            ([left[i:i + 1] for i in range(len(left))],
             [right[i:i + 1] for i in range(len(right))]),
            ([left[:10], b"", left[10:37], b"", left[37:]],
             [b"", b"", right, b""]),
            (list(left[i:i + 7] for i in range(0, len(left), 7)),
             list(right[i:i + 11] for i in range(0, len(right), 11))),
            ((c for c in [left[:5], left[5:]]), iter([right])),
        ]
        expected = self.other.compare_bytes(left, right)
        self.assertEqual(
            (expected["ok"], expected["equal"], expected["tenant"],
             expected["seq"], expected["reason"]),
            (True, False, "a", 3, "different"))
        for left_chunks, right_chunks in chunkings:
            self.assertEqual(
                self.other.compare_chunks(left_chunks, right_chunks),
                expected)

    def test_matches_compare_bytes_at_every_split_point(self):
        self.chain.append("t", {"msg": "héllo→世界"})
        self.chain.append("u", {"msg": "✓" * 40})
        self.chain.append("t", {"i": 2})
        data = self.snapshot()
        expected = self.other.compare_bytes(data, data)
        self.assertTrue(expected["equal"])
        # cut each side at every single byte offset, incl. mid-UTF-8-sequence
        for cut in range(len(data) + 1):
            r = self.other.compare_chunks([data[:cut], data[cut:]],
                                          [data[:cut], data[cut:]])
            self.assertEqual(r, expected, cut)

    def test_empty_containers_are_empty_snapshots(self):
        for chunks in ([], (), iter([]), [b""], [b"", b""]):
            self.assertEqual(
                self.other.compare_chunks(chunks, chunks),
                {"ok": True, "equal": True, "tenants": []})
        self.seed(self.chain, ("a", {"i": 1}))
        data = self.snapshot()
        r = self.other.compare_chunks([], [data])
        self.assertEqual((r["ok"], r["equal"], r["tenant"], r["seq"],
                          r["reason"], r["left"]),
                         (True, False, "a", 1, "missing_left", None))
        r = self.other.compare_chunks([data[:9], data[9:]], [b""])
        self.assertEqual((r["ok"], r["equal"], r["reason"], r["right"]),
                         (True, False, "missing_right", None))

    # --- container and element boundary: left first, stop at first bad ---

    def test_bare_bytes_container_is_value_error(self):
        for bad in (b"", b"{}", bytearray(b""), bytearray(b"{}")):
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
        for bad in ([b"", "x"], [b"", 1], [b"", None], [b"", bytearray(b"")],
                    [b"", [b""]], [b"", b"", 1.0]):
            with self.assertRaises(ValueError):
                self.other.compare_chunks(bad, [])
            with self.assertRaises(ValueError):
                self.other.compare_chunks([], bad)

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

    def test_left_boundary_error_precedes_right_and_content(self):
        # The left container is fully checked before the right one is touched
        # at all: a left boundary error raises even when the right container
        # is illegal too, and the right iterable is never pulled.
        pulled = []

        def right_gen():
            pulled.append(1)
            yield b""

        with self.assertRaises(ValueError):
            self.other.compare_chunks(b"left", right_gen())
        self.assertEqual(pulled, [])
        with self.assertRaises(ValueError):
            self.other.compare_chunks([b"", bytearray(b"")], "not iterable")
        self.assertEqual(pulled, [])
        # A boundary error on either side beats a corrupt snapshot content
        # verdict: the corrupt side is never parsed.
        with self.assertRaises(ValueError):
            self.other.compare_chunks("left", [b"\xff"])
        with self.assertRaises(ValueError):
            self.other.compare_chunks([b"\xff"], "right")

    # --- content verdicts mirror compare_bytes exactly ---

    def test_corrupt_snapshot_reports_side_left_first(self):
        _i, good = self.row("t", 1)
        r = self.other.compare_chunks([b"{oops\n"], [good[:5], good[5:]])
        self.assertEqual(r, {"ok": False, "side": "left", "at": 1,
                             "tenant": None, "reason": "missing"})
        r = self.other.compare_chunks([good], [b"\xff"])
        self.assertEqual(r, {"ok": False, "side": "right", "at": 1,
                             "tenant": None, "reason": "missing"})
        # both corrupt: fixed left-first
        r = self.other.compare_chunks([b"{oops\n"], [b"\xff"])
        self.assertEqual(r["side"], "left")
        self.assertEqual((r["at"], r["tenant"], r["reason"]),
                         (1, None, "missing"))

    def test_corruption_reasons_match_compare_bytes(self):
        _t, good1 = self.row("t", 1)
        gap = {"tenant": "t", "seq": 3, "event": {}, "prev": "9" * 64}
        gap["hash"] = AuditChain._hash(gap)
        raw = good1 + record_bytes(gap)
        chunks = [raw[:20], b"", raw[20:]]
        r = self.other.compare_chunks(chunks, [good1])
        self.assertEqual((r["ok"], r["side"], r["at"], r["tenant"],
                          r["reason"]),
                         (False, "left", 2, "t", "sequence"))
        self.assertEqual(r, self.other.compare_bytes(raw, good1))
        r = self.other.compare_chunks([good1], [good1, b"\xff"])
        self.assertEqual((r["ok"], r["side"], r["at"], r["tenant"],
                          r["reason"]),
                         (False, "right", 2, None, "missing"))
        self.assertEqual(r, self.other.compare_bytes(good1, good1 + b"\xff"))

    def test_difference_results_match_compare_bytes(self):
        _i, left = self.row("t", 1, event={"v": 1})
        _i, right = self.row("t", 1, event={"v": 2})
        r = self.other.compare_chunks([left[:7], left[7:]], [right])
        self.assertEqual((r["ok"], r["equal"], r["tenant"], r["seq"],
                          r["reason"]),
                         (True, False, "t", 1, "different"))
        self.assertEqual(r, self.other.compare_bytes(left, right))
        self.assertEqual(r["left"]["event"], {"v": 1})
        self.assertEqual(r["right"]["event"], {"v": 2})

        _i1, one = self.row("t", 1)
        i2, two = self.row("t", 2, prev=_i1["hash"], event={"i": 2})
        r = self.other.compare_chunks([one, two], [one])
        self.assertEqual((r["tenant"], r["seq"], r["reason"]),
                         ("t", 2, "missing_right"))
        self.assertIsNone(r["right"])
        self.assertEqual(r["left"], i2)
        r = self.other.compare_chunks([one], [one[:3], one[3:], two])
        self.assertEqual((r["tenant"], r["seq"], r["reason"]),
                         ("t", 2, "missing_left"))
        self.assertIsNone(r["left"])
        self.assertEqual(r["right"], i2)

    def test_interleaved_orders_and_tenant_summary(self):
        left_chain = AuditChain(self.path.with_name("l.jsonl"))
        self.seed(left_chain, ("a", {"i": 1}), ("z", {"i": 1}),
                  ("m", {"i": 1}))
        right_chain = AuditChain(self.path.with_name("r.jsonl"))
        self.seed(right_chain, ("z", {"i": 1}), ("a", {"i": 1}),
                  ("m", {"i": 1}))
        left, right = self.snapshot(left_chain), self.snapshot(right_chain)
        r = self.other.compare_chunks([left[:13], left[13:]],
                                      [right[:2], right[2:40], right[40:]])
        self.assertEqual(r, self.other.compare_bytes(left, right))
        self.assertTrue(r["equal"])
        self.assertEqual([e["tenant"] for e in r["tenants"]],
                         ["a", "z", "m"])

    # --- offline purity ---

    def test_never_reads_creates_or_modifies_constructor_path(self):
        phantom = self.path.with_name("phantom.jsonl")
        chain = AuditChain(phantom)
        self.assertEqual(chain.compare_chunks([], [])["equal"], True)
        self.assertFalse(phantom.exists())
        self.seed(self.chain, ("a", {"i": 1}), ("b", {"i": 1}),
                  ("a", {"i": 2}))
        data = self.snapshot()
        self.assertTrue(chain.compare_chunks([data[:11], data[11:]],
                                             [data])["equal"])
        self.assertFalse(phantom.exists())
        self.assertEqual(self.snapshot(), data)

    def test_does_not_mutate_chunks(self):
        self.seed(self.chain, ("a", {"i": 1}))
        data = self.snapshot()
        left = [data[:5], data[5:]]
        right = [data]
        self.other.compare_chunks(left, right)
        self.assertEqual(b"".join(left), data)
        self.assertEqual(b"".join(right), data)


if __name__ == "__main__":
    unittest.main()
