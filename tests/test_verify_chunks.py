import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app import AuditChain

ZERO = "0" * 64


def record_bytes(item):
    return (json.dumps(item, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


class VerifyChunksTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.path = Path(self.tmp.name) / "audit.jsonl"
        self.chain = AuditChain(self.path)
        self.other = AuditChain(self.path.with_name("never-touched.jsonl"))

    def tearDown(self):
        self.tmp.cleanup()

    def seed(self):
        self.chain.append("a", {"i": 1})
        self.chain.append("b", {"i": 1})
        self.chain.append("a", {"i": 2})
        self.chain.append("a", {"i": 3})

    def snapshot(self):
        return self.path.read_bytes()

    def valid_row(self, tenant="t", seq=1, prev=ZERO, event=None):
        row = {"tenant": tenant, "seq": seq, "event": event or {}, "prev": prev}
        row["hash"] = AuditChain._hash(row)
        return record_bytes(row)

    # --- equivalence with verify_bytes / verify_all_bytes ---

    def test_matches_verify_bytes_for_many_chunkings(self):
        self.seed()
        data = self.snapshot()
        chunkings = [
            [data],
            [data[:1], data[1:]],
            [data[i:i + 1] for i in range(len(data))],
            [data[:10], b"", data[10:37], b"", data[37:]],
            [b"", b"", data, b""],
            list(data[i:i + 7] for i in range(0, len(data), 7)),
        ]
        for chunks in chunkings:
            for tenant in ("a", "b", "zzz"):
                self.assertEqual(self.other.verify_chunks(chunks, tenant),
                                 self.other.verify_bytes(data, tenant), chunks)
            self.assertEqual(self.other.verify_all_chunks(chunks),
                             self.other.verify_all_bytes(data), chunks)

    def test_matches_verify_bytes_with_expected_count(self):
        self.seed()
        data = self.chain.export_tenant("a")
        chunks = [data[:5], data[5:]]
        for expected in (None, 0, 2, 3, 4):
            self.assertEqual(
                self.other.verify_chunks(chunks, "a", expected),
                self.other.verify_bytes(data, "a", expected), expected)
        self.assertEqual(self.other.verify_chunks(chunks, "a", 3),
                         {"ok": True, "count": 3})
        r = self.other.verify_chunks(chunks, "a", 4)
        self.assertEqual((r["ok"], r["at"], r["reason"]), (False, 4, "missing"))
        r = self.other.verify_chunks(chunks, "a", 2)
        self.assertEqual((r["ok"], r["at"], r["reason"]), (False, 3, "sequence"))

    def test_chunk_may_split_multibyte_utf8(self):
        self.chain.append("t", {"msg": "héllo→世界"})
        self.chain.append("t", {"msg": "✓" * 40})
        data = self.snapshot()
        # cut at every single byte offset; each split must verify identically
        for cut in range(len(data) + 1):
            chunks = [data[:cut], data[cut:]]
            self.assertEqual(self.other.verify_chunks(chunks, "t"),
                             {"ok": True, "count": 2}, cut)
            self.assertEqual(self.other.verify_all_chunks(chunks),
                             self.other.verify_all_bytes(data), cut)

    def test_empty_sequence_is_empty_history(self):
        for chunks in ([], (), iter([]), [b""], [b"", b""]):
            self.assertEqual(self.other.verify_chunks(chunks, "t"),
                             {"ok": True, "count": 0})
            self.assertEqual(self.other.verify_chunks(chunks, "t", 0),
                             {"ok": True, "count": 0})
            self.assertEqual(self.other.verify_all_chunks(chunks),
                             {"ok": True, "tenants": []})

    def test_accepts_any_iterable_container(self):
        self.seed()
        data = self.snapshot()
        halves = [data[:len(data) // 2], data[len(data) // 2:]]
        for container in (halves, tuple(halves), iter(halves),
                          (c for c in halves)):
            self.assertEqual(self.other.verify_chunks(container, "a"),
                             {"ok": True, "count": 3})
            self.assertTrue(self.other.verify_all_chunks(container)["ok"])

    # --- verdicts mirror verify_bytes exactly, at = physical line ---

    def test_error_classification_and_physical_line_numbers(self):
        good = self.valid_row("t", 1)
        cases = []
        # missing: unparseable, blank, non-object, missing field, bad utf-8,
        # duplicate key, non-standard number, truncated tail record
        row_no_hash = {"tenant": "t", "seq": 1, "event": {}, "prev": ZERO}
        cases += [
            b"{not json\n",
            b"\n",
            b"[1,2]\n",
            (json.dumps(row_no_hash) + "\n").encode(),
            good + b"\xff\n",
            b'{"tenant":"t","tenant":"t","seq":1,"event":{},'
            b'"prev":"' + ZERO.encode() + b'","hash":"x"}\n',
            b'{"tenant":"t","seq":1e999,"event":{},"prev":"'
            + ZERO.encode() + b'","hash":"x"}\n',
            good + b'{"tenant":"t","seq":2',  # incomplete final record
        ]
        for raw in cases:
            chunks = [raw[:3], b"", raw[3:len(raw) // 2], raw[len(raw) // 2:]]
            self.assertEqual(self.other.verify_chunks(chunks, "t"),
                             self.other.verify_bytes(raw, "t"), raw)
            self.assertEqual(self.other.verify_all_chunks(chunks),
                             self.other.verify_all_bytes(raw), raw)
            self.assertFalse(self.other.verify_chunks(chunks, "t")["ok"], raw)

    def test_sequence_and_digest_classification(self):
        good = self.valid_row("t", 1)
        gap_row = {"tenant": "t", "seq": 3, "event": {}, "prev": "x" * 64}
        gap_row["hash"] = AuditChain._hash(gap_row)
        raw = good + record_bytes(gap_row)
        chunks = [raw[:20], raw[20:]]
        self.assertEqual(self.other.verify_chunks(chunks, "t"),
                         {"ok": False, "at": 2, "reason": "sequence"})

        bad_prev = {"tenant": "t", "seq": 2, "event": {}, "prev": "9" * 64}
        bad_prev["hash"] = AuditChain._hash(bad_prev)
        raw = good + record_bytes(bad_prev)
        chunks = [raw[:20], raw[20:]]
        self.assertEqual(self.other.verify_chunks(chunks, "t"),
                         {"ok": False, "at": 2, "reason": "digest"})

    def test_first_error_wins_regardless_of_later_chunks(self):
        # digest error on line 1 beats illegal utf-8 arriving in a later chunk
        tampered = {"tenant": "t", "seq": 1, "event": {"z": 9}, "prev": ZERO}
        tampered["hash"] = AuditChain._hash(
            {"tenant": "t", "seq": 1, "event": {}, "prev": ZERO})
        raw = record_bytes(tampered) + b"\xff\n"
        chunks = [record_bytes(tampered), b"\xff", b"\n"]
        r = self.other.verify_chunks(chunks, "t")
        self.assertEqual((r["ok"], r["at"], r["reason"]), (False, 1, "digest"))
        self.assertEqual(r, self.other.verify_bytes(raw, "t"))

    def test_verify_all_chunks_failure_fields(self):
        good = {"tenant": "a", "seq": 1, "event": {}, "prev": ZERO}
        good["hash"] = AuditChain._hash(good)
        bad = {"tenant": "b", "seq": 1, "event": {}, "prev": "f" * 64}
        bad["hash"] = AuditChain._hash(bad)
        raw = record_bytes(good) + record_bytes(bad)
        chunks = [raw[:11], raw[11:]]
        self.assertEqual(self.other.verify_all_chunks(chunks),
                         {"ok": False, "at": 2, "tenant": "b",
                          "reason": "digest"})
        self.assertEqual(self.other.verify_all_chunks([b"{oops\n"]),
                         {"ok": False, "at": 1, "tenant": None,
                          "reason": "missing"})
        self.assertEqual(self.other.verify_all_chunks([b"\xff"]),
                         {"ok": False, "at": 1, "tenant": None,
                          "reason": "missing"})

    def test_tenant_identities_and_ordering(self):
        for t in (1, 1.0, True, "1"):
            self.chain.append(t, {})
            self.chain.append(t, {})
        data = self.snapshot()
        chunks = [data[:13], data[13:64], data[64:]]
        for t in (1, 1.0, True, "1"):
            self.assertEqual(self.other.verify_chunks(chunks, t),
                             {"ok": True, "count": 2})
        self.assertEqual(self.other.verify_all_chunks(chunks),
                         self.other.verify_all_bytes(data))

    # --- container and element boundary ---

    def test_bare_bytes_container_is_value_error(self):
        for bad in (b"", b"{}", bytearray(b"")):
            with self.assertRaises(ValueError):
                self.other.verify_chunks(bad, "t")
            with self.assertRaises(ValueError):
                self.other.verify_all_chunks(bad)

    def test_non_iterable_container_is_value_error(self):
        for bad in (1, None, object()):
            with self.assertRaises(ValueError):
                self.other.verify_chunks(bad, "t")
            with self.assertRaises(ValueError):
                self.other.verify_all_chunks(bad)

    def test_non_bytes_element_is_value_error(self):
        for bad in ([b"", "x"], [b"", 1], [b"", None], [b"", bytearray(b"")],
                    [b"", [b""]]):
            with self.assertRaises(ValueError):
                self.other.verify_chunks(bad, "t")
            with self.assertRaises(ValueError):
                self.other.verify_all_chunks(bad)

    def test_bad_element_stops_consumption(self):
        pulled = []

        def gen():
            pulled.append(1)
            yield b""
            pulled.append(2)
            yield "not bytes"
            pulled.append(3)  # must never be reached
            yield b""

        with self.assertRaises(ValueError):
            self.other.verify_chunks(gen(), "t")
        self.assertEqual(pulled, [1, 2])

        pulled.clear()
        with self.assertRaises(ValueError):
            self.other.verify_all_chunks(gen())
        self.assertEqual(pulled, [1, 2])

    def test_tenant_and_expected_count_boundary(self):
        for bad in (float("nan"), float("inf"), {"k": float("nan")},
                    {1: "x"}, object(), b"x", {1, 2}):
            with self.assertRaises(ValueError):
                self.other.verify_chunks([], bad)
        for bad in (-1, 1.0, True, False, "1", object()):
            with self.assertRaises(ValueError):
                self.other.verify_chunks([], "t", bad)
        # bad tenant/count raise before the container is consumed at all
        pulled = []

        def gen():
            pulled.append(1)
            yield b""

        with self.assertRaises(ValueError):
            self.other.verify_chunks(gen(), float("nan"))
        with self.assertRaises(ValueError):
            self.other.verify_chunks(gen(), "t", -1)
        self.assertEqual(pulled, [])

    # --- offline purity ---

    def test_never_reads_creates_or_modifies_constructor_path(self):
        phantom = self.path.with_name("phantom.jsonl")
        chain = AuditChain(phantom)
        self.assertEqual(chain.verify_chunks([], "t"), {"ok": True, "count": 0})
        self.assertEqual(chain.verify_all_chunks([]),
                         {"ok": True, "tenants": []})
        self.assertFalse(phantom.exists())
        self.seed()
        before = self.snapshot()
        chunks = [before[:9], before[9:]]
        self.assertTrue(chain.verify_chunks(chunks, "a")["ok"])
        self.assertTrue(chain.verify_all_chunks(chunks)["ok"])
        self.assertEqual(self.snapshot(), before)
        self.assertFalse(phantom.exists())

    def test_does_not_mutate_chunks(self):
        self.seed()
        data = self.snapshot()
        chunks = [data[:5], data[5:]]
        self.other.verify_chunks(chunks, "a")
        self.other.verify_all_chunks(chunks)
        self.assertEqual(b"".join(chunks), data)


if __name__ == "__main__":
    unittest.main()
