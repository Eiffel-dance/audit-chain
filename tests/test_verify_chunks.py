import json
import random
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app import AuditChain

ZERO = "0" * 64


def record_bytes(item):
    return (json.dumps(item, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def chunkings(data, n=48):
    # Deterministic pseudo-random splittings of data, including empty chunks
    # and cuts inside multi-byte UTF-8 sequences.
    yield [data]
    yield [data[i:i + 1] for i in range(len(data))]
    rng = random.Random(20261004)
    for _ in range(n):
        cuts = sorted(rng.randrange(len(data) + 1)
                      for _ in range(rng.randrange(0, 8)))
        chunks, pos = [], 0
        for c in cuts:
            chunks.append(data[pos:c])
            pos = c
        chunks.append(data[pos:])
        yield chunks


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
        self.chain.append("a", {"i": 2, "text": "héllo 你🙂"})
        self.chain.append("a", {"i": 3})

    def snapshot(self):
        return self.path.read_bytes()

    # --- never touches the configured path ---

    def test_never_reads_creates_or_modifies_constructor_path(self):
        chain = AuditChain(self.path.with_name("phantom.jsonl"))
        self.assertEqual(chain.verify_chunks([], "t"), {"ok": True, "count": 0})
        self.assertEqual(chain.verify_all_chunks([]),
                         {"ok": True, "tenants": []})
        self.assertFalse(self.path.with_name("phantom.jsonl").exists())
        self.seed()
        before = self.snapshot()
        self.assertTrue(chain.verify_chunks([before], "a")["ok"])
        self.assertTrue(chain.verify_all_chunks([before])["ok"])
        self.assertEqual(self.snapshot(), before)
        self.assertFalse(self.path.with_name("phantom.jsonl").exists())

    def test_does_not_mutate_input(self):
        self.seed()
        chunks = [self.snapshot()]
        self.other.verify_chunks(chunks, "a")
        self.other.verify_all_chunks(chunks)
        self.assertEqual(chunks, [self.snapshot()])

    # --- equivalence with verify_bytes / verify_all_bytes ---

    def test_matches_verify_bytes_across_chunkings(self):
        self.seed()
        data = self.snapshot()
        for chunks in chunkings(data):
            for tenant in ("a", "b", "zzz"):
                self.assertEqual(self.other.verify_chunks(chunks, tenant),
                                 self.other.verify_bytes(data, tenant),
                                 (tenant, chunks))
            self.assertEqual(self.other.verify_all_chunks(chunks),
                             self.other.verify_all_bytes(data), chunks)

    def test_matches_verify_bytes_expected_count(self):
        self.seed()
        data = self.chain.export_tenant("a")
        for chunks in chunkings(data, n=8):
            for expected in (None, 0, 2, 3, 4):
                self.assertEqual(
                    self.other.verify_chunks(chunks, "a", expected),
                    self.other.verify_bytes(data, "a", expected),
                    (expected, chunks))

    def test_empty_sequence_and_empty_chunks_are_empty_history(self):
        for chunks in ([], [b""], [b"", b""], iter([b""])):
            self.assertEqual(self.other.verify_chunks(chunks, "t"),
                             {"ok": True, "count": 0})
            self.assertEqual(self.other.verify_chunks(chunks, "t", 0),
                             {"ok": True, "count": 0})
            self.assertEqual(self.other.verify_all_chunks(chunks),
                             {"ok": True, "tenants": []})

    def test_accepts_any_iterable_container(self):
        self.seed()
        data = self.snapshot()
        for container in ((data,), iter([data]), (c for c in [data]),
                          {data: None}.keys()):
            self.assertEqual(self.other.verify_chunks(container, "a"),
                             {"ok": True, "count": 3})

    # --- verdicts mirror verify_bytes exactly on corrupt histories ---

    def valid_row(self, tenant="t", seq=1, prev=ZERO, event=None):
        row = {"tenant": tenant, "seq": seq, "event": event or {}, "prev": prev}
        row["hash"] = AuditChain._hash(row)
        return record_bytes(row)

    def corrupt_histories(self):
        good = self.valid_row()
        gap = {"tenant": "t", "seq": 3, "event": {}, "prev": "x" * 64}
        gap["hash"] = AuditChain._hash(gap)
        bad_prev = {"tenant": "t", "seq": 2, "event": {}, "prev": "9" * 64}
        bad_prev["hash"] = AuditChain._hash(bad_prev)
        missing_field = {"tenant": "t", "seq": 1, "event": {}, "prev": ZERO}
        return [
            b"\n", b"   \n", b"[1,2]\n", b"{not json\n",
            b'{"tenant":"t","tenant":"t","seq":1,"event":{},'
            b'"prev":"' + ZERO.encode() + b'","hash":"x"}\n',
            b'{"tenant":"t","seq":1,"event":{},"prev":"' + ZERO.encode()
            + b'","hash":"x","x":NaN}\n',
            b'{"tenant":"t","seq":1e999,"event":{},"prev":"'
            + ZERO.encode() + b'","hash":"x"}\n',
            b"\xff", good + b'{"tenant":\xff}', good + b"abc\xc2\n",
            good + b"\xc3", b"a\r\xff", "x y\n\xff".encode(),
            (json.dumps(missing_field) + "\n").encode(),
            good + record_bytes(gap),
            good + record_bytes(bad_prev),
            good,                      # no trailing newline
            good + b"{",               # truncated trailing record
            good + b"\n" + good,       # blank line, then a duplicate seq
        ]

    def test_corrupt_verdicts_match_across_chunkings(self):
        for data in self.corrupt_histories():
            want_t = self.other.verify_bytes(data, "t")
            want_all = self.other.verify_all_bytes(data)
            for chunks in chunkings(data, n=16):
                self.assertEqual(self.other.verify_chunks(chunks, "t"),
                                 want_t, (data, chunks))
                self.assertEqual(self.other.verify_all_chunks(chunks),
                                 want_all, (data, chunks))

    def test_first_error_not_changed_by_later_chunks(self):
        good = self.valid_row()
        gap = {"tenant": "t", "seq": 3, "event": {}, "prev": "x" * 64}
        gap["hash"] = AuditChain._hash(gap)
        broken = good + record_bytes(gap)
        verdict = self.other.verify_chunks([broken], "t")
        self.assertEqual((verdict["ok"], verdict["reason"]),
                         (False, "sequence"))
        # Appending any later chunks -- valid or corrupt -- changes nothing.
        for tail in (self.valid_row("u"), b"\xff", b"{not json\n"):
            self.assertEqual(
                self.other.verify_chunks([broken, tail], "t"), verdict)
            self.assertEqual(
                self.other.verify_chunks([broken[:5], broken[5:], tail], "t"),
                verdict)

    def test_split_inside_utf8_sequence(self):
        self.chain.append("t", {"text": "héllo 你🙂"})
        self.chain.append("t", {"text": "wörld"})
        data = self.chain.export_tenant("t")
        want = self.other.verify_bytes(data, "t")
        self.assertEqual(want, {"ok": True, "count": 2})
        for chunks in chunkings(data, n=64):
            self.assertEqual(self.other.verify_chunks(chunks, "t"), want)
            self.assertEqual(self.other.verify_all_chunks(chunks),
                             self.other.verify_all_bytes(data))

    # --- container and element boundary ---

    def test_single_bytes_as_container_is_value_error(self):
        self.seed()
        data = self.snapshot()
        for bad in (data, b"", bytearray(data), "abc", ""):
            with self.assertRaises(ValueError):
                self.other.verify_chunks(bad, "t")
            with self.assertRaises(ValueError):
                self.other.verify_all_chunks(bad)

    def test_non_iterable_container_is_value_error(self):
        for bad in (None, 1, 1.5, object()):
            with self.assertRaises(ValueError):
                self.other.verify_chunks(bad, "t")
            with self.assertRaises(ValueError):
                self.other.verify_all_chunks(bad)

    def test_non_bytes_element_is_value_error(self):
        good = self.valid_row()
        for bad_element in ("x", 1, None, bytearray(good),
                            memoryview(good), [good], {"x": 1}):
            with self.assertRaises(ValueError):
                self.other.verify_chunks([good, bad_element], "t")
            with self.assertRaises(ValueError):
                self.other.verify_all_chunks([bad_element, good])

    def test_value_error_stops_consuming_at_bad_element(self):
        pulled = []

        def gen():
            pulled.append(1)
            yield self.valid_row()
            pulled.append(2)
            yield 123
            pulled.append(3)
            yield self.valid_row()

        with self.assertRaises(ValueError):
            self.other.verify_chunks(gen(), "t")
        self.assertEqual(pulled, [1, 2])

    def test_tenant_and_expected_count_boundary(self):
        for bad in (float("nan"), {"k": float("inf")}, {1: "x"}, object()):
            with self.assertRaises(ValueError):
                self.other.verify_chunks([], bad)
        for bad in (-1, 1.0, True, "1", object()):
            with self.assertRaises(ValueError):
                self.other.verify_chunks([], "t", bad)
        self.assertEqual(self.other.verify_chunks([], "t", None),
                         {"ok": True, "count": 0})

    def test_value_error_beats_corrupt_history(self):
        # a bad container or element raises even though the bytes are corrupt
        with self.assertRaises(ValueError):
            self.other.verify_chunks(b"\xff", "t")
        with self.assertRaises(ValueError):
            self.other.verify_chunks([b"\xff", 123], "t")
        with self.assertRaises(ValueError):
            self.other.verify_all_chunks([b"\xff", "not-bytes"])


if __name__ == "__main__":
    unittest.main()
