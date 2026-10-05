import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app import AuditChain, AuditChainStateError, MANIFEST_VERSION

ZERO = "0" * 64


def record_bytes(item):
    return (json.dumps(item, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


class VerifyStreamTest(unittest.TestCase):
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
        self.chain.append("b", {"i": 2})

    def snapshot(self):
        return self.path.read_bytes()

    def valid_row(self, tenant="t", seq=1, prev=ZERO, event=None):
        row = {"tenant": tenant, "seq": seq, "event": event or {}, "prev": prev}
        row["hash"] = AuditChain._hash(row)
        return record_bytes(row)

    # --- equivalence with the bytes entries across many chunkings ---

    def test_matches_bytes_entries_for_many_chunkings(self):
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
                self.assertEqual(
                    self.other.verify_stream(iter(chunks), tenant),
                    self.other.verify_bytes(data, tenant), chunks)
            self.assertEqual(
                self.other.verify_all_stream(iter(chunks)),
                self.other.verify_all_bytes(data), chunks)
            self.assertEqual(
                self.other.heads_stream(iter(chunks)),
                self.other.heads_bytes(data), chunks)
            self.assertEqual(
                self.other.manifest_stream(iter(chunks)),
                self.other.manifest_bytes(data), chunks)

    def test_matches_verify_bytes_with_expected_count(self):
        self.seed()
        data = self.chain.export_tenant("a")
        chunks = [data[:5], data[5:]]
        for expected in (None, 0, 2, 3, 4):
            self.assertEqual(
                self.other.verify_stream(iter(chunks), "a", expected),
                self.other.verify_bytes(data, "a", expected), expected)
        self.assertEqual(
            self.other.verify_stream(iter(chunks), "a", 3),
            {"ok": True, "count": 3})
        r = self.other.verify_stream(iter(chunks), "a", 4)
        self.assertEqual((r["ok"], r["at"], r["reason"]), (False, 4, "missing"))
        r = self.other.verify_stream(iter(chunks), "a", 2)
        self.assertEqual((r["ok"], r["at"], r["reason"]), (False, 3, "sequence"))

    def test_chunk_may_split_multibyte_utf8(self):
        self.chain.append("t", {"msg": "héllo→世界"})
        self.chain.append("t", {"msg": "✓" * 40})
        data = self.snapshot()
        for cut in range(len(data) + 1):
            chunks = [data[:cut], data[cut:]]
            self.assertEqual(
                self.other.verify_stream(iter(chunks), "t"),
                {"ok": True, "count": 2}, cut)
            self.assertEqual(
                self.other.verify_all_stream(iter(chunks)),
                self.other.verify_all_bytes(data), cut)
            self.assertEqual(
                self.other.heads_stream(iter(chunks)),
                self.other.heads_bytes(data), cut)
            self.assertEqual(
                self.other.manifest_stream(iter(chunks)),
                self.other.manifest_bytes(data), cut)

    def test_chunk_may_split_crlf_and_unicode_separators(self):
        # CRLF is one physical line; NEL/U+2028/U+2029 inside a JSON string
        # are content, not line boundaries -- whatever the chunk cut.
        row = {"tenant": "t", "seq": 1, "event": "a\x85  ",
               "prev": ZERO}
        row["hash"] = AuditChain._hash(row)
        raw = (json.dumps(row, ensure_ascii=False) + "\r\n").encode("utf-8")
        self.other.verify_bytes(raw, "t")  # the bytes entry accepts it
        for cut in range(len(raw) + 1):
            chunks = [raw[:cut], raw[cut:]]
            self.assertEqual(
                self.other.verify_stream(iter(chunks), "t"),
                self.other.verify_bytes(raw, "t"), cut)
            self.assertEqual(
                self.other.verify_all_stream(iter(chunks)),
                self.other.verify_all_bytes(raw), cut)
            m = self.other.manifest_stream(iter(chunks))
            self.assertEqual(m["byte_length"], len(raw))
            self.assertEqual(m["byte_sha256"],
                             hashlib.sha256(raw).hexdigest())

    def test_empty_sequence_is_empty_history(self):
        for chunks in ([], (), iter([]), [b""], [b"", b""]):
            self.assertEqual(
                self.other.verify_stream(iter(list(chunks))
                                         if not isinstance(chunks, list)
                                         else chunks, "t"),
                {"ok": True, "count": 0})
            self.assertEqual(
                self.other.verify_all_stream(iter(list(chunks))),
                {"ok": True, "tenants": []})
            self.assertEqual(
                self.other.heads_stream(iter(list(chunks))),
                {"ok": True, "tenants": []})
            self.assertEqual(
                self.other.manifest_stream(iter(list(chunks))),
                self.other.manifest_bytes(b""))

    def test_accepts_any_iterable_container(self):
        self.seed()
        data = self.snapshot()
        halves = [data[:len(data) // 2], data[len(data) // 2:]]
        for container in (halves, tuple(halves), iter(halves),
                          (c for c in halves)):
            self.assertEqual(self.other.verify_stream(container, "a"),
                             {"ok": True, "count": 3})
        for container in (tuple(halves), iter(halves),
                          (c for c in halves)):
            self.assertTrue(self.other.verify_all_stream(container)["ok"])
        for container in (tuple(halves), iter(halves),
                          (c for c in halves)):
            self.assertEqual(self.other.heads_stream(container)["tenants"][0],
                             self.other.heads_bytes(data)["tenants"][0])
        for container in (tuple(halves), iter(halves),
                          (c for c in halves)):
            self.assertEqual(self.other.manifest_stream(container),
                             self.other.manifest_bytes(data))

    def test_single_consumption_iterator_pulled_once(self):
        self.seed()
        data = self.snapshot()
        it = iter([data[:4], b"", data[4:]])
        self.assertTrue(self.other.verify_all_stream(it)["ok"])
        # The iterator is exhausted: a second pass over the same object is an
        # empty history, exactly what a once-only stream contract implies.
        self.assertEqual(self.other.verify_all_stream(it),
                         {"ok": True, "tenants": []})

    # --- verdicts mirror the bytes entries exactly ---

    def test_error_classification_and_physical_line_numbers(self):
        good = self.valid_row("t", 1)
        row_no_hash = {"tenant": "t", "seq": 1, "event": {}, "prev": ZERO}
        cases = [
            b"{not json\n",
            b"\n",
            b"[1,2]\n",
            (json.dumps(row_no_hash) + "\n").encode(),
            good + b"\xff\n",
            b'{"tenant":"t","tenant":"t","seq":1,"event":{},'
            b'"prev":"' + ZERO.encode() + b'","hash":"x"}\n',
            b'{"tenant":"t","seq":1e999,"event":{},"prev":"'
            + ZERO.encode() + b'","hash":"x"}\n',
            good + b'{"tenant":"t","seq":2',  # incomplete tail, no final LF
        ]
        for raw in cases:
            chunks = [raw[:3], b"", raw[3:len(raw) // 2], raw[len(raw) // 2:]]
            self.assertEqual(
                self.other.verify_stream(iter(chunks), "t"),
                self.other.verify_bytes(raw, "t"), raw)
            self.assertEqual(
                self.other.verify_all_stream(iter(chunks)),
                self.other.verify_all_bytes(raw), raw)
            self.assertEqual(
                self.other.heads_stream(iter(chunks)),
                self.other.heads_bytes(raw), raw)
            self.assertFalse(
                self.other.verify_stream(
                    iter([raw[:2], raw[2:]]), "t")["ok"], raw)

    def test_sequence_and_digest_classification(self):
        good = self.valid_row("t", 1)
        gap_row = {"tenant": "t", "seq": 3, "event": {}, "prev": "x" * 64}
        gap_row["hash"] = AuditChain._hash(gap_row)
        raw = good + record_bytes(gap_row)
        self.assertEqual(
            self.other.verify_stream(iter([raw[:20], raw[20:]]), "t"),
            {"ok": False, "at": 2, "reason": "sequence"})
        self.assertEqual(
            self.other.verify_all_stream(iter([raw[:20], raw[20:]])),
            {"ok": False, "at": 2, "tenant": "t", "reason": "sequence"})

        bad_prev = {"tenant": "t", "seq": 2, "event": {}, "prev": "9" * 64}
        bad_prev["hash"] = AuditChain._hash(bad_prev)
        raw = good + record_bytes(bad_prev)
        for fn, expected in (
            (lambda c: self.other.verify_stream(c, "t"),
             {"ok": False, "at": 2, "reason": "digest"}),
            (lambda c: self.other.verify_all_stream(c),
             {"ok": False, "at": 2, "tenant": "t", "reason": "digest"}),
            (lambda c: self.other.heads_stream(c),
             {"ok": False, "at": 2, "tenant": "t", "reason": "digest"}),
        ):
            self.assertEqual(fn(iter([raw[:20], raw[20:]])), expected)

    def test_first_error_wins_over_later_chunks(self):
        # digest error on line 1 beats illegal utf-8 arriving in a later chunk
        tampered = {"tenant": "t", "seq": 1, "event": {"z": 9}, "prev": ZERO}
        tampered["hash"] = AuditChain._hash(
            {"tenant": "t", "seq": 1, "event": {}, "prev": ZERO})
        raw = record_bytes(tampered) + b"\xff\n"
        r = self.other.verify_stream(iter([record_bytes(tampered), b"\xff", b"\n"]), "t")
        self.assertEqual((r["ok"], r["at"], r["reason"]), (False, 1, "digest"))
        self.assertEqual(r, self.other.verify_bytes(raw, "t"))
        r = self.other.verify_all_stream(
            iter([record_bytes(tampered), b"\xff", b"\n"]))
        self.assertEqual(r, self.other.verify_all_bytes(raw))

    def test_verify_all_stream_failure_fields(self):
        good = {"tenant": "a", "seq": 1, "event": {}, "prev": ZERO}
        good["hash"] = AuditChain._hash(good)
        bad = {"tenant": "b", "seq": 1, "event": {}, "prev": "f" * 64}
        bad["hash"] = AuditChain._hash(bad)
        raw = record_bytes(good) + record_bytes(bad)
        expected = {"ok": False, "at": 2, "tenant": "b", "reason": "digest"}
        self.assertEqual(
            self.other.verify_all_stream(iter([raw[:11], raw[11:]])), expected)
        self.assertEqual(
            self.other.heads_stream(iter([raw[:11], raw[11:]])), expected)
        self.assertEqual(
            self.other.verify_all_stream(iter([b"{oops\n"])),
            {"ok": False, "at": 1, "tenant": None, "reason": "missing"})
        self.assertEqual(
            self.other.verify_all_stream(iter([b"\xff"])),
            {"ok": False, "at": 1, "tenant": None, "reason": "missing"})

    def test_tenant_identities_and_ordering(self):
        for t in (1, 1.0, True, "1"):
            self.chain.append(t, {})
            self.chain.append(t, {})
        data = self.snapshot()
        chunks = [data[:13], data[13:64], data[64:]]
        for t in (1, 1.0, True, "1"):
            self.assertEqual(
                self.other.verify_stream(iter(chunks), t),
                {"ok": True, "count": 2})
        self.assertEqual(
            self.other.verify_all_stream(iter(chunks)),
            self.other.verify_all_bytes(data))
        self.assertEqual(
            self.other.heads_stream(iter(chunks)),
            self.other.heads_bytes(data))

    # --- single-pass consumption: a defect stops the pull, success drains ---

    def test_defect_stops_requesting_later_chunks(self):
        tampered = {"tenant": "t", "seq": 1, "event": {"z": 9}, "prev": ZERO}
        tampered["hash"] = AuditChain._hash(
            {"tenant": "t", "seq": 1, "event": {}, "prev": ZERO})

        def make_gen(pulled):
            def gen():
                pulled.append(1)
                yield record_bytes(tampered)  # line 1 complete, digest defect
                pulled.append(2)
                yield b'{"tenant":"t","seq":2}\n'  # must never be requested
                pulled.append(3)
            return gen

        for call in (
            lambda g: self.other.verify_stream(g, "t"),
            lambda g: self.other.verify_all_stream(g),
            lambda g: self.other.heads_stream(g),
        ):
            pulled = []
            r = call(make_gen(pulled)())
            self.assertFalse(r["ok"])
            self.assertEqual(r["reason"], "digest")
            self.assertEqual(pulled, [1])
            with self.assertRaises(AuditChainStateError) as cm:
                self.other.manifest_stream(make_gen(pulled := [])())
            self.assertEqual(cm.exception.reason, "digest")
            self.assertEqual(pulled, [1])

    def test_defect_chunk_without_lf_still_pulls_only_the_lf_chunk(self):
        # The bad UTF-8 byte cannot be located until its physical line ends:
        # the chunk carrying the terminating LF is still pulled, then the pull
        # stops before any later chunk.
        pulled = []

        def gen():
            pulled.append(1)
            yield self.valid_row("t", 1) + b'{"tenant":\xff'
            pulled.append(2)
            yield b"}\n"
            pulled.append(3)
            yield b'{"tenant":"t","seq":3}\n'
            pulled.append(4)

        r = self.other.verify_all_stream(gen())
        self.assertEqual((r["ok"], r["at"], r["tenant"], r["reason"]),
                         (False, 2, None, "missing"))
        self.assertEqual(pulled, [1, 2])

    def test_valid_history_is_consumed_to_the_end(self):
        self.seed()
        data = self.snapshot()
        pulled = []

        def gen():
            for i, part in enumerate((data[:9], b"", data[9:50], data[50:]), 1):
                pulled.append(i)
                yield part

        self.assertTrue(self.other.verify_all_stream(gen())["ok"])
        self.assertEqual(pulled, [1, 2, 3, 4])

        pulled.clear()

        def gen2():
            for i, part in enumerate((data[:9], data[9:]), 1):
                pulled.append(i)
                yield part

        self.other.manifest_stream(gen2())
        self.assertEqual(pulled, [1, 2])

    def test_no_partial_tenants_on_failure(self):
        good = {"tenant": "a", "seq": 1, "event": {}, "prev": ZERO}
        good["hash"] = AuditChain._hash(good)
        raw = record_bytes(good) + b"{oops\n"
        for result in (
            self.other.verify_all_stream(iter([raw])),
            self.other.heads_stream(iter([raw])),
        ):
            self.assertFalse(result["ok"])
            self.assertNotIn("tenants", result)
        with self.assertRaises(AuditChainStateError):
            self.other.manifest_stream(iter([raw]))

    # --- manifest stream specifics ---

    def test_manifest_fields_describe_exact_original_bytes(self):
        self.seed()
        data = self.snapshot()
        m = self.other.manifest_stream(iter([data[:7], b"", data[7:]]))
        self.assertEqual(m["version"], MANIFEST_VERSION)
        self.assertEqual(m["byte_length"], len(data))
        self.assertEqual(m["byte_sha256"], hashlib.sha256(data).hexdigest())
        self.assertEqual(m, self.other.manifest_bytes(data))
        self.assertEqual(
            [(t["tenant"], t["count"], t["hash"]) for t in m["tenants"]],
            [("a", 3, self.chain.head("a")["hash"]),
             ("b", 2, self.chain.head("b")["hash"])],
        )

    def test_manifest_stream_empty(self):
        m = self.other.manifest_stream(iter([]))
        self.assertEqual(m, {
            "version": MANIFEST_VERSION,
            "byte_length": 0,
            "byte_sha256": hashlib.sha256(b"").hexdigest(),
            "tenants": [],
        })

    def test_manifest_stream_corruption_matches_manifest_bytes(self):
        good = self.valid_row("t", 1)
        row_missing_field = {"tenant": "t", "seq": 2, "event": {}, "prev": ZERO}
        gap = {"tenant": "t", "seq": 9, "event": {}, "prev": "x" * 64}
        gap["hash"] = AuditChain._hash(gap)
        bad_digest = {"tenant": "t", "seq": 2, "event": {}, "prev": "9" * 64}
        bad_digest["hash"] = AuditChain._hash(bad_digest)
        cases = [
            (b"{not json\n", (None, None, "missing", 1)),
            (b"\n", (None, None, "missing", 1)),
            (b"[1]\n", (None, None, "missing", 1)),
            (b"\xff", (None, None, "missing", 1)),
            (good + b"\xff\n", (None, None, "missing", 2)),
            (good + (json.dumps(row_missing_field) + "\n").encode(),
             ("t", 2, "missing", 2)),
            (good + record_bytes(gap), ("t", 2, "sequence", 2)),
            (good + record_bytes(bad_digest), ("t", 2, "digest", 2)),
        ]
        for raw, expected in cases:
            chunks = [raw[:2], b"", raw[2:]]
            with self.assertRaises(AuditChainStateError) as cm_s:
                self.other.manifest_stream(iter(chunks))
            with self.assertRaises(AuditChainStateError) as cm_b:
                self.other.manifest_bytes(raw)
            for attr in ("tenant", "seq", "reason", "line"):
                self.assertEqual(getattr(cm_s.exception, attr),
                                 getattr(cm_b.exception, attr), (raw, attr))
            self.assertEqual(
                (cm_s.exception.tenant, cm_s.exception.seq,
                 cm_s.exception.reason, cm_s.exception.line),
                expected, raw)
            self.assertIn(cm_s.exception.reason,
                          ("missing", "sequence", "digest"))

    def test_manifest_stream_does_not_leak_underlying_exceptions(self):
        for raw in (b"{not json\n", b"\xff"):
            try:
                self.other.manifest_stream(iter([raw]))
            except AuditChainStateError as exc:
                message = str(exc)
                self.assertNotIn("UnicodeDecodeError", message)
                self.assertNotIn("JSONDecodeError", message)
                self.assertNotIn("Expecting value", message)
            else:
                self.fail("AuditChainStateError expected")

    # --- container and element boundary ---

    def test_bare_bytes_container_is_value_error(self):
        for bad in (b"", b"{}", bytearray(b""), bytearray(b"{}")):
            with self.assertRaises(ValueError):
                self.other.verify_stream(bad, "t")
            with self.assertRaises(ValueError):
                self.other.verify_all_stream(bad)
            with self.assertRaises(ValueError):
                self.other.heads_stream(bad)
            with self.assertRaises(ValueError):
                self.other.manifest_stream(bad)

    def test_non_iterable_container_is_value_error(self):
        for bad in (1, None, object()):
            with self.assertRaises(ValueError):
                self.other.verify_stream(bad, "t")
            with self.assertRaises(ValueError):
                self.other.verify_all_stream(bad)
            with self.assertRaises(ValueError):
                self.other.heads_stream(bad)
            with self.assertRaises(ValueError):
                self.other.manifest_stream(bad)

    def test_non_bytes_member_is_value_error(self):
        for bad in ([b"", "x"], [b"", 1], [b"", None], [b"", bytearray(b"")],
                    [b"", [b""]], ["x"], [bytearray(b"")]):
            with self.assertRaises(ValueError):
                self.other.verify_stream(iter(bad), "t")
            with self.assertRaises(ValueError):
                self.other.verify_all_stream(iter(bad))
            with self.assertRaises(ValueError):
                self.other.heads_stream(iter(bad))
            with self.assertRaises(ValueError):
                self.other.manifest_stream(iter(bad))

    def test_bad_member_stops_consumption(self):
        for call in (
            lambda g: self.other.verify_stream(g, "t"),
            lambda g: self.other.verify_all_stream(g),
            lambda g: self.other.heads_stream(g),
            lambda g: self.other.manifest_stream(g),
        ):
            pulled = []

            def gen():
                pulled.append(1)
                yield b""
                pulled.append(2)
                yield "not bytes"
                pulled.append(3)  # must never be reached
                yield b""

            with self.assertRaises(ValueError):
                call(gen())
            self.assertEqual(pulled, [1, 2])

    def test_tenant_and_expected_count_boundary(self):
        for bad in (float("nan"), float("inf"), {"k": float("nan")},
                    {1: "x"}, object(), b"x", {1, 2}):
            with self.assertRaises(ValueError):
                self.other.verify_stream([], bad)
        for bad in (-1, 1.0, True, False, "1", object()):
            with self.assertRaises(ValueError):
                self.other.verify_stream([], "t", bad)
        # bad tenant/count raise before the container is consumed at all
        pulled = []

        def gen():
            pulled.append(1)
            yield b""

        with self.assertRaises(ValueError):
            self.other.verify_stream(gen(), float("nan"))
        with self.assertRaises(ValueError):
            self.other.verify_stream(gen(), "t", -1)
        self.assertEqual(pulled, [])

    # --- offline purity ---

    def test_never_reads_creates_or_modifies_constructor_path(self):
        phantom = self.path.with_name("phantom.jsonl")
        chain = AuditChain(phantom)
        self.assertEqual(chain.verify_stream(iter([]), "t"),
                         {"ok": True, "count": 0})
        self.assertEqual(chain.verify_all_stream(iter([])),
                         {"ok": True, "tenants": []})
        self.assertEqual(chain.heads_stream(iter([])),
                         {"ok": True, "tenants": []})
        self.assertEqual(chain.manifest_stream(iter([])),
                         self.other.manifest_bytes(b""))
        self.assertFalse(phantom.exists())
        self.seed()
        before = self.snapshot()
        chunks = [before[:9], before[9:]]
        self.assertTrue(chain.verify_stream(iter(chunks), "a")["ok"])
        self.assertTrue(chain.verify_all_stream(iter(chunks))["ok"])
        self.assertTrue(chain.heads_stream(iter(chunks))["ok"])
        chain.manifest_stream(iter(chunks))
        self.assertEqual(self.snapshot(), before)
        self.assertFalse(phantom.exists())

    def test_does_not_mutate_chunks(self):
        self.seed()
        data = self.snapshot()
        chunks_a = [data[:5], data[5:]]
        self.other.verify_stream(iter(chunks_a), "a")
        chunks_b = [data[:5], b"", data[5:]]
        self.other.verify_all_stream(iter(chunks_b))
        self.other.heads_stream(iter([data[:5], data[5:]]))
        self.other.manifest_stream(iter([data[:5], data[5:]]))
        self.assertEqual(b"".join(chunks_a), data)
        self.assertEqual(b"".join(chunks_b), data)


if __name__ == "__main__":
    unittest.main()
