import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app import AuditChain, MANIFEST_VERSION, ZERO


def record_bytes(item):
    return (json.dumps(item, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def empty_manifest():
    return {
        "version": MANIFEST_VERSION,
        "byte_length": 0,
        "byte_sha256": hashlib.sha256(b"").hexdigest(),
        "tenants": [],
    }


class VerifyManifestStreamTest(unittest.TestCase):
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

    def expected_manifest(self, data):
        return {
            "version": 1,
            "byte_length": len(data),
            "byte_sha256": hashlib.sha256(data).hexdigest(),
            "tenants": [
                {"tenant": "a", "count": 3,
                 "hash": self.chain.head("a")["hash"]},
                {"tenant": "b", "count": 2,
                 "hash": self.chain.head("b")["hash"]},
            ],
        }

    # --- equivalence with the bytes/chunks entries across many chunkings ---

    def test_matches_bytes_and_chunks_entries_for_many_chunkings(self):
        self.seed()
        data = self.snapshot()
        good = self.expected_manifest(data)
        altered = [
            good,
            dict(good, byte_length=good["byte_length"] + 1),
            dict(good, byte_sha256="0" * 64),
            dict(good, tenants=list(reversed(good["tenants"]))),
            dict(good, tenants=[dict(good["tenants"][0], count=4),
                                good["tenants"][1]]),
            dict(good, tenants=[dict(good["tenants"][0], hash="1" * 64),
                                good["tenants"][1]]),
            empty_manifest(),
        ]
        chunkings = [
            [data],
            [data[:1], data[1:]],
            [data[i:i + 1] for i in range(len(data))],
            [data[:10], b"", data[10:37], b"", data[37:]],
            [b"", b"", data, b""],
            list(data[i:i + 7] for i in range(0, len(data), 7)),
        ]
        for m in altered:
            for chunks in chunkings:
                expected = self.other.verify_manifest_bytes(data, m)
                self.assertEqual(
                    self.other.verify_manifest_stream(iter(chunks), m),
                    expected, (m, chunks))
                self.assertEqual(
                    self.other.verify_manifest_stream(iter(chunks), m),
                    self.other.verify_manifest_chunks(chunks, m), (m, chunks))

    def test_success_echoes_manifest_computed_from_stream(self):
        self.seed()
        data = self.snapshot()
        m = self.expected_manifest(data)
        result = self.other.verify_manifest_stream(
            iter([data[:5], b"", data[5:]]), m)
        self.assertEqual(result, {"ok": True, "manifest": m})
        # The echoed manifest is derived from the stream, not the argument.
        self.assertIsNot(result["manifest"], m)
        self.assertEqual(result["manifest"],
                         self.other.manifest_bytes(data))

    def test_empty_stream_against_empty_manifest(self):
        m = empty_manifest()
        for chunks in ([], (), iter([]), [b""], [b"", b""]):
            self.assertEqual(
                self.other.verify_manifest_stream(
                    iter(list(chunks))
                    if not isinstance(chunks, list) else chunks, m),
                {"ok": True, "manifest": m}, chunks)

    def test_chunk_may_split_multibyte_utf8(self):
        self.chain.append("t", {"msg": "héllo→世界"})
        self.chain.append("t", {"msg": "✓" * 40})
        data = self.snapshot()
        m = self.other.manifest_bytes(data)
        for cut in range(len(data) + 1):
            chunks = [data[:cut], data[cut:]]
            self.assertEqual(
                self.other.verify_manifest_stream(iter(chunks), m),
                {"ok": True, "manifest": m}, cut)

    def test_byte_fields_describe_exact_original_bytes(self):
        self.seed()
        data = self.snapshot()
        m = self.expected_manifest(data)
        result = self.other.verify_manifest_stream(
            iter([data[:7], b"", data[7:]]), m)
        echoed = result["manifest"]
        self.assertEqual(echoed["version"], MANIFEST_VERSION)
        self.assertEqual(echoed["byte_length"], len(data))
        self.assertEqual(echoed["byte_sha256"],
                         hashlib.sha256(data).hexdigest())

    # --- chain corruption beats the manifest comparison ---

    def test_corrupt_stream_returns_verify_all_failure_object(self):
        row_no_hash = {"tenant": "t", "seq": 1, "event": {}, "prev": ZERO}
        cases = [
            b"{not json\n",
            b"\n",
            b"[1,2]\n",
            b"\xff",
            (json.dumps(row_no_hash) + "\n").encode(),
            self.valid_row("t", 1) + b"\xff\n",
        ]
        # The manifest deliberately disagrees on every comparable field.
        bad = {"version": 1, "byte_length": 999,
               "byte_sha256": "f" * 64, "tenants": []}
        for raw in cases:
            chunks = [raw[:3], b"", raw[3:len(raw) // 2], raw[len(raw) // 2:]]
            self.assertEqual(
                self.other.verify_manifest_stream(iter(chunks), bad),
                self.other.verify_all_bytes(raw), raw)
            r = self.other.verify_manifest_stream(iter(chunks), bad)
            self.assertFalse(r["ok"], raw)
            self.assertIn(r["reason"], ("missing", "sequence", "digest"))
            self.assertNotIn("field", r)

    def test_sequence_and_digest_corruption_beat_manifest_comparison(self):
        good = self.valid_row("t", 1)
        bad = empty_manifest()
        gap = {"tenant": "t", "seq": 3, "event": {}, "prev": "1" * 64}
        gap["hash"] = AuditChain._hash(gap)
        raw = good + record_bytes(gap)
        self.assertEqual(
            self.other.verify_manifest_stream(iter([raw[:20], raw[20:]]), bad),
            {"ok": False, "at": 2, "tenant": "t", "reason": "sequence"})
        bad_prev = {"tenant": "t", "seq": 2, "event": {}, "prev": "9" * 64}
        bad_prev["hash"] = AuditChain._hash(bad_prev)
        raw = good + record_bytes(bad_prev)
        self.assertEqual(
            self.other.verify_manifest_stream(iter([raw[:20], raw[20:]]), bad),
            {"ok": False, "at": 2, "tenant": "t", "reason": "digest"})

    def test_defect_stops_requesting_later_chunks(self):
        tampered = {"tenant": "t", "seq": 1, "event": {"z": 9}, "prev": ZERO}
        tampered["hash"] = AuditChain._hash(
            {"tenant": "t", "seq": 1, "event": {}, "prev": ZERO})
        pulled = []

        def gen():
            pulled.append(1)
            yield record_bytes(tampered)  # line 1 complete, digest defect
            pulled.append(2)
            yield b'{"tenant":"t","seq":2}\n'  # must never be requested
            pulled.append(3)

        r = self.other.verify_manifest_stream(gen(), empty_manifest())
        self.assertEqual((r["ok"], r["at"], r["tenant"], r["reason"]),
                         (False, 1, "t", "digest"))
        self.assertEqual(pulled, [1])

    def test_valid_stream_is_consumed_to_the_end(self):
        self.seed()
        data = self.snapshot()
        m = self.expected_manifest(data)
        pulled = []

        def gen():
            for i, part in enumerate((data[:9], b"", data[9:50], data[50:]), 1):
                pulled.append(i)
                yield part

        self.assertTrue(self.other.verify_manifest_stream(gen(), m)["ok"])
        self.assertEqual(pulled, [1, 2, 3, 4])

    def test_iterator_exception_propagates_verbatim(self):
        class Boom(Exception):
            pass

        def gen():
            yield self.valid_row("t", 1)
            raise Boom("producer failed")

        with self.assertRaises(Boom):
            self.other.verify_manifest_stream(gen(), empty_manifest())

    # --- manifest comparison: fixed field order and locators ---

    def test_manifest_mismatch_verdicts_match_bytes_entry(self):
        self.seed()
        data = self.snapshot()
        good = self.expected_manifest(data)
        cases = [
            dict(good, byte_length=good["byte_length"] + 1),
            dict(good, byte_length=good["byte_length"] - 1),
            dict(good, byte_sha256="f" * 64),
            dict(good, byte_length=good["byte_length"] + 1,
                 byte_sha256="f" * 64),  # length wins over sha256
            dict(good, tenants=list(reversed(good["tenants"]))),
            dict(good, tenants=good["tenants"] + [
                {"tenant": "zzz", "count": 0, "hash": ZERO}]),
            dict(good, tenants=good["tenants"][:1]),
            dict(good, tenants=[dict(good["tenants"][0], count=1),
                                good["tenants"][1]]),
            dict(good, tenants=[good["tenants"][0],
                                dict(good["tenants"][1], hash="2" * 64)]),
            empty_manifest(),
        ]
        for m in cases:
            chunks = [data[:11], b"", data[11:]]
            expected = self.other.verify_manifest_bytes(data, m)
            self.assertFalse(expected["ok"], m)
            self.assertEqual(expected["reason"], "manifest", m)
            self.assertEqual(
                self.other.verify_manifest_stream(iter(chunks), m),
                expected, m)

    def test_fixed_comparison_order_and_field_set(self):
        self.seed()
        data = self.snapshot()
        good = self.expected_manifest(data)
        r = self.other.verify_manifest_stream(
            iter([data]), dict(good, byte_length=good["byte_length"] + 5))
        self.assertEqual(r, {"ok": False, "reason": "manifest",
                             "field": "byte_length",
                             "expected": good["byte_length"] + 5,
                             "actual": good["byte_length"]})
        r = self.other.verify_manifest_stream(
            iter([data]), dict(good, byte_sha256="f" * 64))
        self.assertEqual(r, {"ok": False, "reason": "manifest",
                             "field": "byte_sha256",
                             "expected": "f" * 64,
                             "actual": good["byte_sha256"]})
        r = self.other.verify_manifest_stream(
            iter([data]), dict(good, tenants=list(reversed(good["tenants"]))))
        self.assertEqual(r, {"ok": False, "reason": "manifest",
                             "field": "tenant_order", "index": 0,
                             "expected": "b", "actual": "a"})
        r = self.other.verify_manifest_stream(
            iter([data]),
            dict(good, tenants=[dict(good["tenants"][0], count=9),
                                good["tenants"][1]]))
        self.assertEqual(r, {"ok": False, "reason": "manifest",
                             "field": "tenant_count", "tenant": "a",
                             "expected": 9, "actual": 3})
        r = self.other.verify_manifest_stream(
            iter([data]),
            dict(good, tenants=[good["tenants"][0],
                                dict(good["tenants"][1], hash="3" * 64)]))
        self.assertEqual(r, {"ok": False, "reason": "manifest",
                             "field": "tenant_hash", "tenant": "b",
                             "expected": "3" * 64,
                             "actual": good["tenants"][1]["hash"]})

    # --- boundaries ---

    def test_manifest_boundary_matches_verify_manifest(self):
        self.seed()
        data = self.snapshot()
        good = self.expected_manifest(data)
        bad_manifests = [
            None, [], "x", 1,
            dict(good, extra=1),
            {k: v for k, v in good.items() if k != "tenants"},
            dict(good, version=2),
            dict(good, version=True),
            dict(good, byte_length=-1),
            dict(good, byte_length=1.0),
            dict(good, byte_sha256="F" * 64),
            dict(good, byte_sha256="f" * 63),
            dict(good, tenants={}),
            dict(good, tenants=[{"tenant": "a", "count": 3}]),
            dict(good, tenants=[{"tenant": float("nan"), "count": 1,
                                 "hash": ZERO}]),
            dict(good, tenants=[{"tenant": "a", "count": True,
                                 "hash": ZERO}]),
            dict(good, tenants=[{"tenant": "a", "count": 1, "hash": "z" * 64}]),
            dict(good, tenants=[{"tenant": {"x": 1, "y": 2}, "count": 1,
                                 "hash": ZERO},
                                {"tenant": {"y": 2, "x": 1}, "count": 1,
                                 "hash": ZERO}]),
        ]
        for m in bad_manifests:
            with self.assertRaises(ValueError, msg=repr(m)):
                self.other.verify_manifest_stream(iter([data]), m)

    def test_manifest_validated_before_any_chunk_is_pulled(self):
        pulled = []

        def gen():
            pulled.append(1)
            yield b""

        for bad in (None, {"version": 2}, empty_manifest() | {"version": 2}):
            with self.assertRaises(ValueError):
                self.other.verify_manifest_stream(gen(), bad)
        self.assertEqual(pulled, [])

    def test_bare_bytes_container_is_value_error(self):
        m = empty_manifest()
        for bad in (b"", b"{}", bytearray(b""), bytearray(b"{}")):
            with self.assertRaises(ValueError):
                self.other.verify_manifest_stream(bad, m)

    def test_non_iterable_container_is_value_error(self):
        m = empty_manifest()
        for bad in (1, None, object()):
            with self.assertRaises(ValueError):
                self.other.verify_manifest_stream(bad, m)

    def test_non_bytes_member_is_value_error(self):
        m = empty_manifest()
        for bad in ([b"", "x"], [b"", 1], [b"", None], [b"", bytearray(b"")],
                    [b"", [b""]], ["x"], [bytearray(b"")]):
            with self.assertRaises(ValueError):
                self.other.verify_manifest_stream(iter(bad), m)

    def test_bad_member_stops_consumption(self):
        pulled = []

        def gen():
            pulled.append(1)
            yield b""
            pulled.append(2)
            yield "not bytes"
            pulled.append(3)  # must never be reached
            yield b""

        with self.assertRaises(ValueError):
            self.other.verify_manifest_stream(gen(), empty_manifest())
        self.assertEqual(pulled, [1, 2])

    # --- offline purity ---

    def test_never_reads_creates_or_modifies_constructor_path(self):
        phantom = self.path.with_name("phantom.jsonl")
        chain = AuditChain(phantom)
        m = empty_manifest()
        self.assertEqual(chain.verify_manifest_stream(iter([]), m),
                         {"ok": True, "manifest": m})
        self.assertFalse(phantom.exists())
        self.seed()
        before = self.snapshot()
        good = self.expected_manifest(before)
        self.assertTrue(chain.verify_manifest_stream(
            iter([before[:9], before[9:]]), good)["ok"])
        self.assertEqual(self.snapshot(), before)
        self.assertFalse(phantom.exists())

    def test_does_not_mutate_chunks_or_manifest(self):
        self.seed()
        data = self.snapshot()
        m = self.expected_manifest(data)
        snapshot_of_m = json.dumps(m, sort_keys=True)
        chunks = [data[:5], b"", data[5:]]
        self.other.verify_manifest_stream(iter(chunks), m)
        self.assertEqual(b"".join(chunks), data)
        self.assertEqual(json.dumps(m, sort_keys=True), snapshot_of_m)


if __name__ == "__main__":
    unittest.main()
