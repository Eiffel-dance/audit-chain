import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app import AuditChain

ZERO = "0" * 64


def record_bytes(item):
    return (json.dumps(item, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


class HeadsBytesChunksTest(unittest.TestCase):
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

    # --- success shape: identical to heads() ---

    def test_matches_heads_on_interleaved_log(self):
        self.seed()
        data = self.snapshot()
        self.assertEqual(self.other.heads_bytes(data), self.chain.heads())
        r = self.other.heads_bytes(data)
        self.assertTrue(r["ok"])
        self.assertEqual(r, {"ok": True, "tenants": [
            {"tenant": "a", "count": 3, "hash": r["tenants"][0]["hash"]},
            {"tenant": "b", "count": 1, "hash": r["tenants"][1]["hash"]},
        ]})

    def test_first_appearance_order_counts_and_hashes(self):
        b1 = self.chain.append("b", {})
        a1 = self.chain.append("a", {})
        b2 = self.chain.append("b", {})
        a2 = self.chain.append("a", {})
        c1 = self.chain.append("c", {})
        r = self.other.heads_bytes(self.snapshot())
        self.assertEqual(r, {"ok": True, "tenants": [
            {"tenant": "b", "count": 2, "hash": b2["hash"]},
            {"tenant": "a", "count": 2, "hash": a2["hash"]},
            {"tenant": "c", "count": 1, "hash": c1["hash"]},
        ]})
        self.assertNotEqual(b1["hash"], b2["hash"])
        self.assertEqual(a1["hash"], a1["hash"])

    def test_tenant_types_kept_distinct_and_verbatim(self):
        for t in (1, 1.0, True, "1"):
            self.chain.append(t, {})
        data = self.snapshot()
        r = self.other.heads_bytes(data)
        self.assertEqual([e["tenant"] for e in r["tenants"]],
                         [1, 1.0, True, "1"])
        self.assertTrue(all(len(e["hash"]) == 64 for e in r["tenants"]))

    def test_result_feeds_offline_conditional_append(self):
        self.seed()
        r = self.other.heads_bytes(self.snapshot())
        for e in r["tenants"]:
            item = self.chain.append_if_head(
                e["tenant"], {"more": True}, e["count"], e["hash"])
            self.assertEqual(item["seq"], e["count"] + 1)
            self.assertEqual(item["prev"], e["hash"])

    def test_empty_snapshot_is_empty_tenant_list(self):
        self.assertEqual(self.other.heads_bytes(b""),
                         {"ok": True, "tenants": []})
        for chunks in ([], (), iter([]), [b""], [b"", b""]):
            self.assertEqual(self.other.heads_chunks(chunks),
                             {"ok": True, "tenants": []})

    # --- chunks: field-for-field equal to heads_bytes ---

    def test_heads_chunks_matches_heads_bytes_for_many_chunkings(self):
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
            self.assertEqual(self.other.heads_chunks(chunks),
                             self.other.heads_bytes(data), chunks)

    def test_chunk_may_split_multibyte_utf8(self):
        self.chain.append("t", {"msg": "héllo→世界"})
        self.chain.append("t", {"msg": "✓" * 40})
        data = self.snapshot()
        for cut in range(len(data) + 1):
            chunks = [data[:cut], data[cut:]]
            self.assertEqual(self.other.heads_chunks(chunks),
                             self.other.heads_bytes(data), cut)

    def test_chunk_split_through_json_and_lf_keeps_semantics(self):
        self.seed()
        data = self.snapshot()
        # cut at every single byte offset; semantics never change
        for cut in range(len(data) + 1):
            self.assertEqual(self.other.heads_chunks([data[:cut], data[cut:]]),
                             self.other.heads_bytes(data), cut)

    def test_accepts_any_iterable_container(self):
        self.seed()
        data = self.snapshot()
        halves = [data[:len(data) // 2], data[len(data) // 2:]]
        for container in (halves, tuple(halves), iter(halves),
                          (c for c in halves)):
            self.assertEqual(self.other.heads_chunks(container),
                             self.other.heads_bytes(data))

    # --- failures: verify_all_bytes's exact object, no partial heads ---

    def test_missing_classifications_and_physical_lines(self):
        good = self.valid_row("x")
        cases = [
            (b"{not json\n", 1, None),
            (b"\n", 1, None),
            (b"[1,2]\n", 1, None),
            (good + b"{not json\n", 2, None),
            (good + b"\xff\n", 2, None),
        ]
        for raw, at, tenant in cases:
            chunks = [raw[:3], b"", raw[3:]]
            expected = {"ok": False, "at": at, "tenant": tenant,
                        "reason": "missing"}
            self.assertEqual(self.other.heads_bytes(raw), expected, raw)
            self.assertEqual(self.other.heads_chunks(chunks), expected, raw)

    def test_missing_field_reports_tenant(self):
        row = {"tenant": "t", "seq": 1, "event": {}, "prev": ZERO}
        raw = (json.dumps(row) + "\n").encode()
        expected = {"ok": False, "at": 1, "tenant": "t", "reason": "missing"}
        self.assertEqual(self.other.heads_bytes(raw), expected)
        self.assertEqual(self.other.heads_chunks([raw[:4], raw[4:]]), expected)

    def test_duplicate_key_and_non_standard_number_are_missing(self):
        # the whole line fails to parse, so the tenant cannot be determined:
        # the same tenant=None verdict verify_all_bytes gives
        for raw in (b'{"tenant":"t","tenant":"t","seq":1,"event":{},'
                    b'"prev":"' + ZERO.encode() + b'","hash":"x"}\n',
                    b'{"tenant":"t","seq":1e999,"event":{},"prev":"'
                    + ZERO.encode() + b'","hash":"x"}\n'):
            self.assertEqual(self.other.heads_bytes(raw),
                             {"ok": False, "at": 1, "tenant": None,
                              "reason": "missing"}, raw)
            self.assertEqual(self.other.heads_bytes(raw),
                             self.other.verify_all_bytes(raw), raw)

    def test_sequence_error_reports_expected_seq(self):
        self.chain.append("a", {})
        row = {"tenant": "b", "seq": 2, "event": {}, "prev": ZERO}
        row["hash"] = AuditChain._hash(row)
        raw = self.snapshot() + record_bytes(row)
        self.assertEqual(self.other.heads_bytes(raw),
                         {"ok": False, "at": 2, "tenant": "b",
                          "reason": "sequence"})

    def test_digest_error_stops_at_first_physical_line(self):
        good = {"tenant": "a", "seq": 1, "event": {}, "prev": ZERO}
        good["hash"] = AuditChain._hash(good)
        bad = {"tenant": "b", "seq": 1, "event": {}, "prev": "f" * 64}
        bad["hash"] = AuditChain._hash(bad)
        raw = record_bytes(good) + record_bytes(bad)
        expected = {"ok": False, "at": 2, "tenant": "b", "reason": "digest"}
        self.assertEqual(self.other.heads_bytes(raw), expected)
        self.assertEqual(self.other.heads_chunks([raw[:11], raw[11:]]),
                         expected)

    def test_failure_equals_verify_all_bytes_with_no_partial_tenants(self):
        good = self.valid_row("a")
        tampered = {"tenant": "t", "seq": 1, "event": {"z": 9}, "prev": ZERO}
        tampered["hash"] = AuditChain._hash(
            {"tenant": "t", "seq": 1, "event": {}, "prev": ZERO})
        raw = good + record_bytes(tampered)
        r = self.other.heads_bytes(raw)
        self.assertFalse(r["ok"])
        self.assertNotIn("tenants", r)
        self.assertEqual(set(r), {"ok", "at", "tenant", "reason"})
        self.assertEqual(r, self.other.verify_all_bytes(raw))

    def test_heads_bytes_matches_heads_on_corrupt_snapshot(self):
        self.chain.append("t", {})
        rows = [json.loads(l) for l in self.snapshot().decode().splitlines()]
        rows[0]["event"] = {"v": 2}
        with self.path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, sort_keys=True) + "\n")
        data = self.snapshot()
        self.assertEqual(self.other.heads_bytes(data), self.chain.heads())
        self.assertEqual(self.other.heads_bytes(data),
                         self.other.verify_all_bytes(data))

    # --- argument boundary ---

    def test_heads_bytes_data_must_be_bytes(self):
        for bad in ('{"x":1}', bytearray(b""), bytearray(b"{}"),
                    1, None, [b""]):
            with self.assertRaises(ValueError):
                self.other.heads_bytes(bad)

    def test_bare_bytes_container_is_value_error(self):
        for bad in (b"", b"{}", bytearray(b"")):
            with self.assertRaises(ValueError):
                self.other.heads_chunks(bad)

    def test_non_iterable_container_is_value_error(self):
        for bad in (1, None, object()):
            with self.assertRaises(ValueError):
                self.other.heads_chunks(bad)

    def test_non_bytes_element_is_value_error(self):
        for bad in ([b"", "x"], [b"", 1], [b"", None],
                    [b"", bytearray(b"")], [b"", [b""]]):
            with self.assertRaises(ValueError):
                self.other.heads_chunks(bad)

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
            self.other.heads_chunks(gen())
        self.assertEqual(pulled, [1, 2])

    # --- offline purity ---

    def test_never_reads_creates_or_modifies_constructor_path(self):
        phantom = self.path.with_name("phantom.jsonl")
        chain = AuditChain(phantom)
        self.assertEqual(chain.heads_bytes(b""), {"ok": True, "tenants": []})
        self.assertEqual(chain.heads_chunks([]), {"ok": True, "tenants": []})
        self.assertFalse(phantom.exists())
        self.seed()
        before = self.snapshot()
        chunks = [before[:9], before[9:]]
        self.assertTrue(chain.heads_bytes(before)["ok"])
        self.assertTrue(chain.heads_chunks(chunks)["ok"])
        self.assertEqual(self.snapshot(), before)
        self.assertFalse(phantom.exists())

    def test_does_not_mutate_inputs(self):
        self.seed()
        data = self.snapshot()
        chunks = [data[:5], data[5:]]
        self.other.heads_bytes(data)
        self.other.heads_chunks(chunks)
        self.assertEqual(data, self.snapshot())
        self.assertEqual(b"".join(chunks), data)


if __name__ == "__main__":
    unittest.main()
