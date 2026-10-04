import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app import AuditChain

ZERO = "0" * 64


def record_bytes(item):
    return (json.dumps(item, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


class HeadsOfflineTest(unittest.TestCase):
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

    # --- offline purity ---

    def test_never_reads_creates_or_modifies_constructor_path(self):
        phantom = self.path.with_name("phantom.jsonl")
        chain = AuditChain(phantom)
        self.assertEqual(chain.heads_bytes(b""),
                         {"ok": True, "tenants": []})
        self.assertEqual(chain.heads_chunks([]),
                         {"ok": True, "tenants": []})
        self.assertFalse(phantom.exists())
        self.seed()
        before = self.snapshot()
        self.assertTrue(chain.heads_bytes(before)["ok"])
        self.assertTrue(chain.heads_chunks([before[:7], b"", before[7:]])["ok"])
        self.assertEqual(self.snapshot(), before)
        self.assertFalse(phantom.exists())

    def test_does_not_mutate_inputs(self):
        self.seed()
        data = self.snapshot()
        chunks = [data[:9], b"", data[9:]]
        self.other.heads_bytes(data)
        self.other.heads_chunks(chunks)
        self.assertEqual(data, self.snapshot())
        self.assertEqual(b"".join(chunks), data)

    # --- success shape and equivalence with heads() ---

    def test_heads_bytes_matches_heads(self):
        self.seed()
        data = self.snapshot()
        self.assertEqual(self.other.heads_bytes(data), self.chain.heads())
        self.assertEqual(self.other.heads_bytes(data), {
            "ok": True,
            "tenants": [
                {"tenant": "a", "count": 3,
                 "hash": self.chain.head("a")["hash"]},
                {"tenant": "b", "count": 2,
                 "hash": self.chain.head("b")["hash"]},
            ],
        })

    def test_empty_snapshot_is_empty_history(self):
        self.assertEqual(self.other.heads_bytes(b""),
                         {"ok": True, "tenants": []})
        for chunks in ([], (), iter([]), [b""], [b"", b""]):
            self.assertEqual(self.other.heads_chunks(chunks),
                             {"ok": True, "tenants": []})

    def test_first_appearance_physical_order_with_original_tenant(self):
        for tenant, event in (({"z": 1}, 0), ("a", 0), ([1, 2], 0),
                              ({"z": 1}, 1), ("a", 1), (None, 0), ([1, 2], 1)):
            self.chain.append(tenant, {"i": event})
        data = self.snapshot()
        result = self.other.heads_bytes(data)
        self.assertTrue(result["ok"])
        self.assertEqual([t["tenant"] for t in result["tenants"]],
                         [{"z": 1}, "a", [1, 2], None])
        self.assertEqual([t["count"] for t in result["tenants"]],
                         [2, 2, 2, 1])
        for entry in result["tenants"]:
            self.assertEqual(
                entry["hash"],
                self.chain.head(entry["tenant"])["hash"])

    def test_distinct_json_identities_stay_partitioned(self):
        for t in (1, 1.0, True, "1"):
            self.chain.append(t, {})
            self.chain.append(t, {})
        data = self.snapshot()
        result = self.other.heads_bytes(data)
        self.assertEqual(
            [(t["tenant"], t["count"]) for t in result["tenants"]],
            [(1, 2), (1.0, 2), (True, 2), ("1", 2)])
        for entry in result["tenants"]:
            self.assertEqual(entry["hash"],
                             self.chain.head(entry["tenant"])["hash"])

    def test_object_tenant_key_order_normalized_but_echoed_verbatim(self):
        self.chain.append({"a": 1, "b": 2}, {})
        self.chain.append({"b": 2, "a": 1}, {})
        result = self.other.heads_bytes(self.snapshot())
        self.assertEqual(len(result["tenants"]), 1)
        self.assertEqual(result["tenants"][0]["tenant"], {"a": 1, "b": 2})
        self.assertEqual(result["tenants"][0]["count"], 2)

    def test_hash_and_count_are_directly_usable_for_conditional_append(self):
        self.seed()
        data = self.snapshot()
        directory = self.other.heads_bytes(data)
        # The reported head pair must be exactly what append_if_head asserts;
        # appending to a separate log carrying the same bytes with that pair
        # succeeds and the new tail then matches another offline heads pass.
        target_path = self.path.with_name("target.jsonl")
        target_path.write_bytes(data)
        target = AuditChain(target_path)
        for entry in directory["tenants"]:
            item = target.append_if_head(
                entry["tenant"], {"more": True},
                entry["count"], entry["hash"])
            self.assertEqual(item["seq"], entry["count"] + 1)
            self.assertEqual(item["prev"], entry["hash"])
        after = self.other.heads_bytes(target_path.read_bytes())
        self.assertEqual(
            {t["tenant"]: t["count"] for t in after["tenants"]},
            {"a": 4, "b": 3})

    # --- failure verdicts mirror verify_all_bytes exactly ---

    def test_missing_classification(self):
        row_no_hash = {"tenant": "t", "seq": 1, "event": {}, "prev": ZERO}
        cases = [
            (b"{not json\n", None),
            (b"\n", None),
            (b"[1,2]\n", None),
            (b'{"seq":1}\n', None),
            ((json.dumps(row_no_hash) + "\n").encode(), "t"),
            (b'{"tenant":"t","tenant":"t","seq":1,"event":{},'
             b'"prev":"' + ZERO.encode() + b'","hash":"x"}\n', None),
            (b'{"tenant":"t","seq":1e999,"event":{},"prev":"'
             + ZERO.encode() + b'","hash":"x"}\n', None),
            (b"\xff", None),
        ]
        for raw, tenant in cases:
            self.assertEqual(
                self.other.heads_bytes(raw),
                self.other.verify_all_bytes(raw), raw)
            r = self.other.heads_bytes(raw)
            self.assertEqual(
                (r["ok"], r["at"], r["tenant"], r["reason"]),
                (False, 1, tenant, "missing"), raw)
            self.assertNotIn("tenants", r)

    def test_bad_utf8_physical_line_number(self):
        cases = [
            (self.valid_row() + b'{"tenant":\xff}', 2),
            (self.valid_row("x") + self.valid_row("y") + b"abc\xc2\n", 3),
        ]
        for raw, at in cases:
            r = self.other.heads_bytes(raw)
            self.assertEqual(
                (r["ok"], r["at"], r["tenant"], r["reason"]),
                (False, at, None, "missing"), raw)

    def test_sequence_and_digest_classification(self):
        good = self.valid_row("t", 1)
        gap_row = {"tenant": "t", "seq": 3, "event": {}, "prev": "x" * 64}
        gap_row["hash"] = AuditChain._hash(gap_row)
        raw = good + record_bytes(gap_row)
        self.assertEqual(
            self.other.heads_bytes(raw),
            {"ok": False, "at": 2, "tenant": "t", "reason": "sequence"})

        bad_prev = {"tenant": "t", "seq": 2, "event": {}, "prev": "9" * 64}
        bad_prev["hash"] = AuditChain._hash(bad_prev)
        raw = good + record_bytes(bad_prev)
        self.assertEqual(
            self.other.heads_bytes(raw),
            {"ok": False, "at": 2, "tenant": "t", "reason": "digest"})

    def test_first_error_wins_and_names_its_tenant(self):
        good_a = {"tenant": "a", "seq": 1, "event": {}, "prev": ZERO}
        good_a["hash"] = AuditChain._hash(good_a)
        bad_b = {"tenant": "b", "seq": 1, "event": {}, "prev": "f" * 64}
        bad_b["hash"] = AuditChain._hash(bad_b)
        raw = record_bytes(good_a) + record_bytes(bad_b)
        self.assertEqual(
            self.other.heads_bytes(raw),
            {"ok": False, "at": 2, "tenant": "b", "reason": "digest"})

    # --- heads_bytes boundary ---

    def test_data_must_be_bytes(self):
        for bad in ("", "{}", bytearray(b""), bytearray(b"{}"), 1, None,
                    [b""]):
            with self.assertRaises(ValueError):
                self.other.heads_bytes(bad)

    # --- heads_chunks equivalence and boundary ---

    def test_heads_chunks_matches_heads_bytes_for_many_chunkings(self):
        self.seed()
        data = self.snapshot()
        chunkings = [
            [data],
            [data[:1], data[1:]],
            [data[i:i + 1] for i in range(len(data))],
            [data[:10], b"", data[10:37], b"", data[37:]],
            [b"", b"", data, b""],
            [data[i:i + 7] for i in range(0, len(data), 7)],
        ]
        for chunks in chunkings:
            self.assertEqual(self.other.heads_chunks(chunks),
                             self.other.heads_bytes(data), chunks)

    def test_chunk_may_split_multibyte_utf8_json_or_lf(self):
        self.chain.append("t", {"msg": "héllo→世界"})
        self.chain.append("t", {"msg": "✓" * 40})
        data = self.snapshot()
        for cut in range(len(data) + 1):
            chunks = [data[:cut], data[cut:]]
            self.assertEqual(self.other.heads_chunks(chunks),
                             self.other.heads_bytes(data), cut)

    def test_accepts_any_iterable_container(self):
        self.seed()
        data = self.snapshot()
        halves = [data[:len(data) // 2], data[len(data) // 2:]]
        for container in (halves, tuple(halves), iter(halves),
                          (c for c in halves)):
            self.assertEqual(self.other.heads_chunks(container),
                             self.other.heads_bytes(data))

    def test_chunks_failure_matches_bytes_failure(self):
        good = self.valid_row("t", 1)
        cases = [
            b"{not json\n",
            good + b"\xff\n",
            good + self.valid_row("t", seq=3, prev="x" * 64),
        ]
        for raw in cases:
            chunks = [raw[:3], b"", raw[3:len(raw) // 2], raw[len(raw) // 2:]]
            self.assertEqual(self.other.heads_chunks(chunks),
                             self.other.heads_bytes(raw), raw)

    def test_bare_bytes_container_is_value_error(self):
        for bad in (b"", b"{}", bytearray(b""), bytearray(b"{}")):
            with self.assertRaises(ValueError):
                self.other.heads_chunks(bad)

    def test_non_iterable_container_is_value_error(self):
        for bad in (1, None, object()):
            with self.assertRaises(ValueError):
                self.other.heads_chunks(bad)

    def test_first_non_bytes_member_is_value_error(self):
        for bad in ([b"", "x"], [b"", 1], [b"", None], [b"", bytearray(b"")],
                    [b"", [b""]], ["x"], [bytearray(b"")]):
            with self.assertRaises(ValueError):
                self.other.heads_chunks(bad)

    def test_bad_member_stops_consumption_immediately(self):
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


if __name__ == "__main__":
    unittest.main()
