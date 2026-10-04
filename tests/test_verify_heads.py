import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app import AuditChain, AuditChainConflictError

ZERO = "0" * 64


def record_bytes(item):
    return (json.dumps(item, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


class VerifyHeadsTest(unittest.TestCase):
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

    def directory(self):
        return [
            {"tenant": t["tenant"], "count": t["count"], "hash": t["hash"]}
            for t in self.chain.heads()["tenants"]
        ]

    def valid_row(self, tenant="t", seq=1, prev=ZERO, event=None):
        row = {"tenant": tenant, "seq": seq, "event": event or {}, "prev": prev}
        row["hash"] = AuditChain._hash(row)
        return record_bytes(row)

    # --- success: result is the heads success structure ---

    def test_file_entry_matches_heads_structure(self):
        self.seed()
        expected = self.directory()
        result = self.chain.verify_heads(expected)
        self.assertEqual(result, self.chain.heads())
        self.assertEqual(
            [t["tenant"] for t in result["tenants"]], ["a", "b"])
        self.assertEqual([t["count"] for t in result["tenants"]], [3, 2])

    def test_empty_log_and_empty_directory(self):
        self.assertEqual(self.chain.verify_heads([]),
                         {"ok": True, "tenants": []})
        self.assertFalse(self.path.exists())
        self.assertEqual(self.other.verify_heads_bytes(b"", []),
                         {"ok": True, "tenants": []})
        self.assertEqual(self.other.verify_heads_chunks([], []),
                         {"ok": True, "tenants": []})

    def test_bytes_and_chunks_entries_match_file_entry(self):
        self.seed()
        data = self.snapshot()
        expected = self.directory()
        self.assertEqual(self.other.verify_heads_bytes(data, expected),
                         self.chain.verify_heads(expected))
        chunks = [data[:5], b"", data[5:40], data[40:]]
        self.assertEqual(self.other.verify_heads_chunks(chunks, expected),
                         self.chain.verify_heads(expected))

    def test_asserting_empty_head_for_absent_tenant_succeeds(self):
        self.seed()
        expected = self.directory() + [
            {"tenant": "never-seen", "count": 0, "hash": ZERO}]
        result = self.chain.verify_heads(expected)
        self.assertTrue(result["ok"])
        self.assertEqual(
            [t["tenant"] for t in result["tenants"]], ["a", "b"])

    def test_original_tenant_values_preserved(self):
        for tenant in ({"z": 1}, "a", [1, 2], None):
            self.chain.append(tenant, {})
        expected = self.directory()
        result = self.other.verify_heads_bytes(self.snapshot(), expected)
        self.assertEqual([t["tenant"] for t in result["tenants"]],
                         [{"z": 1}, "a", [1, 2], None])

    # --- conflict semantics ---

    def test_wrong_count_conflicts_with_actual_tail(self):
        self.seed()
        expected = self.directory()
        expected[0]["count"] = 2
        with self.assertRaises(AuditChainConflictError) as ctx:
            self.chain.verify_heads(expected)
        err = ctx.exception
        self.assertEqual(err.reason, "conflict")
        self.assertEqual(err.tenant, "a")
        self.assertEqual(err.expected_count, 2)
        self.assertEqual(err.expected_hash, self.directory()[0]["hash"])
        self.assertEqual(err.actual_count, 3)
        self.assertEqual(err.actual_hash, self.directory()[0]["hash"])

    def test_wrong_hash_conflicts(self):
        self.seed()
        expected = self.directory()
        expected[1]["hash"] = "f" * 64
        with self.assertRaises(AuditChainConflictError) as ctx:
            self.chain.verify_heads(expected)
        err = ctx.exception
        self.assertEqual(err.tenant, "b")
        self.assertEqual(err.expected_hash, "f" * 64)
        self.assertEqual(err.actual_count, 2)
        self.assertEqual(err.actual_hash, self.directory()[1]["hash"])

    def test_missing_tenant_actual_is_zero_head(self):
        self.seed()
        expected = self.directory() + [
            {"tenant": "ghost", "count": 1, "hash": "a" * 64}]
        with self.assertRaises(AuditChainConflictError) as ctx:
            self.chain.verify_heads(expected)
        err = ctx.exception
        self.assertEqual(err.tenant, "ghost")
        self.assertEqual((err.expected_count, err.expected_hash),
                         (1, "a" * 64))
        self.assertEqual((err.actual_count, err.actual_hash), (0, ZERO))

    def test_unlisted_snapshot_tenant_conflicts_as_expected_empty(self):
        self.seed()
        expected = [self.directory()[0]]  # only "a"; "b" is not listed
        with self.assertRaises(AuditChainConflictError) as ctx:
            self.chain.verify_heads(expected)
        err = ctx.exception
        self.assertEqual(err.tenant, "b")
        self.assertEqual((err.expected_count, err.expected_hash), (0, ZERO))
        self.assertEqual(err.actual_count, 2)
        self.assertEqual(err.actual_hash, self.directory()[1]["hash"])

    def test_list_order_decides_conflict_priority(self):
        self.seed()
        expected = self.directory()
        expected[0]["count"] = 99       # "a" mismatches
        expected[1]["count"] = 98       # "b" mismatches too
        with self.assertRaises(AuditChainConflictError) as ctx:
            self.chain.verify_heads(expected)
        self.assertEqual(ctx.exception.tenant, "a")
        # Reversed order: "b" is now first.
        with self.assertRaises(AuditChainConflictError) as ctx:
            self.chain.verify_heads([expected[1], expected[0]])
        self.assertEqual(ctx.exception.tenant, "b")

    def test_conflict_never_writes_bytes(self):
        self.seed()
        before = self.snapshot()
        expected = self.directory()
        expected[0]["hash"] = "0" * 63 + "1"
        with self.assertRaises(AuditChainConflictError):
            self.chain.verify_heads(expected)
        self.assertEqual(self.snapshot(), before)
        phantom = self.path.with_name("phantom.jsonl")
        with self.assertRaises(AuditChainConflictError):
            AuditChain(phantom).verify_heads(
                [{"tenant": "t", "count": 1, "hash": "a" * 64}])
        self.assertFalse(phantom.exists())

    # --- corrupt snapshot: verify_all failure object, no comparison ---

    def test_corrupt_snapshot_returns_failure_object_not_conflict(self):
        good = self.valid_row("a", 1)
        bad_prev = {"tenant": "a", "seq": 2, "event": {}, "prev": "9" * 64}
        bad_prev["hash"] = AuditChain._hash(bad_prev)
        raw = good + record_bytes(bad_prev)
        expected = [{"tenant": "a", "count": 2, "hash": bad_prev["hash"]}]
        for result in (
            self.other.verify_heads_bytes(raw, expected),
            self.other.verify_heads_chunks([raw[:7], raw[7:]], expected),
        ):
            self.assertEqual(
                result,
                {"ok": False, "at": 2, "tenant": "a", "reason": "digest"})
            self.assertEqual(result, self.other.verify_all_bytes(raw))
            self.assertEqual(result, self.other.heads_bytes(raw))

    def test_corrupt_file_returns_failure_object(self):
        self.seed()
        expected = self.directory()
        with self.path.open("ab") as f:
            f.write(b"{not json\n")
        result = self.chain.verify_heads(expected)
        self.assertEqual(
            result,
            {"ok": False, "at": 6, "tenant": None, "reason": "missing"})
        self.assertEqual(result, self.chain.verify_all())

    def test_first_defect_wins_regardless_of_directory(self):
        raw = self.valid_row("a", 1) + b"\xff\n"
        # Even a directory that would conflict on "a" must not be consulted.
        expected = [{"tenant": "a", "count": 5, "hash": "b" * 64}]
        self.assertEqual(
            self.other.verify_heads_bytes(raw, expected),
            {"ok": False, "at": 2, "tenant": None, "reason": "missing"})

    # --- expected_heads boundary (ValueError before any read/parse) ---

    def test_expected_heads_must_be_a_list(self):
        for bad in (None, 1, "x", {}, ()):
            with self.assertRaises(ValueError):
                self.other.verify_heads_bytes(b"", bad)

    def test_member_shape_and_keys(self):
        for bad in ([None], [1], ["x"], [[]], [{}],
                    [{"tenant": "t", "count": 0}],
                    [{"tenant": "t", "count": 0, "hash": ZERO, "x": 1}],
                    [{"tenant": "t", "count": 0, "hash": ZERO, 1: 2}]):
            with self.assertRaises(ValueError):
                self.other.verify_heads_bytes(b"", bad)

    def test_tenant_standard_json_boundary(self):
        for bad_tenant in (float("nan"), float("inf"), {1: "x"}):
            with self.assertRaises(ValueError):
                self.other.verify_heads_bytes(
                    b"", [{"tenant": bad_tenant, "count": 0, "hash": ZERO}])

    def test_count_boundary(self):
        for bad_count in (True, False, -1, 1.0, "1", None, [0]):
            with self.assertRaises(ValueError):
                self.other.verify_heads_bytes(
                    b"", [{"tenant": "t", "count": bad_count, "hash": ZERO}])

    def test_hash_boundary(self):
        for bad_hash in ("", "0" * 63, "0" * 65, "A" * 64, "g" * 64,
                         0, None, ["0" * 64]):
            with self.assertRaises(ValueError):
                self.other.verify_heads_bytes(
                    b"", [{"tenant": "t", "count": 0, "hash": bad_hash}])

    def test_duplicate_tenant_identity_rejected(self):
        head = {"tenant": {"a": 1, "b": 2}, "count": 0, "hash": ZERO}
        same = {"tenant": {"b": 2, "a": 1}, "count": 0, "hash": ZERO}
        with self.assertRaises(ValueError):
            self.other.verify_heads_bytes(b"", [head, same])
        with self.assertRaises(ValueError):
            self.other.verify_heads_bytes(b"", [head, head])
        # Distinct JSON identities are distinct tenants, not duplicates.
        heads = [{"tenant": t, "count": 0, "hash": ZERO}
                 for t in (1, 1.0, True, "1")]
        self.assertEqual(self.other.verify_heads_bytes(b"", heads),
                         {"ok": True, "tenants": []})

    def test_boundary_errors_precede_any_read_or_parse(self):
        # A corrupt snapshot plus an illegal directory is still ValueError.
        raw = b"{not json\n"
        with self.assertRaises(ValueError):
            self.other.verify_heads_bytes(raw, "not a list")
        with self.assertRaises(ValueError):
            self.other.verify_heads(raw)  # data not bytes at all
        # File entry: illegal directory must not even create/touch the log.
        phantom = self.path.with_name("boundary.jsonl")
        chain = AuditChain(phantom)
        with self.assertRaises(ValueError):
            chain.verify_heads([{"tenant": "t", "count": -1, "hash": ZERO}])
        self.assertFalse(phantom.exists())

    # --- bytes/chunks entry boundaries and offline purity ---

    def test_data_must_be_bytes(self):
        for bad in ("", "{}", bytearray(b""), 1, None, [b""]):
            with self.assertRaises(ValueError):
                self.other.verify_heads_bytes(bad, [])

    def test_chunks_container_boundary(self):
        for bad in (b"", b"{}", bytearray(b""), 1, None, object()):
            with self.assertRaises(ValueError):
                self.other.verify_heads_chunks(bad, [])

    def test_chunks_member_boundary(self):
        for bad in ([b"", "x"], [b"", bytearray(b"")], ["x"], [b"", None]):
            with self.assertRaises(ValueError):
                self.other.verify_heads_chunks(bad, [])

    def test_offline_entries_never_touch_constructor_path(self):
        phantom = self.path.with_name("offline.jsonl")
        chain = AuditChain(phantom)
        self.seed()
        data = self.snapshot()
        expected = self.directory()
        chain.verify_heads_bytes(data, expected)
        chain.verify_heads_chunks([data[:3], b"", data[3:]], expected)
        self.assertFalse(phantom.exists())
        self.assertEqual(self.snapshot(), data)

    def test_chunks_equivalence_for_many_chunkings(self):
        self.seed()
        data = self.snapshot()
        expected = self.directory()
        want = self.other.verify_heads_bytes(data, expected)
        chunkings = [
            [data],
            [data[:1], data[1:]],
            [data[i:i + 1] for i in range(len(data))],
            [data[:10], b"", data[10:37], b"", data[37:]],
            iter([data[:len(data) // 2], data[len(data) // 2:]]),
        ]
        for chunks in chunkings:
            self.assertEqual(
                self.other.verify_heads_chunks(chunks, expected), want)


if __name__ == "__main__":
    unittest.main()
