import copy
import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app import AuditChain, AuditChainStateError, MANIFEST_VERSION, ZERO


def record_bytes(item):
    return (json.dumps(item, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def empty_manifest():
    return {
        "version": MANIFEST_VERSION,
        "byte_length": 0,
        "byte_sha256": hashlib.sha256(b"").hexdigest(),
        "tenants": [],
    }


class ManifestTest(unittest.TestCase):
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

    # --- generation shape ---

    def test_manifest_matches_bytes_and_heads(self):
        self.seed()
        data = self.snapshot()
        m = self.chain.manifest()
        self.assertEqual(m, self.expected_manifest(data))
        self.assertEqual(m, self.other.manifest_bytes(data))
        # byte-level fields describe the exact snapshot bytes.
        self.assertEqual(m["byte_length"], len(data))
        self.assertEqual(m["byte_sha256"], hashlib.sha256(data).hexdigest())
        # tenants keep physical first-appearance order, count and last hash.
        self.assertEqual([t["tenant"] for t in m["tenants"]], ["a", "b"])
        self.assertEqual([t["count"] for t in m["tenants"]], [3, 2])

    def test_manifest_tenants_in_physical_first_appearance_order(self):
        for tenant, event in (({"z": 1}, 0), ("a", 0), ([1, 2], 0),
                              ({"z": 1}, 1), ("a", 1), (None, 0)):
            self.chain.append(tenant, {"i": event})
        data = self.snapshot()
        m = self.other.manifest_bytes(data)
        self.assertEqual([t["tenant"] for t in m["tenants"]],
                         [{"z": 1}, "a", [1, 2], None])
        self.assertEqual([t["count"] for t in m["tenants"]], [2, 2, 1, 1])
        for entry in m["tenants"]:
            self.assertEqual(entry["hash"],
                             self.chain.head(entry["tenant"])["hash"])

    def test_distinct_json_identities_stay_partitioned(self):
        for t in (1, 1.0, True, "1"):
            self.chain.append(t, {})
            self.chain.append(t, {})
        m = self.other.manifest_bytes(self.snapshot())
        self.assertEqual(
            [(t["tenant"], t["count"]) for t in m["tenants"]],
            [(1, 2), (1.0, 2), (True, 2), ("1", 2)])

    def test_empty_snapshot_yields_empty_manifest(self):
        for generated in (self.chain.manifest(),
                         self.other.manifest_bytes(b""),
                         self.other.manifest_chunks([]),
                         self.other.manifest_chunks(iter([])),
                         self.other.manifest_chunks([b"", b""])):
            self.assertEqual(generated, empty_manifest())
        # A missing log is a legitimate empty snapshot and creates nothing.
        phantom = self.path.with_name("phantom.jsonl")
        self.assertEqual(AuditChain(phantom).manifest(), empty_manifest())
        self.assertFalse(phantom.exists())

    def test_manifest_chunks_matches_manifest_bytes_for_every_split(self):
        self.chain.append("t", {"msg": "héllo→世界"})
        self.chain.append("t", {"msg": "✓" * 40})
        data = self.snapshot()
        m = self.other.manifest_bytes(data)
        for chunking in ([data], [data[:1], data[1:]],
                         [data[i:i + 1] for i in range(len(data))],
                         [data[:10], b"", data[10:37], b"", data[37:]],
                         [data[i:i + 7] for i in range(0, len(data), 7)]):
            self.assertEqual(
                self.other.manifest_chunks(chunking), m, chunking)

    def test_generation_is_read_only(self):
        self.seed()
        before = self.snapshot()
        self.chain.manifest()
        self.other.manifest_bytes(before)
        self.other.manifest_chunks([before[:9], b"", before[9:]])
        self.assertEqual(self.snapshot(), before)
        self.assertFalse(self.path.with_name("never-touched.jsonl").exists())

    def test_generation_does_not_mutate_input(self):
        self.seed()
        data = self.snapshot()
        chunks = [data[:9], b"", data[9:]]
        self.other.manifest_bytes(data)
        self.other.manifest_chunks(chunks)
        self.assertEqual(data, self.snapshot())
        self.assertEqual(b"".join(chunks), data)

    # --- generation on corruption: AuditChainStateError, no partial result ---

    def test_manifest_bytes_corrupt_raises_state_error(self):
        row_no_hash = {"tenant": "t", "seq": 1, "event": {}, "prev": ZERO}
        cases = [
            (b"{not json\n", None, None, "missing"),
            (b"\n", None, None, "missing"),
            (b"\xff", None, None, "missing"),
            ((json.dumps(row_no_hash) + "\n").encode(), "t", 1, "missing"),
        ]
        for raw, tenant, seq, reason in cases:
            with self.assertRaises(AuditChainStateError) as ctx:
                self.other.manifest_bytes(raw)
            err = ctx.exception
            self.assertEqual(err.reason, reason, raw)
            self.assertEqual(err.tenant, tenant, raw)
            if seq is not None:
                self.assertEqual(err.seq, seq, raw)

    def test_manifest_sequence_and_digest_corruption_raises(self):
        good = self.valid_row("t", 1)
        gap = {"tenant": "t", "seq": 3, "event": {}, "prev": "1" * 64}
        gap["hash"] = AuditChain._hash(gap)
        with self.assertRaises(AuditChainStateError) as ctx:
            self.other.manifest_bytes(good + record_bytes(gap))
        self.assertEqual((ctx.exception.tenant, ctx.exception.seq,
                          ctx.exception.reason, ctx.exception.line),
                         ("t", 2, "sequence", 2))
        bad_prev = {"tenant": "t", "seq": 2, "event": {}, "prev": "9" * 64}
        bad_prev["hash"] = AuditChain._hash(bad_prev)
        with self.assertRaises(AuditChainStateError) as ctx:
            self.other.manifest_bytes(good + record_bytes(bad_prev))
        self.assertEqual(ctx.exception.reason, "digest")

    def test_manifest_chunks_corruption_matches_bytes(self):
        raw = self.valid_row("t", 1) + b"\xff\n"
        with self.assertRaises(AuditChainStateError) as e1:
            self.other.manifest_bytes(raw)
        with self.assertRaises(AuditChainStateError) as e2:
            self.other.manifest_chunks([raw[:3], b"", raw[3:]])
        for attr in ("tenant", "seq", "reason", "line"):
            self.assertEqual(getattr(e1.exception, attr),
                             getattr(e2.exception, attr), attr)

    def test_manifest_file_corrupt_raises(self):
        self.path.write_bytes(b"{broken\n")
        with self.assertRaises(AuditChainStateError) as ctx:
            self.chain.manifest()
        self.assertEqual((ctx.exception.tenant, ctx.exception.seq,
                          ctx.exception.reason, ctx.exception.line),
                         (None, None, "missing", 1))
        self.assertEqual(self.snapshot(), b"{broken\n")

    # --- verification success ---

    def test_verify_manifest_success_echoes_actual_manifest(self):
        self.seed()
        data = self.snapshot()
        m = self.expected_manifest(data)
        self.assertEqual(
            self.other.verify_manifest_bytes(data, m),
            {"ok": True, "manifest": self.other.manifest_bytes(data)})
        self.assertEqual(self.chain.verify_manifest(m),
                         {"ok": True, "manifest": m})
        for chunking in ([data], [data[:1], data[1:]],
                         [data[i:i + 1] for i in range(len(data))]):
            self.assertEqual(
                self.other.verify_manifest_chunks(chunking, m),
                {"ok": True, "manifest": m}, chunking)

    def test_verify_empty_manifest_against_empty_snapshot(self):
        m = empty_manifest()
        for result in (self.other.verify_manifest_bytes(b"", m),
                       self.other.verify_manifest_chunks([], m),
                       self.other.verify_manifest_chunks(iter([]), m),
                       self.other.verify_manifest_chunks([b"", b""], m)):
            self.assertEqual(result, {"ok": True, "manifest": m})
        phantom = self.path.with_name("phantom.jsonl")
        chain = AuditChain(phantom)
        self.assertEqual(chain.verify_manifest(m), {"ok": True, "manifest": m})
        self.assertFalse(phantom.exists())

    def test_round_trip_generate_then_verify_for_every_append(self):
        manifests = []
        for i in range(6):
            self.chain.append("a" if i % 2 == 0 else "b", {"i": i})
            data = self.snapshot()
            m = self.other.manifest_bytes(data)
            manifests.append(m)
            self.assertTrue(self.other.verify_manifest_bytes(data, m)["ok"])
            self.assertTrue(
                self.other.verify_manifest_chunks([data[:3], data[3:]], m)
                ["ok"])
        # Every historical snapshot still verifies against its own manifest.
        for m in manifests:
            self.assertEqual(m["version"], 1)

    # --- chain corruption beats the manifest comparison ---

    def test_corrupt_snapshot_returns_verify_all_failure_object(self):
        row_no_hash = {"tenant": "t", "seq": 1, "event": {}, "prev": ZERO}
        cases = [
            (b"{not json\n", (1, None, "missing")),
            (b"\n", (1, None, "missing")),
            (b"\xff", (1, None, "missing")),
            ((json.dumps(row_no_hash) + "\n").encode(), (1, "t", "missing")),
        ]
        # The manifest deliberately disagrees on every comparable field.
        bad = {"version": 1, "byte_length": 999,
               "byte_sha256": "f" * 64, "tenants": []}
        for raw, (at, tenant, reason) in cases:
            self.assertEqual(
                self.other.verify_manifest_bytes(raw, bad),
                self.other.verify_all_bytes(raw), raw)
            r = self.other.verify_manifest_bytes(raw, bad)
            self.assertEqual(
                (r["ok"], r["at"], r["tenant"], r["reason"]),
                (False, at, tenant, reason), raw)
            self.assertNotIn("field", r)

    def test_sequence_digest_corruption_beats_manifest_comparison(self):
        good = self.valid_row("t", 1)
        gap = {"tenant": "t", "seq": 3, "event": {}, "prev": "1" * 64}
        gap["hash"] = AuditChain._hash(gap)
        raw = good + record_bytes(gap)
        bad = self.other.manifest_bytes(b"")
        bad["byte_length"] = 999
        self.assertEqual(
            self.other.verify_manifest_bytes(raw, bad),
            {"ok": False, "at": 2, "tenant": "t", "reason": "sequence"})

        bad_prev = {"tenant": "t", "seq": 2, "event": {}, "prev": "9" * 64}
        bad_prev["hash"] = AuditChain._hash(bad_prev)
        raw = good + record_bytes(bad_prev)
        self.assertEqual(
            self.other.verify_manifest_bytes(raw, bad),
            {"ok": False, "at": 2, "tenant": "t", "reason": "digest"})

    def test_chunks_corruption_matches_bytes_corruption(self):
        raw = self.valid_row("t", 1) + b"\xff\n"
        bad = self.other.manifest_bytes(b"")
        chunks = [raw[:3], b"", raw[3:]]
        self.assertEqual(
            self.other.verify_manifest_chunks(chunks, bad),
            self.other.verify_manifest_bytes(raw, bad))

    def test_file_corrupt_snapshot_returns_failure(self):
        self.path.write_bytes(b"{broken\n")
        bad = {"version": 1, "byte_length": 9, "byte_sha256": "f" * 64,
               "tenants": []}
        self.assertEqual(
            self.chain.verify_manifest(bad),
            {"ok": False, "at": 1, "tenant": None, "reason": "missing"})
        self.assertEqual(self.chain.verify_manifest(bad),
                         self.chain.verify_all())
        self.assertEqual(self.snapshot(), b"{broken\n")

    # --- manifest comparison: fixed field order and locators ---

    def alter(self, **changes):
        self.seed()
        data = self.snapshot()
        m = self.expected_manifest(data)
        m.update(changes)
        return data, m

    def alter_tenant(self, index, **changes):
        self.seed()
        data = self.snapshot()
        m = self.expected_manifest(data)
        m["tenants"][index].update(changes)
        return data, m

    def test_byte_length_mismatch(self):
        self.seed()
        data = self.snapshot()
        m = self.expected_manifest(data)
        m["byte_length"] = len(data) + 1
        r = self.other.verify_manifest_bytes(data, m)
        self.assertEqual(r, {
            "ok": False, "reason": "manifest", "field": "byte_length",
            "expected": len(data) + 1, "actual": len(data)})
        self.assertNotIn("index", r)
        self.assertNotIn("tenant", r)

    def test_byte_length_short_mismatch(self):
        data, m = self.alter(byte_length=0)
        r = self.other.verify_manifest_bytes(data, m)
        self.assertEqual(r["field"], "byte_length")
        self.assertEqual((r["expected"], r["actual"]), (0, len(data)))

    def test_byte_sha256_mismatch(self):
        data, m = self.alter(byte_sha256="f" * 64)
        r = self.other.verify_manifest_bytes(data, m)
        self.assertEqual(r["field"], "byte_sha256")
        self.assertEqual(r["expected"], "f" * 64)
        self.assertEqual(r["actual"], hashlib.sha256(data).hexdigest())

    def test_byte_length_takes_priority_over_sha256(self):
        data, m = self.alter(byte_length=1, byte_sha256="f" * 64)
        r = self.other.verify_manifest_bytes(data, m)
        self.assertEqual(r["field"], "byte_length")

    def test_byte_fields_take_priority_over_tenant_fields(self):
        data, m = self.alter(byte_length=1)
        m["tenants"] = []
        r = self.other.verify_manifest_bytes(data, m)
        self.assertEqual(r["field"], "byte_length")

    def test_tenant_order_reversed(self):
        self.seed()
        data = self.snapshot()
        m = self.expected_manifest(data)
        m["tenants"] = list(reversed(m["tenants"]))
        r = self.other.verify_manifest_bytes(data, m)
        self.assertEqual(r, {
            "ok": False, "reason": "manifest", "field": "tenant_order",
            "index": 0, "expected": "b", "actual": "a"})

    def test_tenant_order_extra_tenant_at_end(self):
        self.seed()
        data = self.snapshot()
        m = self.expected_manifest(data)
        m["tenants"].append(
            {"tenant": "ghost", "count": 1, "hash": "9" * 64})
        r = self.other.verify_manifest_bytes(data, m)
        self.assertEqual(
            (r["field"], r["index"], r["expected"], r["actual"]),
            ("tenant_order", 2, "ghost", None))

    def test_tenant_order_missing_tenant_at_end(self):
        self.seed()
        data = self.snapshot()
        m = self.expected_manifest(data)
        m["tenants"].pop()
        r = self.other.verify_manifest_bytes(data, m)
        self.assertEqual(
            (r["field"], r["index"], r["expected"], r["actual"]),
            ("tenant_order", 1, None, "b"))

    def test_tenant_order_empty_manifest_against_nonempty_snapshot(self):
        self.seed()
        data = self.snapshot()
        # Byte fields must still match for the comparison to reach order.
        m = {
            "version": 1,
            "byte_length": len(data),
            "byte_sha256": hashlib.sha256(data).hexdigest(),
            "tenants": [],
        }
        r = self.other.verify_manifest_bytes(data, m)
        self.assertEqual(
            (r["field"], r["index"], r["expected"], r["actual"]),
            ("tenant_order", 0, None, "a"))

    def test_tenant_order_mismatch_beats_count_and_hash(self):
        self.seed()
        data = self.snapshot()
        m = self.expected_manifest(data)
        m["tenants"] = list(reversed(m["tenants"]))
        m["tenants"][0]["count"] = 99
        m["tenants"][0]["hash"] = "9" * 64
        r = self.other.verify_manifest_bytes(data, m)
        self.assertEqual(r["field"], "tenant_order")

    def test_tenant_count_mismatch_fields(self):
        data, m = self.alter_tenant(0, count=9)
        r = self.other.verify_manifest_bytes(data, m)
        self.assertEqual(r, {
            "ok": False, "reason": "manifest", "field": "tenant_count",
            "tenant": "a", "expected": 9, "actual": 3})
        self.assertNotIn("index", r)

    def test_tenant_count_first_tenant_in_manifest_order(self):
        data, m = self.alter_tenant(1, count=9)
        r = self.other.verify_manifest_bytes(data, m)
        self.assertEqual(
            (r["field"], r["tenant"], r["expected"], r["actual"]),
            ("tenant_count", "b", 9, 2))

    def test_tenant_count_takes_priority_over_hash(self):
        data, m = self.alter_tenant(0, count=9, hash="9" * 64)
        r = self.other.verify_manifest_bytes(data, m)
        self.assertEqual(r["field"], "tenant_count")

    def test_tenant_hash_mismatch_fields(self):
        data, m = self.alter_tenant(0, hash="9" * 64)
        r = self.other.verify_manifest_bytes(data, m)
        self.assertEqual(r, {
            "ok": False, "reason": "manifest", "field": "tenant_hash",
            "tenant": "a", "expected": "9" * 64,
            "actual": self.chain.head("a")["hash"]})

    def test_only_one_difference_is_reported(self):
        # A wrong count on "a" hides a wrong hash on "b".
        self.seed()
        data = self.snapshot()
        m = self.expected_manifest(data)
        m["tenants"][0]["count"] = 9
        m["tenants"][1]["hash"] = "9" * 64
        r = self.other.verify_manifest_bytes(data, m)
        self.assertEqual((r["field"], r["tenant"]), ("tenant_count", "a"))

    def test_field_names_are_the_only_allowed_set(self):
        self.seed()
        data = self.snapshot()
        # order mismatch
        m = self.expected_manifest(data)
        m["tenants"] = list(reversed(m["tenants"]))
        self.assertIn(
            self.other.verify_manifest_bytes(data, m)["field"],
            {"byte_length", "byte_sha256", "tenant_order",
             "tenant_count", "tenant_hash"})

    def test_file_entry_mismatch_verdict_matches_bytes_entry(self):
        self.seed()
        data, m = self.alter_tenant(0, hash="9" * 64)
        self.assertEqual(self.chain.verify_manifest(m),
                         self.other.verify_manifest_bytes(data, m))

    def test_chunks_mismatch_matches_bytes_mismatch(self):
        self.seed()
        data, m = self.alter(byte_sha256="f" * 64)
        r1 = self.other.verify_manifest_bytes(data, m)
        r2 = self.other.verify_manifest_chunks(
            [data[:7], b"", data[7:]], m)
        self.assertEqual(r1, r2)

    # --- manifest input boundary: ValueError before reading history ---

    def test_manifest_must_be_exact_four_key_object(self):
        good = empty_manifest()
        for bad in (None, 1, "x", [], [good],
                    {"version": 1},
                    {"version": 1, "byte_length": 0,
                     "byte_sha256": good["byte_sha256"]},
                    {**good, "extra": 1},
                    {"version": "1", "byte_length": 0,
                     "byte_sha256": good["byte_sha256"], "tenants": []}):
            with self.assertRaises(ValueError):
                self.other.verify_manifest_bytes(b"", bad)
            with self.assertRaises(ValueError):
                self.chain.verify_manifest(bad)
            with self.assertRaises(ValueError):
                self.other.verify_manifest_chunks([], bad)

    def test_version_must_be_1(self):
        m = empty_manifest()
        for bad in (0, 2, -1, 1.0, True, False, None, "1"):
            m_bad = dict(m, version=bad)
            with self.assertRaises(ValueError):
                self.other.verify_manifest_bytes(b"", m_bad)

    def test_byte_length_boundary(self):
        m = empty_manifest()
        for bad in (-1, 1.0, True, False, "0", None, 1.5, [0]):
            with self.assertRaises(ValueError):
                self.other.verify_manifest_bytes(b"", dict(m, byte_length=bad))

    def test_byte_sha256_boundary(self):
        m = empty_manifest()
        for bad in ("", "x", "F" * 64, "0" * 63, "0" * 65, 1, None,
                    bytes(64), ZERO.encode()):
            with self.assertRaises(ValueError):
                self.other.verify_manifest_bytes(b"", dict(m, byte_sha256=bad))

    def test_tenants_must_be_list(self):
        m = empty_manifest()
        for bad in ((), {}, "", 1, None):
            with self.assertRaises(ValueError):
                self.other.verify_manifest_bytes(b"", dict(m, tenants=bad))

    def test_tenant_member_must_be_exact_three_key_object(self):
        good = {"tenant": "t", "count": 0, "hash": ZERO}
        m = empty_manifest()
        for bad in (None, 1, "x", [], [good, 1],
                    {"tenant": "t"},
                    {"tenant": "t", "count": 0},
                    {"tenant": "t", "count": 0, "hash": ZERO, "x": 1},
                    {"tenant": "t", "count": 0, "digest": ZERO}):
            with self.assertRaises(ValueError):
                self.other.verify_manifest_bytes(b"", dict(m, tenants=[bad]))

    def test_tenant_value_boundary(self):
        for bad in (float("nan"), float("inf"), {"k": float("nan")},
                    {1: "x"}, object(), b"x", {1, 2}):
            m = empty_manifest()
            m["tenants"] = [{"tenant": bad, "count": 0, "hash": ZERO}]
            with self.assertRaises(ValueError):
                self.other.verify_manifest_bytes(b"", m)

    def test_tenant_count_boundary(self):
        for bad in (-1, 1.0, True, False, "1", None, 1.5, [1]):
            m = empty_manifest()
            m["tenants"] = [{"tenant": "t", "count": bad, "hash": ZERO}]
            with self.assertRaises(ValueError):
                self.other.verify_manifest_bytes(b"", m)

    def test_tenant_hash_boundary(self):
        for bad in ("", "x", "A" * 64, "0" * 63, "0" * 65, 1, None,
                    bytes(64), ZERO.encode()):
            m = empty_manifest()
            m["tenants"] = [{"tenant": "t", "count": 0, "hash": bad}]
            with self.assertRaises(ValueError):
                self.other.verify_manifest_bytes(b"", m)

    def test_duplicate_canonical_tenant_rejected(self):
        head = {"count": 0, "hash": ZERO}
        for dup in (
            [{"tenant": "a", **head}, {"tenant": "a", **head}],
            [{"tenant": {"a": 1, "b": 2}, **head},
             {"tenant": {"b": 2, "a": 1}, **head}],
            [{"tenant": 1, **head}, {"tenant": 1, **head}],
        ):
            m = empty_manifest()
            m["tenants"] = dup
            with self.assertRaises(ValueError):
                self.other.verify_manifest_bytes(b"", m)
        # Distinct JSON identities are not duplicates even if loosely equal.
        for t in (1, 1.0, True, "1"):
            self.chain.append(t, {})
        data = self.snapshot()
        m = self.other.manifest_bytes(data)
        self.assertTrue(self.other.verify_manifest_bytes(data, m)["ok"])

    def test_boundary_value_error_beats_corrupt_snapshot(self):
        # A malformed manifest ends as ValueError even though the snapshot is
        # corrupt, and no byte is read/parsed for the comparison.
        corrupt = b"{not json\n"
        good_hash = "0" * 64
        bad_manifests = (
            None, 1, [], "x",
            {"version": 2, "byte_length": 0, "byte_sha256": good_hash,
             "tenants": []},
            {"version": True, "byte_length": 0, "byte_sha256": good_hash,
             "tenants": []},
            {"version": 1, "byte_length": -1, "byte_sha256": good_hash,
             "tenants": []},
            {"version": 1, "byte_length": 1.0, "byte_sha256": good_hash,
             "tenants": []},
            {"version": 1, "byte_length": 0, "byte_sha256": "Z" * 64,
             "tenants": []},
            {"version": 1, "byte_length": 0, "byte_sha256": good_hash},
            {"version": 1, "byte_length": 0, "byte_sha256": good_hash,
             "tenants": [1]},
            {"version": 1, "byte_length": 0, "byte_sha256": good_hash,
             "tenants": [{"tenant": "t", "count": True, "hash": good_hash}]},
            {"version": 1, "byte_length": 0, "byte_sha256": good_hash,
             "tenants": [
                 {"tenant": "a", "count": 0, "hash": good_hash},
                 {"tenant": "a", "count": 0, "hash": good_hash}]},
            {"version": 1, "byte_length": 0, "byte_sha256": good_hash,
             "tenants": [
                 {"tenant": float("nan"), "count": 0, "hash": good_hash}]},
        )
        for bad in bad_manifests:
            with self.assertRaises(ValueError):
                self.other.verify_manifest_bytes(corrupt, bad)
            with self.assertRaises(ValueError):
                self.other.verify_manifest_chunks([corrupt], bad)

    def test_underlying_parse_exception_is_not_leaked(self):
        # Malformed manifests surface as a plain ValueError, never a
        # TypeError/KeyError/RecursionError escaping from a downstream json
        # call on the manifest.
        cyclic = {}
        cyclic["self"] = cyclic
        m = empty_manifest()
        m["tenants"] = [{"tenant": cyclic, "count": 0, "hash": ZERO}]
        for bad in (m, {**empty_manifest(), "tenants": {}}):
            with self.assertRaises(ValueError) as ctx:
                self.other.verify_manifest_bytes(b"", bad)
            self.assertEqual(type(ctx.exception), ValueError)
            self.assertNotIsInstance(
                ctx.exception, (KeyError, TypeError, RecursionError))

    # --- data / chunks container boundary ---

    def test_data_must_be_bytes(self):
        m = empty_manifest()
        for bad in ("", "{}", bytearray(b""), bytearray(b"{}"), 1, None,
                    [b""]):
            with self.assertRaises(ValueError):
                self.other.verify_manifest_bytes(bad, m)
            with self.assertRaises(ValueError):
                self.other.manifest_bytes(bad)

    def test_bare_bytes_container_is_value_error(self):
        m = empty_manifest()
        for bad in (b"", b"{}", bytearray(b""), bytearray(b"{}")):
            with self.assertRaises(ValueError):
                self.other.verify_manifest_chunks(bad, m)
            with self.assertRaises(ValueError):
                self.other.manifest_chunks(bad)

    def test_non_iterable_container_is_value_error(self):
        m = empty_manifest()
        for bad in (1, None, object()):
            with self.assertRaises(ValueError):
                self.other.verify_manifest_chunks(bad, m)

    def test_first_non_bytes_member_is_value_error(self):
        m = empty_manifest()
        for bad in ([b"", "x"], [b"", 1], [b"", None], [b"", bytearray(b"")],
                    [b"", [b""]], ["x"], [bytearray(b"")]):
            with self.assertRaises(ValueError):
                self.other.verify_manifest_chunks(bad, m)
            with self.assertRaises(ValueError):
                self.other.manifest_chunks(bad)

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
            self.other.verify_manifest_chunks(gen(), empty_manifest())
        self.assertEqual(pulled, [1, 2])

    def test_manifest_validated_before_chunks_consumed(self):
        pulled = []

        def gen():
            pulled.append(1)
            yield b""

        with self.assertRaises(ValueError):
            self.other.verify_manifest_chunks(gen(), {"version": 2})
        self.assertEqual(pulled, [])

    # --- offline purity ---

    def test_offline_entries_never_touch_constructor_path(self):
        phantom = self.path.with_name("phantom.jsonl")
        chain = AuditChain(phantom)
        self.assertEqual(chain.manifest_bytes(b""), empty_manifest())
        self.assertEqual(chain.manifest_chunks([]), empty_manifest())
        self.assertFalse(phantom.exists())
        self.seed()
        before = self.snapshot()
        m = chain.manifest_bytes(before)
        self.assertTrue(chain.verify_manifest_bytes(before, m)["ok"])
        self.assertTrue(
            chain.verify_manifest_chunks([before[:7], b"", before[7:]], m)
            ["ok"])
        self.assertEqual(self.snapshot(), before)
        self.assertFalse(phantom.exists())

    def test_file_entry_is_read_only(self):
        self.seed()
        before = self.snapshot()
        m = self.chain.manifest()
        self.chain.verify_manifest(m)
        m_bad = dict(m, byte_length=1)
        r = self.chain.verify_manifest(m_bad)
        self.assertFalse(r["ok"])
        self.assertEqual(self.snapshot(), before)

    def test_inputs_are_not_mutated(self):
        self.seed()
        data = self.snapshot()
        chunks = [data[:9], b"", data[9:]]
        m = self.other.manifest_bytes(data)
        m_copy = copy.deepcopy(m)
        self.other.verify_manifest_bytes(data, m)
        self.other.verify_manifest_chunks(chunks, m)
        self.assertEqual(data, self.snapshot())
        self.assertEqual(b"".join(chunks), data)
        self.assertEqual(m, m_copy)


if __name__ == "__main__":
    unittest.main()
