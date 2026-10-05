import copy
import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app import AuditChain, AuditChainStateError, ZERO


EMPTY_SHA = hashlib.sha256(b"").hexdigest()


def record_bytes(item):
    return (json.dumps(item, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def row(tenant="t", seq=1, prev=ZERO, event=None, hash_=None):
    item = {"tenant": tenant, "seq": seq, "event": event or {}, "prev": prev}
    item["hash"] = hash_ if hash_ is not None else AuditChain._hash(item)
    return item, record_bytes(item)


class ManifestTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.path = Path(self.tmp.name) / "audit.jsonl"
        self.chain = AuditChain(self.path)
        # Pure in-memory entries are exercised through a chain whose path
        # must never be read, created or modified.
        self.other = AuditChain(self.path.with_name("never-touched.jsonl"))

    def tearDown(self):
        self.tmp.cleanup()

    def seed(self, chain, *tenant_events):
        for tenant, event in tenant_events:
            chain.append(tenant, event)

    def snapshot(self, chain=None):
        chain = chain or self.chain
        return Path(chain.path).read_bytes()

    def empty_manifest(self):
        return {"version": 1, "byte_length": 0,
                "byte_sha256": EMPTY_SHA, "tenants": []}

    # --- generation: shape, order and values ---

    def test_empty_snapshot_manifest(self):
        # Missing file, empty bytes and empty chunks all agree.
        self.assertFalse(self.path.exists())
        self.assertEqual(self.chain.manifest(), self.empty_manifest())
        self.path.write_bytes(b"")
        self.assertEqual(self.chain.manifest(), self.empty_manifest())
        self.assertEqual(self.other.manifest_bytes(b""), self.empty_manifest())
        for chunks in ([], (), iter([]), [b""], [b"", b""]):
            self.assertEqual(self.other.manifest_chunks(chunks),
                             self.empty_manifest())

    def test_manifest_fields_and_physical_first_appearance_order(self):
        self.seed(self.chain,
                  ("a", {"i": 1}), ("b", {"i": 1}), ("a", {"i": 2}),
                  ("a", {"i": 3}), ("b", {"i": 2}))
        data = self.snapshot()
        m = self.chain.manifest()
        self.assertEqual(m["version"], 1)
        self.assertEqual(m["byte_length"], len(data))
        self.assertEqual(m["byte_sha256"],
                         hashlib.sha256(data).hexdigest())
        self.assertEqual(
            m["tenants"],
            [{"tenant": "a", "count": 3,
              "hash": self.chain.head("a")["hash"]},
             {"tenant": "b", "count": 2,
              "hash": self.chain.head("b")["hash"]}],
        )
        # The tenant directory is exactly heads()' directory.
        self.assertEqual(m["tenants"], self.chain.heads()["tenants"])
        # bytes and chunks generators produce the identical object.
        self.assertEqual(self.other.manifest_bytes(data), m)
        self.assertEqual(
            self.other.manifest_chunks([data[:5], b"", data[5:]]), m)

    def test_last_hash_is_zero_for_no_records_only_via_heads(self):
        # A tenant present always has count >= 1, so its hash is the last
        # record's; the empty manifest simply has no tenant entries.
        self.seed(self.chain, ("t", {"i": 1}))
        m = self.chain.manifest()
        self.assertEqual(m["tenants"][0]["hash"],
                         self.chain.head("t")["hash"])
        self.assertNotEqual(m["tenants"][0]["hash"], ZERO)

    def test_original_tenant_values_are_preserved(self):
        self.seed(self.chain, ({"k": 1}, {}), (1, {}), (1.0, {}),
                  (True, {}), ("1", {}))
        m = self.chain.manifest()
        self.assertEqual(
            [t["tenant"] for t in m["tenants"]],
            [{"k": 1}, 1, 1.0, True, "1"],
        )
        self.assertTrue(all(t["count"] == 1 for t in m["tenants"]))

    # --- generation: chunks boundaries ---

    def test_chunks_match_bytes_for_many_chunkings(self):
        self.seed(self.chain, ("a", {"i": 1}), ("b", {"i": 1}),
                  ("a", {"i": 2}))
        self.chain.append("t", {"msg": "héllo→世界✓" * 3})
        data = self.snapshot()
        chunkings = [
            [data],
            [data[:1], data[1:]],
            [data[i:i + 1] for i in range(len(data))],
            [data[:10], b"", data[10:37], b"", data[37:]],
            [b"", b"", data, b""],
            list(data[i:i + 7] for i in range(0, len(data), 7)),
        ]
        expected = self.other.manifest_bytes(data)
        for chunks in chunkings:
            self.assertEqual(self.other.manifest_chunks(chunks), expected)
        for container in (list(data[i:i + 3] for i in range(0, len(data), 3)),
                          tuple(data[i:i + 3]
                                for i in range(0, len(data), 3)),
                          iter([data[:4], data[4:]])):
            self.assertEqual(self.other.manifest_chunks(container), expected)

    def test_generation_chunk_container_boundary(self):
        for bad in (b"", b"{}", bytearray(b"")):
            with self.assertRaises(ValueError):
                self.other.manifest_chunks(bad)
        for bad in (1, None, object()):
            with self.assertRaises(ValueError):
                self.other.manifest_chunks(bad)
        for bad in ([b"", "x"], [b"", 1], [b"", None], [b"", bytearray(b"")]):
            with self.assertRaises(ValueError):
                self.other.manifest_chunks(bad)

    def test_generation_bytes_boundary(self):
        for bad in ("", bytearray(b""), bytearray(b"{}"), 1, 1.0, None,
                    True, [b""]):
            with self.assertRaises(ValueError):
                self.other.manifest_bytes(bad)

    def test_bad_chunk_stops_consumption(self):
        pulled = []

        def gen():
            pulled.append(1)
            yield b""
            pulled.append(2)
            yield "not bytes"
            pulled.append(3)
            yield b""

        with self.assertRaises(ValueError):
            self.other.manifest_chunks(gen())
        self.assertEqual(pulled, [1, 2])

    # --- generation: corruption -> AuditChainStateError, no partial result ---

    def test_generation_corrupt_snapshot_raises_state_error(self):
        _i, good = row("t", 1)
        cases = [
            (b"{oops\n", None, None, "missing"),
            (b"\xff\n", None, None, "missing"),
            (good + b"{oops\n", None, None, "missing"),
        ]
        for raw, tenant, seq, reason in cases:
            with self.assertRaises(AuditChainStateError) as cm:
                self.other.manifest_bytes(raw)
            self.assertEqual(cm.exception.reason, reason)
            self.assertIsNone(cm.exception.tenant)
        # sequence/digest defects name the tenant and seq
        gap = {"tenant": "t", "seq": 3, "event": {}, "prev": "x" * 64}
        gap["hash"] = AuditChain._hash(gap)
        with self.assertRaises(AuditChainStateError) as cm:
            self.other.manifest_bytes(good + record_bytes(gap))
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("t", 2, "sequence", 2))
        bad_prev = {"tenant": "t", "seq": 2, "event": {},
                    "prev": "9" * 64}
        bad_prev["hash"] = AuditChain._hash(bad_prev)
        with self.assertRaises(AuditChainStateError) as cm:
            self.other.manifest_bytes(good + record_bytes(bad_prev))
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason),
                         ("t", 2, "digest"))
        # same through the file and chunks entries
        self.path.write_bytes(good + record_bytes(gap))
        with self.assertRaises(AuditChainStateError):
            self.chain.manifest()
        with self.assertRaises(AuditChainStateError):
            self.other.manifest_chunks([good, record_bytes(gap)])

    # --- verification: success ---

    def test_verify_success_echoes_actual_manifest(self):
        self.seed(self.chain, ("a", {"i": 1}), ("b", {"i": 1}),
                  ("a", {"i": 2}))
        data = self.snapshot()
        m = self.chain.manifest()
        self.assertEqual(self.chain.verify_manifest(m),
                         {"ok": True, "manifest": m})
        self.assertEqual(self.other.verify_manifest_bytes(data, m),
                         {"ok": True, "manifest": m})
        self.assertEqual(
            self.other.verify_manifest_chunks([data[:3], data[3:]], m),
            {"ok": True, "manifest": m})
        # An equal but independently built manifest still verifies; the echo
        # is the actual manifest, not the caller's object.
        equal = copy.deepcopy(m)
        r = self.other.verify_manifest_bytes(data, equal)
        self.assertTrue(r["ok"])
        self.assertEqual(r["manifest"], m)
        self.assertIsNot(r["manifest"], equal)

    def test_verify_empty(self):
        m = self.empty_manifest()
        self.assertEqual(self.chain.verify_manifest(m),
                         {"ok": True, "manifest": m})
        self.assertEqual(self.other.verify_manifest_bytes(b"", m),
                         {"ok": True, "manifest": m})
        self.assertEqual(self.other.verify_manifest_chunks([], m),
                         {"ok": True, "manifest": m})

    # --- verification: field mismatches and fixed first-difference order ---

    def test_byte_length_mismatch(self):
        self.seed(self.chain, ("a", {"i": 1}))
        data = self.snapshot()
        m = self.other.manifest_bytes(data)
        bad = copy.deepcopy(m)
        bad["byte_length"] = len(data) + 5
        r = self.other.verify_manifest_bytes(data, bad)
        self.assertEqual(r, {
            "ok": False, "reason": "manifest", "field": "byte_length",
            "expected": len(data) + 5, "actual": len(data),
        })
        self.assertNotIn("index", r)
        self.assertNotIn("tenant", r)

    def test_byte_sha256_mismatch(self):
        self.seed(self.chain, ("a", {"i": 1}))
        data = self.snapshot()
        m = self.other.manifest_bytes(data)
        bad = copy.deepcopy(m)
        bad["byte_sha256"] = "f" * 64
        r = self.other.verify_manifest_bytes(data, bad)
        self.assertEqual(r["field"], "byte_sha256")
        self.assertEqual(r["expected"], "f" * 64)
        self.assertEqual(r["actual"], m["byte_sha256"])
        self.assertNotIn("index", r)

    def test_tenant_order_reordered(self):
        self.seed(self.chain, ("a", {}), ("b", {}))
        data = self.snapshot()
        m = self.other.manifest_bytes(data)
        bad = copy.deepcopy(m)
        bad["tenants"] = list(reversed(bad["tenants"]))
        r = self.other.verify_manifest_bytes(data, bad)
        self.assertEqual(r["ok"], False)
        self.assertEqual(r["reason"], "manifest")
        self.assertEqual(r["field"], "tenant_order")
        self.assertEqual(r["index"], 0)
        self.assertEqual(r["tenant"], "a")
        self.assertEqual(r["expected"], bad["tenants"])
        self.assertEqual(r["actual"], m["tenants"])

    def test_tenant_order_extra_actual_tenant(self):
        self.seed(self.chain, ("a", {}), ("b", {}), ("c", {}))
        data = self.snapshot()
        m = self.other.manifest_bytes(data)
        bad = copy.deepcopy(m)
        bad["tenants"].pop()  # manifest only knows a, b
        r = self.other.verify_manifest_bytes(data, bad)
        self.assertEqual(r["field"], "tenant_order")
        self.assertEqual((r["index"], r["tenant"]), (2, "c"))
        self.assertEqual(r["actual"], m["tenants"])

    def test_tenant_order_extra_expected_tenant(self):
        self.seed(self.chain, ("a", {}))
        data = self.snapshot()
        m = self.other.manifest_bytes(data)
        bad = copy.deepcopy(m)
        bad["tenants"].append(
            {"tenant": "z", "count": 1, "hash": "d" * 64})
        r = self.other.verify_manifest_bytes(data, bad)
        self.assertEqual(r["field"], "tenant_order")
        self.assertEqual((r["index"], r["tenant"]), (1, "z"))
        self.assertEqual(r["expected"], bad["tenants"])

    def test_tenant_order_empty_snapshot_against_nonempty_manifest(self):
        bad = {"version": 1, "byte_length": 0, "byte_sha256": EMPTY_SHA,
               "tenants": [{"tenant": "q", "count": 1, "hash": "d" * 64}]}
        r = self.other.verify_manifest_bytes(b"", bad)
        self.assertEqual(r["field"], "tenant_order")
        self.assertEqual((r["index"], r["tenant"]), (0, "q"))

    def test_tenant_count_mismatch(self):
        self.seed(self.chain, ("a", {}), ("b", {}), ("a", {}))
        data = self.snapshot()
        m = self.other.manifest_bytes(data)
        bad = copy.deepcopy(m)
        bad["tenants"][1]["count"] = 5
        r = self.other.verify_manifest_bytes(data, bad)
        self.assertEqual(r, {
            "ok": False, "reason": "manifest", "field": "tenant_count",
            "index": 1, "tenant": "b", "expected": 5, "actual": 1,
        })

    def test_tenant_hash_mismatch(self):
        self.seed(self.chain, ("a", {}), ("b", {}))
        data = self.snapshot()
        m = self.other.manifest_bytes(data)
        bad = copy.deepcopy(m)
        bad["tenants"][0]["hash"] = "e" * 64
        r = self.other.verify_manifest_bytes(data, bad)
        self.assertEqual(r, {
            "ok": False, "reason": "manifest", "field": "tenant_hash",
            "index": 0, "tenant": "a",
            "expected": "e" * 64, "actual": m["tenants"][0]["hash"],
        })

    def test_count_mismatch_reported_before_hash_mismatch(self):
        self.seed(self.chain, ("a", {}))
        data = self.snapshot()
        m = self.other.manifest_bytes(data)
        bad = copy.deepcopy(m)
        bad["tenants"][0]["count"] = 9
        bad["tenants"][0]["hash"] = "e" * 64
        r = self.other.verify_manifest_bytes(data, bad)
        self.assertEqual(r["field"], "tenant_count")

    def test_field_precedence_is_fixed(self):
        self.seed(self.chain, ("a", {}), ("b", {}))
        data = self.snapshot()
        m = self.other.manifest_bytes(data)

        def corrupt_manifest():
            bad = copy.deepcopy(m)
            bad["byte_length"] = 1
            bad["byte_sha256"] = "f" * 64
            bad["tenants"] = []
            return bad

        # everything wrong at once -> byte_length
        self.assertEqual(
            self.other.verify_manifest_bytes(data, corrupt_manifest())[
                "field"],
            "byte_length")
        # length fixed, rest wrong -> byte_sha256
        bad = corrupt_manifest()
        bad["byte_length"] = m["byte_length"]
        self.assertEqual(
            self.other.verify_manifest_bytes(data, bad)["field"],
            "byte_sha256")
        # length+sha fixed, tenants wrong -> tenant_order
        bad = corrupt_manifest()
        bad["byte_length"] = m["byte_length"]
        bad["byte_sha256"] = m["byte_sha256"]
        self.assertEqual(
            self.other.verify_manifest_bytes(data, bad)["field"],
            "tenant_order")

    def test_first_tenant_mismatch_wins(self):
        self.seed(self.chain, ("a", {}), ("b", {}), ("c", {}))
        data = self.snapshot()
        m = self.other.manifest_bytes(data)
        bad = copy.deepcopy(m)
        bad["tenants"][0]["count"] = 2
        bad["tenants"][1]["hash"] = "e" * 64
        r = self.other.verify_manifest_bytes(data, bad)
        self.assertEqual((r["field"], r["index"]), ("tenant_count", 0))

    # --- verification: chain damage wins over manifest comparison ---

    def test_chain_damage_wins_over_manifest_mismatch(self):
        _i, good = row("t", 1)
        # An obviously wrong manifest would fail every comparison, yet the
        # corrupt chain on line 2 decides.
        bogus = {"version": 1, "byte_length": 999,
                 "byte_sha256": "0" * 64, "tenants": []}
        r = self.other.verify_manifest_bytes(good + b"{oops\n", bogus)
        self.assertEqual(r, {"ok": False, "at": 2, "tenant": None,
                             "reason": "missing"})
        gap = {"tenant": "t", "seq": 3, "event": {}, "prev": "x" * 64}
        gap["hash"] = AuditChain._hash(gap)
        r = self.other.verify_manifest_bytes(good + record_bytes(gap), bogus)
        self.assertEqual(r, {"ok": False, "at": 2, "tenant": "t",
                             "reason": "sequence"})
        bad_prev = {"tenant": "t", "seq": 2, "event": {},
                    "prev": "9" * 64}
        bad_prev["hash"] = AuditChain._hash(bad_prev)
        r = self.other.verify_manifest_bytes(
            good + record_bytes(bad_prev), bogus)
        self.assertEqual(r, {"ok": False, "at": 2, "tenant": "t",
                             "reason": "digest"})
        # Illegal UTF-8: tenant cannot be determined.
        r = self.other.verify_manifest_bytes(good + b"\xff", bogus)
        self.assertEqual(r, {"ok": False, "at": 2, "tenant": None,
                             "reason": "missing"})

    def test_chain_damage_priority_physical_order(self):
        # digest defect on line 1 beats manifest mismatch and illegal UTF-8
        # arriving later.
        tampered = {"tenant": "t", "seq": 1, "event": {"z": 9}, "prev": ZERO}
        tampered["hash"] = AuditChain._hash(
            {"tenant": "t", "seq": 1, "event": {}, "prev": ZERO})
        raw = record_bytes(tampered) + b"\xff\n"
        bogus = {"version": 1, "byte_length": 1, "byte_sha256": "0" * 64,
                 "tenants": []}
        r = self.other.verify_manifest_bytes(raw, bogus)
        self.assertEqual(r, {"ok": False, "at": 1, "tenant": "t",
                             "reason": "digest"})

    def test_file_entry_chain_damage_returns_failure_object(self):
        _i, good = row("t", 1)
        self.path.write_bytes(good + b"{oops\n")
        bogus = {"version": 1, "byte_length": 1, "byte_sha256": "0" * 64,
                 "tenants": []}
        r = self.chain.verify_manifest(bogus)
        self.assertEqual(r, {"ok": False, "at": 2, "tenant": None,
                             "reason": "missing"})

    # --- manifest input boundary: ValueError before history is read ---

    def test_manifest_boundary(self):
        valid_base = {"version": 1, "byte_length": 0,
                      "byte_sha256": EMPTY_SHA, "tenants": []}
        bad_manifests = [
            None, 1, 1.0, True, "x", [], (), object(),
            {},
            {"byte_length": 0, "byte_sha256": EMPTY_SHA, "tenants": []},
            {"version": 1, "byte_length": 0, "byte_sha256": EMPTY_SHA},
            {**valid_base, "extra": 1},
            {**valid_base, "version": 2},
            {**valid_base, "version": 0},
            {**valid_base, "version": True},
            {**valid_base, "version": False},
            {**valid_base, "version": 1.0},
            {**valid_base, "version": "1"},
            {**valid_base, "version": None},
            {**valid_base, "byte_length": -1},
            {**valid_base, "byte_length": True},
            {**valid_base, "byte_length": 1.0},
            {**valid_base, "byte_length": "1"},
            {**valid_base, "byte_length": None},
            {**valid_base, "byte_sha256": "A" * 64},
            {**valid_base, "byte_sha256": "0" * 63},
            {**valid_base, "byte_sha256": "0" * 65},
            {**valid_base, "byte_sha256": "g" * 64},
            {**valid_base, "byte_sha256": None},
            {**valid_base, "byte_sha256": b"0" * 64},
            {**valid_base, "tenants": {}},
            {**valid_base, "tenants": None},
            {**valid_base, "tenants": "x"},
            {**valid_base, "tenants": [None]},
            {**valid_base, "tenants": [[]]},
            {**valid_base, "tenants": [{"tenant": "a", "count": 0}]},
            {**valid_base, "tenants": [
                {"tenant": "a", "count": 0, "hash": "0" * 64, "x": 1}]},
            {**valid_base, "tenants": [
                {"tenant": "a", "count": False, "hash": "0" * 64}]},
            {**valid_base, "tenants": [
                {"tenant": "a", "count": 1.0, "hash": "0" * 64}]},
            {**valid_base, "tenants": [
                {"tenant": "a", "count": -3, "hash": "0" * 64}]},
            {**valid_base, "tenants": [
                {"tenant": "a", "count": "1", "hash": "0" * 64}]},
            {**valid_base, "tenants": [
                {"tenant": "a", "count": None, "hash": "0" * 64}]},
            {**valid_base, "tenants": [
                {"tenant": "a", "count": 0, "hash": "Z" * 64}]},
            {**valid_base, "tenants": [
                {"tenant": "a", "count": 0, "hash": "0" * 63}]},
            {**valid_base, "tenants": [
                {"tenant": "a", "count": 0, "hash": None}]},
            {**valid_base, "tenants": [
                {"tenant": float("nan"), "count": 0, "hash": "0" * 64}]},
            {**valid_base, "tenants": [
                {"tenant": {1: "x"}, "count": 0, "hash": "0" * 64}]},
        ]
        for bad in bad_manifests:
            with self.assertRaises(ValueError, msg=repr(bad)):
                self.other.verify_manifest_bytes(b"", bad)
            with self.assertRaises(ValueError, msg=repr(bad)):
                self.other.verify_manifest_chunks([], bad)
            with self.assertRaises(ValueError, msg=repr(bad)):
                self.chain.verify_manifest(bad)

    def test_duplicate_canonical_tenant_rejected_distinct_identities_ok(self):
        base = {"version": 1, "byte_length": 0, "byte_sha256": EMPTY_SHA}
        dup = {**base, "tenants": [
            {"tenant": {"a": 1, "b": 2}, "count": 0, "hash": ZERO},
            {"tenant": {"b": 2, "a": 1}, "count": 0, "hash": ZERO}]}
        for entry in (self.chain.verify_manifest, ):
            with self.assertRaises(ValueError):
                entry(dup)
        with self.assertRaises(ValueError):
            self.other.verify_manifest_bytes(b"", dup)
        # 1, 1.0, true, "1" are four different tenants and coexist.
        distinct = {**base, "tenants": [
            {"tenant": t, "count": 0, "hash": ZERO}
            for t in (1, 1.0, True, "1")]}
        r = self.other.verify_manifest_bytes(b"", distinct)
        # They do not match the empty snapshot (tenant_order differs), but the
        # manifest itself is valid: no ValueError, a manifest verdict returns.
        self.assertEqual(r["field"], "tenant_order")

    def test_cyclic_tenant_rejected_without_leaking_exception(self):
        cyc = []
        cyc.append(cyc)
        bad = {"version": 1, "byte_length": 0, "byte_sha256": EMPTY_SHA,
               "tenants": [
                   {"tenant": cyc, "count": 0, "hash": "0" * 64}]}
        with self.assertRaises(ValueError):
            self.other.verify_manifest_bytes(b"", bad)

    def test_boundary_priority_over_corrupt_history(self):
        # A malformed manifest against a corrupt file is ValueError, and no
        # file byte needs reading to reach it.
        _i, good = row("t", 1)
        self.path.write_bytes(good + b"{oops\n")
        with self.assertRaises(ValueError):
            self.chain.verify_manifest({"version": 2})
        with self.assertRaises(ValueError):
            self.other.verify_manifest_bytes(good + b"{oops\n",
                                             {"version": 2})

    def test_verify_bytes_data_boundary(self):
        m = self.empty_manifest()
        for bad in ("", bytearray(b""), bytearray(b"{}"), 1, None, True,
                    [b""]):
            with self.assertRaises(ValueError):
                self.other.verify_manifest_bytes(bad, m)

    def test_verify_chunks_container_boundary(self):
        m = self.empty_manifest()
        for bad in (b"", b"{}", bytearray(b"")):
            with self.assertRaises(ValueError):
                self.other.verify_manifest_chunks(bad, m)
        for bad in (1, None, object()):
            with self.assertRaises(ValueError):
                self.other.verify_manifest_chunks(bad, m)
        for bad in ([b"", "x"], [b"", 1], [b"", bytearray(b"")]):
            with self.assertRaises(ValueError):
                self.other.verify_manifest_chunks(bad, m)

    def test_malformed_manifest_pulls_no_chunks(self):
        pulled = []

        def gen():
            pulled.append(1)
            yield b""

        with self.assertRaises(ValueError):
            self.other.verify_manifest_chunks(
                gen(), {"version": 9, "byte_length": 0,
                        "byte_sha256": "0" * 64, "tenants": []})
        self.assertEqual(pulled, [])

    # --- offline purity ---

    def test_never_reads_creates_or_modifies_constructor_path(self):
        phantom = self.path.with_name("phantom.jsonl")
        chain = AuditChain(phantom)
        self.seed(self.chain, ("a", {"i": 1}), ("b", {"i": 1}),
                  ("a", {"i": 2}))
        data = self.snapshot()
        m = chain.manifest_bytes(data)
        self.assertFalse(phantom.exists())
        self.assertTrue(chain.verify_manifest_bytes(data, m)["ok"])
        self.assertFalse(phantom.exists())
        self.assertTrue(
            chain.verify_manifest_chunks([data[:9], data[9:]], m)["ok"])
        self.assertFalse(phantom.exists())
        self.assertEqual(chain.manifest_bytes(b""), self.empty_manifest())
        self.assertFalse(phantom.exists())

    def test_manifest_generation_is_read_only_on_file(self):
        self.seed(self.chain, ("a", {"i": 1}))
        before = self.snapshot()
        self.chain.manifest()
        self.assertEqual(self.snapshot(), before)

    def test_does_not_mutate_inputs(self):
        self.seed(self.chain, ("a", {"i": 1}))
        data = bytearray(self.snapshot())
        chunks = [bytes(data[:5]), bytes(data[5:])]
        m = self.other.manifest_chunks(chunks)
        self.other.verify_manifest_chunks(chunks, copy.deepcopy(m))
        self.assertEqual(b"".join(chunks), bytes(data))
        manifest_copy = copy.deepcopy(m)
        self.other.verify_manifest_chunks(chunks, manifest_copy)
        self.assertEqual(manifest_copy, m)


if __name__ == "__main__":
    unittest.main()
