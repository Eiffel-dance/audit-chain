import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app import AuditChain, MANIFEST_VERSION, ZERO


def record_bytes(item):
    return (json.dumps(item, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


class MergeBytesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.path = Path(self.tmp.name) / "audit.jsonl"
        self.chain = AuditChain(self.path)
        # Every merge runs through an AuditChain pointed at a path that must
        # never be read, created or modified by the pure in-memory entry.
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
        item = {"tenant": tenant, "seq": seq, "event": event or {},
                "prev": prev}
        item["hash"] = hash_ if hash_ is not None else AuditChain._hash(item)
        return item, record_bytes(item)

    # --- argument boundary: exact bytes, ValueError before any parsing ---

    def test_arguments_must_be_bytes(self):
        for bad in ("", bytearray(b""), bytearray(b"{}"), 1, 1.0, None,
                    True, [b""], {b"": 1}):
            with self.assertRaises(ValueError):
                self.other.merge_bytes(bad, b"")
            with self.assertRaises(ValueError):
                self.other.merge_bytes(b"", bad)

    def test_left_type_checked_first(self):
        with self.assertRaises(ValueError):
            self.other.merge_bytes("left", b"\xff")
        with self.assertRaises(ValueError):
            self.other.merge_bytes("left", "right")

    def test_never_reads_creates_or_modifies_constructor_path(self):
        phantom = self.path.with_name("phantom.jsonl")
        chain = AuditChain(phantom)
        self.seed(self.chain, ("a", {"i": 1}), ("b", {"i": 1}),
                  ("a", {"i": 2}))
        data = self.snapshot()
        r = chain.merge_bytes(b"", b"")
        self.assertTrue(r["ok"])
        self.assertFalse(phantom.exists())
        r = chain.merge_bytes(data, data)
        self.assertTrue(r["ok"])
        self.assertFalse(phantom.exists())

    def test_does_not_mutate_buffers(self):
        self.seed(self.chain, ("a", {"i": 1}))
        left = bytearray(self.snapshot())
        before = bytes(left)
        with self.assertRaises(ValueError):
            self.other.merge_bytes(left, b"")
        self.assertEqual(bytes(left), before)

    # --- empty snapshots ---

    def test_merge_two_empty_snapshots(self):
        r = self.other.merge_bytes(b"", b"")
        self.assertEqual(r["ok"], True)
        self.assertEqual(r["data"], b"")
        self.assertEqual(r["manifest"], {
            "version": MANIFEST_VERSION,
            "byte_length": 0,
            "byte_sha256": hashlib.sha256(b"").hexdigest(),
            "tenants": [],
        })

    def test_empty_side_keeps_the_other_canonically(self):
        self.seed(self.chain, ("a", {"i": 1}), ("b", {"i": 1}))
        data = self.snapshot()
        r = self.other.merge_bytes(b"", data)
        self.assertTrue(r["ok"])
        self.assertEqual(r["data"], data)
        r = self.other.merge_bytes(data, b"")
        self.assertTrue(r["ok"])
        self.assertEqual(r["data"], data)

    # --- corruption: compare_bytes-shaped verdict, left first ---

    def test_corrupt_side_is_not_merged(self):
        item, good = self.row("t", 1)
        r = self.other.merge_bytes(b"{oops\n", good)
        self.assertEqual(r, {
            "ok": False, "side": "left", "at": 1,
            "tenant": None, "reason": "missing",
        })
        r = self.other.merge_bytes(good, b"\xff")
        self.assertEqual(r, {
            "ok": False, "side": "right", "at": 1,
            "tenant": None, "reason": "missing",
        })

    def test_left_corruption_reported_when_both_sides_corrupt(self):
        _t, good = self.row("t", 1)
        bad_digest, _ = self.row("t", 1, prev="9" * 64)
        r = self.other.merge_bytes(record_bytes(bad_digest) + b"\xff",
                                   good + b"{later\n")
        self.assertEqual(r, {
            "ok": False, "side": "left", "at": 1,
            "tenant": "t", "reason": "digest",
        })
        r = self.other.merge_bytes(b"{oops\n", b"\xff")
        self.assertEqual((r["side"], r["at"], r["tenant"], r["reason"]),
                         ("left", 1, None, "missing"))

    def test_corruption_reasons(self):
        _t, good1 = self.row("t", 1)
        gap = {"tenant": "t", "seq": 3, "event": {}, "prev": "9" * 64}
        gap["hash"] = AuditChain._hash(gap)
        r = self.other.merge_bytes(good1 + record_bytes(gap), good1)
        self.assertEqual((r["ok"], r["side"], r["at"], r["tenant"],
                          r["reason"]),
                         (False, "left", 2, "t", "sequence"))
        r = self.other.merge_bytes(good1, good1 + b"\xff")
        self.assertEqual((r["ok"], r["side"], r["at"], r["tenant"],
                          r["reason"]),
                         (False, "right", 2, None, "missing"))

    def test_extra_field_rejected_as_missing_even_compare_accepts_it(self):
        # The merge output has to satisfy import_all's exact-five-field
        # contract, so an extra field makes the side corrupt here even
        # though the looser verify_all_bytes pass (and compare_bytes) treats
        # it as valid.
        item, _ = self.row("t", 1)
        item["extra"] = True
        raw = record_bytes(item)
        self.assertTrue(self.other.compare_bytes(raw, raw)["equal"])
        _t, good = self.row("t", 1)
        r = self.other.merge_bytes(raw, good)
        self.assertEqual((r["ok"], r["side"], r["at"], r["tenant"],
                          r["reason"]),
                         (False, "left", 1, "t", "missing"))
        r = self.other.merge_bytes(good, raw)
        self.assertEqual(r["side"], "right")

    # --- prefix merges ---

    def test_identical_inputs_merge_to_one_copy(self):
        self.seed(self.chain, ("a", {"i": 1}), ("a", {"i": 2}))
        data = self.snapshot()
        r = self.other.merge_bytes(data, bytes(data))
        self.assertTrue(r["ok"])
        self.assertEqual(r["data"], data)

    def test_right_suffix_extends_the_left_prefix(self):
        left_chain = AuditChain(self.path.with_name("l.jsonl"))
        right_chain = AuditChain(self.path.with_name("r.jsonl"))
        self.seed(left_chain, ("a", {"i": 1}), ("b", {"i": 1}),
                  ("a", {"i": 2}))
        self.seed(right_chain, ("a", {"i": 1}), ("a", {"i": 2}),
                  ("b", {"i": 1}), ("b", {"i": 2}), ("c", {"i": 1}))
        left, right = self.snapshot(left_chain), self.snapshot(right_chain)
        r = self.other.merge_bytes(left, right)
        self.assertTrue(r["ok"])
        expected_chain = AuditChain(self.path.with_name("m.jsonl"))
        expected_chain.import_all(r["data"])
        heads = {h["tenant"]: (h["count"], h["hash"])
                 for h in expected_chain.heads()["tenants"]}
        self.assertEqual(
            {t: c for t, (c, _h) in heads.items()},
            {"a": 2, "b": 2, "c": 1},
        )
        # Left records keep their physical order; the b suffix (b2) precedes
        # the right-only c record because that is their right physical order.
        self.seq_order = [
            (it["tenant"], it["seq"])
            for line in r["data"].decode("utf-8").splitlines()
            for it in [json.loads(line)]
        ]
        self.assertEqual(self.seq_order,
                         [("a", 1), ("b", 1), ("a", 2), ("b", 2), ("c", 1)])

    def test_left_longer_chain_is_a_valid_prefix(self):
        left_chain = AuditChain(self.path.with_name("l.jsonl"))
        right_chain = AuditChain(self.path.with_name("r.jsonl"))
        self.seed(left_chain, ("a", {"i": 1}), ("a", {"i": 2}),
                  ("a", {"i": 3}))
        self.seed(right_chain, ("a", {"i": 1}))
        r = self.other.merge_bytes(self.snapshot(left_chain),
                                   self.snapshot(right_chain))
        self.assertTrue(r["ok"])
        self.assertEqual(r["data"], self.snapshot(left_chain))

    def test_physical_interleaving_never_causes_a_conflict(self):
        left_chain = AuditChain(self.path.with_name("l.jsonl"))
        right_chain = AuditChain(self.path.with_name("r.jsonl"))
        self.seed(left_chain, ("a", {"i": 1}), ("b", {"i": 1}),
                  ("a", {"i": 2}), ("a", {"i": 3}), ("b", {"i": 2}))
        self.seed(right_chain, ("b", {"i": 1}), ("b", {"i": 2}),
                  ("a", {"i": 1}), ("a", {"i": 2}), ("a", {"i": 3}),
                  ("c", {"i": 1}))
        r = self.other.merge_bytes(self.snapshot(left_chain),
                                   self.snapshot(right_chain))
        self.assertTrue(r["ok"])
        # the c suffix is appended after all left records, regardless of its
        # early physical position on the right
        self.assertEqual(
            [(json.loads(line)["tenant"], json.loads(line)["seq"])
             for line in r["data"].decode("utf-8").splitlines()],
            [("a", 1), ("b", 1), ("a", 2), ("a", 3), ("b", 2), ("c", 1)],
        )

    def test_canonical_tenant_identity_aligns_records(self):
        _i, left = self.row({"a": 1, "b": 2}, 1, event={"v": 1})
        _j, right_prefix = self.row({"b": 2, "a": 1}, 1, event={"v": 1})
        _k, right_extra = self.row({"b": 2, "a": 1}, 2,
                                   prev=_j["hash"], event={"v": 2})
        r = self.other.merge_bytes(left, right_prefix + right_extra)
        self.assertTrue(r["ok"])
        self.assertEqual(
            [json.loads(line)["tenant"] for line in
             r["data"].decode("utf-8").splitlines()],
            [{"a": 1, "b": 2}, {"a": 1, "b": 2}],
        )

    def test_input_is_normalized_to_canonical_jsonl(self):
        item, canonical = self.row("t", 1, event={"msg": "世界"})
        # Same record with a scrambled key order, interior whitespace and a
        # CRLF physical end: still strict JSON on an LF physical line, but
        # not canonical.
        raw = (
            '{"hash": ' + json.dumps(item["hash"])
            + ', "tenant": ' + json.dumps(item["tenant"])
            + ', "seq": 1, "event": ' + json.dumps(item["event"],
                                                   allow_nan=False)
            + ', "prev": ' + json.dumps(ZERO) + '}\r\n'
        ).encode("utf-8")
        self.assertNotEqual(raw, canonical)
        r = self.other.merge_bytes(raw, b"")
        self.assertTrue(r["ok"])
        self.assertEqual(r["data"], canonical)
        self.assertEqual(r["manifest"], self.other.manifest_bytes(canonical))

    # --- conflicts ---

    def test_conflicting_event_returns_both_records_and_no_output(self):
        _i, left = self.row("t", 1, event={"v": 1})
        _j, right = self.row("t", 1, event={"v": 2})
        r = self.other.merge_bytes(left, right)
        self.assertEqual(set(r), {"ok", "reason", "tenant", "seq",
                                  "left", "right"})
        self.assertEqual((r["ok"], r["reason"], r["tenant"], r["seq"]),
                         (False, "conflict", "t", 1))
        self.assertEqual(r["left"]["event"], {"v": 1})
        self.assertEqual(r["right"]["event"], {"v": 2})

    def test_conflict_at_later_seq_after_shared_prefix(self):
        _i1, l1 = self.row("t", 1, event={"v": 1})
        _i2, l2 = self.row("t", 2, prev=_i1["hash"], event={"v": 2})
        _j1, r1 = self.row("t", 1, event={"v": 1})
        _j2, r2 = self.row("t", 2, prev=_j1["hash"], event={"v": 9})
        r = self.other.merge_bytes(l1 + l2, r1 + r2)
        self.assertEqual((r["ok"], r["reason"], r["seq"]),
                         (False, "conflict", 2))
        self.assertEqual(r["left"]["hash"], _i2["hash"])
        self.assertEqual(r["right"]["hash"], _j2["hash"])

    def test_conflict_tenant_order_follows_left_first_appearance(self):
        # Both a and b conflict; the left side's first-appearance order picks
        # the one reported even though the right physically names b first.
        lc = AuditChain(self.path.with_name("l.jsonl"))
        rc = AuditChain(self.path.with_name("r.jsonl"))
        self.seed(lc, ("a", {"v": 1}), ("b", {"v": 1}))
        self.seed(rc, ("b", {"v": 9}), ("a", {"v": 9}))
        r = self.other.merge_bytes(self.snapshot(lc), self.snapshot(rc))
        self.assertEqual((r["reason"], r["tenant"], r["seq"]),
                         ("conflict", "a", 1))
        lc2 = AuditChain(self.path.with_name("l2.jsonl"))
        self.seed(lc2, ("b", {"v": 1}), ("a", {"v": 1}))
        r = self.other.merge_bytes(self.snapshot(lc2), self.snapshot(rc))
        self.assertEqual(r["tenant"], "b")

    def test_conflict_checked_after_all_matching_left_tenants(self):
        # z matches (left names it first), the right-only tenant is appended,
        # and the conflict on shared tenant a (which the left names later)
        # still wins in the left first-appearance ordering; here the left
        # order is [z, a], so z matches before a's conflict is reported.
        lc = AuditChain(self.path.with_name("l.jsonl"))
        rc = AuditChain(self.path.with_name("r.jsonl"))
        self.seed(lc, ("z", {"v": 1}), ("a", {"v": 1}))
        self.seed(rc, ("a", {"v": 9}), ("z", {"v": 1}))
        r = self.other.merge_bytes(self.snapshot(lc), self.snapshot(rc))
        self.assertEqual((r["reason"], r["tenant"]), ("conflict", "a"))

    def test_conflict_tenant_value_uses_left_first_appearance(self):
        _ia, l1 = self.row({"k": 1}, 1, event={"v": 1})
        _ib, l2 = self.row({"k": 1}, 2, prev=_ia["hash"], event={"v": 2})
        _j1, r1 = self.row({"k": 1}, 1, event={"v": 1})
        _j2, r2 = self.row({"k": 1}, 2, prev=_j1["hash"], event={"v": 9})
        r = self.other.merge_bytes(l1 + l2, r1 + r2)
        self.assertEqual((r["reason"], r["seq"]), ("conflict", 2))
        self.assertEqual(r["tenant"], {"k": 1})

    # --- success result: data, manifest, downstream compatibility ---

    def test_manifest_describes_the_merged_data_byte_for_byte(self):
        left_chain = AuditChain(self.path.with_name("l.jsonl"))
        right_chain = AuditChain(self.path.with_name("r.jsonl"))
        self.seed(left_chain, ("a", {"i": 1}), ("b", {"i": 1}))
        self.seed(right_chain, ("a", {"i": 1}), ("a", {"i": 2}),
                  ("c", {"i": 1}))
        r = self.other.merge_bytes(self.snapshot(left_chain),
                                   self.snapshot(right_chain))
        self.assertTrue(r["ok"])
        data, manifest = r["data"], r["manifest"]
        self.assertEqual(manifest["version"], MANIFEST_VERSION)
        self.assertEqual(set(manifest),
                         {"version", "byte_length", "byte_sha256",
                          "tenants"})
        self.assertEqual(manifest["byte_length"], len(data))
        self.assertEqual(manifest["byte_sha256"],
                         hashlib.sha256(data).hexdigest())
        # tenants follow the merged physical first-appearance order: the left
        # order (a, b) then the right-only tenant c
        self.assertEqual([t["tenant"] for t in manifest["tenants"]],
                         ["a", "b", "c"])
        a_tail = AuditChain._hash(
            {"tenant": "a", "seq": 2, "event": {"i": 2},
             "prev": AuditChain._hash(
                 {"tenant": "a", "seq": 1, "event": {"i": 1},
                  "prev": ZERO})})
        self.assertEqual(manifest["tenants"], [
            {"tenant": "a", "count": 2, "hash": a_tail},
            {"tenant": "b", "count": 1,
             "hash": AuditChain._hash(
                 {"tenant": "b", "seq": 1, "event": {"i": 1},
                  "prev": ZERO})},
            {"tenant": "c", "count": 1,
             "hash": AuditChain._hash(
                 {"tenant": "c", "seq": 1, "event": {"i": 1},
                  "prev": ZERO})},
        ])
        # the manifest is exactly what the existing manifest entry computes
        self.assertEqual(manifest, self.other.manifest_bytes(data))

    def test_merged_data_verifies_offline_and_matches_manifest(self):
        left_chain = AuditChain(self.path.with_name("l.jsonl"))
        right_chain = AuditChain(self.path.with_name("r.jsonl"))
        self.seed(left_chain, ("a", {"i": 1}), ("b", {"i": 1}),
                  ("a", {"i": 2}))
        self.seed(right_chain, ("a", {"i": 1}), ("a", {"i": 2}),
                  ("a", {"i": 3}), ("c", {"i": 1}))
        r = self.other.merge_bytes(self.snapshot(left_chain),
                                   self.snapshot(right_chain))
        data, manifest = r["data"], r["manifest"]
        self.assertTrue(self.other.verify_all_bytes(data)["ok"])
        check = self.other.verify_manifest_bytes(data, manifest)
        self.assertTrue(check["ok"])
        self.assertEqual(check["manifest"], manifest)

    def test_merged_data_imports_into_a_fresh_chain(self):
        left_chain = AuditChain(self.path.with_name("l.jsonl"))
        right_chain = AuditChain(self.path.with_name("r.jsonl"))
        self.seed(left_chain, ("a", {"i": 1}), ("b", {"i": 1}))
        self.seed(right_chain, ("a", {"i": 1}), ("a", {"i": 2}),
                  ("c", {"i": 1}))
        r = self.other.merge_bytes(self.snapshot(left_chain),
                                   self.snapshot(right_chain))
        target = AuditChain(self.path.with_name("target.jsonl"))
        target.import_all(r["data"])
        verify = target.verify_all()
        self.assertTrue(verify["ok"])
        self.assertEqual(
            [(t["tenant"], t["count"]) for t in verify["tenants"]],
            [("a", 2), ("b", 1), ("c", 1)],
        )
        # appends continue from the merged tail
        record = target.append("a", {"i": 3})
        self.assertEqual((record["seq"], record["prev"]),
                         (3, r["manifest"]["tenants"][0]["hash"]))
        target.verify("a", expected_count=3)

    def test_digest_values_are_never_rewritten(self):
        left_chain = AuditChain(self.path.with_name("l.jsonl"))
        right_chain = AuditChain(self.path.with_name("r.jsonl"))
        self.seed(left_chain, ("a", {"i": 1}))
        self.seed(right_chain, ("a", {"i": 1}), ("a", {"i": 2}))
        r = self.other.merge_bytes(self.snapshot(left_chain),
                                   self.snapshot(right_chain))
        items = [json.loads(line)
                 for line in r["data"].decode("utf-8").splitlines()]
        source = [json.loads(line)
                  for line in self.snapshot(right_chain).decode("utf-8")
                  .splitlines()]
        self.assertEqual([it["hash"] for it in items],
                         [it["hash"] for it in source])
        self.assertEqual([it["prev"] for it in items],
                         [it["prev"] for it in source])


if __name__ == "__main__":
    unittest.main()
