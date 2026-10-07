import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app import AuditChain, ZERO


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
        item = {"tenant": tenant, "seq": seq, "event": event or {}, "prev": prev}
        item["hash"] = hash_ if hash_ is not None else AuditChain._hash(item)
        return item, record_bytes(item)

    def chain_bytes(self, tenant, events):
        out = []
        prev = ZERO
        for seq, event in enumerate(events, 1):
            item, raw = self.row(tenant, seq, prev=prev, event=event)
            out.append(raw)
            prev = item["hash"]
        return b"".join(out)

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
        self.seed(self.chain, ("a", {"i": 1}), ("b", {"i": 1}))
        data = self.snapshot()
        r = chain.merge_bytes(data, b"")
        self.assertTrue(r["ok"])
        self.assertFalse(phantom.exists())

    # --- empty snapshots ---

    def test_empty_snapshots_merge_to_empty(self):
        r = self.other.merge_bytes(b"", b"")
        self.assertEqual(r["ok"], True)
        self.assertEqual(r["data"], b"")
        self.assertEqual(r["manifest"], {
            "version": 1,
            "byte_length": 0,
            "byte_sha256": hashlib.sha256(b"").hexdigest(),
            "tenants": [],
        })

    def test_empty_side_yields_other_side_canonical(self):
        self.seed(self.chain, ("a", {"i": 1}), ("b", {"i": 1}),
                  ("a", {"i": 2}))
        data = self.snapshot()
        for left, right in ((data, b""), (b"", data)):
            r = self.other.merge_bytes(left, right)
            self.assertTrue(r["ok"])
            self.assertEqual(r["data"], data)  # already canonical
            self.assertEqual(r["manifest"],
                             self.other.manifest_bytes(data))

    # --- corruption: compare_bytes' first-defect report, left first ---

    def test_corrupt_snapshot_is_not_merged(self):
        _i, good = self.row("t", 1)
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
        r = self.other.merge_bytes(b"{oops\n", b"\xff")
        self.assertEqual(r["side"], "left")
        self.assertEqual((r["at"], r["tenant"], r["reason"]),
                         (1, None, "missing"))

    def test_corruption_reasons_match_compare_bytes(self):
        _i, good1 = self.row("t", 1)
        gap = {"tenant": "t", "seq": 3, "event": {}, "prev": "9" * 64}
        gap["hash"] = AuditChain._hash(gap)
        r = self.other.merge_bytes(good1 + record_bytes(gap), good1)
        self.assertEqual((r["ok"], r["side"], r["at"], r["tenant"],
                          r["reason"]),
                         (False, "left", 2, "t", "sequence"))
        bad_hash = {"tenant": "t", "seq": 1, "event": {}, "prev": ZERO,
                    "hash": "a" * 64}
        r = self.other.merge_bytes(good1, record_bytes(bad_hash))
        self.assertEqual((r["ok"], r["side"], r["at"], r["tenant"],
                          r["reason"]),
                         (False, "right", 1, "t", "digest"))

    def test_extra_field_is_a_missing_defect(self):
        # Merge re-serializes its output, so the fixed five-field contract
        # applies to the inputs: an extra key is a missing-class defect.
        item, _ = self.row("t", 1)
        item["extra"] = 1
        r = self.other.merge_bytes(record_bytes(item), b"")
        self.assertEqual((r["ok"], r["side"], r["at"], r["tenant"],
                          r["reason"]),
                         (False, "left", 1, "t", "missing"))

    # --- conflicts ---

    def test_divergent_event_is_a_conflict(self):
        _i, left = self.row("t", 1, event={"v": 1})
        _i, right = self.row("t", 1, event={"v": 2})
        r = self.other.merge_bytes(left, right)
        self.assertEqual((r["ok"], r["reason"], r["tenant"], r["seq"]),
                         (False, "conflict", "t", 1))
        self.assertEqual(r["left"]["event"], {"v": 1})
        self.assertEqual(r["right"]["event"], {"v": 2})
        self.assertNotIn("data", r)

    def test_conflict_at_later_seq_reports_first_divergence(self):
        left = self.chain_bytes("t", [{"i": 1}, {"i": 2}, {"i": 3}])
        right = self.chain_bytes("t", [{"i": 1}, {"i": 2}, {"i": 30}])
        r = self.other.merge_bytes(left, right)
        self.assertEqual((r["ok"], r["reason"], r["tenant"], r["seq"]),
                         (False, "conflict", "t", 3))
        self.assertEqual(r["left"]["event"], {"i": 3})
        self.assertEqual(r["right"]["event"], {"i": 30})

    def test_conflict_tenant_order_prefers_left_first_appearance(self):
        lc = AuditChain(self.path.with_name("l.jsonl"))
        rc = AuditChain(self.path.with_name("r.jsonl"))
        self.seed(lc, ("a", {"v": 1}), ("b", {"v": 1}))
        self.seed(rc, ("b", {"v": 9}), ("a", {"v": 9}))
        r = self.other.merge_bytes(self.snapshot(lc), self.snapshot(rc))
        self.assertEqual((r["reason"], r["tenant"], r["seq"]),
                         ("conflict", "a", 1))

    def test_physical_interleaving_is_not_a_conflict(self):
        # Same logical chains, different physical interleaving: no conflict.
        lc = AuditChain(self.path.with_name("l.jsonl"))
        rc = AuditChain(self.path.with_name("r.jsonl"))
        self.seed(lc, ("a", {"i": 1}), ("b", {"i": 1}), ("a", {"i": 2}))
        self.seed(rc, ("b", {"i": 1}), ("a", {"i": 1}), ("a", {"i": 2}))
        r = self.other.merge_bytes(self.snapshot(lc), self.snapshot(rc))
        self.assertTrue(r["ok"])
        # left physical order is kept; the right adds nothing
        self.assertEqual(r["data"], self.snapshot(lc))

    # --- successful merges ---

    def test_prefix_merge_appends_only_the_suffix(self):
        left = self.chain_bytes("t", [{"i": 1}, {"i": 2}])
        right = self.chain_bytes("t", [{"i": 1}, {"i": 2}, {"i": 3}])
        r = self.other.merge_bytes(left, right)
        self.assertTrue(r["ok"])
        self.assertEqual(r["data"], right)  # shared prefix kept once
        # and the symmetric call: left is the longer side
        r = self.other.merge_bytes(right, left)
        self.assertTrue(r["ok"])
        self.assertEqual(r["data"], right)

    def test_disjoint_tenants_merge_left_then_right_order(self):
        left = self.chain_bytes("a", [{"i": 1}]) + self.chain_bytes("b", [{"j": 1}])
        right = self.chain_bytes("c", [{"k": 1}]) + self.chain_bytes("d", [{"l": 1}])
        r = self.other.merge_bytes(left, right)
        self.assertTrue(r["ok"])
        self.assertEqual(r["data"], left + right)

    def test_right_suffix_keeps_right_physical_order(self):
        # Left has a and b at seq 1; right extends both and interleaves the
        # suffix records c1, a2, b2 -- the suffix keeps that physical order.
        lc = AuditChain(self.path.with_name("l.jsonl"))
        self.seed(lc, ("a", {"i": 1}), ("b", {"j": 1}))
        rc = AuditChain(self.path.with_name("r.jsonl"))
        self.seed(rc, ("a", {"i": 1}), ("b", {"j": 1}), ("c", {"k": 1}),
                  ("a", {"i": 2}), ("b", {"j": 2}))
        left, right = self.snapshot(lc), self.snapshot(rc)
        r = self.other.merge_bytes(left, right)
        self.assertTrue(r["ok"])
        a2 = self.chain_bytes("a", [{"i": 1}, {"i": 2}])
        b2 = self.chain_bytes("b", [{"j": 1}, {"j": 2}])
        a1 = self.chain_bytes("a", [{"i": 1}])
        b1 = self.chain_bytes("b", [{"j": 1}])
        suffix = (self.chain_bytes("c", [{"k": 1}])
                  + a2[len(a1):] + b2[len(b1):])
        self.assertEqual(r["data"], left + suffix)

    def test_merged_output_verifies_and_imports(self):
        lc = AuditChain(self.path.with_name("l.jsonl"))
        rc = AuditChain(self.path.with_name("r.jsonl"))
        self.seed(lc, ("a", {"i": 1}), ("b", {"j": 1}))
        self.seed(rc, ("b", {"j": 1}), ("a", {"i": 1}), ("a", {"i": 2}),
                  ("c", {"k": 1}))
        r = self.other.merge_bytes(self.snapshot(lc), self.snapshot(rc))
        self.assertTrue(r["ok"])
        # The merged snapshot passes the existing offline verification.
        v = self.other.verify_all_bytes(r["data"])
        self.assertTrue(v["ok"])
        self.assertEqual([e["tenant"] for e in v["tenants"]],
                         ["a", "b", "c"])
        # ... and imports cleanly into a fresh log.
        target = AuditChain(self.path.with_name("target.jsonl"))
        target.import_all(r["data"])
        self.assertEqual(target.export_all(), r["data"])

    def test_field_values_and_digests_are_not_rewritten(self):
        event = {"nested": [1, 2.5, {"x": True}], "s": "héllo"}
        left = self.chain_bytes("t", [event])
        right = self.chain_bytes("t", [event, {"i": 2}])
        r = self.other.merge_bytes(left, right)
        self.assertTrue(r["ok"])
        lines = r["data"].decode("utf-8").splitlines()
        first = json.loads(lines[0])
        second = json.loads(lines[1])
        self.assertEqual(first["event"], event)
        originals = [json.loads(raw) for raw in
                     right.decode("utf-8").splitlines()]
        self.assertEqual([first, second], originals)
        self.assertEqual(first["hash"], originals[0]["hash"])
        self.assertEqual(second["prev"], first["hash"])

    def test_canonical_tenant_identity_aligns_across_spellings(self):
        # Same object tenant spelled with different key orders on each side:
        # one shared chain, emitted once with the left spelling.
        _i, left = self.row({"a": 1, "b": 2}, 1)
        _i, right = self.row({"b": 2, "a": 1}, 1)
        r = self.other.merge_bytes(left, right)
        self.assertTrue(r["ok"])
        self.assertEqual(r["data"], left)
        self.assertEqual(r["manifest"]["tenants"][0]["tenant"],
                         {"a": 1, "b": 2})

    def test_manifest_describes_merged_bytes(self):
        lc = AuditChain(self.path.with_name("l.jsonl"))
        rc = AuditChain(self.path.with_name("r.jsonl"))
        self.seed(lc, ("a", {"i": 1}), ("b", {"j": 1}))
        self.seed(rc, ("a", {"i": 1}), ("a", {"i": 2}), ("c", {"k": 1}))
        r = self.other.merge_bytes(self.snapshot(lc), self.snapshot(rc))
        self.assertTrue(r["ok"])
        m = r["manifest"]
        self.assertEqual(m["version"], 1)
        self.assertEqual(m["byte_length"], len(r["data"]))
        self.assertEqual(m["byte_sha256"],
                         hashlib.sha256(r["data"]).hexdigest())
        self.assertEqual([e["tenant"] for e in m["tenants"]],
                         ["a", "b", "c"])
        self.assertEqual([e["count"] for e in m["tenants"]], [2, 1, 1])
        heads = self.other.heads_bytes(r["data"])["tenants"]
        self.assertEqual([e["hash"] for e in m["tenants"]],
                         [e["hash"] for e in heads])
        # identical to the manifest the existing entry computes over data
        self.assertEqual(m, self.other.manifest_bytes(r["data"]))

    def test_noncanonical_input_is_reemitted_canonically(self):
        # A valid but non-canonical line (extra whitespace, unsorted keys)
        # merges into canonical JSONL without changing any value.
        item, canonical = self.row("t", 1, event={"v": 1})
        messy = ('{ "hash": "%s", "prev": "%s", "event": {"v": 1}, '
                 '"seq": 1, "tenant": "t" }\n' % (item["hash"], ZERO)
                 ).encode("utf-8")
        r = self.other.merge_bytes(messy, b"")
        self.assertTrue(r["ok"])
        self.assertEqual(r["data"], canonical)


if __name__ == "__main__":
    unittest.main()
