import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import app
from app import AuditChain, AuditChainStateError


ZERO = "0" * 64


class AuditChainTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.path = Path(self.tmp.name) / "audit.jsonl"
        self.chain = AuditChain(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, rows):
        with self.path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(r if isinstance(r, str) else json.dumps(r, sort_keys=True))
                f.write("\n")

    def read(self):
        return self.path.read_text(encoding="utf-8")

    def test_empty_verify(self):
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": 0})

    def test_append_starts_at_seq_one_with_zero_prev(self):
        item = self.chain.append("t", {"a": 1})
        self.assertEqual(item["seq"], 1)
        self.assertEqual(item["prev"], ZERO)
        self.assertEqual(len(item["hash"]), 64)
        self.assertEqual(item["hash"], item["hash"].lower())
        row = json.loads(self.read())
        self.assertEqual(set(row), {"tenant", "seq", "event", "prev", "hash"})

    def test_append_chains(self):
        a = self.chain.append("t", {"i": 1})
        b = self.chain.append("t", {"i": 2})
        self.assertEqual(b["seq"], 2)
        self.assertEqual(b["prev"], a["hash"])
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": 2})

    def test_interleaved_tenants(self):
        a1 = self.chain.append("a", {})
        b1 = self.chain.append("b", {})
        a2 = self.chain.append("a", {})
        self.assertEqual(a2["seq"], 2)
        self.assertEqual(a2["prev"], a1["hash"])
        self.assertEqual(b1["seq"], 1)
        self.assertEqual(self.chain.verify("a"), {"ok": True, "count": 2})
        self.assertEqual(self.chain.verify("b"), {"ok": True, "count": 1})

    def test_expected_count_match_short_and_exact(self):
        self.chain.append("t", {})
        self.chain.append("t", {})
        self.assertEqual(self.chain.verify("t", 2), {"ok": True, "count": 2})
        r = self.chain.verify("t", 3)
        self.assertEqual((r["ok"], r["at"], r["reason"]), (False, 3, "missing"))
        r = self.chain.verify("t", 1)
        self.assertEqual((r["ok"], r["at"], r["reason"]), (False, 2, "sequence"))

    def test_expected_count_interleaved(self):
        self.chain.append("a", {})
        self.chain.append("b", {})
        self.assertEqual(self.chain.verify("a", 1), {"ok": True, "count": 1})

    # --- corruption: verify classification ---

    def test_unparseable_line_at_line_number(self):
        other = {"tenant": "x", "seq": 1, "event": {}, "prev": ZERO}
        other["hash"] = AuditChain._hash(other)
        with self.path.open("w", encoding="utf-8") as f:
            f.write(json.dumps(other, sort_keys=True) + "\n")
            f.write("{not json\n")
        r = self.chain.verify("t")
        self.assertFalse(r["ok"])
        self.assertEqual(r["at"], 2)  # file line number, not expected seq 1
        self.assertEqual(r["reason"], "missing")

    def test_unparseable_other_tenant_line_still_fatal(self):
        self.write(['{"tenant": "x", broken'])
        r = self.chain.verify("t")
        self.assertEqual((False, 1, "missing"), (r["ok"], r["at"], r["reason"]))

    def test_non_object_line(self):
        self.write(["[1,2,3]"])
        r = self.chain.verify("t")
        self.assertEqual((r["at"], r["reason"]), (1, "missing"))

    def test_missing_field(self):
        self.write([{"tenant": "t", "seq": 1, "event": {}, "prev": ZERO}])  # no hash
        r = self.chain.verify("t")
        self.assertEqual((r["at"], r["reason"]), (1, "missing"))

    def test_bad_first_prev_is_digest(self):
        row = {"tenant": "t", "seq": 1, "event": {}, "prev": "f" * 64}
        row["hash"] = AuditChain._hash(row)
        self.write([row])
        r = self.chain.verify("t")
        self.assertEqual((r["at"], r["reason"]), (1, "digest"))

    def test_bad_first_seq_is_sequence(self):
        row = {"tenant": "t", "seq": 2, "event": {}, "prev": ZERO}
        row["hash"] = AuditChain._hash(row)
        self.write([row])
        r = self.chain.verify("t")
        self.assertEqual((r["at"], r["reason"]), (1, "sequence"))

    def test_gap_is_sequence_at_expected(self):
        good = {"tenant": "t", "seq": 1, "event": {}, "prev": ZERO}
        good["hash"] = AuditChain._hash(good)
        row = {"tenant": "t", "seq": 3, "event": {}, "prev": good["hash"]}
        row["hash"] = AuditChain._hash(row)
        self.write([good, row])
        r = self.chain.verify("t")
        self.assertEqual((r["at"], r["reason"]), (2, "sequence"))

    def test_bad_prev_link_is_digest(self):
        self.chain.append("t", {})
        row = {"tenant": "t", "seq": 2, "event": {}, "prev": "9" * 64}
        row["hash"] = AuditChain._hash(row)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        r = self.chain.verify("t")
        self.assertEqual((r["at"], r["reason"]), (2, "digest"))

    def test_tampered_hash_is_digest(self):
        self.chain.append("t", {})
        row = {"tenant": "t", "seq": 2, "event": {}, "prev": ZERO}
        row["hash"] = "deadbeef" * 8
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        r = self.chain.verify("t")
        self.assertEqual((r["at"], r["reason"]), (2, "digest"))

    def test_tampered_event_is_digest(self):
        item = self.chain.append("t", {"v": 1})
        rows = [json.loads(l) for l in self.read().splitlines()]
        rows[0]["event"] = {"v": 2}
        self.write(rows)
        r = self.chain.verify("t")
        self.assertEqual((r["at"], r["reason"]), (1, "digest"))

    def test_first_error_reported_with_interleaving(self):
        # bad seq record for a at seq2 position, b in between is fine
        good = {"tenant": "a", "seq": 1, "event": {}, "prev": ZERO}
        good["hash"] = AuditChain._hash(good)
        b = {"tenant": "b", "seq": 1, "event": {}, "prev": ZERO}
        b["hash"] = AuditChain._hash(b)
        bad = {"tenant": "a", "seq": 3, "event": {}, "prev": good["hash"]}
        bad["hash"] = AuditChain._hash(bad)
        self.write([good, b, bad])
        r = self.chain.verify("a")
        self.assertEqual((r["at"], r["reason"]), (2, "sequence"))
        self.assertEqual(self.chain.verify("b"), {"ok": True, "count": 1})

    # --- append must reject corrupt history without modifying file ---

    def test_append_rejects_and_keeps_file(self):
        self.chain.append("t", {})
        rows = [json.loads(l) for l in self.read().splitlines()]
        rows[0]["event"] = {"x": 9}
        self.write(rows)
        before = self.read()
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.append("t", {})
        self.assertEqual(cm.exception.tenant, "t")
        self.assertEqual(cm.exception.seq, 1)
        self.assertEqual(self.read(), before)

    def test_append_error_seq_falls_back_to_expected(self):
        # unparseable line: seq unknown -> first affected seq is expected (1)
        self.write(["{oops"])
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.append("t", {})
        self.assertEqual(cm.exception.seq, 1)

    def test_append_after_other_tenant_corruption_irrelevant(self):
        # corruption belongs to another tenant only: t appends fine
        self.chain.append("b", {})
        rows = [json.loads(l) for l in self.read().splitlines()]
        rows[0]["event"] = {"x": 1}
        self.write(rows)
        item = self.chain.append("t", {})
        self.assertEqual(item["seq"], 1)
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": 1})

    def test_unicode_event_hash_roundtrip(self):
        item = self.chain.append("t", {"name": "审计"})
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": 1})
        # independently recompute from the file
        row = json.loads(self.read())
        payload = json.dumps(
            {k: row[k] for k in ("tenant", "seq", "event", "prev")},
            sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
        import hashlib
        self.assertEqual(hashlib.sha256(payload).hexdigest(), row["hash"])

    # --- verify_all: whole-file, all tenants ---

    def test_verify_all_missing_path_is_empty(self):
        self.assertFalse(self.path.exists())
        self.assertEqual(self.chain.verify_all(), {"ok": True, "tenants": []})

    def test_verify_all_empty_file(self):
        self.write([])
        self.assertEqual(self.chain.verify_all(), {"ok": True, "tenants": []})

    def test_verify_all_interleaved_first_appearance_order(self):
        self.chain.append("a", {})
        self.chain.append("b", {})
        self.chain.append("a", {})
        self.assertEqual(
            self.chain.verify_all(),
            {"ok": True, "tenants": [
                {"tenant": "a", "count": 2},
                {"tenant": "b", "count": 1},
            ]},
        )

    def test_verify_all_preserves_tenant_types_as_distinct(self):
        self.write([
            self.hashed(1, 1),
            self.hashed("1", 1),
            self.hashed(True, 1),
        ])
        self.assertEqual(
            self.chain.verify_all(),
            {"ok": True, "tenants": [
                {"tenant": 1, "count": 1},
                {"tenant": "1", "count": 1},
                {"tenant": True, "count": 1},
            ]},
        )

    def hashed(self, tenant, seq, prev=ZERO):
        row = {"tenant": tenant, "seq": seq, "event": {}, "prev": prev}
        row["hash"] = AuditChain._hash(row)
        return row

    def test_verify_all_sequence_error_reports_physical_line_and_tenant(self):
        good = self.hashed("a", 1)
        b = self.hashed("b", 1)
        bad = {"tenant": "a", "seq": 3, "event": {}, "prev": good["hash"]}
        bad["hash"] = AuditChain._hash(bad)
        self.write([good, b, bad])
        self.assertEqual(
            self.chain.verify_all(),
            {"ok": False, "at": 3, "tenant": "a", "reason": "sequence"},
        )

    def test_verify_all_digest_error(self):
        good = self.chain.append("t", {})
        bad = {"tenant": "t", "seq": 2, "event": {}, "prev": "9" * 64}
        bad["hash"] = AuditChain._hash(bad)
        self.write([json.loads(self.read()), bad])
        self.assertEqual(
            self.chain.verify_all(),
            {"ok": False, "at": 2, "tenant": "t", "reason": "digest"},
        )

    def test_verify_all_unparseable_is_missing_with_null_tenant(self):
        self.write([self.hashed("x", 1), "{broken"])
        self.assertEqual(
            self.chain.verify_all(),
            {"ok": False, "at": 2, "tenant": None, "reason": "missing"},
        )

    def test_verify_all_non_object_and_missing_field(self):
        self.write(["[1]"])
        self.assertEqual(
            self.chain.verify_all(),
            {"ok": False, "at": 1, "tenant": None, "reason": "missing"},
        )
        self.write([{"tenant": "t", "seq": 1, "event": {}, "prev": ZERO}])
        self.assertEqual(
            self.chain.verify_all(),
            {"ok": False, "at": 1, "tenant": "t", "reason": "missing"},
        )

    def test_verify_all_first_error_wins_and_is_read_only(self):
        self.write(["{oops", json.dumps(self.hashed("a", 1)), "{also bad"])
        before = self.read()
        self.assertEqual(
            self.chain.verify_all(),
            {"ok": False, "at": 1, "tenant": None, "reason": "missing"},
        )
        self.assertEqual(self.read(), before)


if __name__ == "__main__":
    unittest.main()
