import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app import AuditChain, AuditChainStateError


ZERO = "0" * 64


class HeadTest(unittest.TestCase):
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

    def test_head_missing_log_is_zero_head_and_creates_nothing(self):
        self.assertEqual(self.chain.head("t"),
                         {"tenant": "t", "count": 0, "hash": ZERO})
        self.assertFalse(self.path.exists())

    def test_head_empty_log_is_zero_head(self):
        self.path.write_bytes(b"")
        self.assertEqual(self.chain.head("t"),
                         {"tenant": "t", "count": 0, "hash": ZERO})

    def test_head_unknown_tenant_is_zero_head(self):
        self.chain.append("a", {})
        self.assertEqual(self.chain.head("t"),
                         {"tenant": "t", "count": 0, "hash": ZERO})

    def test_head_returns_verified_tail(self):
        a = self.chain.append("t", {"i": 1})
        b = self.chain.append("t", {"i": 2})
        self.assertEqual(self.chain.head("t"),
                         {"tenant": "t", "count": 2, "hash": b["hash"]})
        self.assertNotEqual(a["hash"], b["hash"])

    def test_head_ignores_interleaved_tenants(self):
        self.chain.append("a", {})
        b1 = self.chain.append("b", {})
        self.chain.append("a", {})
        b2 = self.chain.append("b", {})
        self.assertEqual(self.chain.head("b"),
                         {"tenant": "b", "count": 2, "hash": b2["hash"]})
        self.assertEqual(self.chain.head("a")["count"], 2)
        self.assertEqual(self.chain.head("c"),
                         {"tenant": "c", "count": 0, "hash": ZERO})
        self.assertNotEqual(b1["hash"], b2["hash"])

    def test_head_keeps_distinct_json_identities_apart(self):
        for t in (1, 1.0, True, "1"):
            self.chain.append(t, {})
        for t in (1, 1.0, True, "1"):
            h = self.chain.head(t)
            self.assertEqual(h["tenant"], t)
            self.assertEqual(h["count"], 1)

    def test_head_result_feeds_append_if_head(self):
        self.chain.append("t", {})
        h = self.chain.head("t")
        item = self.chain.append_if_head("t", {"i": 2}, h["count"], h["hash"])
        self.assertEqual(item["seq"], 2)
        h2 = self.chain.head("t")
        self.assertEqual((h2["count"], h2["hash"]), (2, item["hash"]))

    def test_head_result_feeds_append_batch_if_head(self):
        self.chain.append("t", {})
        h = self.chain.head("t")
        items = self.chain.append_batch_if_head(
            "t", [{"i": 2}, {"i": 3}], h["count"], h["hash"])
        self.assertEqual([i["seq"] for i in items], [2, 3])
        self.assertEqual(self.chain.head("t")["hash"], items[-1]["hash"])

    def test_head_empty_chain_result_feeds_append_if_head(self):
        h = self.chain.head("t")
        item = self.chain.append_if_head("t", {}, h["count"], h["hash"])
        self.assertEqual((item["seq"], item["prev"]), (1, ZERO))

    # --- tenant input boundary: same standard-JSON rules as append/verify ---

    def test_head_rejects_illegal_tenant_before_reading(self):
        for bad in (float("nan"), float("inf"), float("-inf"),
                    {"k": float("nan")}, {1: "x"}, object(), b"x", (1,)):
            with self.assertRaises(ValueError):
                self.chain.head(bad)
        self.assertFalse(self.path.exists())

    def test_head_rejects_cyclic_tenant(self):
        cyc = []
        cyc.append(cyc)
        with self.assertRaises(ValueError):
            self.chain.head(cyc)

    def test_head_value_error_beats_corrupt_history_and_keeps_bytes(self):
        row = {"tenant": "t", "seq": 1, "event": {}, "prev": ZERO}
        row["hash"] = AuditChain._hash(row)
        row["event"] = {"tampered": True}
        self.write([row])
        before = self.path.read_bytes()
        with self.assertRaises(ValueError):
            self.chain.head(float("nan"))
        self.assertEqual(self.path.read_bytes(), before)

    # --- corruption: full-chain validation, no partial heads ---

    def test_head_corrupt_prefix_raises_state_error(self):
        row = {"tenant": "t", "seq": 1, "event": {}, "prev": ZERO}
        row["hash"] = AuditChain._hash(row)
        self.write([row, {"tenant": "t", "seq": 2, "event": {}, "prev": row["hash"],
                          "hash": "deadbeef" * 8}])
        before = self.path.read_bytes()
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.head("t")
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("t", 2, "digest", 2))
        self.assertEqual(self.path.read_bytes(), before)

    def test_head_corrupt_tail_raises_state_error(self):
        self.chain.append("t", {})
        self.chain.append("t", {})
        with self.path.open("a", encoding="utf-8") as f:
            f.write("{oops\n")
        before = self.path.read_bytes()
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.head("t")
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("t", 3, "missing", 3))
        self.assertEqual(self.path.read_bytes(), before)

    def test_head_sequence_error_fields_match_append(self):
        row = {"tenant": "t", "seq": 2, "event": {}, "prev": ZERO}
        row["hash"] = AuditChain._hash(row)
        self.write([row])
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.head("t")
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("t", 1, "sequence", 1))

    def test_head_unparseable_other_tenant_line_still_fatal(self):
        self.write(['{"tenant": "x", broken'])
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.head("t")
        self.assertEqual((cm.exception.reason, cm.exception.line),
                         ("missing", 1))


class HeadsTest(unittest.TestCase):
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

    def test_heads_missing_log_is_empty_and_creates_nothing(self):
        self.assertEqual(self.chain.heads(), {"ok": True, "tenants": []})
        self.assertFalse(self.path.exists())

    def test_heads_empty_log_is_empty(self):
        self.path.write_bytes(b"")
        self.assertEqual(self.chain.heads(), {"ok": True, "tenants": []})

    def test_heads_first_appearance_order_with_hashes(self):
        b1 = self.chain.append("b", {})
        a1 = self.chain.append("a", {})
        b2 = self.chain.append("b", {})
        c1 = self.chain.append("c", {})
        self.assertEqual(self.chain.heads(), {"ok": True, "tenants": [
            {"tenant": "b", "count": 2, "hash": b2["hash"]},
            {"tenant": "a", "count": 1, "hash": a1["hash"]},
            {"tenant": "c", "count": 1, "hash": c1["hash"]},
        ]})
        self.assertNotEqual(b1["hash"], b2["hash"])

    def test_heads_distinct_json_identities_verbatim(self):
        for t in (1, 1.0, True, "1"):
            self.chain.append(t, {})
        r = self.chain.heads()
        self.assertTrue(r["ok"])
        self.assertEqual([t["tenant"] for t in r["tenants"]],
                         [1, 1.0, True, "1"])
        self.assertTrue(all(t["count"] == 1 for t in r["tenants"]))
        self.assertTrue(all(len(t["hash"]) == 64 for t in r["tenants"]))

    def test_heads_match_per_tenant_head(self):
        for t in ("a", "b"):
            self.chain.append(t, {})
            self.chain.append(t, {})
        r = self.chain.heads()
        self.assertTrue(r["ok"])
        for entry in r["tenants"]:
            h = self.chain.head(entry["tenant"])
            self.assertEqual((entry["count"], entry["hash"]),
                             (h["count"], h["hash"]))

    def test_heads_result_feeds_append_if_head(self):
        self.chain.append("t", {})
        entry = self.chain.heads()["tenants"][0]
        item = self.chain.append_if_head(
            "t", {}, entry["count"], entry["hash"])
        self.assertEqual(item["seq"], 2)

    # --- failure: same fields and physical-line priority as verify_all ---

    def test_heads_failure_matches_verify_all(self):
        self.chain.append("t", {})
        with self.path.open("a", encoding="utf-8") as f:
            f.write("{oops\n")
        expected = self.chain.verify_all()
        self.assertEqual(self.chain.heads(), expected)
        self.assertEqual(expected,
                         {"ok": False, "at": 2, "tenant": None, "reason": "missing"})

    def test_heads_sequence_failure_reports_tenant_and_line(self):
        self.chain.append("a", {})
        row = {"tenant": "b", "seq": 2, "event": {}, "prev": ZERO}
        row["hash"] = AuditChain._hash(row)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        self.assertEqual(self.chain.heads(),
                         {"ok": False, "at": 2, "tenant": "b",
                          "reason": "sequence"})

    def test_heads_digest_failure_stops_at_first_and_returns_no_heads(self):
        good = {"tenant": "a", "seq": 1, "event": {}, "prev": ZERO}
        good["hash"] = AuditChain._hash(good)
        bad = {"tenant": "b", "seq": 1, "event": {}, "prev": "f" * 64}
        bad["hash"] = AuditChain._hash(bad)
        self.write([good, bad])
        r = self.chain.heads()
        self.assertEqual(r, {"ok": False, "at": 2, "tenant": "b",
                             "reason": "digest"})
        self.assertNotIn("tenants", r)

    def test_heads_failure_is_read_only(self):
        row = {"tenant": "t", "seq": 1, "event": {}, "prev": ZERO}
        row["hash"] = AuditChain._hash(row)
        row["event"] = {"tampered": True}
        self.write([row])
        before = self.path.read_bytes()
        self.assertEqual(self.chain.heads(),
                         {"ok": False, "at": 1, "tenant": "t",
                          "reason": "digest"})
        self.assertEqual(self.path.read_bytes(), before)

    def test_heads_illegal_utf8_matches_verify_all(self):
        self.path.write_bytes(b"\xff")
        self.assertEqual(self.chain.heads(), self.chain.verify_all())
        self.assertEqual(self.chain.heads(),
                         {"ok": False, "at": 1, "tenant": None,
                          "reason": "missing"})


if __name__ == "__main__":
    unittest.main()
