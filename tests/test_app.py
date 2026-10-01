import json
import tempfile
import unittest
from pathlib import Path

import app
from app import AuditChain, AuditChainStateError


class AuditChainTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name) / "audit.jsonl"
        self.chain = AuditChain(self.path)

    def tearDown(self):
        self.dir.cleanup()

    def lines(self):
        return self.path.read_text(encoding="utf-8").splitlines()

    def records(self):
        return [json.loads(x) for x in self.lines()]

    def write_raw(self, *lines):
        self.path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def test_import(self):
        self.assertTrue(app)

    def test_append_chains_and_verifies(self):
        first = self.chain.append("tenant-a", {"action": "create"})
        second = self.chain.append("tenant-a", {"action": "update"})
        self.assertEqual(first["seq"], 1)
        self.assertEqual(first["prev"], "0" * 64)
        self.assertEqual(second["seq"], 2)
        self.assertEqual(second["prev"], first["hash"])
        self.assertEqual(second["hash"], AuditChain._hash(second))
        self.assertEqual(self.chain.verify("tenant-a"), {"ok": True, "count": 2})

    def test_hash_is_recomputable(self):
        item = self.chain.append("tenant-a", {"action": "create", "id": 1})
        payload = {k: item[k] for k in ("tenant", "seq", "event", "prev")}
        import hashlib
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        self.assertEqual(item["hash"], digest)

    def test_interleaved_tenants_keep_independent_chains(self):
        a1 = self.chain.append("a", "a1")
        self.chain.append("b", "b1")
        a2 = self.chain.append("a", "a2")
        self.assertEqual(a2["seq"], 2)
        self.assertEqual(a2["prev"], a1["hash"])
        self.assertEqual(self.chain.verify("a"), {"ok": True, "count": 2})
        self.assertEqual(self.chain.verify("b"), {"ok": True, "count": 1})

    def test_verify_empty_tenant(self):
        self.assertEqual(self.chain.verify("nobody"), {"ok": True, "count": 0})

    def test_verify_expected_count(self):
        self.chain.append("a", 1)
        self.chain.append("a", 2)
        self.assertEqual(self.chain.verify("a", 2), {"ok": True, "count": 2})
        self.assertEqual(
            self.chain.verify("a", 3), {"ok": False, "at": 3, "reason": "missing"}
        )
        too_many = self.chain.verify("a", 1)
        self.assertFalse(too_many["ok"])
        self.assertEqual(too_many["reason"], "sequence")
        self.assertEqual(too_many["at"], 2)

    def test_verify_unparseable_line_reports_line_number(self):
        self.chain.append("a", 1)
        with self.path.open("a", encoding="utf-8") as f:
            f.write("not json\n")
        result = self.chain.verify("a")
        self.assertEqual(result, {"ok": False, "at": 2, "reason": "missing"})

    def test_verify_missing_field(self):
        self.chain.append("a", 1)
        rows = self.records()
        del rows[0]["event"]
        self.write_raw(*[json.dumps(r) for r in rows])
        result = self.chain.verify("a")
        self.assertEqual(result["reason"], "missing")
        self.assertEqual(result["at"], 1)

    def test_verify_sequence_and_digest(self):
        self.chain.append("a", 1)
        self.chain.append("a", 2)
        rows = self.records()

        broken = [dict(rows[0]), dict(rows[1], seq=5)]
        self.write_raw(*[json.dumps(r) for r in broken])
        self.assertEqual(
            self.chain.verify("a"), {"ok": False, "at": 2, "reason": "sequence"}
        )

        tampered = [dict(rows[0]), dict(rows[1], event="forged")]
        self.write_raw(*[json.dumps(r) for r in tampered])
        self.assertEqual(
            self.chain.verify("a"), {"ok": False, "at": 2, "reason": "digest"}
        )

        relinked = [dict(rows[0]), dict(rows[1], prev="0" * 64)]
        self.write_raw(*[json.dumps(r) for r in relinked])
        self.assertEqual(
            self.chain.verify("a"), {"ok": False, "at": 2, "reason": "digest"}
        )

    def test_append_rejects_corrupt_history_without_modifying_file(self):
        self.chain.append("a", 1)
        rows = self.records()
        rows[0]["hash"] = "f" * 64
        self.write_raw(json.dumps(rows[0]))
        before = self.lines()
        with self.assertRaises(AuditChainStateError) as ctx:
            self.chain.append("a", 2)
        self.assertEqual(ctx.exception.tenant, "a")
        self.assertEqual(ctx.exception.seq, 1)
        self.assertEqual(self.lines(), before)

    def test_append_rejects_unparseable_line(self):
        self.write_raw("not json")
        with self.assertRaises(AuditChainStateError) as ctx:
            self.chain.append("a", 1)
        self.assertEqual(ctx.exception.seq, 1)
        self.assertEqual(self.lines(), ["not json"])

    def test_append_ignores_other_tenant_records(self):
        self.chain.append("b", "b1")
        item = self.chain.append("a", "a1")
        self.assertEqual(item["seq"], 1)
        self.assertEqual(item["prev"], "0" * 64)


if __name__ == "__main__":
    unittest.main()
