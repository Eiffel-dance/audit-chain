import hashlib
import json
import threading
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


class CompactTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.path = Path(self.tmp.name) / "audit.jsonl"
        self.chain = AuditChain(self.path)

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
        return row

    # --- empty history ---

    def test_missing_path_returns_empty_manifest_and_creates_nothing(self):
        phantom = Path(self.tmp.name) / "sub" / "phantom.jsonl"
        chain = AuditChain(phantom)
        self.assertEqual(chain.compact(), empty_manifest())
        self.assertFalse(phantom.exists())
        self.assertFalse(phantom.parent.exists())

    def test_empty_file_returns_empty_manifest_and_stays_untouched(self):
        self.path.write_bytes(b"")
        self.assertEqual(self.chain.compact(), empty_manifest())
        self.assertEqual(self.snapshot(), b"")

    # --- canonical rewrite ---

    def test_compact_rewrites_crlf_whitespace_and_key_order(self):
        row1 = self.valid_row("a", 1, event={"x": 1})
        row2 = self.valid_row("b", 1, event=[1, 2])
        row3 = self.valid_row("a", 2, prev=row1["hash"], event="héllo→世界")
        # Redundant but valid spellings: CRLF endings, extra whitespace,
        # non-canonical key order, raw (unescaped) UTF-8 content.
        raw = (
            b'{ "tenant": "a", "seq": 1, "event": {"x": 1}, "prev": "'
            + ZERO.encode() + b'", "hash": "' + row1["hash"].encode()
            + b'" }\r\n'
            + b'{"hash": "' + row2["hash"].encode()
            + b'", "prev": "' + ZERO.encode()
            + b'", "event": [1, 2], "seq": 1, "tenant": "b"}\n'
            + ('{"tenant": "a", "seq": 2, "event": "héllo→世界", "prev": "'
               + row1["hash"] + '", "hash": "' + row3["hash"] + '"}\n'
               ).encode("utf-8")
        )
        self.path.write_bytes(raw)
        heads_before = self.chain.heads()
        m = self.chain.compact()
        expected = record_bytes(row1) + record_bytes(row2) + record_bytes(row3)
        self.assertEqual(self.snapshot(), expected)
        # The returned manifest describes the rewritten bytes.
        self.assertEqual(m, self.chain.manifest())
        self.assertEqual(m["byte_length"], len(expected))
        self.assertEqual(m["byte_sha256"], hashlib.sha256(expected).hexdigest())
        self.assertEqual([t["tenant"] for t in m["tenants"]], ["a", "b"])
        self.assertEqual([t["count"] for t in m["tenants"]], [2, 1])
        # Tenant heads survive the rewrite unchanged.
        self.assertEqual(self.chain.heads(), heads_before)
        self.assertEqual(
            [(t["tenant"], t["count"], t["hash"]) for t in m["tenants"]],
            [(t["tenant"], t["count"], t["hash"])
             for t in heads_before["tenants"]])

    def test_compact_preserves_physical_interleave_and_values(self):
        tenants = [{"z": 1}, "a", [1, 2], 1, 1.0, True, "1", None]
        for i, tenant in enumerate(tenants * 2):
            self.chain.append(tenant, {"i": i, "nested": {"k": [None, True]}})
        before = self.snapshot()
        heads_before = self.chain.heads()
        m = self.chain.compact()
        # append output is already canonical: compaction changes nothing.
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(m["byte_length"], len(before))
        self.assertEqual(m["byte_sha256"], hashlib.sha256(before).hexdigest())
        self.assertEqual(self.chain.heads(), heads_before)
        # Physical order and record values are exactly the parsed originals.
        lines = [json.loads(raw) for raw in before.decode("utf-8").splitlines()]
        after = [json.loads(raw)
                 for raw in self.snapshot().decode("utf-8").splitlines()]
        self.assertEqual(lines, after)
        self.assertEqual([r["tenant"] for r in after],
                         [r["tenant"] for r in lines])

    def test_compact_is_a_fixed_point(self):
        self.seed()
        m1 = self.chain.compact()
        data1 = self.snapshot()
        m2 = self.chain.compact()
        self.assertEqual(m1, m2)
        self.assertEqual(self.snapshot(), data1)
        self.assertEqual(m1, self.chain.manifest())

    def test_log_remains_appendable_and_verifiable_after_compact(self):
        self.seed()
        heads_before = self.chain.heads()
        self.chain.compact()
        item = self.chain.append("a", {"i": 4})
        self.assertEqual(item["seq"], 4)
        self.assertEqual(item["prev"], heads_before["tenants"][0]["hash"])
        self.assertEqual(self.chain.verify_all()["ok"], True)
        self.assertEqual(self.chain.verify("a"), {"ok": True, "count": 4})
        # Offline entries agree on the rewritten bytes.
        data = self.snapshot()
        other = AuditChain(self.path.with_name("never-touched.jsonl"))
        self.assertEqual(self.chain.export_all(), data)
        self.assertTrue(other.verify_all_bytes(data)["ok"])
        self.assertEqual(other.manifest_bytes(data), self.chain.manifest())
        self.assertFalse(self.path.with_name("never-touched.jsonl").exists())

    # --- corruption: AuditChainStateError, file untouched ---

    def corrupt_cases(self):
        good1 = record_bytes(self.valid_row("t", 1))
        row_no_hash = {"tenant": "t", "seq": 1, "event": {}, "prev": ZERO}
        extra = dict(self.valid_row("t", 1), extra=1)
        gap = {"tenant": "t", "seq": 3, "event": {}, "prev": "1" * 64}
        gap["hash"] = AuditChain._hash(gap)
        bad_prev = {"tenant": "t", "seq": 2, "event": {}, "prev": "9" * 64}
        bad_prev["hash"] = AuditChain._hash(bad_prev)
        bad_hash = self.valid_row("t", 2, prev=self.valid_row("t", 1)["hash"])
        bad_hash["hash"] = "0" * 64
        return [
            (b"{not json\n", None, None, "missing", 1),
            (b"\n", None, None, "missing", 1),
            (b"\xff", None, None, "missing", 1),
            (good1 + b"\xff\n", None, None, "missing", 2),
            ((json.dumps(row_no_hash) + "\n").encode(), "t", 1, "missing", 1),
            (record_bytes(extra), "t", 1, "missing", 1),
            (good1 + record_bytes(gap), "t", 2, "sequence", 2),
            (good1 + record_bytes(bad_prev), "t", 2, "digest", 2),
            (good1 + record_bytes(bad_hash), "t", 2, "digest", 2),
            (b'{"tenant": "t", "seq": 1, "event": NaN, "prev": "' + ZERO.encode()
             + b'", "hash": "' + "0".encode() * 64 + b'"}\n',
             None, None, "missing", 1),
        ]

    def test_corrupt_history_raises_and_leaves_file_untouched(self):
        for raw, tenant, seq, reason, line in self.corrupt_cases():
            self.path.write_bytes(raw)
            with self.assertRaises(AuditChainStateError) as ctx:
                self.chain.compact()
            err = ctx.exception
            self.assertEqual(
                (err.tenant, err.seq, err.reason, err.line),
                (tenant, seq, reason, line), raw)
            self.assertEqual(self.snapshot(), raw)

    def test_error_fields_match_export_all(self):
        for raw, _tenant, _seq, _reason, _line in self.corrupt_cases():
            if b"extra" in raw:
                # export_all tolerates extra keys; compact does not. That
                # contract difference is covered by its own test.
                continue
            self.path.write_bytes(raw)
            with self.assertRaises(AuditChainStateError) as c1:
                self.chain.compact()
            with self.assertRaises(AuditChainStateError) as c2:
                self.chain.export_all()
            for attr in ("tenant", "seq", "reason", "line"):
                self.assertEqual(getattr(c1.exception, attr),
                                 getattr(c2.exception, attr), (attr, raw))

    def test_extra_field_is_missing_class_defect(self):
        row = dict(self.valid_row("t", 1), extra="x")
        self.path.write_bytes(record_bytes(row))
        with self.assertRaises(AuditChainStateError) as ctx:
            self.chain.compact()
        self.assertEqual(
            (ctx.exception.tenant, ctx.exception.seq,
             ctx.exception.reason, ctx.exception.line),
            ("t", 1, "missing", 1))

    def test_first_defect_in_physical_order_wins(self):
        good = record_bytes(self.valid_row("a", 1))
        gap = {"tenant": "b", "seq": 5, "event": {}, "prev": "1" * 64}
        gap["hash"] = AuditChain._hash(gap)
        raw = good + b"{broken\n" + record_bytes(gap)
        self.path.write_bytes(raw)
        with self.assertRaises(AuditChainStateError) as ctx:
            self.chain.compact()
        self.assertEqual(
            (ctx.exception.tenant, ctx.exception.seq,
             ctx.exception.reason, ctx.exception.line),
            (None, None, "missing", 2))
        self.assertEqual(self.snapshot(), raw)

    # --- concurrency: observers only see complete snapshots ---

    def test_concurrent_appends_and_compacts_leave_valid_history(self):
        self.seed()
        self.path.write_bytes(  # force a real rewrite during compact
            self.snapshot().replace(b"\n", b"\r\n"))
        errors = []

        def appends():
            try:
                for i in range(20):
                    self.chain.append("c", {"i": i})
            except Exception as exc:  # pragma: no cover - failure path
                errors.append(exc)

        def compacts():
            try:
                for _ in range(5):
                    self.chain.compact()
            except Exception as exc:  # pragma: no cover - failure path
                errors.append(exc)

        threads = ([threading.Thread(target=appends) for _ in range(3)]
                   + [threading.Thread(target=compacts) for _ in range(2)])
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        result = self.chain.verify_all()
        self.assertTrue(result["ok"])
        counts = {t["tenant"]: t["count"] for t in result["tenants"]}
        self.assertEqual(counts, {"a": 3, "b": 2, "c": 60})
        # The final file is canonical and matches its own manifest.
        data = self.snapshot()
        m = self.chain.compact()
        self.assertEqual(self.snapshot(), data)
        self.assertEqual(m["byte_sha256"], hashlib.sha256(data).hexdigest())


if __name__ == "__main__":
    unittest.main()
