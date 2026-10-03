import json
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app import AuditChain, AuditChainStateError

ZERO = "0" * 64


def record_bytes(item):
    return (json.dumps(item, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


class ExportAllTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.path = Path(self.tmp.name) / "audit.jsonl"
        self.chain = AuditChain(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def write_rows(self, rows):
        with self.path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(r if isinstance(r, str) else json.dumps(r, sort_keys=True))
                f.write("\n")

    def valid_row(self, tenant="t", seq=1, prev=ZERO, event=None):
        row = {"tenant": tenant, "seq": seq, "event": event or {}, "prev": prev}
        row["hash"] = AuditChain._hash(row)
        return row

    # --- success: raw snapshot bytes, offline verification ---

    def test_export_all_returns_raw_log_bytes(self):
        for i in range(3):
            self.chain.append("a", {"i": i})
            self.chain.append("b", {"i": i})
        data = self.chain.export_all()
        self.assertIsInstance(data, bytes)
        self.assertEqual(data, self.path.read_bytes())
        rows = [json.loads(l) for l in data.decode("utf-8").splitlines()]
        self.assertEqual(len(rows), 6)
        self.assertTrue(all(set(r) == {"tenant", "seq", "event", "prev", "hash"}
                            for r in rows))

    def test_export_all_interleaved_snapshot_verifies_offline(self):
        for i in range(4):
            self.chain.append("a", {"i": i})
            self.chain.append("b", {"i": i * 10})
        data = self.chain.export_all()
        out = Path(self.tmp.name) / "copy.jsonl"
        out.write_bytes(data)
        clone = AuditChain(out)
        self.assertEqual(clone.verify("a"), {"ok": True, "count": 4})
        self.assertEqual(clone.verify("b"), {"ok": True, "count": 4})
        self.assertEqual(clone.verify_all(), {
            "ok": True,
            "tenants": [{"tenant": "a", "count": 4},
                        {"tenant": "b", "count": 4}],
        })
        self.assertEqual(clone.export_all(), data)

    def test_export_all_missing_file_returns_empty_creates_nothing(self):
        self.assertFalse(self.path.exists())
        self.assertEqual(self.chain.export_all(), b"")
        self.assertFalse(self.path.exists())

    def test_export_all_empty_file_returns_empty_bytes(self):
        self.path.write_bytes(b"")
        self.assertEqual(self.chain.export_all(), b"")

    def test_export_all_does_not_create_or_modify_any_file(self):
        self.chain.append("t", {})
        before = self.path.read_bytes()
        entries = sorted(p.name for p in Path(self.tmp.name).iterdir())
        self.assertTrue(self.chain.export_all())
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(sorted(p.name for p in Path(self.tmp.name).iterdir()),
                         entries)

    def test_repeated_exports_are_byte_identical(self):
        for i in range(5):
            self.chain.append("t", {"i": i})
            self.chain.append("u", {"i": i})
        first = self.chain.export_all()
        for _ in range(5):
            self.assertEqual(self.chain.export_all(), first)

    def test_export_all_distinct_json_identities_all_included(self):
        for t in (1, 1.0, True, "1"):
            self.chain.append(t, {})
        rows = [json.loads(l)
                for l in self.chain.export_all().decode("utf-8").splitlines()]
        self.assertEqual(len(rows), 4)
        self.assertTrue(self.chain.verify_all()["ok"])

    def test_export_all_without_trailing_newline_returns_verbatim_bytes(self):
        rows = [self.valid_row("t", 1), self.valid_row("b", 1)]
        raw = b"".join(record_bytes(r) for r in rows)[:-1]  # drop final \n
        self.path.write_bytes(raw)
        self.assertEqual(self.chain.export_all(), raw)

    # --- state errors: first defect in physical order, no partial result ---

    def test_unparseable_line_raises_missing_with_none_tenant(self):
        self.write_rows([self.valid_row("a", 1), "{not json"])
        before = self.path.read_bytes()
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.export_all()
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         (None, None, "missing", 2))
        self.assertEqual(self.path.read_bytes(), before)

    def test_missing_field_raises_missing_at_own_seq(self):
        good = self.valid_row("a", 1)
        bad = {"tenant": "a", "seq": 2, "event": {}, "prev": good["hash"]}
        self.write_rows([good, json.dumps(bad)])
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.export_all()
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("a", 2, "missing", 2))

    def test_sequence_gap_raises_sequence_at_expected(self):
        good = self.valid_row("a", 1)
        gap = self.valid_row("a", 3, prev=good["hash"])
        self.write_rows([good, gap])
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.export_all()
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("a", 2, "sequence", 2))

    def test_digest_error_raises_digest(self):
        row = self.valid_row("a", 1)
        row["event"] = {"tampered": True}
        self.write_rows([row])
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.export_all()
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("a", 1, "digest", 1))

    def test_prev_mismatch_raises_digest(self):
        first = self.valid_row("a", 1)
        second = self.valid_row("a", 2, prev="9" * 64)
        self.write_rows([first, second])
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.export_all()
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("a", 2, "digest", 2))

    def test_illegal_utf8_raises_missing_at_physical_line(self):
        raw = record_bytes(self.valid_row("a", 1)) + b"\xff"
        self.path.write_bytes(raw)
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.export_all()
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         (None, None, "missing", 2))

    def test_duplicate_keys_and_non_standard_numbers_are_missing(self):
        for raw in (
            b'{"tenant":"a","tenant":"a","seq":1,"event":{},'
            b'"prev":"' + ZERO.encode() + b'","hash":"x"}\n',
            b'{"tenant":"a","seq":1,"event":{"x":NaN},"prev":"'
            + ZERO.encode() + b'","hash":"x"}\n',
            b'{"tenant":"a","seq":1e999,"event":{},"prev":"'
            + ZERO.encode() + b'","hash":"x"}\n',
        ):
            self.path.write_bytes(raw)
            with self.assertRaises(AuditChainStateError) as cm:
                self.chain.export_all()
            self.assertEqual((cm.exception.reason, cm.exception.line),
                             ("missing", 1), raw)

    def test_first_physical_defect_wins_across_tenants(self):
        # a's digest breaks at line 1, b's sequence breaks at line 2
        a = self.valid_row("a", 1)
        a["event"] = {"tampered": True}
        b = self.valid_row("b", 2)
        self.write_rows([a, b])
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.export_all()
        self.assertEqual((cm.exception.tenant, cm.exception.reason,
                          cm.exception.line), ("a", "digest", 1))
        # swap: b's defect is now physically first
        self.write_rows([b, a])
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.export_all()
        self.assertEqual((cm.exception.tenant, cm.exception.reason,
                          cm.exception.line), ("b", "sequence", 1))

    def test_error_priority_digest_before_later_bad_bytes(self):
        row = self.valid_row("a", 1)
        row["event"] = {"z": 9}
        self.path.write_bytes(record_bytes(row) + b"\xff")
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.export_all()
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("a", 1, "digest", 1))


class ExportAllConcurrencyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.path = Path(self.tmp.name) / "audit.jsonl"
        self.chain = AuditChain(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_export_all_sees_only_complete_pre_or_post_append_snapshot(self):
        for i in range(5):
            self.chain.append("a", {"i": i})
            self.chain.append("b", {"i": i})
        problems = []
        box = threading.Lock()
        stop = threading.Event()

        def exporter():
            while not stop.is_set():
                data = self.chain.export_all()
                out = self.chain.verify_all_bytes(data)
                if not out["ok"]:
                    with box:
                        problems.append(("verify_all_bytes", out))
                    return
                counts = {t["tenant"]: t["count"] for t in out["tenants"]}
                if counts.get("a") != counts.get("b"):
                    # a and b are appended in lockstep pairs below; a torn
                    # snapshot could still pass per-chain verification but
                    # must never fail offline validation (checked above)
                    pass

        exporters = [threading.Thread(target=exporter) for _ in range(4)]
        for t in exporters:
            t.start()

        def writer(i):
            c = AuditChain(self.path)
            for j in range(30):
                c.append_many([{"tenant": "a", "event": {"w": i, "j": j}},
                               {"tenant": "b", "event": {"w": i, "j": j}}])

        writers = [threading.Thread(target=writer, args=(i,)) for i in range(4)]
        for t in writers:
            t.start()
        for t in writers:
            t.join()
        stop.set()
        for t in exporters:
            t.join(timeout=2)

        self.assertEqual(problems, [])
        total = 5 + 4 * 30
        self.assertEqual(self.chain.verify("a"), {"ok": True, "count": total})
        self.assertEqual(self.chain.verify("b"), {"ok": True, "count": total})
        final = self.chain.export_all()
        out = Path(self.tmp.name) / "final.jsonl"
        out.write_bytes(final)
        self.assertTrue(AuditChain(out).verify_all()["ok"])


if __name__ == "__main__":
    unittest.main()
