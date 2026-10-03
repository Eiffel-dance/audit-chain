import json
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app import AuditChain, AuditChainStateError, ZERO


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

    # --- success: the log's own bytes, all tenants, physical order ---

    def test_export_all_returns_the_logs_own_bytes(self):
        for i in range(3):
            self.chain.append("a", {"i": i})
            self.chain.append("b", {"i": i})
        data = self.chain.export_all()
        self.assertIsInstance(data, bytes)
        self.assertEqual(data, self.path.read_bytes())
        rows = [json.loads(l) for l in data.decode("utf-8").splitlines()]
        self.assertEqual([r["tenant"] for r in rows], ["a", "b"] * 3)
        self.assertTrue(all(set(r) == {"tenant", "seq", "event", "prev", "hash"}
                            for r in rows))

    def test_export_all_round_trips_through_import_all(self):
        for i in range(4):
            self.chain.append("a", {"i": i})
            self.chain.append({"k": [1, 2]}, {"i": i})
        data = self.chain.export_all()
        out = Path(self.tmp.name) / "copy.jsonl"
        copied = AuditChain(out)
        records = copied.import_all(data)
        self.assertEqual(len(records), 8)
        self.assertEqual(out.read_bytes(), data)
        self.assertEqual(copied.export_all(), data)
        self.assertEqual(copied.verify_all(), self.chain.verify_all())

    def test_export_all_missing_file_returns_empty_bytes_creates_nothing(self):
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
        self.assertEqual(self.chain.export_all(), before)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(sorted(p.name for p in Path(self.tmp.name).iterdir()),
                         entries)

    def test_repeated_exports_are_byte_identical(self):
        for i in range(4):
            self.chain.append("t", {"i": i})
        first = self.chain.export_all()
        for _ in range(3):
            self.assertEqual(self.chain.export_all(), first)

    # --- corruption anywhere in the snapshot: no partial result ---

    def test_unparseable_line_raises_missing_with_none_tenant(self):
        self.write_rows([self.valid_row("a", 1), "{not json"])
        before = self.path.read_bytes()
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.export_all()
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         (None, None, "missing", 2))
        self.assertEqual(self.path.read_bytes(), before)

    def test_any_tenants_digest_error_is_reported(self):
        good = self.valid_row("a", 1)
        bad = self.valid_row("b", 1)
        bad["event"] = {"tampered": True}
        self.write_rows([good, bad])
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.export_all()
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("b", 1, "digest", 2))

    def test_sequence_error_of_second_tenant_is_reported(self):
        a1 = self.valid_row("a", 1)
        b1 = self.valid_row("b", 1)
        a2 = self.valid_row("a", 2, prev=a1["hash"])
        b2 = self.valid_row("b", 3, prev=b1["hash"])  # gap in b's chain
        self.write_rows([a1, b1, a2, b2])
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.export_all()
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("b", 2, "sequence", 4))

    def test_first_physical_error_wins_across_tenants(self):
        bad_b = self.valid_row("b", 1)
        bad_b["event"] = {"x": 1}  # digest broken at line 1
        gap_a = self.valid_row("a", 5)  # sequence broken at line 2
        self.write_rows([bad_b, gap_a])
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.export_all()
        self.assertEqual((cm.exception.tenant, cm.exception.reason,
                          cm.exception.line), ("b", "digest", 1))

    def test_missing_field_raises_missing_at_own_seq(self):
        row = {"tenant": "t", "seq": 1, "event": {}, "prev": ZERO}  # no hash
        self.write_rows([row])
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.export_all()
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("t", 1, "missing", 1))

    def test_duplicate_keys_and_non_standard_numbers_are_missing(self):
        for raw in (
            b'{"tenant":"t","tenant":"t","seq":1,"event":{},'
            b'"prev":"' + ZERO.encode() + b'","hash":"x"}\n',
            b'{"tenant":"t","seq":1,"event":{"x":NaN},"prev":"'
            + ZERO.encode() + b'","hash":"x"}\n',
            b'{"tenant":"t","seq":1e999,"event":{},"prev":"'
            + ZERO.encode() + b'","hash":"x"}\n',
        ):
            self.path.write_bytes(raw)
            with self.assertRaises(AuditChainStateError) as cm:
                self.chain.export_all()
            self.assertEqual((cm.exception.reason, cm.exception.line),
                             ("missing", 1), raw)

    def test_illegal_utf8_raises_missing_with_physical_line(self):
        raw = record_bytes(self.valid_row("t", 1)) + b"\xff"
        self.path.write_bytes(raw)
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.export_all()
        self.assertEqual((cm.exception.tenant, cm.exception.reason,
                          cm.exception.line), (None, "missing", 2))

    def test_non_integer_seq_spelling_is_sequence(self):
        row = self.valid_row("t", 1)
        raw = json.dumps(row, sort_keys=True).replace('"seq": 1', '"seq": 1.0')
        self.path.write_bytes(raw.encode() + b"\n")
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.export_all()
        self.assertEqual((cm.exception.reason, cm.exception.line),
                         ("sequence", 1))


class ExportAllConcurrencyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.path = Path(self.tmp.name) / "audit.jsonl"
        self.chain = AuditChain(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_export_all_sees_only_complete_pre_or_post_append_views(self):
        for i in range(5):
            self.chain.append("t", {"i": i})
        problems = []
        box = threading.Lock()
        stop = threading.Event()

        def exporter():
            while not stop.is_set():
                data = self.chain.export_all()
                r = self.chain.verify_all_bytes(data)
                if not r["ok"]:
                    with box:
                        problems.append(r)
                    return

        exporters = [threading.Thread(target=exporter) for _ in range(4)]
        for t in exporters:
            t.start()

        def writer(i):
            c = AuditChain(self.path)
            for j in range(30):
                c.append("t", {"w": i, "j": j})
                c.append("u", {"w": i, "j": j})

        writers = [threading.Thread(target=writer, args=(i,)) for i in range(4)]
        for t in writers:
            t.start()
        for t in writers:
            t.join()
        stop.set()
        for t in exporters:
            t.join(timeout=2)

        self.assertEqual(problems, [])
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": 125})
        self.assertEqual(self.chain.verify("u"), {"ok": True, "count": 120})


if __name__ == "__main__":
    unittest.main()
