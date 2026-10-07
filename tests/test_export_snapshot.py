import hashlib
import json
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app import AuditChain, AuditChainStateError

ZERO = "0" * 64
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


def record_bytes(item):
    return (json.dumps(item, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


class ExportSnapshotTest(unittest.TestCase):
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

    # --- result shape: exactly data and manifest from one snapshot ---

    def test_result_contains_exactly_data_and_manifest(self):
        self.chain.append("a", {"i": 1})
        result = self.chain.export_snapshot()
        self.assertEqual(set(result), {"data", "manifest"})
        self.assertIsInstance(result["data"], bytes)
        self.assertIsInstance(result["manifest"], dict)

    def test_data_is_raw_log_bytes_verbatim(self):
        for i in range(3):
            self.chain.append("a", {"i": i})
            self.chain.append("b", {"i": i * 10})
        result = self.chain.export_snapshot()
        self.assertEqual(result["data"], self.path.read_bytes())
        self.assertEqual(result["data"], self.chain.export_all())

    def test_data_without_trailing_newline_returned_verbatim(self):
        rows = [self.valid_row("t", 1), self.valid_row("b", 1)]
        raw = b"".join(record_bytes(r) for r in rows)[:-1]  # drop final \n
        self.path.write_bytes(raw)
        self.assertEqual(self.chain.export_snapshot()["data"], raw)

    def test_manifest_matches_manifest_of_same_snapshot(self):
        for i in range(4):
            self.chain.append("a", {"i": i})
            self.chain.append("b", {"i": i})
        result = self.chain.export_snapshot()
        self.assertEqual(result["manifest"], self.chain.manifest())
        self.assertEqual(result["manifest"],
                         self.chain.manifest_bytes(result["data"]))
        self.assertEqual(result["manifest"]["version"], 1)
        self.assertEqual(result["manifest"]["byte_length"],
                         len(result["data"]))
        self.assertEqual(result["manifest"]["byte_sha256"],
                         hashlib.sha256(result["data"]).hexdigest())
        self.assertEqual(
            [t["tenant"] for t in result["manifest"]["tenants"]], ["a", "b"])
        self.assertTrue(all(t["count"] == 4
                            for t in result["manifest"]["tenants"]))
        heads = self.chain.heads()["tenants"]
        self.assertEqual([t["hash"] for t in result["manifest"]["tenants"]],
                         [h["hash"] for h in heads])

    def test_manifest_tenants_follow_physical_first_appearance_order(self):
        for t in ("b", "a", "c", "a", "b"):
            self.chain.append(t, {})
        result = self.chain.export_snapshot()
        self.assertEqual([t["tenant"] for t in result["manifest"]["tenants"]],
                         ["b", "a", "c"])

    def test_data_and_manifest_describe_one_snapshot(self):
        # The manifest must be computed from the returned data itself, so
        # verifying the manifest against the data always succeeds.
        for i in range(3):
            self.chain.append("t", {"i": i})
        result = self.chain.export_snapshot()
        out = self.chain.verify_manifest_bytes(result["data"],
                                               result["manifest"])
        self.assertTrue(out["ok"])

    # --- empty snapshots ---

    def test_missing_path_is_empty_snapshot_and_creates_nothing(self):
        self.assertFalse(self.path.exists())
        result = self.chain.export_snapshot()
        self.assertEqual(result["data"], b"")
        self.assertEqual(result["manifest"], {
            "version": 1,
            "byte_length": 0,
            "byte_sha256": EMPTY_SHA256,
            "tenants": [],
        })
        self.assertFalse(self.path.exists())

    def test_empty_file_is_empty_snapshot(self):
        self.path.write_bytes(b"")
        result = self.chain.export_snapshot()
        self.assertEqual(result["data"], b"")
        self.assertEqual(result["manifest"]["byte_length"], 0)
        self.assertEqual(result["manifest"]["byte_sha256"], EMPTY_SHA256)
        self.assertEqual(result["manifest"]["tenants"], [])

    # --- read-only ---

    def test_export_snapshot_does_not_create_or_modify_any_file(self):
        self.chain.append("t", {})
        before = self.path.read_bytes()
        entries = sorted(p.name for p in Path(self.tmp.name).iterdir())
        self.assertTrue(self.chain.export_snapshot()["data"])
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(sorted(p.name for p in Path(self.tmp.name).iterdir()),
                         entries)

    def test_repeated_snapshots_are_identical(self):
        for i in range(3):
            self.chain.append("t", {"i": i})
            self.chain.append("u", {"i": i})
        first = self.chain.export_snapshot()
        for _ in range(3):
            self.assertEqual(self.chain.export_snapshot(), first)

    # --- state errors: same first-defect fields as export_all ---

    def test_unparseable_line_raises_missing_with_none_tenant(self):
        self.write_rows([self.valid_row("a", 1), "{not json"])
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.export_snapshot()
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         (None, None, "missing", 2))

    def test_missing_field_raises_missing_at_own_seq(self):
        good = self.valid_row("a", 1)
        bad = {"tenant": "a", "seq": 2, "event": {}, "prev": good["hash"]}
        self.write_rows([good, json.dumps(bad)])
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.export_snapshot()
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("a", 2, "missing", 2))

    def test_sequence_gap_raises_sequence_at_expected(self):
        good = self.valid_row("a", 1)
        gap = self.valid_row("a", 3, prev=good["hash"])
        self.write_rows([good, gap])
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.export_snapshot()
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("a", 2, "sequence", 2))

    def test_digest_error_raises_digest(self):
        row = self.valid_row("a", 1)
        row["event"] = {"tampered": True}
        self.write_rows([row])
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.export_snapshot()
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("a", 1, "digest", 1))

    def test_illegal_utf8_raises_missing_at_physical_line(self):
        raw = record_bytes(self.valid_row("a", 1)) + b"\xff"
        self.path.write_bytes(raw)
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.export_snapshot()
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         (None, None, "missing", 2))

    def test_error_matches_export_all_error(self):
        good = self.valid_row("a", 1)
        bad = self.valid_row("a", 2, prev="9" * 64)
        self.write_rows([good, bad])
        with self.assertRaises(AuditChainStateError) as snap:
            self.chain.export_snapshot()
        with self.assertRaises(AuditChainStateError) as all_:
            self.chain.export_all()
        for exc in (snap.exception, all_.exception):
            self.assertEqual((exc.tenant, exc.seq, exc.reason, exc.line),
                             ("a", 2, "digest", 2))

    # --- the snapshot feeds the offline entry points directly ---

    def test_data_feeds_offline_entries(self):
        for i in range(3):
            self.chain.append("a", {"i": i})
            self.chain.append("b", {"i": i})
        result = self.chain.export_snapshot()
        data = result["data"]
        self.assertEqual(self.chain.verify_all_bytes(data),
                         self.chain.verify_all())
        self.assertEqual(self.chain.heads_bytes(data), self.chain.heads())
        self.assertEqual(self.chain.manifest_bytes(data),
                         result["manifest"])
        self.assertEqual(self.chain.compare_bytes(data, data)["equal"], True)
        out = Path(self.tmp.name) / "clone.jsonl"
        clone = AuditChain(out)
        self.assertEqual(len(clone.import_all(data)), 6)
        self.assertEqual(clone.export_all(), data)


class ExportSnapshotChunksTest(unittest.TestCase):
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

    # --- result shape and chunking ---

    def test_result_contains_exactly_manifest_and_chunks(self):
        self.chain.append("a", {})
        result = self.chain.export_snapshot_chunks(10)
        self.assertEqual(set(result), {"manifest", "chunks"})
        self.assertIsInstance(result["manifest"], dict)

    def test_chunks_reassemble_to_snapshot_data(self):
        for i in range(6):
            self.chain.append("a", {"i": i})
            self.chain.append("b", {"i": i})
        snapshot = self.chain.export_snapshot()
        for size in (1, 2, 7, 64, 10_000):
            result = self.chain.export_snapshot_chunks(size)
            chunks = list(result["chunks"])
            self.assertTrue(all(isinstance(c, bytes) for c in chunks))
            self.assertTrue(all(len(c) <= size for c in chunks))
            self.assertEqual(b"".join(chunks), snapshot["data"])
            self.assertEqual(result["manifest"], snapshot["manifest"])

    def test_chunks_iterator_is_one_shot(self):
        self.chain.append("a", {})
        result = self.chain.export_snapshot_chunks(4)
        first = list(result["chunks"])
        self.assertTrue(first)
        self.assertEqual(list(result["chunks"]), [])

    def test_empty_snapshot_yields_no_chunks(self):
        self.assertFalse(self.path.exists())
        result = self.chain.export_snapshot_chunks(5)
        self.assertEqual(list(result["chunks"]), [])
        self.assertEqual(result["manifest"], {
            "version": 1,
            "byte_length": 0,
            "byte_sha256": EMPTY_SHA256,
            "tenants": [],
        })
        self.assertFalse(self.path.exists())
        self.path.write_bytes(b"")
        result = self.chain.export_snapshot_chunks(5)
        self.assertEqual(list(result["chunks"]), [])
        self.assertEqual(result["manifest"]["byte_length"], 0)

    def test_manifest_matches_snapshot_manifest(self):
        for i in range(3):
            self.chain.append("t", {"i": i})
        result = self.chain.export_snapshot_chunks(3)
        data = b"".join(result["chunks"])
        self.assertEqual(result["manifest"]["byte_length"], len(data))
        self.assertEqual(result["manifest"]["byte_sha256"],
                         hashlib.sha256(data).hexdigest())
        self.assertEqual(result["manifest"], self.chain.manifest())

    # --- chunk_size boundary, before any history is read ---

    def test_invalid_chunk_size_raises_value_error(self):
        self.chain.append("a", {})
        for bad in (True, False, 0, -1, 1.5, "4", None, b"4"):
            with self.assertRaises(ValueError, msg=repr(bad)):
                self.chain.export_snapshot_chunks(bad)

    def test_invalid_chunk_size_wins_over_corrupt_history(self):
        self.write_rows(["{not json"])
        for bad in (True, 0, -3, 2.0):
            with self.assertRaises(ValueError, msg=repr(bad)):
                self.chain.export_snapshot_chunks(bad)

    def test_invalid_chunk_size_reads_no_history(self):
        # A missing path must stay missing: the boundary error comes first.
        self.assertFalse(self.path.exists())
        with self.assertRaises(ValueError):
            self.chain.export_snapshot_chunks(0)
        self.assertFalse(self.path.exists())

    # --- state errors: no partial chunk stream ---

    def test_corrupt_history_raises_without_yielding_chunks(self):
        good = self.valid_row("a", 1)
        gap = self.valid_row("a", 3, prev=good["hash"])
        self.write_rows([good, gap])
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.export_snapshot_chunks(4)
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("a", 2, "sequence", 2))

    def test_error_fields_match_export_all(self):
        row = self.valid_row("a", 1)
        row["event"] = {"tampered": True}
        self.write_rows([row])
        with self.assertRaises(AuditChainStateError) as chunks:
            self.chain.export_snapshot_chunks(4)
        with self.assertRaises(AuditChainStateError) as all_:
            self.chain.export_all()
        for exc in (chunks.exception, all_.exception):
            self.assertEqual((exc.tenant, exc.seq, exc.reason, exc.line),
                             ("a", 1, "digest", 1))

    # --- read-only ---

    def test_export_snapshot_chunks_does_not_create_or_modify_any_file(self):
        self.chain.append("t", {})
        before = self.path.read_bytes()
        entries = sorted(p.name for p in Path(self.tmp.name).iterdir())
        result = self.chain.export_snapshot_chunks(2)
        self.assertTrue(list(result["chunks"]))
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(sorted(p.name for p in Path(self.tmp.name).iterdir()),
                         entries)

    # --- the reassembled chunks feed the offline entry points ---

    def test_reassembled_chunks_feed_offline_entries(self):
        for i in range(4):
            self.chain.append("a", {"i": i})
            self.chain.append("b", {"i": i})
        result = self.chain.export_snapshot_chunks(5)
        data = b"".join(result["chunks"])
        self.assertEqual(self.chain.verify_all_bytes(data),
                         self.chain.verify_all())
        self.assertEqual(self.chain.heads_bytes(data), self.chain.heads())
        self.assertEqual(self.chain.manifest_bytes(data), result["manifest"])
        self.assertEqual(self.chain.compare_bytes(data, data)["equal"], True)
        out = Path(self.tmp.name) / "clone.jsonl"
        clone = AuditChain(out)
        self.assertEqual(len(clone.import_all(data)), 8)
        self.assertEqual(clone.export_all(), data)


class ExportSnapshotConcurrencyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.path = Path(self.tmp.name) / "audit.jsonl"
        self.chain = AuditChain(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_snapshot_pairs_data_and_manifest_from_one_window(self):
        for i in range(5):
            self.chain.append("a", {"i": i})
            self.chain.append("b", {"i": i})
        problems = []
        box = threading.Lock()
        stop = threading.Event()

        def exporter():
            while not stop.is_set():
                result = self.chain.export_snapshot()
                data, manifest = result["data"], result["manifest"]
                # data and manifest must always describe one snapshot.
                if manifest["byte_length"] != len(data) or \
                        manifest["byte_sha256"] != hashlib.sha256(data).hexdigest():
                    with box:
                        problems.append(("manifest/data mismatch", manifest))
                    return
                out = self.chain.verify_all_bytes(data)
                if not out["ok"]:
                    with box:
                        problems.append(("verify_all_bytes", out))
                    return
                counts = {t["tenant"]: t["count"] for t in out["tenants"]}
                mcounts = {t["tenant"]: t["count"]
                           for t in manifest["tenants"]}
                if counts != mcounts:
                    with box:
                        problems.append(("tenant counts", counts, mcounts))
                    return
                chunked = self.chain.export_snapshot_chunks(13)
                joined = b"".join(chunked["chunks"])
                cmanifest = chunked["manifest"]
                if cmanifest["byte_length"] != len(joined) or \
                        cmanifest["byte_sha256"] != hashlib.sha256(joined).hexdigest():
                    with box:
                        problems.append(("chunks/manifest mismatch",))
                    return

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


if __name__ == "__main__":
    unittest.main()
