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


class ExportSnapshotTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.path = Path(self.tmp.name) / "audit.jsonl"
        self.chain = AuditChain(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def seed(self):
        # Interleaved append order: a, b, a, a, b.
        self.chain.append("a", {"i": 1})
        self.chain.append("b", {"i": 1})
        self.chain.append("a", {"i": 2})
        self.chain.append("a", {"i": 3})
        self.chain.append("b", {"i": 2})

    def write_rows(self, rows):
        with self.path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(r if isinstance(r, str) else json.dumps(r, sort_keys=True))
                f.write("\n")

    def valid_row(self, tenant="t", seq=1, prev=ZERO, event=None):
        row = {"tenant": tenant, "seq": seq, "event": event or {}, "prev": prev}
        row["hash"] = AuditChain._hash(row)
        return row

    # --- export_snapshot: shape, verbatim data, matching manifest ---

    def test_result_carries_exactly_data_and_manifest(self):
        self.seed()
        result = self.chain.export_snapshot()
        self.assertEqual(set(result), {"data", "manifest"})
        self.assertIsInstance(result["data"], bytes)
        self.assertIsInstance(result["manifest"], dict)

    def test_data_is_the_raw_snapshot_verbatim(self):
        self.seed()
        result = self.chain.export_snapshot()
        self.assertEqual(result["data"], self.path.read_bytes())
        self.assertEqual(result["data"], self.chain.export_all())

    def test_manifest_describes_the_same_bytes(self):
        self.seed()
        result = self.chain.export_snapshot()
        data, manifest = result["data"], result["manifest"]
        self.assertEqual(manifest, {
            "version": MANIFEST_VERSION,
            "byte_length": len(data),
            "byte_sha256": hashlib.sha256(data).hexdigest(),
            "tenants": manifest["tenants"],
        })
        self.assertEqual(manifest, self.chain.manifest_bytes(data))
        # Physical first-appearance order: a, b.
        self.assertEqual([t["tenant"] for t in manifest["tenants"]], ["a", "b"])
        self.assertEqual(
            {t["tenant"]: t["count"] for t in manifest["tenants"]},
            {"a": 3, "b": 2},
        )
        heads = self.chain.heads_bytes(data)
        head_by_tenant = {h["tenant"]: h["hash"] for h in heads["tenants"]}
        for entry in manifest["tenants"]:
            self.assertEqual(entry["hash"], head_by_tenant[entry["tenant"]])

    def test_data_and_manifest_come_from_one_snapshot(self):
        # A concurrent append may land before or after the call, never
        # between data and manifest.
        self.seed()
        stop = threading.Event()
        errors = []

        def writer():
            i = 0
            while not stop.is_set():
                try:
                    self.chain.append("c", {"i": i})
                except Exception as exc:  # pragma: no cover - defensive
                    errors.append(exc)
                    return
                i += 1

        t = threading.Thread(target=writer)
        t.start()
        try:
            for _ in range(50):
                result = self.chain.export_snapshot()
                self.assertEqual(
                    result["manifest"],
                    self.chain.manifest_bytes(result["data"]),
                )
                self.assertEqual(
                    result["manifest"]["byte_length"], len(result["data"]))
                self.assertEqual(
                    result["manifest"]["byte_sha256"],
                    hashlib.sha256(result["data"]).hexdigest(),
                )
        finally:
            stop.set()
            t.join()
        self.assertEqual(errors, [])

    def test_missing_log_is_an_empty_snapshot(self):
        result = self.chain.export_snapshot()
        self.assertEqual(result, {"data": b"", "manifest": empty_manifest()})
        self.assertFalse(self.path.exists())

    def test_empty_log_is_an_empty_snapshot(self):
        self.path.write_bytes(b"")
        result = self.chain.export_snapshot()
        self.assertEqual(result, {"data": b"", "manifest": empty_manifest()})

    def test_data_feeds_the_offline_entries(self):
        self.seed()
        result = self.chain.export_snapshot()
        data = result["data"]
        self.assertEqual(self.chain.verify_all_bytes(data)["ok"], True)
        self.assertEqual(self.chain.manifest_bytes(data), result["manifest"])
        self.assertEqual(
            self.chain.compare_bytes(data, data),
            self.chain.compare_bytes(self.chain.export_all(), data),
        )
        clone_path = Path(self.tmp.name) / "clone.jsonl"
        clone = AuditChain(clone_path)
        imported = clone.import_all(data)
        self.assertEqual(len(imported), 5)
        self.assertEqual(clone_path.read_bytes(), data)

    # --- corrupt history: AuditChainStateError, no partial result ---

    def test_undecodable_bytes_raise_state_error(self):
        self.seed()
        with self.path.open("ab") as f:
            f.write(b"\xff\xfe")
        with self.assertRaises(AuditChainStateError) as ctx:
            self.chain.export_snapshot()
        self.assertEqual(ctx.exception.reason, "missing")
        self.assertIsNone(ctx.exception.tenant)
        self.assertIsNone(ctx.exception.seq)
        self.assertEqual(ctx.exception.line, 6)

    def test_unparseable_line_raises_state_error(self):
        self.seed()
        with self.path.open("a", encoding="utf-8") as f:
            f.write("not json\n")
        with self.assertRaises(AuditChainStateError) as ctx:
            self.chain.export_snapshot()
        self.assertEqual(ctx.exception.reason, "missing")
        self.assertEqual(ctx.exception.line, 6)

    def test_missing_field_raises_state_error(self):
        row = self.valid_row("t", 1)
        del row["hash"]
        self.write_rows([row])
        with self.assertRaises(AuditChainStateError) as ctx:
            self.chain.export_snapshot()
        self.assertEqual(ctx.exception.reason, "missing")
        self.assertEqual(ctx.exception.tenant, "t")
        self.assertEqual(ctx.exception.seq, 1)
        self.assertEqual(ctx.exception.line, 1)

    def test_sequence_gap_raises_state_error(self):
        first = self.valid_row("t", 1)
        second = self.valid_row("t", 3, prev=first["hash"])
        self.write_rows([first, second])
        with self.assertRaises(AuditChainStateError) as ctx:
            self.chain.export_snapshot()
        self.assertEqual(ctx.exception.reason, "sequence")
        self.assertEqual(ctx.exception.tenant, "t")
        self.assertEqual(ctx.exception.seq, 2)
        self.assertEqual(ctx.exception.line, 2)

    def test_digest_mismatch_raises_state_error(self):
        first = self.valid_row("t", 1)
        second = self.valid_row("t", 2, prev=ZERO)
        self.write_rows([first, second])
        with self.assertRaises(AuditChainStateError) as ctx:
            self.chain.export_snapshot()
        self.assertEqual(ctx.exception.reason, "digest")
        self.assertEqual(ctx.exception.tenant, "t")
        self.assertEqual(ctx.exception.seq, 2)
        self.assertEqual(ctx.exception.line, 2)

    def test_state_error_matches_export_all(self):
        first = self.valid_row("t", 1)
        second = self.valid_row("t", 2, prev=ZERO)
        self.write_rows([first, second])
        with self.assertRaises(AuditChainStateError) as export_ctx:
            self.chain.export_all()
        with self.assertRaises(AuditChainStateError) as snap_ctx:
            self.chain.export_snapshot()
        for field in ("tenant", "seq", "reason", "line"):
            self.assertEqual(getattr(snap_ctx.exception, field),
                             getattr(export_ctx.exception, field))

    # --- export_snapshot_chunks ---

    def test_chunks_result_carries_exactly_manifest_and_chunks(self):
        self.seed()
        result = self.chain.export_snapshot_chunks(7)
        self.assertEqual(set(result), {"manifest", "chunks"})
        chunks = list(result["chunks"])
        self.assertTrue(chunks)
        self.assertTrue(all(isinstance(c, bytes) for c in chunks))
        data = b"".join(chunks)
        self.assertEqual(data, self.chain.export_snapshot()["data"])
        self.assertEqual(result["manifest"],
                         self.chain.manifest_bytes(data))

    def test_chunks_split_on_plain_byte_offsets(self):
        self.seed()
        data = self.chain.export_all()
        for size in (1, 2, 3, len(data), len(data) + 10):
            result = self.chain.export_snapshot_chunks(size)
            chunks = list(result["chunks"])
            self.assertEqual(b"".join(chunks), data)
            for c in chunks[:-1]:
                self.assertEqual(len(c), size)
            self.assertLessEqual(len(chunks[-1]), size)

    def test_chunks_iterator_is_one_shot(self):
        self.seed()
        result = self.chain.export_snapshot_chunks(4)
        first = list(result["chunks"])
        self.assertTrue(first)
        self.assertEqual(list(result["chunks"]), [])

    def test_chunks_empty_snapshot_yields_no_chunks(self):
        result = self.chain.export_snapshot_chunks(5)
        self.assertEqual(result["manifest"], empty_manifest())
        self.assertEqual(list(result["chunks"]), [])
        self.assertFalse(self.path.exists())

    def test_chunks_fixed_to_one_snapshot(self):
        self.seed()
        result = self.chain.export_snapshot_chunks(3)
        self.chain.append("z", {"i": 99})
        data = b"".join(result["chunks"])
        self.assertNotEqual(data, self.path.read_bytes())
        self.assertEqual(result["manifest"],
                         self.chain.manifest_bytes(data))

    def test_chunks_invalid_chunk_size_rejected_before_reading(self):
        # Corrupt history on disk: the boundary error must still win.
        self.path.write_bytes(b"\xff\xfe")
        for bad in (True, False, 0, -1, 2.5, "4", None, b"4"):
            with self.assertRaises(ValueError, msg=repr(bad)):
                self.chain.export_snapshot_chunks(bad)

    def test_chunks_corrupt_history_raises_before_any_chunk(self):
        first = self.valid_row("t", 1)
        second = self.valid_row("t", 2, prev=ZERO)
        self.write_rows([first, second])
        with self.assertRaises(AuditChainStateError) as ctx:
            self.chain.export_snapshot_chunks(2)
        self.assertEqual(ctx.exception.reason, "digest")
        self.assertEqual(ctx.exception.line, 2)

    def test_chunks_reconnected_feeds_the_offline_entries(self):
        self.seed()
        result = self.chain.export_snapshot_chunks(5)
        data = b"".join(result["chunks"])
        self.assertEqual(self.chain.verify_all_bytes(data)["ok"], True)
        self.assertEqual(self.chain.manifest_bytes(data), result["manifest"])
        clone_path = Path(self.tmp.name) / "clone.jsonl"
        clone = AuditChain(clone_path)
        clone.import_all(data)
        self.assertEqual(clone_path.read_bytes(), data)


if __name__ == "__main__":
    unittest.main()
