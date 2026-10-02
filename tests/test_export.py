import hashlib
import json
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app import AuditChain, AuditChainStateError, ZERO


def _row(tenant="t", seq=1, event=None, prev=ZERO):
    row = {"tenant": tenant, "seq": seq,
           "event": {} if event is None else event, "prev": prev}
    row["hash"] = AuditChain._hash(row)
    return row


def _valid_blob(row):
    return (json.dumps(row, sort_keys=True) + "\n").encode("utf-8")


class ExportTenantTest(unittest.TestCase):
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

    def source_rows(self):
        return [json.loads(l) for l in self.path.read_text().splitlines()]

    def rehydrate(self, blob):
        # Write the exported bytes the way a migration caller would, then
        # open them as a brand new independent AuditChain.
        out = Path(self.tmp.name) / "exported.jsonl"
        out.write_bytes(blob)
        return out, AuditChain(out)

    # --- empty snapshots ---

    def test_missing_file_exports_empty_bytes_and_creates_nothing(self):
        self.assertFalse(self.path.exists())
        self.assertEqual(self.chain.export_tenant("t"), b"")
        self.assertFalse(self.path.exists())
        self.assertEqual(list(Path(self.tmp.name).iterdir()), [])

    def test_empty_file_exports_empty_bytes(self):
        self.path.write_bytes(b"")
        self.assertEqual(self.chain.export_tenant("t"), b"")

    def test_unseen_tenant_in_populated_file_exports_empty_bytes(self):
        self.chain.append("a", {})
        self.chain.append("b", {})
        before = self.path.read_bytes()
        self.assertEqual(self.chain.export_tenant("t"), b"")
        self.assertEqual(self.path.read_bytes(), before)

    def test_returns_bytes_instance_utf8(self):
        self.chain.append("t", {"n": "审计"})
        blob = self.chain.export_tenant("t")
        self.assertIsInstance(blob, bytes)
        blob.decode("utf-8")  # must be valid UTF-8

    # --- successful export semantics ---

    def test_export_contains_only_target_rows_with_fields_and_values(self):
        items_t = [self.chain.append("t", {"i": i}) for i in range(3)]
        self.chain.append("u", {"j": 0})
        blob = self.chain.export_tenant("t")
        rows = [json.loads(l) for l in blob.splitlines()]
        self.assertEqual(rows, items_t)
        self.assertTrue(all(set(r) == {"tenant", "seq", "event", "prev", "hash"}
                            for r in rows))
        # fields and values are exactly the verified source records, and no
        # other tenant's record is carried along
        self.assertEqual(rows,
                         [r for r in self.source_rows() if r["tenant"] == "t"])
        self.assertTrue(all(r["tenant"] == "t" for r in rows))

    def test_export_uses_physical_order_despite_interleaving(self):
        self.chain.append("a", {"i": 1})
        self.chain.append("b", {"i": 1})
        self.chain.append_batch("a", [{"i": 2}, {"i": 3}])
        self.chain.append("b", {"i": 2})
        blob = self.chain.export_tenant("a")
        rows = [json.loads(l) for l in blob.splitlines()]
        self.assertEqual([r["seq"] for r in rows], [1, 2, 3])
        self.assertEqual([r["prev"] for r in rows],
                         [ZERO, rows[0]["hash"], rows[1]["hash"]])
        # physical order of appearance in the source, not re-sorted
        src = [r for r in self.source_rows() if r["tenant"] == "a"]
        self.assertEqual(rows, src)

    def test_export_byte_equivalent_to_chain_without_other_tenants(self):
        self.chain.append("a", {})
        self.chain.append_batch("t", [{"i": 1}, {"i": 2}])
        self.chain.append("z", {})
        self.chain.append("t", {"i": 3})
        self.chain.append_batch("z", [{}, {}])
        blob = self.chain.export_tenant("t")

        solo = AuditChain(self.path.with_name("solo.jsonl"))
        for e in ({"i": 1}, {"i": 2}, {"i": 3}):
            solo.append("t", e)
        self.assertEqual(blob, solo.path.read_bytes())

    def test_exported_blob_verifies_offline_with_matching_history(self):
        first = self.chain.append("t", {"v": "审计"})
        self.chain.append("other", {})
        second = self.chain.append("t", {"v": [1, 2, {"x": True}]})
        self.chain.append_batch("t", [None, 1.5, "str"])
        self.chain.append("other", {})
        blob = self.chain.export_tenant("t")

        out, migrated = self.rehydrate(blob)
        self.assertEqual(migrated.verify("t"), {"ok": True, "count": 5})
        r = migrated.verify_all()
        self.assertEqual(r, {"ok": True, "tenants": [
            {"tenant": "t", "count": 5}]})
        rows = [json.loads(l) for l in out.read_text().splitlines()]
        self.assertEqual(rows[0], first)
        self.assertEqual(rows[1], second)
        self.assertEqual([r["seq"] for r in rows], [1, 2, 3, 4, 5])
        self.assertEqual([r["prev"] for r in rows],
                         [ZERO, first["hash"], second["hash"],
                          rows[2]["hash"], rows[3]["hash"]])
        # no other tenant hitched a ride
        self.assertTrue(all(r["tenant"] == "t" for r in rows))
        # the source file itself was never modified
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": 5})

    def test_repeated_exports_are_byte_identical(self):
        self.chain.append_batch("t", [{"i": i} for i in range(4)])
        self.chain.append("u", {})
        first = self.chain.export_tenant("t")
        for _ in range(5):
            self.assertEqual(self.chain.export_tenant("t"), first)

    def test_serialization_and_newline_rules_match_append(self):
        self.chain.append("t", {"b": 2, "a": 1})
        blob = self.chain.export_tenant("t")
        # exactly one trailing newline per record, sorted-key serialization
        self.assertTrue(blob.endswith(b"\n"))
        self.assertFalse(blob.endswith(b"\n\n"))
        self.assertEqual(len(blob.splitlines()), 1)
        self.assertEqual(
            blob,
            (json.dumps(json.loads(blob), sort_keys=True, allow_nan=False)
             + "\n").encode("utf-8"))

    def test_legacy_source_without_trailing_newline_exports_with_newlines(self):
        self.chain.append_batch("t", [{"v": 1}, {"v": 2}])
        self.path.write_bytes(self.path.read_bytes().rstrip(b"\n"))
        blob = self.chain.export_tenant("t")
        self.assertEqual(len(blob.splitlines()), 2)
        self.assertTrue(blob.endswith(b"\n"))
        out, migrated = self.rehydrate(blob)
        self.assertEqual(migrated.verify("t"), {"ok": True, "count": 2})

    # --- distinct JSON identities export separately ---

    def test_distinct_json_identities_are_separate_exports(self):
        for t in (1, 1.0, True, "1"):
            self.chain.append(t, {})
            self.chain.append(t, {})
        blobs = {json.dumps(t, sort_keys=True): self.chain.export_tenant(t)
                 for t in (1, 1.0, True, "1")}
        for key, blob in blobs.items():
            rows = [json.loads(l) for l in blob.splitlines()]
            self.assertEqual(len(rows), 2, key)
            self.assertTrue(all(json.dumps(r["tenant"], sort_keys=True) == key
                                for r in rows), key)
            out, migrated = self.rehydrate(blob)
            self.assertTrue(migrated.verify_all()["ok"])
            out.unlink()

    def test_complex_json_tenant_key_order_normalized(self):
        t1 = {"a": [1, {"x": 1, "y": 2}], "b": {"p": True, "q": None}}
        t2 = {"b": {"q": None, "p": True}, "a": [1, {"y": 2, "x": 1}]}
        self.chain.append(t1, {"i": 1})
        other = {"a": [1, {"x": 1, "y": 2}]}
        self.chain.append(other, {})
        self.chain.append(t2, {"i": 2})
        blob = self.chain.export_tenant({"b": {"q": None, "p": True},
                                        "a": [1, {"y": 2, "x": 1}]})
        rows = [json.loads(l) for l in blob.splitlines()]
        self.assertEqual([r["seq"] for r in rows], [1, 2])
        self.assertEqual(rows[1]["prev"], rows[0]["hash"])
        # the different-looking object is not the same tenant
        self.assertEqual(len(self.chain.export_tenant(other).splitlines()), 1)
        out, migrated = self.rehydrate(blob)
        self.assertEqual(migrated.verify(t1), {"ok": True, "count": 2})

    def test_unicode_tenant_and_events_roundtrip(self):
        t = {"区": ["租户", None]}
        self.chain.append(t, {"名": "审计"})
        self.chain.append("别的租户", {})
        blob = self.chain.export_tenant(t)
        out, migrated = self.rehydrate(blob)
        self.assertEqual(migrated.verify(t), {"ok": True, "count": 1})
        self.assertEqual(migrated.verify_all()["tenants"][0]["tenant"], t)

    # --- source state errors: same mapping/priority as append/verify ---

    def assertStateError(self, ctx, tenant, seq, reason, line):
        e = ctx.exception
        self.assertEqual((e.tenant, e.seq, e.reason, e.line),
                         (tenant, seq, reason, line))

    def test_missing_field_raises_state_error_without_writing(self):
        self.write_rows([{"tenant": "t", "seq": 1, "event": {},
                          "prev": ZERO}])  # hash missing
        before = self.path.read_bytes()
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.export_tenant("t")
        self.assertStateError(cm, "t", 1, "missing", 1)
        self.assertEqual(self.path.read_bytes(), before)

    def test_unparseable_json_falls_back_to_expected_seq(self):
        good = _row("x", 1)
        self.path.write_bytes(_valid_blob(good) + b"{not json\n")
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.export_tenant("t")
        # t unseen -> expected seq 1; physical line is 2
        self.assertStateError(cm, "t", 1, "missing", 2)

    def test_unparseable_other_tenant_line_is_still_fatal(self):
        self.write_rows(['{"tenant": "x", broken'])
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.export_tenant("t")
        self.assertStateError(cm, "t", 1, "missing", 1)

    def test_illegal_utf8_reports_bad_physical_line(self):
        before = _valid_blob(_row("t", 1)) + b"\xffgarbage"
        self.path.write_bytes(before)
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.export_tenant("t")
        self.assertStateError(cm, "t", 2, "missing", 2)
        self.assertEqual(self.path.read_bytes(), before)

    def test_sequence_gap_is_sequence_at_expected(self):
        good = _row("t", 1)
        bad = _row("t", 3, prev=good["hash"])
        self.write_rows([good, bad])
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.export_tenant("t")
        self.assertStateError(cm, "t", 2, "sequence", 2)

    def test_bad_prev_link_is_digest(self):
        self.chain.append("t", {})
        bad = _row("t", 2, prev="9" * 64)
        self.write_rows(self.source_rows() + [bad])
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.export_tenant("t")
        self.assertStateError(cm, "t", 2, "digest", 2)

    def test_tampered_hash_is_digest(self):
        self.chain.append("t", {})
        rows = self.source_rows()
        rows[0]["event"] = {"v": 2}
        self.write_rows(rows)
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.export_tenant("t")
        self.assertStateError(cm, "t", 1, "digest", 1)

    def test_earlier_digest_beats_later_bad_bytes(self):
        tampered = _row("t", 1)
        tampered["event"] = {"z": 9}  # hash stays the valid-row hash
        self.path.write_bytes(
            (json.dumps(tampered, sort_keys=True) + "\n").encode("utf-8")
            + b"\xff\n")
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.export_tenant("t")
        self.assertStateError(cm, "t", 1, "digest", 1)

    def test_earlier_sequence_beats_later_bad_bytes(self):
        row = _row("t", 2)
        self.path.write_bytes(
            (json.dumps(row, sort_keys=True) + "\n").encode("utf-8") + b"\xff")
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.export_tenant("t")
        self.assertStateError(cm, "t", 1, "sequence", 1)

    def test_earlier_unparseable_json_beats_later_bad_bytes(self):
        self.path.write_bytes(b"{oops\n\xff")
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.export_tenant("t")
        self.assertStateError(cm, "t", 1, "missing", 1)

    def test_corruption_of_other_tenant_does_not_block_export(self):
        self.chain.append("a", {})
        self.chain.append("t", {})
        rows = self.source_rows()
        rows[0]["event"] = {"x": 1}  # breaks a's digest; irrelevant to t
        self.write_rows(rows)
        blob = self.chain.export_tenant("t")
        self.assertEqual(len(blob.splitlines()), 1)
        out, migrated = self.rehydrate(blob)
        self.assertEqual(migrated.verify("t"), {"ok": True, "count": 1})
        # and the corrupted target tenant still fails
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.export_tenant("a")
        self.assertStateError(cm, "a", 1, "digest", 1)

    def test_no_partial_prefix_on_failure(self):
        # two good rows then a broken one: export must raise, never return
        # the verified prefix
        self.chain.append_batch("t", [{"i": 1}, {"i": 2}])
        bad = _row("t", 99, prev="0" * 64)
        self.write_rows(self.source_rows() + [bad])
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.export_tenant("t")
        self.assertEqual((cm.exception.seq, cm.exception.reason, cm.exception.line),
                         (3, "sequence", 3))
        self.assertEqual(self.chain.verify("t")["reason"], "sequence")

    # --- tenant input boundary ---

    def test_illegal_tenant_raises_value_error_without_file_access(self):
        for bad in (float("nan"), float("inf"), float("-inf"),
                    {"k": float("nan")}, {1: "x"}, object(),
                    b"bytes", {1, 2}, ("a", 1)):
            with self.assertRaises(ValueError):
                self.chain.export_tenant(bad)
        cyc = []
        cyc.append(cyc)
        with self.assertRaises(ValueError):
            self.chain.export_tenant(cyc)
        # nothing was ever read from or created at the missing path
        self.assertFalse(self.path.exists())

    def test_value_error_beats_corrupt_history_and_keeps_bytes(self):
        bad_row = _row("t", 1, event={"tampered": True})
        self.write_rows([bad_row])
        before = self.path.read_bytes()
        for bad in (float("nan"), {"k": float("inf")}, {1: "x"}):
            with self.assertRaises(ValueError):
                self.chain.export_tenant(bad)
        self.assertEqual(self.path.read_bytes(), before)

    def test_legal_export_after_rejected_call_unaffected(self):
        self.chain.append("t", {})
        with self.assertRaises(ValueError):
            self.chain.export_tenant(float("nan"))
        self.assertEqual(len(self.chain.export_tenant("t").splitlines()), 1)

    # --- concurrent appends: only complete pre/post snapshots ---

    @staticmethod
    def _check_complete_snapshot(blob, total):
        rows = [json.loads(l) for l in blob.splitlines()]
        n = len(rows)
        assert 0 <= n <= total
        prev = ZERO
        for i, r in enumerate(rows, 1):
            assert r["tenant"] == "t"
            assert r["seq"] == i
            assert r["prev"] == prev
            payload = json.dumps(
                {k: r[k] for k in ("tenant", "seq", "event", "prev")},
                sort_keys=True, separators=(",", ":")).encode("utf-8")
            assert r["hash"] == hashlib.sha256(payload).hexdigest()
            prev = r["hash"]
        return n

    def test_concurrent_appends_yield_only_whole_snapshots(self):
        n_writer, per = 8, 25
        total = n_writer * per
        stop = threading.Event()
        sizes = []
        problems = []
        box = threading.Lock()

        def reader():
            while not stop.is_set():
                try:
                    blob = self.chain.export_tenant("t")
                    n = self._check_complete_snapshot(blob, total)
                except Exception as e:  # noqa: BLE001
                    with box:
                        problems.append(repr(e))
                    return
                with box:
                    sizes.append(n)

        readers = [threading.Thread(target=reader, daemon=True)
                   for _ in range(4)]

        def writer(i):
            for j in range(per):
                self.chain.append("t", {"w": i, "j": j})

        threads = [threading.Thread(target=writer, args=(i,))
                   for i in range(n_writer)]
        # a deterministic pre-append snapshot before any writer starts
        self.assertEqual(
            self._check_complete_snapshot(self.chain.export_tenant("t"), total),
            0)
        for t in readers:
            t.start()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        stop.set()
        for t in readers:
            t.join(timeout=2)

        self.assertEqual(problems, [])
        # readers observed snapshots of various sizes, each a fully
        # self-consistent state after k complete appends
        self.assertTrue(sizes)
        self.assertIn(total, sizes)
        self.assertEqual(sizes[-1], total)
        final = self.chain.export_tenant("t")
        self.assertEqual(self._check_complete_snapshot(final, total), total)
        # every export during the race also verifies offline as-is
        self.assertEqual(AuditChain(self.path).verify("t"),
                         {"ok": True, "count": total})

    def test_concurrent_other_tenant_appends_keep_export_stable(self):
        stop = threading.Event()
        blobs = set()
        box = threading.Lock()

        def reader():
            while not stop.is_set():
                blob = self.chain.export_tenant("t")
                with box:
                    blobs.add(blob)

        def writer():
            for i in range(50):
                self.chain.append("noise", {"i": i})

        self.chain.append("t", {"only": 1})
        expected = self.chain.export_tenant("t")
        rthread = threading.Thread(target=reader, daemon=True)
        rthread.start()
        wthread = threading.Thread(target=writer)
        wthread.start()
        wthread.join()
        stop.set()
        rthread.join(timeout=2)

        self.assertEqual(blobs, {expected})


if __name__ == "__main__":
    unittest.main()
