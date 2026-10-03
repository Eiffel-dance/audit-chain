import json
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app import AuditChain, AuditChainRangeError, AuditChainStateError, ZERO


def record_bytes(item):
    return (json.dumps(item, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def valid_row(tenant="t", seq=1, prev=ZERO, event=None):
    row = {"tenant": tenant, "seq": seq, "event": event or {}, "prev": prev}
    row["hash"] = AuditChain._hash(row)
    return row


class ExportTenantRangeTest(unittest.TestCase):
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

    def seed(self, tenant="t", n=6, other=0):
        records = []
        for i in range(n):
            records.append(self.chain.append(tenant, {"i": i, "s": "审计"}))
            for _ in range(other):
                self.chain.append("other", {"i": i})
        return records

    # --- success: bytes, verbatim segment with omitted-prefix head ---

    def test_closed_interval_returns_utf8_bytes_with_original_links(self):
        records = self.seed("t", 6)
        data = self.chain.export_tenant_range("t", 3, 5)
        self.assertIsInstance(data, bytes)
        rows = [json.loads(l) for l in data.decode("utf-8").splitlines()]
        self.assertEqual([r["seq"] for r in rows], [3, 4, 5])
        self.assertTrue(all(r["tenant"] == "t" for r in rows))
        self.assertTrue(all(
            set(r) == {"tenant", "seq", "event", "prev", "hash"} for r in rows))
        # values verbatim: prev/hash are the source chain's, first prev is the
        # omitted prefix tail (record 2's hash), not ZERO
        self.assertEqual(rows[0]["prev"], records[1]["hash"])
        self.assertEqual(rows[1]["prev"], records[2]["hash"])
        self.assertEqual(rows[2]["prev"], records[3]["hash"])
        for r in rows:
            self.assertEqual(r["hash"], AuditChain._hash(r))
        self.assertTrue(data.endswith(b"\n"))

    def test_end_seq_none_runs_through_the_tail(self):
        records = self.seed("t", 4)
        data = self.chain.export_tenant_range("t", 3)
        rows = [json.loads(l) for l in data.decode("utf-8").splitlines()]
        self.assertEqual([r["seq"] for r in rows], [3, 4])
        self.assertEqual(rows[0]["prev"], records[1]["hash"])
        self.assertEqual(rows[-1]["hash"], records[-1]["hash"])

    def test_single_record_interval(self):
        records = self.seed("t", 3)
        data = self.chain.export_tenant_range("t", 2, 2)
        rows = [json.loads(l) for l in data.decode("utf-8").splitlines()]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0], records[1])
        self.assertEqual(rows[0]["prev"], records[0]["hash"])

    def test_full_chain_interval_from_one_equals_export(self):
        self.seed("t", 5)
        self.assertEqual(self.chain.export_tenant_range("t", 1, 5),
                         self.chain.export_tenant("t"))
        self.assertEqual(self.chain.export_tenant_range("t", 1),
                         self.chain.export_tenant("t"))

    def test_interleaved_tenants_are_dropped_without_reordering(self):
        records = self.seed("t", 4, other=1)
        data = self.chain.export_tenant_range("t", 2, 3)
        rows = [json.loads(l) for l in data.decode("utf-8").splitlines()]
        self.assertEqual([r["seq"] for r in rows], [2, 3])
        self.assertEqual(data, record_bytes(records[1]) + record_bytes(records[2]))

    def test_segment_verifies_offline_after_graft(self):
        records = self.seed("t", 4)
        seg = self.chain.export_tenant_range("t", 3, 4)
        out = Path(self.tmp.name) / "seg.jsonl"
        # the segment alone is not a from-ZERO chain; with its asserted prefix
        # head it continues a fresh chain seeded with records 1..2
        prefix = self.chain.export_tenant_range("t", 1, 2)
        out.write_bytes(prefix + seg)
        merged = AuditChain(out)
        self.assertEqual(merged.verify("t"), {"ok": True, "count": 4})
        self.assertEqual(merged.export_tenant("t"),
                         self.chain.export_tenant("t"))
        self.assertEqual(records[1]["hash"], json.loads(prefix.splitlines()[-1])["hash"])

    def test_distinct_json_identities_stay_separate(self):
        for t in (1, 1.0, True, "1"):
            self.chain.append(t, {})
            self.chain.append(t, {})
        for t in (1, 1.0, True, "1"):
            data = self.chain.export_tenant_range(t, 2, 2)
            rows = [json.loads(l) for l in data.decode("utf-8").splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(
                AuditChain._tenant_key(rows[0]["tenant"]),
                AuditChain._tenant_key(t))

    def test_repeated_exports_are_byte_identical(self):
        self.seed("t", 6, other=1)
        first = self.chain.export_tenant_range("t", 2, 5)
        for _ in range(4):
            self.assertEqual(self.chain.export_tenant_range("t", 2, 5), first)

    def test_export_creates_or_modifies_nothing(self):
        self.seed("t", 3, other=1)
        before = self.path.read_bytes()
        entries = sorted(p.name for p in Path(self.tmp.name).iterdir())
        self.chain.export_tenant_range("t", 2, 3)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(sorted(p.name for p in Path(self.tmp.name).iterdir()),
                         entries)

    # --- boundary validation: ValueError before any read ---

    def test_invalid_bounds_are_value_error_before_reading(self):
        self.seed("t", 3)
        corrupt = b"\xff"
        self.path.write_bytes(corrupt)
        bad_bounds = [
            (0, None), (-1, None), (True, None), (False, None),
            (1.0, None), ("1", None), (None, None), ([1], None),
            (1, 0), (2, 1), (1, -1), (1, True), (1, 2.0), (1, "2"),
        ]
        for start, end in bad_bounds:
            with self.assertRaises(ValueError, msg=(start, end)):
                self.chain.export_tenant_range("t", start, end)
        # no byte read means even a corrupt file is left untouched
        self.assertEqual(self.path.read_bytes(), corrupt)

    def test_end_equal_to_start_is_a_valid_one_record_interval(self):
        self.seed("t", 2)
        rows = [json.loads(l) for l in
                self.chain.export_tenant_range("t", 1, 1).decode().splitlines()]
        self.assertEqual([r["seq"] for r in rows], [1])

    def test_illegal_tenant_is_value_error_even_with_corrupt_history(self):
        row = valid_row("t", 1)
        row["event"] = {"tampered": True}
        self.write_rows([row])
        before = self.path.read_bytes()
        for bad in (float("nan"), float("inf"), {1: "x"}, object()):
            with self.assertRaises(ValueError):
                self.chain.export_tenant_range(bad, 1, 1)
        self.assertEqual(self.path.read_bytes(), before)

    def test_missing_path_with_invalid_bound_reads_nothing(self):
        self.assertFalse(self.path.exists())
        with self.assertRaises(ValueError):
            self.chain.export_tenant_range("t", 0)
        self.assertFalse(self.path.exists())

    # --- out of range: AuditChainRangeError ---

    def test_missing_file_empty_chain_is_range_error(self):
        self.assertFalse(self.path.exists())
        with self.assertRaises(AuditChainRangeError) as cm:
            self.chain.export_tenant_range("t", 1)
        e = cm.exception
        self.assertEqual((e.tenant, e.start_seq, e.end_seq, e.count, e.reason),
                         ("t", 1, None, 0, "range"))
        self.assertFalse(self.path.exists())

    def test_unknown_tenant_is_range_error(self):
        self.seed("a", 3)
        with self.assertRaises(AuditChainRangeError) as cm:
            self.chain.export_tenant_range("t", 1)
        self.assertEqual((cm.exception.tenant, cm.exception.count,
                          cm.exception.reason), ("t", 0, "range"))

    def test_start_past_tail_is_range_error(self):
        self.seed("t", 4)
        with self.assertRaises(AuditChainRangeError) as cm:
            self.chain.export_tenant_range("t", 5)
        e = cm.exception
        self.assertEqual((e.start_seq, e.end_seq, e.count), (5, None, 4))
        with self.assertRaises(AuditChainRangeError) as cm2:
            self.chain.export_tenant_range("t", 5, 6)
        self.assertEqual((cm2.exception.start_seq, cm2.exception.end_seq,
                          cm2.exception.count), (5, 6, 4))

    def test_end_past_tail_is_range_error_even_when_start_fits(self):
        self.seed("t", 4)
        with self.assertRaises(AuditChainRangeError) as cm:
            self.chain.export_tenant_range("t", 3, 5)
        e = cm.exception
        self.assertEqual((e.tenant, e.start_seq, e.end_seq, e.count, e.reason),
                         ("t", 3, 5, 4, "range"))

    def test_range_error_returns_no_bytes_and_modifies_nothing(self):
        self.seed("t", 2)
        before = self.path.read_bytes()
        with self.assertRaises(AuditChainRangeError):
            self.chain.export_tenant_range("t", 9, 10)
        self.assertEqual(self.path.read_bytes(), before)

    # --- source corruption: AuditChainStateError, priority over range ---

    def test_corrupt_history_raises_state_error_even_for_valid_interval(self):
        good = valid_row("t", 1)
        bad = valid_row("t", 2, prev=good["hash"])
        bad["event"] = {"tampered": True}
        self.write_rows([good, bad])
        before = self.path.read_bytes()
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.export_tenant_range("t", 1, 1)
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("t", 2, "digest", 2))
        self.assertEqual(self.path.read_bytes(), before)

    def test_corrupt_prefix_is_state_error_before_range_slice(self):
        good = valid_row("t", 1)
        gap = valid_row("t", 3, prev=good["hash"])
        self.write_rows([good, gap])
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.export_tenant_range("t", 3, 3)
        self.assertEqual((cm.exception.seq, cm.exception.reason,
                          cm.exception.line), (2, "sequence", 2))

    def test_other_tenant_corruption_does_not_block_range_export(self):
        bad_other = valid_row("b", 1)
        bad_other["event"] = {"x": 1}
        self.write_rows([bad_other])
        self.seed("t", 3)
        data = self.chain.export_tenant_range("t", 2, 3)
        self.assertEqual(
            [json.loads(l)["seq"] for l in data.decode().splitlines()], [2, 3])


class ExportRangeConcurrencyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.path = Path(self.tmp.name) / "audit.jsonl"
        self.chain = AuditChain(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_range_export_sees_only_complete_snapshots(self):
        for i in range(8):
            self.chain.append("t", {"i": i})
        problems = []
        box = threading.Lock()
        stop = threading.Event()

        def exporter():
            while not stop.is_set():
                try:
                    data = self.chain.export_tenant_range("t", 3)
                except AuditChainRangeError:
                    # tail must never shrink below 8 once baseline is set, and
                    # the baseline already exists, so this must not happen
                    with box:
                        problems.append(("range",))
                    return
                rows = [json.loads(l) for l in data.decode().splitlines()]
                seqs = list(range(3, 3 + len(rows)))
                if [r["seq"] for r in rows] != seqs:
                    with box:
                        problems.append(("seqs", [r["seq"] for r in rows]))
                    return
                if not rows:
                    with box:
                        problems.append(("empty",))
                    return
                for r in rows:
                    if r["hash"] != AuditChain._hash(r):
                        with box:
                            problems.append(("hash", r["seq"]))
                        return

        threads = [threading.Thread(target=exporter) for _ in range(4)]
        for t in threads:
            t.start()

        def writer(i):
            for j in range(40):
                self.chain.append("t", {"w": i, "j": j})

        writers = [threading.Thread(target=writer, args=(i,)) for i in range(6)]
        for w in writers:
            w.start()
        for w in writers:
            w.join()
        stop.set()
        for t in threads:
            t.join(timeout=2)
        self.assertEqual(problems, [])
        total = 8 + 6 * 40
        final = self.chain.export_tenant_range("t", 3)
        rows = [json.loads(l) for l in final.decode().splitlines()]
        self.assertEqual([r["seq"] for r in rows], list(range(3, total + 1)))


if __name__ == "__main__":
    unittest.main()
