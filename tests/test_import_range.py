import json
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app import (
    AuditChain,
    AuditChainConflictError,
    AuditChainStateError,
    ZERO,
)


def record_bytes(item):
    return (json.dumps(item, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def valid_row(tenant="t", seq=1, prev=ZERO, event=None):
    row = {"tenant": tenant, "seq": seq, "event": event or {}, "prev": prev}
    row["hash"] = AuditChain._hash(row)
    return row


class ImportTenantRangeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.path = Path(self.tmp.name) / "audit.jsonl"
        self.chain = AuditChain(self.path)
        self.source = AuditChain(self.path.with_name("source.jsonl"))
        self.records = []
        for i in range(6):
            self.records.append(self.source.append("t", {"i": i, "s": "审计"}))
            self.source.append("o", {"i": i})

    def tearDown(self):
        self.tmp.cleanup()

    def segment(self, lo, hi):
        return self.source.export_tenant_range("t", lo, hi)

    def anchored_segment(self, first_seq, prev_hash, events):
        # Build a self-consistent segment whose first record links to an
        # arbitrary asserted prefix head (used to reach target-side conflict
        # checks, where the input itself must validate but the target head
        # need not equal the asserted one).
        rows, chunks = [], []
        for i, event in enumerate(events):
            row = {"tenant": "t", "seq": first_seq + i, "event": event,
                   "prev": prev_hash}
            row["hash"] = AuditChain._hash(row)
            rows.append(row)
            chunks.append(record_bytes(row))
            prev_hash = row["hash"]
        return rows, b"".join(chunks)

    # --- boundary: validation order, empty no-op, nothing touched ---

    def test_data_must_be_bytes(self):
        h = self.records[0]["hash"]
        for bad in ("", "not jsonl", bytearray(b"x"), 1, None, [], object()):
            with self.assertRaises(ValueError):
                self.chain.import_tenant_range("t", bad, 1, h)
        self.assertFalse(self.path.exists())

    def test_count_must_be_non_negative_plain_int(self):
        data = self.segment(2, 2)
        for bad in (-1, True, False, 1.0, "1", None, [0], 1.0):
            with self.assertRaises(ValueError, msg=bad):
                self.chain.import_tenant_range("t", data, bad,
                                               self.records[0]["hash"])
        self.assertFalse(self.path.exists())

    def test_hash_must_be_64_lowercase_hex(self):
        data = self.segment(1, 1)
        for bad in ("x" * 63, "x" * 65, "A" * 64, "g" * 64, 1, None,
                    b"0" * 64, ""):
            with self.assertRaises(ValueError, msg=bad):
                self.chain.import_tenant_range("t", data, 0, bad)
        self.assertFalse(self.path.exists())

    def test_illegal_tenant_is_value_error_even_for_empty_data(self):
        for bad in (float("nan"), float("inf"), {1: "x"}, object(),
                    b"bytes", {1, 2}):
            with self.assertRaises(ValueError):
                self.chain.import_tenant_range(bad, b"", 0, ZERO)
        self.assertFalse(self.path.exists())

    def test_all_boundaries_checked_before_any_read(self):
        # even a corrupt target is never read when an argument is illegal
        self.path.write_bytes(b"\xff")
        corrupt = self.path.read_bytes()
        data = self.segment(1, 1)
        for call in (
            lambda: self.chain.import_tenant_range("t", data, -1, ZERO),
            lambda: self.chain.import_tenant_range("t", data, True, ZERO),
            lambda: self.chain.import_tenant_range("t", data, 0, "BAD"),
            lambda: self.chain.import_tenant_range("t", "s", 0, ZERO),
            lambda: self.chain.import_tenant_range(float("nan"), data, 0, ZERO),
        ):
            with self.assertRaises(ValueError):
                call()
        self.assertEqual(self.path.read_bytes(), corrupt)

    def test_empty_bytes_is_noop_under_any_assertion(self):
        self.assertFalse(self.path.exists())
        self.assertEqual(
            self.chain.import_tenant_range("t", b"", 3, "a" * 64), [])
        self.assertFalse(self.path.exists())
        # corrupt target: the no-op never reads and leaves the bytes
        self.path.write_bytes(b"\xff")
        corrupt = self.path.read_bytes()
        self.assertEqual(
            self.chain.import_tenant_range("t", b"", 0, ZERO), [])
        self.assertEqual(self.path.read_bytes(), corrupt)
        # healthy target: nothing appended
        self.path.unlink()
        self.chain.append("other", {})
        before = self.path.read_bytes()
        self.assertEqual(
            self.chain.import_tenant_range(
                "t", b"", 5, self.records[4]["hash"]), [])
        self.assertEqual(self.path.read_bytes(), before)

    # --- success: grafting segments, verbatim, continuation ---

    def test_graft_full_chain_under_zero_assertion_creates_log(self):
        data = self.segment(1, 6)
        records = self.chain.import_tenant_range("t", data, 0, ZERO)
        self.assertEqual([r["seq"] for r in records], [1, 2, 3, 4, 5, 6])
        self.assertEqual(self.path.read_bytes(), data)
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": 6})
        self.assertEqual(self.chain.export_tenant("t"),
                         self.source.export_tenant("t"))

    def test_graft_segment_after_existing_prefix(self):
        self.chain.import_tenant_range("t", self.segment(1, 2), 0, ZERO)
        records = self.chain.import_tenant_range(
            "t", self.segment(3, 5), 2, self.records[1]["hash"])
        self.assertEqual([r["seq"] for r in records], [3, 4, 5])
        self.assertEqual(records[0]["prev"], self.records[1]["hash"])
        self.chain.import_tenant_range(
            "t", self.segment(6, 6), 5, self.records[4]["hash"])
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": 6})
        self.assertEqual(self.chain.export_tenant("t"),
                         self.source.export_tenant("t"))
        # continuing with a plain append links to the imported tail
        nxt = self.chain.append("t", {"i": 6})
        self.assertEqual((nxt["seq"], nxt["prev"]),
                         (7, self.records[5]["hash"]))

    def test_single_record_segment_round_trip(self):
        self.chain.import_tenant_range("t", self.segment(1, 3), 0, ZERO)
        records = self.chain.import_tenant_range(
            "t", self.segment(4, 4), 3, self.records[2]["hash"])
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0], self.records[3])

    def test_records_keep_exactly_five_fields_and_values_verbatim(self):
        self.chain.import_tenant_range("t", self.segment(1, 1), 0, ZERO)
        records = self.chain.import_tenant_range(
            "t", self.segment(2, 4), 1, self.records[0]["hash"])
        for r in records:
            self.assertEqual(set(r), {"tenant", "seq", "event", "prev", "hash"})
            self.assertEqual(r["hash"], AuditChain._hash(r))
        self.assertEqual(records, self.records[1:4])

    def test_graft_coexists_with_other_tenants_on_target(self):
        for i in range(2):
            self.chain.append("a", {"i": i})
            self.chain.append("b", {"i": i})
        before = self.path.read_bytes()
        # segment 1..3 anchored at an empty t chain
        records = self.chain.import_tenant_range(
            "t", self.segment(1, 3), 0, ZERO)
        self.assertEqual(len(records), 3)
        self.assertEqual(self.path.read_bytes(), before + self.segment(1, 3))
        self.assertTrue(self.chain.verify_all()["ok"])
        self.assertEqual(self.chain.verify("a"), {"ok": True, "count": 2})
        self.assertEqual(self.chain.export_tenant("t"), self.segment(1, 3))

    def test_target_without_trailing_newline_is_healed(self):
        other = valid_row("a", 1)
        raw = json.dumps(other, sort_keys=True).encode("utf-8")
        self.path.write_bytes(raw)
        self.chain.import_tenant_range("t", self.segment(1, 2), 0, ZERO)
        merged = self.path.read_bytes()
        self.assertTrue(self.chain.verify_all()["ok"])
        self.assertEqual(merged, raw + b"\n" + self.segment(1, 2))

    # --- target-not-exists: only (0, ZERO) may create ---

    def test_non_empty_assertion_on_missing_target_conflicts_creates_nothing(self):
        with self.assertRaises(AuditChainConflictError) as cm:
            self.chain.import_tenant_range(
                "t", self.segment(2, 2), 1, self.records[0]["hash"])
        e = cm.exception
        self.assertEqual(e.tenant, "t")
        self.assertEqual(e.reason, "conflict")
        self.assertEqual((e.expected_count, e.expected_hash),
                         (1, self.records[0]["hash"]))
        self.assertEqual((e.actual_count, e.actual_hash), (0, ZERO))
        self.assertFalse(self.path.exists())

    def test_wrong_zero_hash_on_missing_target_conflicts(self):
        # the input is self-consistent with the asserted (but wrong) head;
        # the missing target is still the real empty head, so this is a
        # target-side conflict rather than an input digest error
        _rows, data = self.anchored_segment(1, "a" * 64, [{}])
        with self.assertRaises(AuditChainConflictError) as cm:
            self.chain.import_tenant_range(
                "t", data, 0, "a" * 64)
        self.assertEqual((cm.exception.actual_count, cm.exception.actual_hash),
                         (0, ZERO))
        self.assertFalse(self.path.exists())

    # --- conflict against an existing head ---

    def test_stale_count_or_hash_conflicts_with_actual_tail(self):
        self.chain.import_tenant_range("t", self.segment(1, 3), 0, ZERO)
        before = self.path.read_bytes()
        # wrong count, right-shaped hash: a source segment starting at seq 3
        # is self-consistent with the asserted (2, record-2-hash) head, but
        # the target already holds 3 records -> conflict at the real tail
        with self.assertRaises(AuditChainConflictError) as cm:
            self.chain.import_tenant_range(
                "t", self.segment(3, 3), 2, self.records[1]["hash"])
        e = cm.exception
        self.assertEqual((e.expected_count, e.expected_hash),
                         (2, self.records[1]["hash"]))
        self.assertEqual((e.actual_count, e.actual_hash),
                         (3, self.records[2]["hash"]))
        # right count, wrong hash: the segment itself links to the asserted
        # bogus prefix head, but the target tail differs
        _rows, data = self.anchored_segment(4, "9" * 64, [{}])
        with self.assertRaises(AuditChainConflictError) as cm2:
            self.chain.import_tenant_range(
                "t", data, 3, "9" * 64)
        self.assertEqual((cm2.exception.actual_count, cm2.exception.actual_hash),
                         (3, self.records[2]["hash"]))
        self.assertEqual(self.path.read_bytes(), before)

    def test_other_tenant_interleaving_still_reports_real_tail(self):
        self.chain.import_tenant_range("t", self.segment(1, 1), 0, ZERO)
        self.chain.append("a", {})
        self.chain.import_tenant_range(
            "t", self.segment(2, 2), 1, self.records[0]["hash"])
        self.chain.append("a", {})
        before = self.path.read_bytes()
        # a stale assertion: the input segment (seq 2 linked to record 1's
        # hash) is self-consistent with (1, hash1), but the target tail has
        # already advanced to (2, hash2)
        with self.assertRaises(AuditChainConflictError) as cm:
            self.chain.import_tenant_range(
                "t", self.segment(2, 2), 1, self.records[0]["hash"])
        self.assertEqual((cm.exception.actual_count, cm.exception.actual_hash),
                         (2, self.records[1]["hash"]))
        self.assertEqual(self.path.read_bytes(), before)

    def test_distinct_json_identities_never_conflict(self):
        self.chain.append(1, {})
        records = self.chain.import_tenant_range(
            "1", record_bytes(valid_row("1", 1)), 0, ZERO)
        self.assertEqual(len(records), 1)
        self.assertTrue(self.chain.verify_all()["ok"])

    # --- input chain validation: missing / sequence / digest at first line ---

    def test_input_wrong_start_seq_is_sequence(self):
        # assertion count 2 means the segment must start at seq 3
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.import_tenant_range(
                "t", self.segment(2, 3), 2, self.records[1]["hash"])
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("t", 3, "sequence", 1))
        self.assertFalse(self.path.exists())

    def test_input_first_prev_must_equal_expected_hash(self):
        row = valid_row("t", 2, prev="f" * 64)
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.import_tenant_range(
                "t", record_bytes(row), 1, self.records[0]["hash"])
        self.assertEqual((cm.exception.seq, cm.exception.reason,
                          cm.exception.line), (2, "digest", 1))

    def test_input_mid_chain_broken_link_reports_its_physical_line(self):
        good2 = valid_row("t", 2, prev=self.records[0]["hash"])
        bad4 = valid_row("t", 4, prev="9" * 64)
        data = record_bytes(good2) + record_bytes(bad4)
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.import_tenant_range(
                "t", data, 1, self.records[0]["hash"])
        self.assertEqual((cm.exception.seq, cm.exception.reason,
                          cm.exception.line), (3, "sequence", 2))
        # a present-but-wrong prev on the second physical line is digest
        good3 = valid_row("t", 3, prev=good2["hash"])
        bad4d = valid_row("t", 4, prev="9" * 64)
        data = record_bytes(good2) + record_bytes(good3) + record_bytes(bad4d)
        with self.assertRaises(AuditChainStateError) as cm2:
            self.chain.import_tenant_range(
                "t", data, 1, self.records[0]["hash"])
        self.assertEqual((cm2.exception.seq, cm2.exception.reason,
                          cm2.exception.line), (4, "digest", 3))

    def test_input_hash_tampering_is_digest(self):
        row = valid_row("t", 1)
        row["event"] = {"tampered": True}
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.import_tenant_range("t", record_bytes(row), 0, ZERO)
        self.assertEqual((cm.exception.seq, cm.exception.reason,
                          cm.exception.line), (1, "digest", 1))

    def test_input_missing_or_extra_field_is_missing(self):
        lacking = json.dumps(
            {"tenant": "t", "seq": 1, "event": {}, "prev": ZERO}).encode() + b"\n"
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.import_tenant_range("t", lacking, 0, ZERO)
        self.assertEqual((cm.exception.seq, cm.exception.reason,
                          cm.exception.line), (1, "missing", 1))
        row = valid_row("t", 1)
        row["extra"] = 1
        with self.assertRaises(AuditChainStateError) as cm2:
            self.chain.import_tenant_range(
                "t", record_bytes(row), 0, ZERO)
        self.assertEqual((cm2.exception.seq, cm2.exception.reason,
                          cm2.exception.line), (1, "missing", 1))

    def test_input_unparseable_duplicate_keys_nan_bad_utf8_are_missing(self):
        cases = [
            (b"\n", 1),
            (b"[1]\n", 1),
            (b"{not json\n", 1),
            (b'{"tenant":"t","tenant":"t","seq":1,"event":{},'
             b'"prev":"' + ZERO.encode() + b'","hash":"x"}\n', 1),
            (b'{"tenant":"t","seq":1,"event":{"x":NaN},"prev":"'
             + ZERO.encode() + b'","hash":"x"}\n', 1),
            (record_bytes(valid_row("t", 1)) + b'{"tenant":\xff}\n', 2),
        ]
        for raw, line in cases:
            with self.assertRaises(AuditChainStateError, msg=raw) as cm:
                self.chain.import_tenant_range("t", raw, 0, ZERO)
            self.assertEqual((cm.exception.reason, cm.exception.line),
                             ("missing", line), raw)

    def test_foreign_tenant_record_is_value_error(self):
        foreign = record_bytes(valid_row("other", 1))
        with self.assertRaises(ValueError):
            self.chain.import_tenant_range("t", foreign, 0, ZERO)
        # a foreign record later in the stream, after a valid prefix
        data = self.segment(1, 1) + record_bytes(valid_row("other", 2))
        with self.assertRaises(ValueError):
            self.chain.import_tenant_range("t", data, 0, ZERO)
        self.assertFalse(self.path.exists())

    def test_foreign_distinct_json_identity_is_value_error(self):
        data = record_bytes(valid_row(1, 1)) + record_bytes(valid_row("1", 2))
        with self.assertRaises(ValueError):
            self.chain.import_tenant_range(1, data, 0, ZERO)

    def test_input_state_error_takes_priority_over_target_conflict(self):
        self.chain.import_tenant_range("t", self.segment(1, 2), 0, ZERO)
        # target already has 2 records (would conflict with a (0, ZERO)
        # assertion), but the input itself is broken: input error wins
        broken = record_bytes(valid_row("t", 2))  # seq 2 where 1 expected
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.import_tenant_range("t", broken, 0, ZERO)
        self.assertEqual(cm.exception.reason, "sequence")

    def test_failed_input_writes_no_byte(self):
        self.chain.append("other", {"keep": True})
        before = self.path.read_bytes()
        row = valid_row("t", 1)
        row["event"] = {"x": 1}
        with self.assertRaises(AuditChainStateError):
            self.chain.import_tenant_range(
                "t", record_bytes(row), 0, ZERO)
        self.assertEqual(self.path.read_bytes(), before)

    # --- target corruption: state error takes priority over conflict ---

    def test_corrupt_target_raises_state_error_and_writes_nothing(self):
        tampered = valid_row("t", 1)
        tampered["event"] = {"tampered": True}
        self.path.write_bytes(record_bytes(tampered))
        before = self.path.read_bytes()
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.import_tenant_range(
                "t", self.segment(2, 2), 1, self.records[0]["hash"])
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("t", 1, "digest", 1))
        self.assertEqual(self.path.read_bytes(), before)

    def test_other_tenant_corruption_does_not_block_segment_import(self):
        bad_other = valid_row("b", 1)
        bad_other["event"] = {"x": 1}
        self.path.write_bytes(record_bytes(bad_other))
        records = self.chain.import_tenant_range(
            "t", self.segment(1, 2), 0, ZERO)
        self.assertEqual(len(records), 2)
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": 2})


class ImportRangeConcurrencyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.path = Path(self.tmp.name) / "audit.jsonl"
        self.chain = AuditChain(self.path)
        self.source = AuditChain(self.path.with_name("source.jsonl"))
        self.records = [self.source.append("t", {"i": i}) for i in range(20)]
        # split into two segments: 1..10 and 11..20
        self.seg_a = self.source.export_tenant_range("t", 1, 10)
        self.seg_b = self.source.export_tenant_range("t", 11, 20)
        self.mid_hash = self.records[9]["hash"]
        self.tail_hash = self.records[-1]["hash"]

    def tearDown(self):
        self.tmp.cleanup()

    def test_parallel_first_segments_one_wins_rest_conflict_at_winner_head(self):
        successes, conflicts, others = [], [], []
        box = threading.Lock()

        def worker():
            try:
                records = self.chain.import_tenant_range(
                    "t", self.seg_a, 0, ZERO)
                with box:
                    successes.append(len(records))
            except AuditChainConflictError as e:
                with box:
                    conflicts.append((e.expected_count, e.expected_hash,
                                      e.actual_count, e.actual_hash))
            except Exception as e:  # noqa: BLE001
                with box:
                    others.append(repr(e))

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(others, [])
        self.assertEqual(successes, [10])
        self.assertEqual(len(conflicts), 7)
        for exp_count, exp_hash, act_count, act_hash in conflicts:
            self.assertEqual((exp_count, exp_hash), (0, ZERO))
            # losers observe the winner's real chain head
            self.assertEqual((act_count, act_hash), (10, self.mid_hash))

    def test_staged_imports_under_contention_converge_once(self):
        # one winner plants segment A; many contenders then race segment B
        # against the (10, mid) head: exactly one wins, the rest conflict at
        # (20, tail); the final log verifies and re-exports the full chain.
        self.chain.import_tenant_range("t", self.seg_a, 0, ZERO)
        outcomes = {"ok": 0, "conflict": 0, "other": []}
        box = threading.Lock()

        def worker():
            try:
                self.chain.import_tenant_range(
                    "t", self.seg_b, 10, self.mid_hash)
                with box:
                    outcomes["ok"] += 1
            except AuditChainConflictError as e:
                with box:
                    outcomes["conflict"] += 1
                    assert (e.actual_count, e.actual_hash) == (
                        20, self.tail_hash)
            except Exception as e:  # noqa: BLE001
                with box:
                    outcomes["other"].append(repr(e))

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(outcomes["other"], [])
        self.assertEqual((outcomes["ok"], outcomes["conflict"]), (1, 7))
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": 20})
        self.assertEqual(self.chain.export_tenant("t"),
                         self.source.export_tenant("t"))

    def test_segment_block_is_indivisible_against_appends_and_readers(self):
        self.chain.import_tenant_range("t", self.seg_a, 0, ZERO)
        stop = threading.Event()
        problems = []
        box = threading.Lock()

        def reader():
            while not stop.is_set():
                r = self.chain.verify("t")
                if not r["ok"]:
                    with box:
                        problems.append(("verify", r))
                    return
                # t is either the 10-record prefix or the full 20-record
                # chain; a partial segment B would be a torn block
                if r["count"] not in (10, 20):
                    with box:
                        problems.append(("partial", r["count"]))
                    return

        readers = [threading.Thread(target=reader) for _ in range(4)]
        for t in readers:
            t.start()

        def other_writer(i):
            c = AuditChain(self.path)
            for j in range(40):
                c.append("other", {"w": i, "j": j})

        writers = [threading.Thread(target=other_writer, args=(i,))
                   for i in range(4)]
        result = {"ok": 0, "conflict": 0}

        def importer():
            try:
                self.chain.import_tenant_range(
                    "t", self.seg_b, 10, self.mid_hash)
                with box:
                    result["ok"] += 1
            except AuditChainConflictError:
                with box:
                    result["conflict"] += 1

        importers = [threading.Thread(target=importer) for _ in range(6)]
        for t in writers + importers:
            t.start()
        for t in writers:
            t.join()
        for t in importers:
            t.join()
        stop.set()
        for t in readers:
            t.join(timeout=2)

        self.assertEqual(problems, [])
        self.assertEqual((result["ok"], result["conflict"]), (1, 5))
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": 20})
        self.assertEqual(self.chain.verify("other"),
                         {"ok": True, "count": 160})
        self.assertTrue(self.chain.verify_all()["ok"])
        self.assertEqual(self.chain.export_tenant("t"),
                         self.source.export_tenant("t"))


if __name__ == "__main__":
    unittest.main()
