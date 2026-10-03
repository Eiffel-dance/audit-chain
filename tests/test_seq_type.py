import json
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import app
from app import AuditChain, AuditChainStateError


ZERO = "0" * 64


def valid_row(tenant, seq, prev=ZERO, event=None):
    row = {"tenant": tenant, "seq": seq,
           "event": {} if event is None else event, "prev": prev}
    row["hash"] = AuditChain._hash(row)
    return row


def raw_seq_line(tenant_json, seq_text, prev=ZERO, event_json="{}"):
    # One physical JSONL record whose seq keeps the exact source spelling
    # (e.g. 1.0 or 1e0 instead of 1). The hash is computed over the parsed
    # values, so the only defect a scan can find in this line is the seq
    # itself -- never a digest mismatch.
    item = {
        "tenant": json.loads(tenant_json),
        "seq": json.loads(seq_text),
        "event": json.loads(event_json),
        "prev": prev,
    }
    item["hash"] = AuditChain._hash(item)
    return (
        '{"event": %s, "hash": "%s", "prev": "%s", "seq": %s, "tenant": %s}'
        % (event_json, item["hash"], prev, seq_text, tenant_json)
    )


class StrictSeqTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.path = Path(self.tmp.name) / "audit.jsonl"
        self.chain = AuditChain(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def write_text(self, text):
        with self.path.open("w", encoding="utf-8") as f:
            f.write(text)

    def append_text(self, text):
        with self.path.open("a", encoding="utf-8") as f:
            f.write(text)

    def assert_state_error(self, tenant, seq, line, fn, *args):
        with self.assertRaises(AuditChainStateError) as cm:
            fn(*args)
        e = cm.exception
        self.assertEqual((e.tenant, e.seq, e.reason, e.line),
                         (tenant, seq, "sequence", line))

    def assert_sequence_everywhere(self, tenant, expected_seq, line):
        # Every public entry point must agree on the same sequence verdict
        # for the corrupt line currently on disk, and no write entry point
        # may change a single byte while reporting it.
        data = self.path.read_bytes()
        verdict = {"ok": False, "at": expected_seq, "reason": "sequence"}
        self.assertEqual(self.chain.verify(tenant), verdict)
        self.assertEqual(self.chain.verify(tenant, 0), verdict)
        self.assertEqual(self.chain.verify_bytes(data, tenant), verdict)
        all_verdict = {"ok": False, "at": line, "tenant": tenant,
                       "reason": "sequence"}
        self.assertEqual(self.chain.verify_all(), all_verdict)
        self.assertEqual(self.chain.heads(), all_verdict)
        self.assertEqual(self.chain.verify_all_bytes(data), all_verdict)
        # read-only raising entry points
        self.assert_state_error(tenant, expected_seq, line,
                                self.chain.head, tenant)
        self.assert_state_error(tenant, expected_seq, line,
                                self.chain.read_tenant, tenant)
        self.assert_state_error(tenant, expected_seq, line,
                                self.chain.export_tenant, tenant)
        # write entry points: state error, no append, no byte changes
        self.assert_state_error(tenant, expected_seq, line,
                                self.chain.append, tenant, {})
        self.assert_state_error(tenant, expected_seq, line,
                                self.chain.append_batch, tenant, [{}])
        self.assert_state_error(tenant, expected_seq, line,
                                self.chain.append_if_head, tenant, {}, 0, ZERO)
        self.assert_state_error(tenant, expected_seq, line,
                                self.chain.append_batch_if_head,
                                tenant, [{}], 0, ZERO)
        self.assert_state_error(tenant, expected_seq, line,
                                self.chain.append_many,
                                [{"tenant": tenant, "event": {}}])
        self.assert_state_error(tenant, expected_seq, line,
                                self.chain.append_many_if_heads,
                                [{"tenant": tenant, "events": [{}],
                                  "expected_count": 0, "expected_hash": ZERO}])
        self.assertEqual(self.path.read_bytes(), data)

    # --- healthy and empty histories are unaffected by the stricter rule ---

    def test_empty_and_missing_log_unaffected(self):
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": 0})
        self.assertEqual(self.chain.verify_all(), {"ok": True, "tenants": []})
        self.assertEqual(self.chain.heads(), {"ok": True, "tenants": []})
        self.assertEqual(self.chain.head("t"),
                         {"tenant": "t", "count": 0, "hash": ZERO})
        self.assertEqual(self.chain.read_tenant("t"), [])
        self.assertEqual(self.chain.export_tenant("t"), b"")
        self.assertEqual(self.chain.verify_bytes(b"", "t"),
                         {"ok": True, "count": 0})
        self.assertEqual(self.chain.verify_all_bytes(b""),
                         {"ok": True, "tenants": []})
        item = self.chain.append("t", {})
        self.assertEqual((item["seq"], item["prev"]), (1, ZERO))
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": 1})

    def test_healthy_history_all_entry_points(self):
        self.chain.append("t", {"i": 1})
        self.chain.append("u", {})
        self.chain.append_batch("t", [{"i": 2}])
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": 2})
        self.assertEqual(self.chain.head("t")["count"], 2)
        self.assertEqual(len(self.chain.heads()["tenants"]), 2)
        self.assertEqual([r["seq"] for r in self.chain.read_tenant("t")],
                         [1, 2])
        data = self.path.read_bytes()
        self.assertEqual(self.chain.verify_bytes(data, "t"),
                         {"ok": True, "count": 2})
        self.assertTrue(self.chain.verify_all_bytes(data)["ok"])
        exported = self.chain.export_tenant("t")
        other = AuditChain(Path(self.tmp.name) / "other.jsonl")
        records = other.import_tenant("t", exported)
        self.assertEqual([r["seq"] for r in records], [1, 2])
        self.assertEqual(other.verify("t"), {"ok": True, "count": 2})
        item = other.append("t", {"i": 3})
        self.assertEqual(item["seq"], 3)

    # --- non-integer seq spellings on the first record ---

    def test_non_integer_seq_spellings_are_sequence(self):
        # 1.0/1e0/1E0/1.00/10e-1 are the spellings host-language loose
        # equality would wrongly admit; the rest were already rejected and
        # must keep the exact same classification.
        variants = ["1.0", "1e0", "1E0", "1.00", "10e-1", "1.5",
                    "true", "false", '"1"', "null", "-1", "0", "2",
                    "[1]", '{"x": 1}']
        for seq_text in variants:
            with self.subTest(seq_text=seq_text):
                self.write_text(raw_seq_line('"t"', seq_text) + "\n")
                self.assert_sequence_everywhere("t", 1, 1)

    def test_missing_seq_field_stays_missing(self):
        # a record without seq (or without any required field) is missing,
        # not sequence, exactly as before
        row = valid_row("t", 1)
        del row["seq"]
        self.write_text(json.dumps(row, sort_keys=True) + "\n")
        self.assertEqual(self.chain.verify("t"),
                         {"ok": False, "at": 1, "reason": "missing"})
        self.assertEqual(
            self.chain.verify_all(),
            {"ok": False, "at": 1, "tenant": "t", "reason": "missing"})
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.append("t", {})
        self.assertEqual((cm.exception.seq, cm.exception.reason,
                          cm.exception.line), (1, "missing", 1))

    def test_unparseable_line_stays_missing(self):
        self.write_text("{not json\n")
        self.assertEqual(self.chain.verify("t"),
                         {"ok": False, "at": 1, "reason": "missing"})

    # --- non-integer seq in the middle of a chain ---

    def test_non_integer_seq_middle_record(self):
        first = self.chain.append("t", {})
        self.append_text(raw_seq_line('"t"', "2.0", prev=first["hash"]) + "\n")
        self.assert_sequence_everywhere("t", 2, 2)

    def test_non_integer_seq_middle_record_other_tenant_unaffected(self):
        first = self.chain.append("t", {})
        self.append_text(raw_seq_line('"t"', "2e0", prev=first["hash"]) + "\n")
        before = self.path.read_bytes()
        # the corrupt record is never renumbered, skipped, rewritten or
        # treated as another tenant: t is broken, u is a separate chain
        item = self.chain.append("u", {})
        self.assertEqual((item["seq"], item["prev"]), (1, ZERO))
        self.assertEqual(self.chain.verify("u"), {"ok": True, "count": 1})
        self.assertEqual(self.chain.verify("t"),
                         {"ok": False, "at": 2, "reason": "sequence"})
        # t's bytes were left exactly as they were (only u's record appended)
        self.assertTrue(self.path.read_bytes().startswith(before))

    # --- interleaved tenants: physical line vs expected seq ---

    def test_interleaved_tenants_sequence_location(self):
        a1 = self.chain.append("a", {})
        self.chain.append("b", {})
        self.append_text(raw_seq_line('"a"', "2.0", prev=a1["hash"]) + "\n")
        # per-tenant verify locates by expected seq, verify_all by line
        self.assertEqual(self.chain.verify("a"),
                         {"ok": False, "at": 2, "reason": "sequence"})
        self.assertEqual(self.chain.verify("b"), {"ok": True, "count": 1})
        self.assertEqual(
            self.chain.verify_all(),
            {"ok": False, "at": 3, "tenant": "a", "reason": "sequence"})
        self.assertEqual(
            self.chain.heads(),
            {"ok": False, "at": 3, "tenant": "a", "reason": "sequence"})
        self.assert_state_error("a", 2, 3, self.chain.append, "a", {})
        self.assert_state_error("a", 2, 3, self.chain.head, "a")
        # the healthy interleaved tenant still appends on its own chain
        item = self.chain.append("b", {})
        self.assertEqual(item["seq"], 2)

    def test_append_many_reports_corrupt_tenant_and_writes_nothing(self):
        self.write_text(raw_seq_line('"t"', "1.0") + "\n")
        before = self.path.read_bytes()
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.append_many([{"tenant": "u", "event": {}},
                                    {"tenant": "t", "event": {}}])
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("t", 1, "sequence", 1))
        self.assertEqual(self.path.read_bytes(), before)

    # --- digest classification and error priority are unchanged ---

    def test_later_digest_corruption_still_digest(self):
        self.chain.append("t", {})
        self.chain.append("t", {})
        rows = [json.loads(l)
                for l in self.path.read_text(encoding="utf-8").splitlines()]
        rows[1]["event"] = {"x": 1}  # break the second record's digest
        self.write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
        self.assertEqual(self.chain.verify("t"),
                         {"ok": False, "at": 2, "reason": "digest"})
        self.assertEqual(
            self.chain.verify_all(),
            {"ok": False, "at": 2, "tenant": "t", "reason": "digest"})

    def test_sequence_checked_before_digest(self):
        # float seq AND a wrong hash in the same record: sequence wins
        line = ('{"event": {}, "hash": "%s", "prev": "%s", '
                '"seq": 1.0, "tenant": "t"}') % ("f" * 64, ZERO)
        self.write_text(line + "\n")
        self.assertEqual(self.chain.verify("t"),
                         {"ok": False, "at": 1, "reason": "sequence"})
        self.assertEqual(
            self.chain.verify_all(),
            {"ok": False, "at": 1, "tenant": "t", "reason": "sequence"})

    def test_earlier_error_beats_later_non_integer_seq(self):
        row = valid_row("t", 1)
        row["event"] = {"tampered": True}  # digest broken at line 1
        self.write_text(json.dumps(row, sort_keys=True) + "\n"
                        + raw_seq_line('"t"', "2.0", prev=row["hash"]) + "\n")
        self.assertEqual(self.chain.verify("t"),
                         {"ok": False, "at": 1, "reason": "digest"})

    # --- expected_count semantics on legal histories are unchanged ---

    def test_expected_count_short_and_over_unchanged(self):
        self.chain.append("t", {})
        self.chain.append("t", {})
        self.assertEqual(self.chain.verify("t", 2), {"ok": True, "count": 2})
        self.assertEqual(self.chain.verify("t", 3),
                         {"ok": False, "at": 3, "reason": "missing"})
        self.assertEqual(self.chain.verify("t", 1),
                         {"ok": False, "at": 2, "reason": "sequence"})
        data = self.path.read_bytes()
        self.assertEqual(self.chain.verify_bytes(data, "t", 3),
                         {"ok": False, "at": 3, "reason": "missing"})
        self.assertEqual(self.chain.verify_bytes(data, "t", 1),
                         {"ok": False, "at": 2, "reason": "sequence"})

    # --- offline byte verification agrees with the file scan ---

    def test_verify_bytes_and_verify_all_bytes_match_file_verdicts(self):
        first = self.chain.append("t", {})
        self.chain.append("u", {})
        self.append_text(raw_seq_line('"t"', "2.0", prev=first["hash"]) + "\n")
        data = self.path.read_bytes()
        self.assertEqual(self.chain.verify_bytes(data, "t"),
                         self.chain.verify("t"))
        self.assertEqual(self.chain.verify_all_bytes(data),
                         self.chain.verify_all())
        # the healthy tenant's export still verifies offline
        exported = self.chain.export_tenant("u")
        self.assertEqual(self.chain.verify_bytes(exported, "u"),
                         {"ok": True, "count": 1})

    def test_verify_bytes_non_integer_seq_without_trailing_newline(self):
        data = raw_seq_line('"t"', "1.0").encode("utf-8")  # no trailing \n
        self.assertEqual(self.chain.verify_bytes(data, "t"),
                         {"ok": False, "at": 1, "reason": "sequence"})
        self.assertEqual(
            self.chain.verify_all_bytes(data),
            {"ok": False, "at": 1, "tenant": "t", "reason": "sequence"})

    # --- import_tenant: corrupt input data and corrupt target ---

    def test_import_rejects_non_integer_seq_in_data(self):
        data = (raw_seq_line('"t"', "1.0") + "\n").encode("utf-8")
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.import_tenant("t", data)
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("t", 1, "sequence", 1))
        self.assertFalse(self.path.exists())  # no partial import

    def test_import_rejects_non_integer_seq_midstream_no_partial(self):
        good = valid_row("t", 1)
        data = (json.dumps(good, sort_keys=True) + "\n"
                + raw_seq_line('"t"', "2e0", prev=good["hash"]) + "\n"
                ).encode("utf-8")
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.import_tenant("t", data)
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("t", 2, "sequence", 2))
        self.assertFalse(self.path.exists())

    def test_import_onto_corrupt_target_leaves_bytes_unchanged(self):
        self.write_text(raw_seq_line('"t"', "1.0") + "\n")
        before = self.path.read_bytes()
        good = valid_row("t", 1)
        data = (json.dumps(good, sort_keys=True) + "\n").encode("utf-8")
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.import_tenant("t", data)
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("t", 1, "sequence", 1))
        self.assertEqual(self.path.read_bytes(), before)

    # --- illegal call arguments still win over the corrupt history ---

    def test_value_error_still_precedes_corrupt_history(self):
        self.write_text(raw_seq_line('"t"', "1.0") + "\n")
        before = self.path.read_bytes()
        with self.assertRaises(ValueError):
            self.chain.append(float("nan"), {})
        with self.assertRaises(ValueError):
            self.chain.verify("t", -1)
        with self.assertRaises(ValueError):
            self.chain.append_if_head("t", {}, True, ZERO)
        with self.assertRaises(ValueError):
            self.chain.append_batch("t", "not-a-list")
        with self.assertRaises(ValueError):
            self.chain.import_tenant("t", "not-bytes")
        self.assertEqual(self.path.read_bytes(), before)

    # --- concurrency: readers and writers agree on the corrupt state ---

    def test_concurrent_reads_and_writes_on_corrupt_log(self):
        first = self.chain.append("t", {})
        self.append_text(raw_seq_line('"t"', "2.0", prev=first["hash"]) + "\n")
        before = self.path.read_bytes()
        errors = []
        verdicts = []
        lock = threading.Lock()

        def writer():
            try:
                self.chain.append("t", {})
            except AuditChainStateError as e:
                with lock:
                    errors.append((e.tenant, e.seq, e.reason, e.line))

        def reader():
            with lock:
                verdicts.append(self.chain.verify("t"))

        threads = []
        for _ in range(8):
            threads.append(threading.Thread(target=writer))
            threads.append(threading.Thread(target=reader))
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        # every writer refused with the identical location, no byte changed
        self.assertEqual(len(errors), 8)
        self.assertEqual(set(errors), {("t", 2, "sequence", 2)})
        # every reader saw the same consistent sequence verdict
        self.assertEqual(len(verdicts), 8)
        for v in verdicts:
            self.assertEqual(v, {"ok": False, "at": 2, "reason": "sequence"})
        self.assertEqual(self.path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
