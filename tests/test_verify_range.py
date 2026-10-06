import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app import AuditChain, ZERO


def record_bytes(item):
    return (json.dumps(item, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def rows_of(data):
    return [json.loads(line) for line in data.decode("utf-8").splitlines()]


class VerifyTenantRangeBytesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.path = Path(self.tmp.name) / "audit.jsonl"
        self.chain = AuditChain(self.path)
        # every offline entry below points at a path that must never be read,
        # created or modified
        self.offline = AuditChain(self.path.with_name("never-touched.jsonl"))
        for i in range(8):
            self.chain.append("t", {"i": i, "s": "审计"})
        self.chain.append("other", {"x": 1})

    def tearDown(self):
        self.tmp.cleanup()

    def segment(self, lo, hi=None):
        return self.chain.export_tenant_range("t", lo, hi)

    def prev_of(self, data):
        return rows_of(data)[0]["prev"]

    # --- boundary: shape, types, and priority before any parsing ---

    def test_data_must_be_bytes(self):
        seg, prev = self.segment(3, 5), self.prev_of(self.segment(3, 5))
        for bad in ("jsonl", bytearray(seg), 1, None, [], object()):
            with self.assertRaises(ValueError):
                self.offline.verify_tenant_range_bytes(
                    bad, "t", 3, 5, prev)
        self.assertFalse(self.offline.path.exists())

    def test_illegal_tenant_is_value_error(self):
        bad_tenants = [
            float("nan"), float("inf"), float("-inf"),
            {"k": float("nan")}, [1, [float("-inf")]],
            {1: "x"}, {"a": {2: 3}}, object(), b"bytes", {1, 2},
        ]
        for bad in bad_tenants:
            with self.assertRaises(ValueError):
                self.offline.verify_tenant_range_bytes(b"x", bad, 3, 5, "f" * 64)
        self.assertFalse(self.offline.path.exists())

    def test_start_seq_must_be_positive_plain_int(self):
        for bad in (True, False, 0, -1, -10, 1.0, 2.5, "3", None, [], {}):
            with self.assertRaises(ValueError):
                self.offline.verify_tenant_range_bytes(
                    b"x", "t", bad, 5, "f" * 64)
        self.assertFalse(self.offline.path.exists())

    def test_end_seq_must_be_none_or_plain_int_not_smaller(self):
        for bad in (True, False, 2.0, 3.5, "5", [], {}, 2):
            with self.assertRaises(ValueError):
                self.offline.verify_tenant_range_bytes(
                    b"x", "t", 3, bad, "f" * 64)
        self.assertFalse(self.offline.path.exists())

    def test_expected_prev_must_be_64_lowercase_hex(self):
        for bad in (None, "", "0" * 63, "0" * 65, "A" * 64, "g" * 64,
                    0, [], {}):
            with self.assertRaises(ValueError):
                self.offline.verify_tenant_range_bytes(b"x", "t", 3, 5, bad)
        self.assertFalse(self.offline.path.exists())

    def test_start_seq_1_accepts_only_zero(self):
        with self.assertRaises(ValueError):
            self.offline.verify_tenant_range_bytes(
                b"x", "t", 1, 2, "f" * 64)
        with self.assertRaises(ValueError):
            self.offline.verify_tenant_range_bytes(
                b"x", "t", 1, None, "0" * 63 + "1")
        self.assertFalse(self.offline.path.exists())
        # the well-formed ZERO case must cross the boundary even on garbage
        # content (it surfaces as a content verdict, not ValueError)
        r = self.offline.verify_tenant_range_bytes(b"x", "t", 1, None, ZERO)
        self.assertEqual(r["reason"], "missing")

    def test_boundary_value_error_beats_corrupt_content_and_path(self):
        bad_prev = "nope"
        with self.assertRaises(ValueError):
            self.offline.verify_tenant_range_bytes(
                b"{not json\n", "t", 3, 5, bad_prev)
        # an existing corrupt file at the configured path is never read
        self.offline.path.write_bytes(b"\xff\xff")
        with self.assertRaises(ValueError):
            self.offline.verify_tenant_range_bytes(
                bytearray(b""), "t", 0, 5, ZERO)
        self.assertEqual(self.offline.path.read_bytes(), b"\xff\xff")

    def test_never_reads_creates_or_modifies_constructor_path(self):
        phantom = AuditChain(self.path.with_name("phantom.jsonl"))
        seg, prev = self.segment(4, 6), self.prev_of(self.segment(4, 6))
        self.assertTrue(
            phantom.verify_tenant_range_bytes(seg, "t", 4, 6, prev)["ok"])
        self.assertFalse(phantom.path.exists())
        before = self.chain.export_tenant("t")
        phantom.verify_tenant_range_bytes(before, "t", 1, 8, ZERO)
        self.assertEqual(self.chain.export_tenant("t"), before)
        self.assertFalse(phantom.path.exists())

    # --- success ---

    def test_closed_interval_success_shape(self):
        seg = self.segment(3, 5)
        rows = rows_of(seg)
        r = self.offline.verify_tenant_range_bytes(
            seg, "t", 3, 5, rows[0]["prev"])
        self.assertEqual(r, {
            "ok": True,
            "tenant": "t",
            "start_seq": 3,
            "end_seq": 5,
            "count": 3,
            "hash": rows[-1]["hash"],
        })

    def test_open_tail_closes_end_seq_at_actual_last(self):
        seg = self.segment(6)  # 6..8
        rows = rows_of(seg)
        r = self.offline.verify_tenant_range_bytes(
            seg, "t", 6, None, rows[0]["prev"])
        self.assertEqual(
            (r["ok"], r["start_seq"], r["end_seq"], r["count"]),
            (True, 6, 8, 3))
        self.assertEqual(r["hash"], rows[-1]["hash"])

    def test_chain_beginning_with_zero_anchor(self):
        seg = self.segment(1, 4)
        r = self.offline.verify_tenant_range_bytes(seg, "t", 1, 4, ZERO)
        self.assertTrue(r["ok"])
        self.assertEqual((r["start_seq"], r["end_seq"], r["count"]), (1, 4, 4))

    def test_trailing_lf_optional_and_crlf_accepted(self):
        seg = self.segment(2, 3)
        rows = rows_of(seg)
        no_lf = seg.rstrip(b"\n")
        r = self.offline.verify_tenant_range_bytes(
            no_lf, "t", 2, 3, rows[0]["prev"])
        self.assertTrue(r["ok"])
        crlf = seg.replace(b"\n", b"\r\n")
        r = self.offline.verify_tenant_range_bytes(
            crlf, "t", 2, 3, rows[0]["prev"])
        self.assertTrue(r["ok"])

    def test_object_tenant_identity_key_order_normalized(self):
        src = AuditChain(self.path.with_name("obj.jsonl"))
        for i in range(3):
            src.append({"a": 1, "b": 2}, {"i": i})
        seg = src.export_tenant_range({"a": 1, "b": 2}, 2, 3)
        r = self.offline.verify_tenant_range_bytes(
            seg, {"b": 2, "a": 1}, 2, 3, rows_of(seg)[0]["prev"])
        self.assertTrue(r["ok"])

    def test_success_feds_import_tenant_range_head_assertion(self):
        head = self.segment(1, 3)
        tail = self.segment(4, 8)
        verdict = self.offline.verify_tenant_range_bytes(
            tail, "t", 4, 8, rows_of(tail)[0]["prev"])
        self.assertTrue(verdict["ok"])
        target = AuditChain(self.path.with_name("target.jsonl"))
        target.import_tenant_range("t", head, 0, ZERO)
        records = target.import_tenant_range(
            "t", tail, verdict["start_seq"] - 1, rows_of(tail)[0]["prev"])
        self.assertEqual([x["seq"] for x in records], [4, 5, 6, 7, 8])
        self.assertEqual(target.verify("t"), {"ok": True, "count": 8})
        self.assertEqual(records[-1]["hash"], verdict["hash"])

    # --- missing: empty fragment, bad lines, fields, declared tail ---

    def fail(self, data, start, end, prev):
        return self.offline.verify_tenant_range_bytes(
            data, "t", start, end, prev)

    def test_empty_fragment_is_missing_at_start_seq(self):
        for end in (5, None):
            r = self.fail(b"", 3, end, "f" * 64)
            self.assertEqual(
                (r["ok"], r["reason"], r["at"], r["tenant"],
                 r["start_seq"], r["end_seq"]),
                (False, "missing", 3, "t", 3, end))
        r = self.fail(b"", 1, None, ZERO)
        self.assertEqual((r["reason"], r["at"]), ("missing", 1))

    def test_unparseable_and_bad_utf8_lines_are_missing_at_line(self):
        seg = self.segment(3, 5)
        prev = rows_of(seg)[0]["prev"]
        cases = [
            (b"\n", 1),
            (b"   \n", 1),
            (b"[1,2]\n", 1),
            (b"{not json\n", 1),
            (seg + b"{bad\n", 4),
            (seg + b"\xff", 4),
            (json.dumps(None).encode() + b"\n", 1),
        ]
        for raw, at in cases:
            r = self.fail(raw, 3, 5, prev)
            self.assertEqual(
                (r["reason"], r["at"]), ("missing", at), raw)
            self.assertEqual((r["start_seq"], r["end_seq"]), (3, 5))

    def test_duplicate_keys_and_nonstandard_numbers_are_missing(self):
        prev = "c" * 64
        row = {"tenant": "t", "seq": 3, "event": {}, "prev": prev}
        raw = json.dumps(row, sort_keys=True, separators=(",", ":"))
        # duplicate the seq key textually
        dup = raw.replace('"seq":3', '"seq":3,"seq":3', 1).encode() + b"\n"
        r = self.fail(dup, 3, 3, prev)
        self.assertEqual((r["reason"], r["at"]), ("missing", 1))
        nan_line = raw.replace("{}", "NaN", 1).encode() + b"\n"
        r = self.fail(nan_line, 3, 3, prev)
        self.assertEqual((r["reason"], r["at"]), ("missing", 1))

    def test_missing_field_fills_at_own_seq_or_expected(self):
        seg = self.segment(3, 5)
        rows = rows_of(seg)
        # row 4 missing its hash: at is the record's own seq 4
        row = {k: rows[1][k] for k in ("tenant", "seq", "event", "prev")}
        data = record_bytes(rows[0]) + record_bytes(row)
        r = self.fail(data, 3, 5, rows[0]["prev"])
        self.assertEqual((r["reason"], r["at"]), ("missing", 4))
        # row 3 missing seq (and hash): at falls back to the expected seq
        row = {k: rows[0][k] for k in ("tenant", "event", "prev")}
        r = self.fail(record_bytes(row), 3, 5, rows[0]["prev"])
        self.assertEqual((r["reason"], r["at"]), ("missing", 3))

    def test_extra_field_is_missing(self):
        row = rows_of(self.segment(3, 3))[0]
        row["extra"] = True
        r = self.fail(record_bytes(row), 3, 3, row["prev"])
        self.assertEqual((r["reason"], r["at"]), ("missing", 3))

    def test_declared_interval_missing_tail_is_missing_at_first_gap(self):
        seg = self.segment(3, 4)  # declares up to 5, ends early
        prev = rows_of(seg)[0]["prev"]
        r = self.fail(seg, 3, 5, prev)
        self.assertEqual((r["reason"], r["at"]), ("missing", 5))
        # one-record gap: only seq 3 present, declared [3,4]
        one = self.segment(3, 3)
        r = self.fail(one, 3, 4, rows_of(one)[0]["prev"])
        self.assertEqual((r["reason"], r["at"]), ("missing", 4))

    # --- sequence ---

    def test_wrong_seq_is_sequence_at_expected(self):
        rows = rows_of(self.segment(3, 5))
        # first record carries seq 9 while seq 3 is expected
        bad = dict(rows[0])
        bad["seq"] = 9
        bad["hash"] = AuditChain._hash(bad)
        r = self.fail(record_bytes(bad), 3, 5, rows[0]["prev"])
        self.assertEqual((r["reason"], r["at"]), ("sequence", 3))
        # internal gap: seq 3 then seq 5
        good0 = rows[0]
        bad = dict(rows[2])
        data = record_bytes(good0) + record_bytes(bad)
        r = self.fail(data, 3, 5, rows[0]["prev"])
        self.assertEqual((r["reason"], r["at"]), ("sequence", 4))

    def test_float_and_bool_seq_spellings_are_sequence(self):
        prev = "c" * 64
        row = {"tenant": "t", "seq": 3, "event": {}, "prev": prev}
        row["hash"] = AuditChain._hash(row)
        line = json.dumps(row, sort_keys=True)
        self.assertIn('"seq": 3', line)
        for spelling in ("3.0", "3e0", "true"):
            raw = line.replace(
                '"seq": 3', '"seq": ' + spelling, 1).encode() + b"\n"
            r = self.fail(raw, 3, 3, prev)
            self.assertEqual((r["reason"], r["at"]), ("sequence", 3), spelling)

    def test_record_past_declared_end_is_sequence(self):
        seg = self.segment(3, 5)  # declares the closed interval [3,4]
        r = self.fail(seg, 3, 4, rows_of(seg)[0]["prev"])
        self.assertEqual((r["reason"], r["at"]), ("sequence", 5))

    # --- digest ---

    def test_wrong_first_prev_is_digest(self):
        seg = self.segment(3, 5)
        r = self.fail(seg, 3, 5, "f" * 64)
        self.assertEqual((r["reason"], r["at"]), ("digest", 3))

    def test_wrong_interior_prev_recomputed_hash_is_digest(self):
        rows = rows_of(self.segment(3, 5))
        bad = dict(rows[1])
        bad["prev"] = "a" * 64
        bad["hash"] = AuditChain._hash(bad)  # hash consistent, prev wrong
        data = record_bytes(rows[0]) + record_bytes(bad)
        r = self.fail(data, 3, 5, rows[0]["prev"])
        self.assertEqual((r["reason"], r["at"]), ("digest", 4))

    def test_tampered_event_hash_mismatch_is_digest(self):
        row = rows_of(self.segment(3, 3))[0]
        row["event"] = {"tampered": True}
        r = self.fail(record_bytes(row), 3, 3, row["prev"])
        self.assertEqual((r["reason"], r["at"]), ("digest", 3))

    def test_sequence_checked_before_digest(self):
        row = rows_of(self.segment(3, 3))[0]
        row = dict(row)
        row["seq"] = 9
        row["prev"] = "f" * 64
        row["hash"] = AuditChain._hash(row)
        r = self.fail(record_bytes(row), 3, 3, "0" * 64)
        self.assertEqual(r["reason"], "sequence")

    # --- foreign tenant: ValueError, first physical line wins ---

    def test_foreign_tenant_record_is_value_error(self):
        rows = rows_of(self.segment(3, 4))
        foreign = dict(rows[1])
        foreign["tenant"] = "other"
        foreign["hash"] = AuditChain._hash(foreign)
        data = record_bytes(rows[0]) + record_bytes(foreign)
        with self.assertRaises(ValueError):
            self.offline.verify_tenant_range_bytes(
                data, "t", 3, 4, rows[0]["prev"])

    def test_distinct_json_identities_are_foreign(self):
        src = AuditChain(self.path.with_name("ids.jsonl"))
        src.append(1, {})
        src.append(1.0, {})
        src.append(True, {})
        src.append("1", {})
        for declared, other in ((1, 1.0), (1, True), (1, "1"),
                                ("1", 1), (1.0, True)):
            seg = src.export_tenant_range(other, 1, 1)
            with self.assertRaises(ValueError):
                self.offline.verify_tenant_range_bytes(
                    seg, declared, 1, 1, ZERO)
        # a segment of the exact identity verifies
        seg = src.export_tenant_range(1, 1, 1)
        self.assertTrue(
            self.offline.verify_tenant_range_bytes(seg, 1, 1, 1, ZERO)["ok"])

    def test_physical_line_order_decides_foreign_against_defect(self):
        rows = rows_of(self.segment(3, 5))
        # foreign record first: ValueError even though a later line is broken
        foreign = dict(rows[0])
        foreign["tenant"] = "other"
        foreign["hash"] = AuditChain._hash(foreign)
        data = record_bytes(foreign) + b"{bad\n"
        with self.assertRaises(ValueError):
            self.offline.verify_tenant_range_bytes(
                data, "t", 3, 4, rows[0]["prev"])
        # defect first: the scan stops there, a later foreign line is never
        # reached and the content verdict wins
        tampered = dict(rows[0])
        tampered["event"] = {"tampered": True}  # hash now wrong
        foreign = dict(rows[1])
        foreign["tenant"] = "other"
        foreign["hash"] = AuditChain._hash(foreign)
        data = record_bytes(tampered) + record_bytes(foreign)
        r = self.offline.verify_tenant_range_bytes(
            data, "t", 3, 4, rows[0]["prev"])
        self.assertEqual((r["reason"], r["at"]), ("digest", 3))


class VerifyTenantRangeChunksTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.path = Path(self.tmp.name) / "audit.jsonl"
        self.chain = AuditChain(self.path)
        for i in range(8):
            self.chain.append("t", {"i": i, "s": "审计"})
        self.offline = AuditChain(self.path.with_name("never-touched.jsonl"))
        self.seg = self.chain.export_tenant_range("t", 3, 6)
        self.rows = rows_of(self.seg)
        self.prev = self.rows[0]["prev"]

    def tearDown(self):
        self.tmp.cleanup()

    def verify(self, chunks, start=3, end=6, prev=None, tenant="t"):
        return self.offline.verify_tenant_range_chunks(
            chunks, tenant, start, end, self.prev if prev is None else prev)

    # --- container/member boundary ---

    def test_bare_bytes_and_bytearray_are_not_containers(self):
        for bad in (self.seg, bytearray(self.seg)):
            with self.assertRaises(ValueError):
                self.verify(bad)
        self.assertFalse(self.offline.path.exists())

    def test_non_iterable_container_is_value_error(self):
        for bad in (None, 1, object()):
            with self.assertRaises(ValueError):
                self.verify(bad)

    def test_non_bytes_members_value_error_and_stop_consumption(self):
        for bad_member in ("nope", 3, bytearray(self.seg), object()):
            pulled = []

            def gen():
                yield self.seg[:5]
                pulled.append(1)
                yield bad_member
                pulled.append(2)
                yield self.seg[5:]

            with self.assertRaises(ValueError):
                self.verify(gen())
            self.assertEqual(pulled, [1])

    def test_empty_iteration_is_missing_never_success(self):
        r = self.verify([])
        self.assertEqual((r["reason"], r["at"]), ("missing", 3))
        r = self.verify([b"", b""])
        self.assertEqual((r["reason"], r["at"]), ("missing", 3))
        r = self.verify(iter([]), end=None)
        self.assertEqual((r["reason"], r["at"]), ("missing", 3))

    def test_parameters_validated_before_any_chunk_pulled(self):
        pulled = []

        def gen():
            pulled.append(1)
            yield self.seg

        with self.assertRaises(ValueError):
            self.offline.verify_tenant_range_chunks(
                gen(), "t", 0, 6, ZERO)
        with self.assertRaises(ValueError):
            self.offline.verify_tenant_range_chunks(
                gen(), "t", 3, 6, "nope")
        with self.assertRaises(ValueError):
            self.offline.verify_tenant_range_chunks(
                gen(), "t", 1, 2, "f" * 64)
        self.assertEqual(pulled, [])

    # --- bytes/chunks field-for-field equality on every cut ---

    def all_cuts(self, data):
        yield [data]
        yield [data, b""]
        yield [b"", data]
        for size in (1, 2, 3, 5, 7, 13, 64, 1000):
            yield [data[i:i + size]
                   for i in range(0, len(data), size)]
        for cut in range(0, len(data) + 1):
            yield [data[:cut], data[cut:]]

    def assert_chunks_match_bytes(self, data, start, end, prev):
        expected = self.offline.verify_tenant_range_bytes(
            data, "t", start, end, prev)
        for pieces in self.all_cuts(data):
            got = self.offline.verify_tenant_range_chunks(
                iter(pieces), "t", start, end, prev)
            self.assertEqual(got, expected, (pieces, got, expected))

    def test_chunks_match_bytes_on_valid_segments(self):
        self.assert_chunks_match_bytes(self.seg, 3, 6, self.prev)
        open_seg = self.chain.export_tenant_range("t", 5)
        self.assert_chunks_match_bytes(
            open_seg, 5, None, rows_of(open_seg)[0]["prev"])
        whole = self.chain.export_tenant_range("t", 1, 8)
        self.assert_chunks_match_bytes(whole, 1, 8, ZERO)

    def test_chunks_match_bytes_on_every_failure_shape(self):
        rows = self.rows
        bad_seq = dict(rows[1]); bad_seq["seq"] = 9
        bad_seq["hash"] = AuditChain._hash(bad_seq)
        bad_prev = dict(rows[1]); bad_prev["prev"] = "a" * 64
        bad_prev["hash"] = AuditChain._hash(bad_prev)
        tampered = dict(rows[0]); tampered["event"] = {"x": True}
        extra = dict(rows[0]); extra["z"] = 1
        payloads = [
            (b"", 3, 6),
            (self.seg, 3, 7),       # declared tail missing
            (self.seg, 3, 5),       # record past declared end
            (record_bytes(rows[0]) + record_bytes(bad_seq), 3, 6),
            (record_bytes(rows[0]) + record_bytes(bad_prev), 3, 6),
            (record_bytes(tampered), 3, 6),
            (record_bytes(extra), 3, 6),
            (b"{bad\n", 3, 6),
            (record_bytes(rows[0]) + b"\xff", 3, 6),
            (b"\n", 3, 6),
        ]
        for data, start, end in payloads:
            self.assert_chunks_match_bytes(data, start, end, self.prev)

    def test_foreign_tenant_matches_value_error_on_every_cut(self):
        foreign = dict(self.rows[0])
        foreign["tenant"] = "other"
        foreign["hash"] = AuditChain._hash(foreign)
        data = record_bytes(foreign)
        for pieces in self.all_cuts(data):
            with self.assertRaises(ValueError):
                self.verify(iter(pieces), start=3, end=3)

    def test_chunk_cuts_split_utf8_json_and_lf(self):
        # a fragment serialized with raw multibyte UTF-8 (a backup need not
        # be ASCII-escaped): the parsed record carries the same value, so its
        # hash recomputes and cuts may land inside the multibyte sequence
        row = {"tenant": "t", "seq": 3, "event": {"s": "审计"},
               "prev": self.prev}
        row["hash"] = AuditChain._hash(row)
        raw = (json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
               ).encode("utf-8")
        self.assertIn("审计".encode("utf-8"), raw)
        for cut in range(len(raw) + 1):
            r = self.offline.verify_tenant_range_chunks(
                [raw[:cut], raw[cut:]], "t", 3, 3, self.prev)
            self.assertTrue(r["ok"], cut)
        self.assertEqual(
            self.offline.verify_tenant_range_bytes(
                raw, "t", 3, 3, self.prev)["ok"], True)

    # --- forward-only single consumption: pull stops at first defect ---

    def test_pull_stops_after_first_defective_completed_line(self):
        pulled = []

        def gen():
            good = record_bytes(self.rows[0])
            bad = b"{broken\n"
            later = record_bytes(self.rows[1])
            for name, chunk in (("g1", good[:4]), ("g2", good[4:]),
                                ("bad", bad), ("later", later)):
                pulled.append(name)
                yield chunk

        r = self.verify(gen())
        self.assertEqual((r["reason"], r["at"]), ("missing", 2))
        # the bad line completed with the "bad" chunk; "later" never pulled
        self.assertEqual(pulled, ["g1", "g2", "bad"])

    def test_pull_stops_after_sequence_verdict_mid_chunk(self):
        pulled = []

        def gen():
            good = record_bytes(self.rows[0])
            bad = record_bytes(self.rows[2])  # seq 5 where 4 expected
            for name, chunk in (("a", good), ("b", bad),
                                ("c", record_bytes(self.rows[1]))):
                pulled.append(name)
                yield chunk

        r = self.verify(gen())
        self.assertEqual((r["reason"], r["at"]), ("sequence", 4))
        self.assertEqual(pulled, ["a", "b"])

    def test_pull_stops_after_foreign_record(self):
        pulled = []
        foreign = dict(self.rows[0])
        foreign["tenant"] = "other"
        foreign["hash"] = AuditChain._hash(foreign)

        def gen():
            for name, chunk in (("foreign", record_bytes(foreign)),
                                ("later", record_bytes(self.rows[1]))):
                pulled.append(name)
                yield chunk

        with self.assertRaises(ValueError):
            self.verify(gen())
        self.assertEqual(pulled, ["foreign"])

    def test_iterator_exception_propagates_verbatim(self):
        class Boom(Exception):
            pass

        def gen():
            yield self.seg[:3]
            raise Boom()

        with self.assertRaises(Boom):
            self.verify(gen())


if __name__ == "__main__":
    unittest.main()
