import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app import AuditChain, ZERO


def record_bytes(item):
    return (json.dumps(item, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def rows_of(data):
    return [json.loads(line) for line in data.decode("utf-8").splitlines()]


class VerifyAllRangeBytesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.path = Path(self.tmp.name) / "audit.jsonl"
        self.chain = AuditChain(self.path)
        # every offline entry below points at a path that must never be read,
        # created or modified
        self.offline = AuditChain(self.path.with_name("never-touched.jsonl"))
        for i in range(8):
            self.chain.append("a", {"i": i, "s": "审计"})
            if i % 2 == 0:
                self.chain.append("b", {"i": i})
        self.chain.append("other", {"x": 1})

    def tearDown(self):
        self.tmp.cleanup()

    def segment(self, tenant, lo, hi=None):
        return self.chain.export_tenant_range(tenant, lo, hi)

    def head_for(self, seg):
        first = rows_of(seg)[0]
        return first["seq"] - 1, first["prev"]

    def heads_for(self, *segs):
        heads = []
        for tenant, seg in segs:
            count, h = self.head_for(seg)
            heads.append({"tenant": tenant, "expected_count": count,
                          "expected_hash": h})
        return heads

    # --- boundary: shape, types, and priority before any parsing ---

    def test_data_must_be_bytes(self):
        seg_a, seg_b = self.segment("a", 4), self.segment("b", 2)
        heads = self.heads_for(("a", seg_a), ("b", seg_b))
        inter = record_bytes(rows_of(seg_a)[0]) + record_bytes(rows_of(seg_b)[0])
        for bad in ("jsonl", bytearray(inter), 1, None, [], object()):
            with self.assertRaises(ValueError):
                self.offline.verify_all_range_bytes(bad, heads)
        self.assertFalse(self.offline.path.exists())

    def test_expected_heads_must_be_list_of_exact_key_objects(self):
        seg = self.segment("a", 2)
        count, h = self.head_for(seg)
        for bad in (None, "x", 1, {}, (), object()):
            with self.assertRaises(ValueError):
                self.offline.verify_all_range_bytes(seg, bad)
        good = {"tenant": "a", "expected_count": count, "expected_hash": h}
        for member in (None, 1, "x", [], (),
                       {"tenant": "a", "expected_count": count},
                       {"tenant": "a", "expected_hash": h},
                       {"expected_count": count, "expected_hash": h},
                       dict(good, extra=1),
                       {"tenant": "a", "expected_count": count,
                        "expected_hash": h, "z": 2}):
            with self.assertRaises(ValueError):
                self.offline.verify_all_range_bytes(seg, [member])

    def test_non_string_member_key_rejected(self):
        seg = self.segment("a", 1)
        with self.assertRaises(ValueError):
            self.offline.verify_all_range_bytes(
                seg, [{"tenant": "a", "expected_count": 0,
                        "expected_hash": ZERO, 1: "x"}])

    def test_count_must_be_non_negative_plain_int(self):
        for bad in (True, False, -1, -10, 1.0, 2.5, "3", None, [], {}):
            with self.assertRaises(ValueError):
                self.offline.verify_all_range_bytes(
                    b"x", [{"tenant": "a", "expected_count": bad,
                            "expected_hash": ZERO}])

    def test_hash_must_be_64_lowercase_hex(self):
        for bad in (None, "", "0" * 63, "0" * 65, "A" * 64, "g" * 64,
                    0, [], {}):
            with self.assertRaises(ValueError):
                self.offline.verify_all_range_bytes(
                    b"x", [{"tenant": "a", "expected_count": 0,
                            "expected_hash": bad}])

    def test_illegal_tenant_is_value_error(self):
        bad_tenants = [
            float("nan"), float("inf"), float("-inf"),
            {"k": float("nan")}, [1, [float("-inf")]],
            b"bytes", object(),
        ]
        for bad in bad_tenants:
            with self.assertRaises(ValueError):
                self.offline.verify_all_range_bytes(
                    b"x", [{"tenant": bad, "expected_count": 0,
                            "expected_hash": ZERO}])

    def test_illegal_tenant_non_string_key_and_cycles(self):
        with self.assertRaises(ValueError):
            self.offline.verify_all_range_bytes(
                b"x", [{"tenant": {1: "x"}, "expected_count": 0,
                        "expected_hash": ZERO}])
        cyc = []
        cyc.append(cyc)
        with self.assertRaises(ValueError):
            self.offline.verify_all_range_bytes(
                b"x", [{"tenant": cyc, "expected_count": 0,
                        "expected_hash": ZERO}])
        cyc_obj = {}
        cyc_obj["self"] = cyc_obj
        with self.assertRaises(ValueError):
            self.offline.verify_all_range_bytes(
                b"x", [{"tenant": cyc_obj, "expected_count": 0,
                        "expected_hash": ZERO}])

    def test_duplicate_canonical_tenant_rejected(self):
        h = {"tenant": {"a": 1, "b": 2}, "expected_count": 0,
             "expected_hash": ZERO}
        h2 = {"tenant": {"b": 2, "a": 1}, "expected_count": 3,
              "expected_hash": "f" * 64}
        with self.assertRaises(ValueError):
            self.offline.verify_all_range_bytes(b"", [h, h2])
        # distinct JSON identities are all allowed together
        heads = [
            {"tenant": 1, "expected_count": 0, "expected_hash": ZERO},
            {"tenant": 1.0, "expected_count": 0, "expected_hash": ZERO},
            {"tenant": True, "expected_count": 0, "expected_hash": ZERO},
            {"tenant": "1", "expected_count": 0, "expected_hash": ZERO},
        ]
        r = self.offline.verify_all_range_bytes(b"", heads)
        self.assertTrue(r["ok"])
        self.assertEqual(
            [t["tenant"] for t in r["tenants"]], [1, 1.0, True, "1"])

    def test_boundary_value_error_beats_corrupt_content_and_path(self):
        # malformed expectation beats corrupt content and is raised
        with self.assertRaises(ValueError):
            self.offline.verify_all_range_bytes(
                b"{not json\n", "not-a-list")
        # an existing corrupt file at the configured path is never read
        self.offline.path.write_bytes(b"\xff\xff")
        with self.assertRaises(ValueError):
            self.offline.verify_all_range_bytes(
                bytearray(b""), [{"tenant": "a", "expected_count": -1,
                                 "expected_hash": ZERO}])
        self.assertEqual(self.offline.path.read_bytes(), b"\xff\xff")

    def test_never_reads_creates_or_modifies_constructor_path(self):
        seg_a, seg_b = self.segment("a", 4, 6), self.segment("b", 2, 4)
        heads = self.heads_for(("a", seg_a), ("b", seg_b))
        rows_a, rows_b = rows_of(seg_a), rows_of(seg_b)
        inter = (record_bytes(rows_a[0]) + record_bytes(rows_b[0])
                 + record_bytes(rows_b[1]) + record_bytes(rows_b[2])
                 + record_bytes(rows_a[1]) + record_bytes(rows_a[2]))
        phantom = AuditChain(self.path.with_name("phantom.jsonl"))
        self.assertTrue(
            phantom.verify_all_range_bytes(inter, heads)["ok"])
        self.assertFalse(phantom.path.exists())

    # --- success ---

    def test_interleaved_success_in_expected_heads_order(self):
        seg_a = self.segment("a", 4, 6)
        seg_b = self.segment("b", 2, 4)
        heads = self.heads_for(("a", seg_a), ("b", seg_b))
        rows_a, rows_b = rows_of(seg_a), rows_of(seg_b)
        inter = (record_bytes(rows_a[0]) + record_bytes(rows_b[0])
                 + record_bytes(rows_b[1]) + record_bytes(rows_a[1])
                 + record_bytes(rows_b[2]) + record_bytes(rows_a[2]))
        r = self.offline.verify_all_range_bytes(inter, heads)
        self.assertEqual(r, {
            "ok": True,
            "tenants": [
                {"tenant": "a", "count": 6, "hash": rows_a[-1]["hash"]},
                {"tenant": "b", "count": 4, "hash": rows_b[-1]["hash"]},
            ],
        })
        # expected_heads order, not physical first-appearance order
        heads_rev = list(reversed(heads))
        r = self.offline.verify_all_range_bytes(inter, heads_rev)
        self.assertEqual(
            [t["tenant"] for t in r["tenants"]], ["b", "a"])

    def test_success_continues_from_declared_mid_chain_tails(self):
        seg_a = self.segment("a", 2, 8)
        heads = self.heads_for(("a", seg_a))
        r = self.offline.verify_all_range_bytes(seg_a, heads)
        self.assertTrue(r["ok"])
        self.assertEqual(r["tenants"][0]["count"], 8)

    def test_trailing_lf_optional_and_crlf_accepted(self):
        seg_a = self.segment("a", 2, 3)
        heads = self.heads_for(("a", seg_a))
        r = self.offline.verify_all_range_bytes(seg_a.rstrip(b"\n"), heads)
        self.assertTrue(r["ok"])
        r = self.offline.verify_all_range_bytes(
            seg_a.replace(b"\n", b"\r\n"), heads)
        self.assertTrue(r["ok"])

    def test_object_tenant_identity_key_order_normalized(self):
        src = AuditChain(self.path.with_name("obj.jsonl"))
        for i in range(3):
            src.append({"a": 1, "b": 2}, {"i": i})
        seg = src.export_tenant_range({"a": 1, "b": 2}, 2, 3)
        heads = [{
            "tenant": {"b": 2, "a": 1},
            "expected_count": self.head_for(seg)[0],
            "expected_hash": self.head_for(seg)[1],
        }]
        self.assertTrue(
            self.offline.verify_all_range_bytes(seg, heads)["ok"])

    def test_empty_fragment_returns_unchanged_heads(self):
        heads = [
            {"tenant": "a", "expected_count": 3, "expected_hash": "f" * 64},
            {"tenant": "b", "expected_count": 0, "expected_hash": ZERO},
        ]
        r = self.offline.verify_all_range_bytes(b"", heads)
        self.assertEqual(r, {"ok": True, "tenants": [
            {"tenant": "a", "count": 3, "hash": "f" * 64},
            {"tenant": "b", "count": 0, "hash": ZERO},
        ]})

    def test_success_result_feeds_next_segment_and_import(self):
        # head prefixes land in a target via import, the tail fragments are
        # first verified offline and then imported as one interleaved block
        head_a = self.segment("a", 1, 3)
        head_b = self.segment("b", 1, 2)
        tail_a = self.segment("a", 4, 6)
        tail_b = self.segment("b", 3, 4)
        ra, rb = rows_of(tail_a), rows_of(tail_b)
        inter = (record_bytes(ra[0]) + record_bytes(rb[0])
                 + record_bytes(rb[1]) + record_bytes(ra[1])
                 + record_bytes(ra[2]))
        mid_heads = [
            {"tenant": "a", "expected_count": 3,
             "expected_hash": rows_of(head_a)[-1]["hash"]},
            {"tenant": "b", "expected_count": 2,
             "expected_hash": rows_of(head_b)[-1]["hash"]},
        ]
        verdict = self.offline.verify_all_range_bytes(inter, mid_heads)
        self.assertTrue(verdict["ok"])
        # the verdict is directly usable as the next segment's predecessor
        next_heads = [
            {"tenant": t["tenant"], "expected_count": t["count"],
             "expected_hash": t["hash"]}
            for t in verdict["tenants"]
        ]
        r2 = self.offline.verify_all_range_bytes(b"", next_heads)
        self.assertEqual(r2["tenants"], verdict["tenants"])
        # and import_all_range accepts the same block against a target
        target = AuditChain(self.path.with_name("target.jsonl"))
        target.import_all(head_a + head_b)
        target.import_all_range(inter, mid_heads)
        self.assertEqual(target.verify_all(), {"ok": True, "tenants": [
            {"tenant": "a", "count": 6},
            {"tenant": "b", "count": 4},
        ]})

    # --- missing ---

    def check_fail(self, data, heads):
        return self.offline.verify_all_range_bytes(data, heads)

    def test_unparseable_and_bad_utf8_lines_are_missing_at_line(self):
        seg_a, seg_b = self.segment("a", 4, 5), self.segment("b", 2, 3)
        heads = self.heads_for(("a", seg_a), ("b", seg_b))
        ra, rb = rows_of(seg_a), rows_of(seg_b)
        prefix = record_bytes(ra[0]) + record_bytes(rb[0])
        cases = [
            (b"\n", 1),
            (b"   \n", 1),
            (b"[1,2]\n", 1),
            (b"{not json\n", 1),
            (prefix + b"{bad\n", 3),
            (prefix + b"\xff", 3),
            ((json.dumps(None) + "\n").encode(), 1),
            ((json.dumps([1, 2]) + "\n").encode(), 1),
        ]
        for raw, at in cases:
            r = self.check_fail(raw, heads)
            self.assertEqual(r["ok"], False)
            self.assertEqual(
                (r["reason"], r["at"], r["tenant"], r["seq"]),
                ("missing", at, None, None), raw)
            self.assertEqual(
                set(r), {"ok", "at", "tenant", "seq", "reason"})

    def test_duplicate_keys_and_nonstandard_numbers_are_missing(self):
        seg = self.segment("a", 3, 3)
        heads = self.heads_for(("a", seg))
        raw = json.dumps(rows_of(seg)[0], sort_keys=True,
                         separators=(",", ":"))
        dup = raw.replace('"seq":3', '"seq":3,"seq":3', 1).encode() + b"\n"
        r = self.check_fail(dup, heads)
        # duplicate keys reject the whole parse: the line cannot name a
        # tenant or seq, so both are null
        self.assertEqual(
            (r["reason"], r["at"], r["tenant"], r["seq"]),
            ("missing", 1, None, None))
        row = {"tenant": "a", "seq": 3, "event": {}, "prev": "c" * 64}
        row["hash"] = AuditChain._hash(row)
        base = json.dumps(row, sort_keys=True, separators=(",", ":"))
        for spelling in ("NaN", "Infinity", "-Infinity", "1e999"):
            raw_line = base.replace("{}", spelling, 1).encode() + b"\n"
            r = self.check_fail(raw_line, heads)
            self.assertEqual((r["reason"], r["at"]), ("missing", 1),
                             spelling)

    def test_missing_field_fills_own_seq_or_expected(self):
        seg = self.segment("a", 3, 5)
        heads = self.heads_for(("a", seg))
        rows = rows_of(seg)
        # row 4 missing its hash: seq is the record's own 4
        row = {k: rows[1][k] for k in ("tenant", "seq", "event", "prev")}
        r = self.check_fail(record_bytes(rows[0]) + record_bytes(row), heads)
        self.assertEqual(
            (r["reason"], r["at"], r["tenant"], r["seq"]),
            ("missing", 2, "a", 4))
        # missing seq: seq falls back to the expected seq
        row = {k: rows[0][k] for k in ("tenant", "event", "prev")}
        r = self.check_fail(record_bytes(row), heads)
        self.assertEqual(
            (r["reason"], r["at"], r["tenant"], r["seq"]),
            ("missing", 1, "a", 3))

    def test_extra_field_is_missing(self):
        seg = self.segment("a", 3, 3)
        heads = self.heads_for(("a", seg))
        row = dict(rows_of(seg)[0])
        row["extra"] = True
        r = self.check_fail(record_bytes(row), heads)
        self.assertEqual((r["reason"], r["at"], r["seq"]),
                         ("missing", 1, 3))

    def test_nonempty_fragment_missing_declared_tenant(self):
        seg_a = self.segment("a", 4, 5)
        seg_b = self.segment("b", 2, 3)
        heads = self.heads_for(("a", seg_a), ("b", seg_b))
        # only a records: b is missing its first owed seq (2), at null
        r = self.check_fail(seg_a, heads)
        self.assertEqual(
            (r["ok"], r["reason"], r["at"], r["tenant"], r["seq"]),
            (False, "missing", None, "b", 2))
        # first listed-but-absent tenant in expected_heads order decides
        heads_rev = list(reversed(heads))
        r = self.check_fail(seg_b, heads_rev)
        self.assertEqual(
            (r["reason"], r["at"], r["tenant"], r["seq"]),
            ("missing", None, "a", 4))

    # --- sequence ---

    def test_wrong_seq_is_sequence_at_expected(self):
        seg = self.segment("a", 3, 5)
        heads = self.heads_for(("a", seg))
        rows = rows_of(seg)
        bad = dict(rows[0])
        bad["seq"] = 9
        bad["hash"] = AuditChain._hash(bad)
        r = self.check_fail(record_bytes(bad), heads)
        self.assertEqual(
            (r["reason"], r["at"], r["tenant"], r["seq"]),
            ("sequence", 1, "a", 3))
        # internal gap: seq 3 then seq 5
        data = record_bytes(rows[0]) + record_bytes(rows[2])
        r = self.check_fail(data, heads)
        self.assertEqual(
            (r["reason"], r["at"], r["tenant"], r["seq"]),
            ("sequence", 2, "a", 4))

    def test_float_and_bool_seq_spellings_are_sequence(self):
        row = {"tenant": "a", "seq": 3, "event": {}, "prev": "c" * 64}
        row["hash"] = AuditChain._hash(row)
        line = json.dumps(row, sort_keys=True)
        heads = [{"tenant": "a", "expected_count": 2,
                  "expected_hash": "c" * 64}]
        for spelling in ("3.0", "3e0", "true"):
            raw = line.replace('"seq": 3', '"seq": ' + spelling,
                               1).encode() + b"\n"
            r = self.check_fail(raw, heads)
            self.assertEqual(
                (r["reason"], r["at"], r["tenant"], r["seq"]),
                ("sequence", 1, "a", 3), spelling)

    # --- digest ---

    def test_wrong_first_prev_is_digest(self):
        seg = self.segment("a", 4, 5)
        heads = [{"tenant": "a", "expected_count": 3,
                  "expected_hash": "f" * 64}]
        r = self.check_fail(seg, heads)
        self.assertEqual(
            (r["reason"], r["at"], r["tenant"], r["seq"]),
            ("digest", 1, "a", 4))

    def test_wrong_interior_prev_and_tampered_hash_are_digest(self):
        seg = self.segment("a", 3, 5)
        heads = self.heads_for(("a", seg))
        rows = rows_of(seg)
        bad = dict(rows[1])
        bad["prev"] = "a" * 64
        bad["hash"] = AuditChain._hash(bad)
        r = self.check_fail(record_bytes(rows[0]) + record_bytes(bad), heads)
        self.assertEqual(
            (r["reason"], r["at"], r["tenant"], r["seq"]),
            ("digest", 2, "a", 4))
        tampered = dict(rows[0])
        tampered["event"] = {"tampered": True}
        r = self.check_fail(record_bytes(tampered), heads)
        self.assertEqual((r["reason"], r["at"], r["seq"]),
                         ("digest", 1, 3))

    def test_sequence_checked_before_digest(self):
        row = rows_of(self.segment("a", 3, 3))[0]
        row = dict(row)
        row["seq"] = 9
        row["prev"] = "f" * 64
        row["hash"] = AuditChain._hash(row)
        heads = [{"tenant": "a", "expected_count": 2,
                  "expected_hash": "c" * 64}]
        r = self.check_fail(record_bytes(row), heads)
        self.assertEqual(r["reason"], "sequence")

    # --- foreign tenant: ValueError, first physical line wins ---

    def test_undeclared_tenant_record_is_value_error(self):
        seg = self.segment("a", 3, 4)
        heads = self.heads_for(("a", seg))
        rows = rows_of(seg)
        foreign = dict(rows[0])
        foreign["tenant"] = "other"
        foreign["hash"] = AuditChain._hash(foreign)
        with self.assertRaises(ValueError):
            self.check_fail(record_bytes(foreign), heads)

    def test_distinct_json_identities_are_foreign(self):
        src = AuditChain(self.path.with_name("ids.jsonl"))
        src.append(1, {})
        src.append(1.0, {})
        src.append(True, {})
        src.append("1", {})
        for declared, other in ((1, 1.0), (1, True), (1, "1"),
                                ("1", 1), (1.0, True)):
            seg = src.export_tenant_range(other, 1, 1)
            heads = [{"tenant": declared, "expected_count": 0,
                      "expected_hash": ZERO}]
            with self.assertRaises(ValueError):
                self.offline.verify_all_range_bytes(seg, heads)
        # the exact identity verifies
        seg = src.export_tenant_range(1, 1, 1)
        heads = [{"tenant": 1, "expected_count": 0, "expected_hash": ZERO}]
        self.assertTrue(
            self.offline.verify_all_range_bytes(seg, heads)["ok"])

    def test_physical_line_order_decides_foreign_against_defect(self):
        seg = self.segment("a", 3, 5)
        heads = self.heads_for(("a", seg))
        rows = rows_of(seg)
        # foreign record first: ValueError even though a later line is broken
        foreign = dict(rows[0])
        foreign["tenant"] = "other"
        foreign["hash"] = AuditChain._hash(foreign)
        with self.assertRaises(ValueError):
            self.check_fail(record_bytes(foreign) + b"{bad\n", heads)
        # defect first: scan stops there, the later foreign line is never
        # reached and the content verdict wins
        tampered = dict(rows[0])
        tampered["event"] = {"tampered": True}  # hash now wrong
        foreign = dict(rows[1])
        foreign["tenant"] = "other"
        foreign["hash"] = AuditChain._hash(foreign)
        r = self.check_fail(
            record_bytes(tampered) + record_bytes(foreign), heads)
        self.assertEqual((r["reason"], r["at"]), ("digest", 1))


class VerifyAllRangeChunksTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.path = Path(self.tmp.name) / "audit.jsonl"
        self.chain = AuditChain(self.path)
        for i in range(8):
            self.chain.append("a", {"i": i, "s": "审计"})
            if i % 2 == 0:
                self.chain.append("b", {"i": i})
        self.chain.append("other", {"x": 1})
        self.offline = AuditChain(self.path.with_name("never-touched.jsonl"))
        self.seg_a = self.chain.export_tenant_range("a", 4, 6)
        self.seg_b = self.chain.export_tenant_range("b", 2, 4)
        self.rows_a, self.rows_b = rows_of(self.seg_a), rows_of(self.seg_b)
        self.heads = self.heads(self.seg_a, self.seg_b)
        self.inter = (
            record_bytes(self.rows_a[0]) + record_bytes(self.rows_b[0])
            + record_bytes(self.rows_b[1]) + record_bytes(self.rows_a[1])
            + record_bytes(self.rows_b[2]) + record_bytes(self.rows_a[2]))

    def tearDown(self):
        self.tmp.cleanup()

    def heads(self, seg_a, seg_b):
        def h(seg):
            first = rows_of(seg)[0]
            return first["seq"] - 1, first["prev"]
        ca, ha = h(seg_a)
        cb, hb = h(seg_b)
        return [
            {"tenant": "a", "expected_count": ca, "expected_hash": ha},
            {"tenant": "b", "expected_count": cb, "expected_hash": hb},
        ]

    def verify(self, chunks, heads=None):
        return self.offline.verify_all_range_chunks(
            chunks, self.heads if heads is None else heads)

    # --- container/member boundary ---

    def test_bare_bytes_and_bytearray_are_not_containers(self):
        for bad in (self.inter, bytearray(self.inter)):
            with self.assertRaises(ValueError):
                self.verify(bad)
        self.assertFalse(self.offline.path.exists())

    def test_non_iterable_container_is_value_error(self):
        for bad in (None, 1, object()):
            with self.assertRaises(ValueError):
                self.verify(bad)

    def test_non_bytes_members_value_error_and_stop_consumption(self):
        for bad_member in ("nope", 3, bytearray(self.inter), object()):
            pulled = []

            def gen():
                yield self.inter[:5]
                pulled.append(1)
                yield bad_member
                pulled.append(2)
                yield self.inter[5:]

            with self.assertRaises(ValueError):
                self.verify(gen())
            self.assertEqual(pulled, [1])

    def test_empty_iteration_is_unchanged_heads(self):
        expected = self.offline.verify_all_range_bytes(b"", self.heads)
        self.assertEqual(self.verify([]), expected)
        self.assertEqual(self.verify([b"", b""]), expected)
        self.assertEqual(self.verify(iter([])), expected)

    def test_heads_validated_before_any_chunk_pulled(self):
        pulled = []

        def gen():
            pulled.append(1)
            yield self.inter

        with self.assertRaises(ValueError):
            self.verify(gen(), "not-a-list")
        with self.assertRaises(ValueError):
            self.verify(
                gen(),
                [{"tenant": "a", "expected_count": -1,
                  "expected_hash": ZERO}])
        with self.assertRaises(ValueError):
            self.verify(
                gen(),
                [{"tenant": "a", "expected_count": 0,
                  "expected_hash": "nope"}])
        self.assertEqual(pulled, [])

    # --- bytes/chunks field-for-field equality on every cut ---

    def all_cuts(self, data):
        yield [data]
        yield [data, b""]
        yield [b"", data]
        yield [b"", data, b""]
        for size in (1, 2, 3, 5, 7, 13, 64, 1000):
            yield [data[i:i + size]
                   for i in range(0, len(data), size)]
        for cut in range(0, len(data) + 1):
            yield [data[:cut], data[cut:]]

    def assert_chunks_match_bytes(self, data, heads):
        expected = self.offline.verify_all_range_bytes(data, heads)
        for pieces in self.all_cuts(data):
            got = self.offline.verify_all_range_chunks(iter(pieces), heads)
            self.assertEqual(got, expected, (pieces, got, expected))

    def test_chunks_match_bytes_on_valid_fragments(self):
        self.assert_chunks_match_bytes(self.inter, self.heads)
        # single-tenant fragment against a multi-tenant expectation fails
        # with the missing-tenant verdict on every cut too
        self.assert_chunks_match_bytes(self.seg_a, self.heads)
        # empty fragment
        self.assert_chunks_match_bytes(b"", self.heads)

    def test_chunks_match_bytes_on_every_failure_shape(self):
        ra, rb = self.rows_a, self.rows_b
        bad_seq = dict(ra[1])
        bad_seq["seq"] = 9
        bad_seq["hash"] = AuditChain._hash(bad_seq)
        bad_prev = dict(ra[1])
        bad_prev["prev"] = "a" * 64
        bad_prev["hash"] = AuditChain._hash(bad_prev)
        tampered = dict(ra[0])
        tampered["event"] = {"x": True}
        extra = dict(ra[0])
        extra["z"] = 1
        foreign = dict(ra[0])
        foreign["tenant"] = "other"
        foreign["hash"] = AuditChain._hash(foreign)
        payloads = [
            record_bytes(ra[0]) + record_bytes(bad_seq),
            record_bytes(ra[0]) + record_bytes(bad_prev),
            record_bytes(tampered),
            record_bytes(extra),
            b"{bad\n",
            record_bytes(ra[0]) + b"\xff",
            b"\n",
            record_bytes(ra[0]) + record_bytes(rb[0]) + b"[1]\n",
        ]
        for data in payloads:
            self.assert_chunks_match_bytes(data, self.heads)
        # foreign tenant: ValueError on every cut
        data = record_bytes(foreign)
        for pieces in self.all_cuts(data):
            with self.assertRaises(ValueError):
                self.verify(iter(pieces))

    def test_chunk_cuts_split_utf8_json_and_lf(self):
        row = {"tenant": "a", "seq": 4, "event": {"s": "审计"},
               "prev": self.rows_a[0]["prev"]}
        row["hash"] = AuditChain._hash(row)
        raw = (json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
               ).encode("utf-8")
        self.assertIn("审计".encode("utf-8"), raw)
        heads = [{"tenant": "a", "expected_count": 3,
                  "expected_hash": self.rows_a[0]["prev"]},
                 {"tenant": "b", "expected_count": 4,
                  "expected_hash": self.rows_b[-1]["hash"]}]
        # non-empty fragment with no b record -> missing b, same on every cut
        expected = self.offline.verify_all_range_bytes(raw, heads)
        self.assertEqual(
            (expected["reason"], expected["tenant"], expected["seq"]),
            ("missing", "b", 5))
        for pieces in self.all_cuts(raw):
            self.assertEqual(self.verify(iter(pieces), heads), expected)

    # --- forward-only single consumption: pull stops at first defect ---

    def test_pull_stops_after_first_defective_completed_line(self):
        pulled = []

        def gen():
            good = record_bytes(self.rows_a[0])
            bad = b"{broken\n"
            later = record_bytes(self.rows_b[0])
            for name, chunk in (("g1", good[:4]), ("g2", good[4:]),
                                ("bad", bad), ("later", later)):
                pulled.append(name)
                yield chunk

        r = self.verify(gen())
        self.assertEqual(
            (r["reason"], r["at"], r["tenant"], r["seq"]),
            ("missing", 2, None, None))
        # the bad line completed with the "bad" chunk; "later" never pulled
        self.assertEqual(pulled, ["g1", "g2", "bad"])

    def test_pull_stops_after_sequence_verdict(self):
        pulled = []
        # a's first fragment record (seq 4) verifies; the next claims 9
        # while the state now expects 5
        good = self.rows_a[0]
        bad = dict(self.rows_a[1])  # seq 5
        bad["seq"] = 9
        bad["hash"] = AuditChain._hash(bad)

        def gen():
            for name, chunk in (
                    ("a4", record_bytes(good)),
                    ("bad", record_bytes(bad)),
                    ("later", record_bytes(self.rows_b[0]))):
                pulled.append(name)
                yield chunk

        r = self.verify(gen())
        self.assertEqual(
            (r["reason"], r["at"], r["tenant"], r["seq"]),
            ("sequence", 2, "a", 5))
        self.assertEqual(pulled, ["a4", "bad"])

    def test_pull_stops_after_foreign_record(self):
        pulled = []
        foreign = dict(self.rows_a[0])
        foreign["tenant"] = "other"
        foreign["hash"] = AuditChain._hash(foreign)

        def gen():
            for name, chunk in (
                    ("foreign", record_bytes(foreign)),
                    ("later", record_bytes(self.rows_b[0]))):
                pulled.append(name)
                yield chunk

        with self.assertRaises(ValueError):
            self.verify(gen())
        self.assertEqual(pulled, ["foreign"])

    def test_iterator_exception_propagates_verbatim(self):
        class Boom(Exception):
            pass

        def gen():
            yield self.inter[:3]
            raise Boom()

        with self.assertRaises(Boom):
            self.verify(gen())

    def test_path_never_touched(self):
        self.assertTrue(self.verify([self.inter])["ok"])
        self.assertFalse(self.offline.path.exists())


if __name__ == "__main__":
    unittest.main()
