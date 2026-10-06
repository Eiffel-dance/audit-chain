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
        # two tenants interleaved, plus an uninvolved third tenant; events
        # carry multibyte UTF-8 so fragment bytes are not ASCII-only
        for i in range(6):
            self.chain.append("a", {"i": i, "s": "审计"})
            self.chain.append("b", {"i": i, "s": "审计"})
        self.chain.append("other", {"x": 1})
        # every offline entry below points at a path that must never be read,
        # created or modified
        self.offline = AuditChain(self.path.with_name("never-touched.jsonl"))

    def tearDown(self):
        self.tmp.cleanup()

    def rows(self, tenant, lo, hi=None):
        return rows_of(self.chain.export_tenant_range(tenant, lo, hi))

    def head(self, tenant, n):
        if n == 0:
            return {"tenant": tenant, "expected_count": 0,
                    "expected_hash": ZERO}
        tail = self.rows(tenant, 1, n)[-1]["hash"]
        return {"tenant": tenant, "expected_count": n,
                "expected_hash": tail}

    def fragment(self, pairs):
        # pairs: list of (tenant, row) concatenated in the given physical
        # order, so tenants may interleave arbitrarily
        return b"".join(record_bytes(row) for _tenant, row in pairs)

    def interleave(self, rows_a, rows_b):
        out = []
        for ra, rb in zip(rows_a, rows_b):
            out.append(("a", ra))
            out.append(("b", rb))
        return out

    # --- boundary: shape, types, and priority before any parsing ---

    def test_data_must_be_bytes(self):
        heads = [self.head("a", 0)]
        row = record_bytes(self.rows("a", 1, 1)[0])
        for bad in ("jsonl", bytearray(row), 1, None, [], object()):
            with self.assertRaises(ValueError):
                self.offline.verify_all_range_bytes(bad, heads)
        self.assertFalse(self.offline.path.exists())

    def test_expected_heads_must_be_list(self):
        for bad in (None, {}, (), "x", 1, object()):
            with self.assertRaises(ValueError):
                self.offline.verify_all_range_bytes(b"", bad)
        self.assertFalse(self.offline.path.exists())

    def test_head_members_must_have_exactly_three_keys(self):
        good = {"tenant": "a", "expected_count": 0, "expected_hash": ZERO}
        for bad in (
            None, 1, "x", [],
            {"tenant": "a", "expected_count": 0},
            {"tenant": "a", "expected_hash": ZERO},
            {"expected_count": 0, "expected_hash": ZERO},
            dict(good, extra=1),
        ):
            with self.assertRaises(ValueError):
                self.offline.verify_all_range_bytes(b"", [bad])
        self.assertFalse(self.offline.path.exists())

    def test_illegal_tenant_is_value_error(self):
        bad_tenants = [
            float("nan"), float("inf"), float("-inf"),
            {"k": float("nan")}, [1, [float("-inf")]],
            {1: "x"}, {"a": {2: 3}}, object(), b"bytes", {1, 2}, ("a", 1),
        ]
        for bad in bad_tenants:
            with self.assertRaises(ValueError):
                self.offline.verify_all_range_bytes(
                    b"", [{"tenant": bad, "expected_count": 0,
                           "expected_hash": ZERO}])
        cyclic = {}
        cyclic["k"] = cyclic
        with self.assertRaises(ValueError):
            self.offline.verify_all_range_bytes(
                b"", [{"tenant": cyclic, "expected_count": 0,
                       "expected_hash": ZERO}])
        self.assertFalse(self.offline.path.exists())

    def test_expected_count_must_be_non_negative_plain_int(self):
        for bad in (True, False, -1, -100, 1.0, 2.5, "3", None, [], {}):
            with self.assertRaises(ValueError):
                self.offline.verify_all_range_bytes(
                    b"", [{"tenant": "a", "expected_count": bad,
                           "expected_hash": ZERO}])
        self.assertFalse(self.offline.path.exists())

    def test_expected_hash_must_be_64_lowercase_hex(self):
        for bad in (None, "", "0" * 63, "0" * 65, "A" * 64, "g" * 64,
                    0, [], {}):
            with self.assertRaises(ValueError):
                self.offline.verify_all_range_bytes(
                    b"", [{"tenant": "a", "expected_count": 0,
                           "expected_hash": bad}])
        self.assertFalse(self.offline.path.exists())

    def test_duplicate_canonical_tenant_is_value_error(self):
        heads = [
            self.head("a", 0),
            {"tenant": "a", "expected_count": 1, "expected_hash": ZERO},
        ]
        with self.assertRaises(ValueError):
            self.offline.verify_all_range_bytes(b"", heads)
        # object key order alone does not change the canonical identity
        heads = [
            {"tenant": {"k": 1, "j": 2}, "expected_count": 0,
             "expected_hash": ZERO},
            {"tenant": {"j": 2, "k": 1}, "expected_count": 3,
             "expected_hash": ZERO},
        ]
        with self.assertRaises(ValueError):
            self.offline.verify_all_range_bytes(b"", heads)
        self.assertFalse(self.offline.path.exists())

    def test_boundary_value_error_beats_corrupt_content_and_path(self):
        # a malformed expected_heads loses no time on corrupt content
        with self.assertRaises(ValueError):
            self.offline.verify_all_range_bytes(
                b"{not json\n", [{"tenant": "a", "expected_count": -1,
                                  "expected_hash": ZERO}])
        # an existing corrupt file at the configured path is never read
        self.offline.path.write_bytes(b"\xff\xff")
        with self.assertRaises(ValueError):
            self.offline.verify_all_range_bytes(
                b"", [{"tenant": "a", "expected_count": 0,
                       "expected_hash": "nope"}])
        self.assertEqual(self.offline.path.read_bytes(), b"\xff\xff")

    def test_never_reads_creates_or_modifies_constructor_path(self):
        phantom = AuditChain(self.path.with_name("phantom.jsonl"))
        heads = [self.head("a", 3), self.head("b", 3)]
        pairs = self.interleave(self.rows("a", 4), self.rows("b", 4))
        self.assertTrue(phantom.verify_all_range_bytes(
            self.fragment(pairs), heads)["ok"])
        self.assertFalse(phantom.path.exists())
        before = self.chain.export_all()
        phantom.verify_all_range_bytes(before, [
            self.head("a", 0), self.head("b", 0), self.head("other", 0)])
        self.assertEqual(self.chain.export_all(), before)
        self.assertFalse(phantom.path.exists())

    # --- success ---

    def test_interleaved_fragment_success_shape(self):
        heads = [self.head("a", 3), self.head("b", 3)]
        rows_a = self.rows("a", 4)
        rows_b = self.rows("b", 4)
        pairs = self.interleave(rows_a, rows_b)
        r = self.offline.verify_all_range_bytes(self.fragment(pairs), heads)
        self.assertEqual(set(r), {"ok", "tenants"})
        self.assertTrue(r["ok"])
        self.assertEqual(
            r["tenants"],
            [
                {"tenant": "a", "count": 6,
                 "hash": rows_a[-1]["hash"]},
                {"tenant": "b", "count": 6,
                 "hash": rows_b[-1]["hash"]},
            ])
        for entry in r["tenants"]:
            self.assertEqual(set(entry), {"tenant", "count", "hash"})

    def test_tenants_follow_expected_heads_order_not_physical_order(self):
        # physical first-appearance is a, then b; expectation lists b first,
        # and the two segments carry different record counts
        heads = [self.head("b", 2), self.head("a", 5)]
        a6 = self.rows("a", 6)
        b_tail = self.rows("b", 3, 6)
        pairs = [("a", a6[0])] + [("b", row) for row in b_tail]
        r = self.offline.verify_all_range_bytes(self.fragment(pairs), heads)
        self.assertTrue(r["ok"])
        self.assertEqual([t["tenant"] for t in r["tenants"]], ["b", "a"])
        self.assertEqual([t["count"] for t in r["tenants"]], [6, 6])
        self.assertEqual(r["tenants"][0]["hash"], b_tail[-1]["hash"])
        self.assertEqual(r["tenants"][1]["hash"], a6[-1]["hash"])

    def test_chain_beginning_with_zero_anchor(self):
        heads = [self.head("a", 0), self.head("b", 0)]
        pairs = self.interleave(self.rows("a", 1, 3), self.rows("b", 1, 3))
        r = self.offline.verify_all_range_bytes(self.fragment(pairs), heads)
        self.assertTrue(r["ok"])
        self.assertEqual([t["count"] for t in r["tenants"]], [3, 3])

    def test_trailing_lf_optional_and_crlf_accepted(self):
        heads = [self.head("a", 0), self.head("b", 0)]
        pairs = self.interleave(self.rows("a", 1, 2), self.rows("b", 1, 2))
        frag = self.fragment(pairs)
        self.assertTrue(self.offline.verify_all_range_bytes(
            frag.rstrip(b"\n"), heads)["ok"])
        self.assertTrue(self.offline.verify_all_range_bytes(
            frag.replace(b"\n", b"\r\n"), heads)["ok"])

    def test_object_tenant_identity_key_order_normalized(self):
        src = AuditChain(self.path.with_name("obj.jsonl"))
        for i in range(3):
            src.append({"a": 1, "b": 2}, {"i": i})
        rows = rows_of(src.export_tenant_range({"a": 1, "b": 2}, 2))
        head_rows = rows_of(src.export_tenant_range({"b": 2, "a": 1}, 1, 1))
        heads = [{
            "tenant": {"b": 2, "a": 1},
            "expected_count": 1,
            "expected_hash": head_rows[-1]["hash"],
        }]
        r = self.offline.verify_all_range_bytes(
            record_bytes(rows[0]), heads)
        self.assertTrue(r["ok"])
        # the tenant value is echoed verbatim from the expectation
        self.assertEqual(r["tenants"][0]["tenant"], {"b": 2, "a": 1})

    def test_distinct_json_identities_are_separate_tenants(self):
        src = AuditChain(self.path.with_name("ids.jsonl"))
        for tenant in (1, 1.0, True, "1"):
            src.append(tenant, {})
        fragments = {}
        for tenant in (1, 1.0, True, "1"):
            fragments[json.dumps(tenant)] = rows_of(
                src.export_tenant_range(tenant, 1, 1))[0]
        heads = [
            {"tenant": 1, "expected_count": 0, "expected_hash": ZERO},
            {"tenant": 1.0, "expected_count": 0, "expected_hash": ZERO},
            {"tenant": True, "expected_count": 0, "expected_hash": ZERO},
            {"tenant": "1", "expected_count": 0, "expected_hash": ZERO},
        ]
        frag = b"".join(record_bytes(fragments[k])
                        for k in ("1", "1.0", "true", '"1"'))
        r = self.offline.verify_all_range_bytes(frag, heads)
        self.assertTrue(r["ok"])
        self.assertEqual(
            [t["count"] for t in r["tenants"]], [1, 1, 1, 1])

    def test_empty_fragment_returns_heads_unchanged(self):
        heads = [self.head("a", 3), self.head("b", 2)]
        r = self.offline.verify_all_range_bytes(b"", heads)
        self.assertEqual(r, {
            "ok": True,
            "tenants": [
                {"tenant": "a", "count": 3,
                 "hash": heads[0]["expected_hash"]},
                {"tenant": "b", "count": 2,
                 "hash": heads[1]["expected_hash"]},
            ]})

    def test_empty_heads_with_empty_bytes_is_empty_fragment(self):
        r = self.offline.verify_all_range_bytes(b"", [])
        self.assertEqual(r, {"ok": True, "tenants": []})

    def test_success_heads_chain_into_next_segment_and_import(self):
        first_heads = [self.head("a", 2), self.head("b", 2)]
        seg1 = self.interleave(self.rows("a", 3, 4), self.rows("b", 3, 4))
        v1 = self.offline.verify_all_range_bytes(
            self.fragment(seg1), first_heads)
        self.assertTrue(v1["ok"])
        next_heads = [
            {"tenant": t["tenant"], "expected_count": t["count"],
             "expected_hash": t["hash"]}
            for t in v1["tenants"]
        ]
        seg2 = self.interleave(self.rows("a", 5, 6), self.rows("b", 5, 6))
        v2 = self.offline.verify_all_range_bytes(
            self.fragment(seg2), next_heads)
        self.assertEqual([t["count"] for t in v2["tenants"]], [6, 6])
        # the very same fragments and head assertions drive import_all_range
        target = AuditChain(self.path.with_name("target.jsonl"))
        prefix = self.interleave(self.rows("a", 1, 2), self.rows("b", 1, 2))
        target.import_all_range(self.fragment(prefix),
                                [self.head("a", 0), self.head("b", 0)])
        target.import_all_range(self.fragment(seg1), first_heads)
        target.import_all_range(self.fragment(seg2), next_heads)
        self.assertEqual(
            target.verify_all()["tenants"],
            [{"tenant": "a", "count": 6}, {"tenant": "b", "count": 6}])

    # --- missing: undeclared-in-fragment, bad lines, fields ---

    def verify(self, data, heads=None):
        if heads is None:
            heads = [self.head("a", 0), self.head("b", 0)]
        return self.offline.verify_all_range_bytes(data, heads)

    def test_failure_object_has_exactly_five_fields(self):
        r = self.verify(b"{bad\n")
        self.assertEqual(set(r), {"ok", "at", "tenant", "seq", "reason"})
        self.assertFalse(r["ok"])

    def test_nonempty_fragment_without_declared_tenant_is_missing(self):
        heads = [self.head("a", 3), self.head("b", 3)]
        # only a's records; b's first owed seq is 4 -- assertions order picks
        # the first absent declaration when several are missing
        r = self.verify(self.fragment([("a", self.rows("a", 4)[0])]), heads)
        self.assertEqual(r, {
            "ok": False, "reason": "missing", "at": 4,
            "tenant": "b", "seq": 4})
        heads_rev = [self.head("b", 3), self.head("a", 3)]
        frag = record_bytes(self.rows("b", 4)[0])
        r = self.offline.verify_all_range_bytes(frag, heads_rev)
        self.assertEqual(
            (r["reason"], r["at"], r["tenant"], r["seq"]),
            ("missing", 4, "a", 4))

    def test_unparseable_and_bad_utf8_lines_are_missing_at_line(self):
        rows = self.rows("a", 1, 2)
        good = record_bytes(rows[0])
        cases = [
            (b"\n", 1),
            (b"   \n", 1),
            (b"[1,2]\n", 1),
            (b"{not json\n", 1),
            (good + b"{bad\n", 2),
            (good + b"\xff", 2),
            (json.dumps(None).encode() + b"\n", 1),
            (json.dumps(42).encode() + b"\n", 1),
        ]
        for raw, at in cases:
            r = self.verify(raw)
            self.assertEqual(r["reason"], "missing", raw)
            self.assertEqual(
                (r["at"], r["tenant"], r["seq"]), (at, None, None), raw)

    def test_duplicate_keys_and_nonstandard_numbers_are_missing(self):
        row = {"tenant": "a", "seq": 1, "event": {}, "prev": ZERO}
        raw = json.dumps(row, sort_keys=True, separators=(",", ":"))
        dup = raw.replace('"seq":1', '"seq":1,"seq":1', 1).encode() + b"\n"
        r = self.verify(dup)
        self.assertEqual(
            (r["reason"], r["at"], r["tenant"], r["seq"]),
            ("missing", 1, None, None))
        nan_line = raw.replace("{}", "NaN", 1).encode() + b"\n"
        r = self.verify(nan_line)
        self.assertEqual((r["reason"], r["at"]), ("missing", 1))

    def test_missing_field_fills_at_own_seq_or_expected(self):
        rows = self.rows("a", 4, 5)
        # first owed record carries seq 4 but omits hash: at/seq are its own
        row = {k: rows[0][k] for k in ("tenant", "seq", "event", "prev")}
        r = self.verify(record_bytes(row),
                        [self.head("a", 3), self.head("b", 3)])
        self.assertEqual(
            (r["reason"], r["at"], r["tenant"], r["seq"]),
            ("missing", 4, "a", 4))
        # seq itself missing: fall back to the expected seq
        row = {k: rows[0][k] for k in ("tenant", "event", "prev")}
        r = self.verify(record_bytes(row),
                        [self.head("a", 3), self.head("b", 3)])
        self.assertEqual(
            (r["reason"], r["at"], r["tenant"], r["seq"]),
            ("missing", 4, "a", 4))

    def test_extra_field_is_missing(self):
        row = self.rows("a", 1, 1)[0]
        row["extra"] = True
        r = self.verify(record_bytes(row))
        self.assertEqual(
            (r["reason"], r["at"], r["tenant"], r["seq"]),
            ("missing", 1, "a", 1))

    def test_line_without_tenant_key_cannot_name_tenant(self):
        row = {"seq": 1, "event": {}, "prev": ZERO,
               "hash": "0" * 64}
        r = self.verify(record_bytes(row))
        self.assertEqual(
            (r["reason"], r["at"], r["tenant"], r["seq"]),
            ("missing", 1, None, None))

    # --- sequence ---

    def test_wrong_seq_is_sequence_at_expected(self):
        rows = self.rows("a", 1, 3)
        bad = dict(rows[0])
        bad["seq"] = 9
        bad["hash"] = AuditChain._hash(bad)
        r = self.verify(record_bytes(bad))
        self.assertEqual(
            (r["reason"], r["at"], r["tenant"], r["seq"]),
            ("sequence", 1, "a", 1))
        # internal gap across interleaving: a1, b1 then a3 (a expects 2)
        b1 = self.rows("b", 1)[0]
        bad = dict(rows[2])
        frag = record_bytes(rows[0]) + record_bytes(b1) + record_bytes(bad)
        r = self.verify(frag)
        self.assertEqual(
            (r["reason"], r["at"], r["tenant"], r["seq"]),
            ("sequence", 2, "a", 2))

    def test_float_and_bool_seq_spellings_are_sequence(self):
        row = {"tenant": "a", "seq": 1, "event": {}, "prev": ZERO}
        row["hash"] = AuditChain._hash(row)
        line = json.dumps(row, sort_keys=True)
        for spelling in ("1.0", "1e0", "true"):
            raw = line.replace(
                '"seq": 1', '"seq": ' + spelling, 1).encode() + b"\n"
            r = self.verify(raw)
            self.assertEqual(
                (r["reason"], r["at"], r["tenant"], r["seq"]),
                ("sequence", 1, "a", 1), spelling)

    # --- digest ---

    def test_wrong_first_prev_is_digest(self):
        row = self.rows("a", 1)[0]
        bad = dict(row)
        bad["prev"] = "f" * 64
        bad["hash"] = AuditChain._hash(bad)
        r = self.verify(record_bytes(bad))
        self.assertEqual(
            (r["reason"], r["at"], r["tenant"], r["seq"]),
            ("digest", 1, "a", 1))

    def test_wrong_interior_prev_is_digest(self):
        rows = self.rows("a", 1, 2)
        bad = dict(rows[1])
        bad["prev"] = "a" * 64
        bad["hash"] = AuditChain._hash(bad)  # hash consistent, prev wrong
        r = self.verify(record_bytes(rows[0]) + record_bytes(bad))
        self.assertEqual(
            (r["reason"], r["at"], r["tenant"], r["seq"]),
            ("digest", 2, "a", 2))

    def test_tampered_event_hash_mismatch_is_digest(self):
        row = self.rows("a", 1)[0]
        row["event"] = {"tampered": True}
        r = self.verify(record_bytes(row))
        self.assertEqual(
            (r["reason"], r["at"], r["tenant"], r["seq"]),
            ("digest", 1, "a", 1))

    def test_sequence_checked_before_digest(self):
        row = self.rows("a", 1)[0]
        row = dict(row)
        row["seq"] = 9
        row["prev"] = "f" * 64
        row["hash"] = AuditChain._hash(row)
        r = self.verify(record_bytes(row))
        self.assertEqual(r["reason"], "sequence")

    # --- undeclared tenant: ValueError, first physical line wins ---

    def test_undeclared_tenant_record_is_value_error(self):
        rows = self.rows("a", 1, 2)
        foreign = dict(rows[0])
        foreign["tenant"] = "other"
        foreign["hash"] = AuditChain._hash(foreign)
        with self.assertRaises(ValueError):
            self.verify(record_bytes(foreign))

    def test_undeclared_identity_with_bad_fields_is_still_value_error(self):
        # the line names an undeclared tenant identity before the field set
        # is examined, exactly _scan_import_ranges' lookup-first rule
        row = self.rows("a", 1)[0]
        row["tenant"] = "other"
        row["extra"] = 1
        with self.assertRaises(ValueError):
            self.verify(record_bytes(row))

    def test_distinct_json_identities_are_undeclared(self):
        for declared, other in ((1, 1.0), (1, True), (1, "1"),
                                ("1", 1), (1.0, True)):
            row = {"tenant": other, "seq": 1, "event": {}, "prev": ZERO}
            row["hash"] = AuditChain._hash(row)
            with self.assertRaises(ValueError):
                self.offline.verify_all_range_bytes(
                    record_bytes(row),
                    [{"tenant": declared, "expected_count": 0,
                      "expected_hash": ZERO}])

    def test_physical_line_order_decides_undeclared_against_defect(self):
        rows = self.rows("a", 1, 2)
        # undeclared record first: ValueError even though a later line is
        # broken
        foreign = dict(rows[0])
        foreign["tenant"] = "other"
        foreign["hash"] = AuditChain._hash(foreign)
        with self.assertRaises(ValueError):
            self.verify(record_bytes(foreign) + b"{bad\n")
        # defect first: the scan stops there, a later undeclared line is
        # never reached and the content verdict wins
        tampered = dict(rows[0])
        tampered["event"] = {"tampered": True}  # hash now wrong
        foreign = dict(rows[1])
        foreign["tenant"] = "other"
        foreign["hash"] = AuditChain._hash(foreign)
        r = self.verify(record_bytes(tampered) + record_bytes(foreign))
        self.assertEqual(r["reason"], "digest")
        self.assertEqual(r["at"], 1)


class VerifyAllRangeChunksTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.path = Path(self.tmp.name) / "audit.jsonl"
        self.chain = AuditChain(self.path)
        for i in range(6):
            self.chain.append("a", {"i": i, "s": "审计"})
            self.chain.append("b", {"i": i, "s": "审计"})
        self.offline = AuditChain(self.path.with_name("never-touched.jsonl"))
        self.heads = [
            self.head("a", 2),
            self.head("b", 2),
        ]
        pairs = []
        ra = rows_of(self.chain.export_tenant_range("a", 3, 5))
        rb = rows_of(self.chain.export_tenant_range("b", 3, 5))
        for x, y in zip(ra, rb):
            pairs.append(x)
            pairs.append(y)
        self.frag = b"".join(record_bytes(r) for r in pairs)

    def tearDown(self):
        self.tmp.cleanup()

    def head(self, tenant, n):
        if n == 0:
            return {"tenant": tenant, "expected_count": 0,
                    "expected_hash": ZERO}
        rows = rows_of(self.chain.export_tenant_range(tenant, 1, n))
        return {"tenant": tenant, "expected_count": n,
                "expected_hash": rows[-1]["hash"]}

    def verify(self, chunks, heads=None):
        return self.offline.verify_all_range_chunks(
            chunks, self.heads if heads is None else heads)

    # --- container/member boundary ---

    def test_bare_bytes_and_bytearray_are_not_containers(self):
        for bad in (self.frag, bytearray(self.frag)):
            with self.assertRaises(ValueError):
                self.verify(bad)
        self.assertFalse(self.offline.path.exists())

    def test_non_iterable_container_is_value_error(self):
        for bad in (None, 1, object()):
            with self.assertRaises(ValueError):
                self.verify(bad)

    def test_non_bytes_members_value_error_and_stop_consumption(self):
        for bad_member in ("nope", 3, bytearray(self.frag), object()):
            pulled = []

            def gen():
                yield self.frag[:5]
                pulled.append(1)
                yield bad_member
                pulled.append(2)
                yield self.frag[5:]

            with self.assertRaises(ValueError):
                self.verify(gen())
            self.assertEqual(pulled, [1])

    def test_empty_iteration_is_empty_fragment_success(self):
        expected = self.offline.verify_all_range_bytes(b"", self.heads)
        for chunks in ([], iter([]), [b""], [b"", b""]):
            self.assertEqual(self.verify(chunks), expected)

    def test_parameters_validated_before_any_chunk_pulled(self):
        pulled = []

        def gen():
            pulled.append(1)
            yield self.frag

        bad_heads = [
            None,
            [{}],
            [{"tenant": "a", "expected_count": True,
              "expected_hash": ZERO}],
            [{"tenant": "a", "expected_count": -1, "expected_hash": ZERO}],
            [{"tenant": "a", "expected_count": 0,
              "expected_hash": "A" * 64}],
            [{"tenant": "a", "expected_count": 0, "expected_hash": ZERO,
              "extra": 1}],
        ]
        for heads in bad_heads:
            with self.assertRaises(ValueError):
                self.offline.verify_all_range_chunks(gen(), heads)
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

    def assert_chunks_match_bytes(self, data, heads):
        expected = self.offline.verify_all_range_bytes(data, heads)
        for pieces in self.all_cuts(data):
            got = self.offline.verify_all_range_chunks(iter(pieces), heads)
            self.assertEqual(got, expected, (pieces, got, expected))

    def test_chunks_match_bytes_on_valid_fragments(self):
        self.assert_chunks_match_bytes(self.frag, self.heads)
        self.assert_chunks_match_bytes(b"", self.heads)
        # heads in reverse order, and a ZERO-anchored chain beginning
        rev = [self.head("b", 0), self.head("a", 0)]
        ra = rows_of(self.chain.export_tenant_range("a", 1, 3))
        rb = rows_of(self.chain.export_tenant_range("b", 1, 3))
        data = b"".join(
            record_bytes(r) for pair in zip(ra, rb) for r in pair)
        self.assert_chunks_match_bytes(data, rev)

    def test_chunks_match_bytes_on_every_failure_shape(self):
        ra = rows_of(self.chain.export_tenant_range("a", 1, 3))
        rb = rows_of(self.chain.export_tenant_range("b", 1, 3))
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
        heads0 = [self.head("a", 0), self.head("b", 0)]
        payloads = [
            (record_bytes(ra[0]), heads0),                 # b missing
            (record_bytes(bad_seq), heads0),
            (record_bytes(bad_prev), heads0),
            (record_bytes(tampered), heads0),
            (record_bytes(extra), heads0),
            (b"{bad\n", heads0),
            (record_bytes(ra[0]) + b"\xff", heads0),
            (b"\n", heads0),
            (record_bytes(ra[0]) + record_bytes(bad_seq), heads0),
        ]
        for data, heads in payloads:
            self.assert_chunks_match_bytes(data, heads)

    def test_undeclared_tenant_matches_value_error_on_every_cut(self):
        row = ra = rows_of(self.chain.export_tenant_range("a", 1, 1))[0]
        foreign = dict(row)
        foreign["tenant"] = "other"
        foreign["hash"] = AuditChain._hash(foreign)
        data = record_bytes(foreign)
        heads0 = [self.head("a", 0), self.head("b", 0)]
        for pieces in self.all_cuts(data):
            with self.assertRaises(ValueError):
                self.offline.verify_all_range_chunks(
                    iter(pieces), heads0)

    def test_chunk_cuts_split_utf8_json_and_lf(self):
        row = {"tenant": "a", "seq": 3, "event": {"s": "审计"},
               "prev": self.head("a", 2)["expected_hash"]}
        row["hash"] = AuditChain._hash(row)
        raw = (json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
               ).encode("utf-8")
        self.assertIn("审计".encode("utf-8"), raw)
        heads = [self.head("a", 2)]
        for cut in range(len(raw) + 1):
            r = self.offline.verify_all_range_chunks(
                [raw[:cut], raw[cut:]], heads)
            self.assertTrue(r["ok"], cut)

    # --- forward-only single consumption: pull stops at first defect ---

    def test_valid_fragment_consumes_every_chunk(self):
        pulled = []

        def gen():
            for i in range(0, len(self.frag), 7):
                pulled.append(i)
                yield self.frag[i:i + 7]

        r = self.verify(gen())
        self.assertTrue(r["ok"])
        self.assertEqual(pulled, list(range(0, len(self.frag), 7)))

    def test_pull_stops_after_first_defective_completed_line(self):
        pulled = []
        rows = rows_of(self.chain.export_tenant_range("a", 3, 4))
        good = record_bytes(rows[0])
        bad = b"{broken\n"
        later = record_bytes(rows[1])

        def gen():
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
        rows = rows_of(self.chain.export_tenant_range("a", 3, 5))
        good = record_bytes(rows[0])
        bad = record_bytes(rows[2])   # seq 5 where 4 expected

        def gen():
            for name, chunk in (("a", good), ("bad", bad),
                                ("c", record_bytes(rows[1]))):
                pulled.append(name)
                yield chunk

        r = self.verify(gen())
        self.assertEqual(
            (r["reason"], r["at"], r["tenant"], r["seq"]),
            ("sequence", 4, "a", 4))
        self.assertEqual(pulled, ["a", "bad"])

    def test_pull_stops_after_undeclared_record(self):
        pulled = []
        row = rows_of(self.chain.export_tenant_range("a", 3))[0]
        foreign = dict(row)
        foreign["tenant"] = "other"
        foreign["hash"] = AuditChain._hash(foreign)

        def gen():
            for name, chunk in (
                ("foreign", record_bytes(foreign)),
                ("later", record_bytes(row)),
            ):
                pulled.append(name)
                yield chunk

        with self.assertRaises(ValueError):
            self.verify(gen())
        self.assertEqual(pulled, ["foreign"])

    def test_iterator_exception_propagates_verbatim(self):
        class Boom(Exception):
            pass

        def gen():
            yield self.frag[:3]
            raise Boom()

        with self.assertRaises(Boom):
            self.verify(gen())


if __name__ == "__main__":
    unittest.main()
