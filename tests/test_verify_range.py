import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app import AuditChain, ZERO


def record_bytes(item):
    return (json.dumps(item, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


class VerifyTenantRangeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.path = Path(self.tmp.name) / "audit.jsonl"
        self.chain = AuditChain(self.path)
        self.other = AuditChain(self.path.with_name("never-touched.jsonl"))

    def tearDown(self):
        self.tmp.cleanup()

    def seed(self, n=6):
        for i in range(n):
            self.chain.append("a", {"i": i, "s": "审计-世界"})
            self.chain.append("b", {"i": i})

    def prev_for(self, tenant, start_seq):
        if start_seq == 1:
            return ZERO
        return self.chain.read_tenant(tenant)[start_seq - 2]["hash"]

    def valid_row(self, tenant="t", seq=1, prev=ZERO, event=None):
        row = {"tenant": tenant, "seq": seq, "event": event or {}, "prev": prev}
        row["hash"] = AuditChain._hash(row)
        return record_bytes(row)

    # --- success shape and range semantics ---

    def test_success_result_shape(self):
        self.seed()
        recs = self.chain.read_tenant("a")
        data = self.chain.export_tenant_range("a", 3, 5)
        prev = recs[1]["hash"]
        self.assertEqual(
            self.other.verify_tenant_range_bytes("a", 3, 5, prev, data),
            {"ok": True, "tenant": "a", "start_seq": 3, "end_seq": 5,
             "count": 3, "hash": recs[4]["hash"]})
        # open tail echoes the declared end_seq (None)
        data = self.chain.export_tenant_range("a", 3)
        self.assertEqual(
            self.other.verify_tenant_range_bytes("a", 3, None, prev, data),
            {"ok": True, "tenant": "a", "start_seq": 3, "end_seq": None,
             "count": 4, "hash": recs[5]["hash"]})

    def test_success_full_chain_and_single_record(self):
        self.seed()
        recs = self.chain.read_tenant("a")
        data = self.chain.export_tenant_range("a", 1, 6)
        self.assertEqual(
            self.other.verify_tenant_range_bytes("a", 1, 6, ZERO, data),
            {"ok": True, "tenant": "a", "start_seq": 1, "end_seq": 6,
             "count": 6, "hash": recs[5]["hash"]})
        data = self.chain.export_tenant_range("a", 4, 4)
        self.assertEqual(
            self.other.verify_tenant_range_bytes("a", 4, 4, recs[2]["hash"],
                                                 data),
            {"ok": True, "tenant": "a", "start_seq": 4, "end_seq": 4,
             "count": 1, "hash": recs[3]["hash"]})

    def test_call_conventions(self):
        self.seed()
        data = self.chain.export_tenant_range("a", 2, 4)
        prev = self.prev_for("a", 2)
        a = self.other.verify_tenant_range_bytes("a", 2, 4, prev, data)
        b = self.other.verify_tenant_range_bytes("a", 2, 4, expected_prev=prev,
                                                 data=data)
        c = self.other.verify_tenant_range_bytes("a", 2, end_seq=4,
                                                 expected_prev=prev, data=data)
        d = self.other.verify_tenant_range_chunks("a", 2, 4, prev, [data])
        self.assertEqual(a, b)
        self.assertEqual(a, c)
        self.assertEqual(a, d)

    def test_result_anchors_import_tenant_range(self):
        self.seed()
        recs = self.chain.read_tenant("a")
        data = self.chain.export_tenant_range("a", 3, 5)
        prev = recs[1]["hash"]
        r = self.other.verify_tenant_range_bytes("a", 3, 5, prev, data)
        self.assertTrue(r["ok"])
        # the verified fragment grafts onto a target whose head is the
        # asserted predecessor (start_seq - 1, expected_prev)
        target = AuditChain(Path(self.tmp.name) / "target.jsonl")
        for item in recs[:2]:
            target.append("a", item["event"])
        imported = target.import_tenant_range("a", data, 2, prev)
        self.assertEqual(len(imported), 3)
        self.assertEqual(target.head("a"),
                         {"tenant": "a", "count": 5, "hash": r["hash"]})

    def test_tenant_identities(self):
        for t in (1, 1.0, True, "1"):
            self.chain.append(t, {})
            self.chain.append(t, {})
        for t in (1, 1.0, True, "1"):
            data = self.chain.export_tenant_range(t, 2, 2)
            prev = self.chain.read_tenant(t)[0]["hash"]
            r = self.other.verify_tenant_range_bytes(t, 2, 2, prev, data)
            self.assertEqual(r["ok"], True, t)
            self.assertEqual(r["tenant"], t)
            self.assertEqual(r["count"], 1)

    def test_object_tenant_identity_ignores_key_order(self):
        self.chain.append({"k": "v", "n": [1, 2]}, {})
        self.chain.append({"k": "v", "n": [1, 2]}, {})
        data = self.chain.export_tenant_range({"n": [1, 2], "k": "v"}, 1, 2)
        r = self.other.verify_tenant_range_bytes({"k": "v", "n": [1, 2]},
                                                 1, 2, ZERO, data)
        self.assertTrue(r["ok"])
        self.assertEqual(r["tenant"], {"k": "v", "n": [1, 2]})

    # --- failure classification ---

    def test_failure_result_shape(self):
        self.assertEqual(
            self.other.verify_tenant_range_bytes("a", 3, 5, ZERO, b""),
            {"ok": False, "at": 3, "tenant": "a", "reason": "missing",
             "start_seq": 3, "end_seq": 5})

    def test_missing_classification(self):
        cases = [
            b"",                        # empty fragment
            b"{not json\n",             # unparseable
            b"\n",                      # blank line
            b"[1,2]\n",                 # non-object
            b'{"seq":3,"event":{},"prev":"' + b"8" * 64
            + b'","hash":"' + b"x" * 64 + b'"}\n',          # no tenant key
            b'{"tenant":"t","seq":3,"event":{},"prev":"' + b"8" * 64
            + b'"}\n',                                       # missing hash
            b'{"tenant":"t","seq":3,"event":{},"prev":"' + b"8" * 64
            + b'","hash":"' + b"x" * 64 + b'","extra":1}\n',  # extra field
            b'{"tenant":"t","tenant":"t","seq":3,"event":{},"prev":"'
            + b"8" * 64 + b'","hash":"' + b"x" * 64 + b'"}\n',  # dup key
            b'{"tenant":"t","seq":1e999,"event":{},"prev":"' + b"8" * 64
            + b'","hash":"' + b"x" * 64 + b'"}\n',          # non-standard
            b"\xff\n",                  # illegal utf-8
        ]
        for raw in cases:
            r = self.other.verify_tenant_range_bytes("t", 3, 3, "8" * 64, raw)
            self.assertFalse(r["ok"], raw)
            self.assertEqual(r["reason"], "missing", raw)
            chunks = [raw[:2], b"", raw[2:]]
            c = self.other.verify_tenant_range_chunks("t", 3, 3, "8" * 64,
                                                      chunks)
            self.assertEqual(c, r, raw)

    def test_missing_at_locations(self):
        # empty fragment: the first expected seq
        r = self.other.verify_tenant_range_bytes("t", 3, 3, "8" * 64, b"")
        self.assertEqual(r["at"], 3)
        # unparseable line: the physical line number
        r = self.other.verify_tenant_range_bytes("t", 3, 3, "8" * 64,
                                                 b"{oops\n")
        self.assertEqual(r["at"], 1)
        # field-defective line carrying its own plain-int seq
        row = b'{"tenant":"t","seq":7,"event":{},"prev":"' + b"8" * 64 + b'"}\n'
        r = self.other.verify_tenant_range_bytes("t", 3, 7, "8" * 64, row)
        self.assertEqual((r["at"], r["reason"]), (7, "missing"))
        # illegal utf-8 on the second physical line
        good = self.valid_row("t", 3, "8" * 64)
        r = self.other.verify_tenant_range_bytes("t", 3, 4, "8" * 64,
                                                 good + b"\xff\n")
        self.assertEqual((r["at"], r["reason"]), (2, "missing"))
        # truncated final record (LF-less tail)
        r = self.other.verify_tenant_range_bytes(
            "t", 3, 4, "8" * 64, good + b'{"tenant":"t","seq":4')
        self.assertEqual((r["at"], r["reason"]), (2, "missing"))

    def test_declared_end_seq_must_be_covered_exactly(self):
        self.seed()
        prev = self.prev_for("a", 3)
        # short fragment: records stop before the declared end_seq
        short = self.chain.export_tenant_range("a", 3, 4)
        r = self.other.verify_tenant_range_bytes("a", 3, 5, prev, short)
        self.assertEqual((r["ok"], r["at"], r["reason"]),
                         (False, 5, "missing"))
        # long fragment: a record beyond the declared end_seq is sequence
        long = self.chain.export_tenant_range("a", 3, 6)
        r = self.other.verify_tenant_range_bytes("a", 3, 5, prev, long)
        self.assertEqual((r["ok"], r["at"], r["reason"]),
                         (False, 6, "sequence"))
        # exact coverage succeeds
        exact = self.chain.export_tenant_range("a", 3, 5)
        self.assertTrue(
            self.other.verify_tenant_range_bytes("a", 3, 5, prev,
                                                 exact)["ok"])

    def test_sequence_classification(self):
        # wrong starting seq
        raw = self.valid_row("t", 4, "8" * 64)
        r = self.other.verify_tenant_range_bytes("t", 3, 4, "8" * 64, raw)
        self.assertEqual((r["ok"], r["at"], r["reason"]),
                         (False, 3, "sequence"))
        # gap after a good record
        raw = self.valid_row("t", 3, "8" * 64) + self.valid_row("t", 5,
                                                                "8" * 64)
        r = self.other.verify_tenant_range_bytes("t", 3, 5, "8" * 64, raw)
        self.assertEqual((r["ok"], r["at"], r["reason"]),
                         (False, 4, "sequence"))
        # non-integer seq spellings are sequence, never renumbered
        for bad_seq in (b"3.0", b"1e0", b"true", b'"3"', b"null", b"[3]",
                        b'{"s":3}'):
            raw = (b'{"tenant":"t","seq":' + bad_seq
                   + b',"event":{},"prev":"' + b"8" * 64
                   + b'","hash":"' + b"x" * 64 + b'"}\n')
            r = self.other.verify_tenant_range_bytes("t", 3, 3, "8" * 64, raw)
            self.assertEqual((r["ok"], r["at"], r["reason"]),
                             (False, 3, "sequence"), bad_seq)

    def test_digest_classification(self):
        # the first record's prev must equal expected_prev
        raw = self.valid_row("t", 3, "9" * 64)
        r = self.other.verify_tenant_range_bytes("t", 3, 3, "8" * 64, raw)
        self.assertEqual((r["ok"], r["at"], r["reason"]),
                         (False, 3, "digest"))
        # mid-chain prev mismatch
        raw = self.valid_row("t", 3, "8" * 64) + self.valid_row("t", 4,
                                                                "9" * 64)
        r = self.other.verify_tenant_range_bytes("t", 3, 4, "8" * 64, raw)
        self.assertEqual((r["ok"], r["at"], r["reason"]),
                         (False, 4, "digest"))
        # tampered hash
        row = {"tenant": "t", "seq": 3, "event": {}, "prev": "8" * 64,
               "hash": "7" * 64}
        r = self.other.verify_tenant_range_bytes("t", 3, 3, "8" * 64,
                                                 record_bytes(row))
        self.assertEqual((r["ok"], r["at"], r["reason"]),
                         (False, 3, "digest"))

    # --- foreign tenant contract ---

    def test_foreign_tenant_record_is_value_error(self):
        raw = self.valid_row("a", 1) + self.valid_row("b", 1)
        with self.assertRaises(ValueError):
            self.other.verify_tenant_range_bytes("a", 1, 2, ZERO, raw)
        with self.assertRaises(ValueError):
            self.other.verify_tenant_range_chunks("a", 1, 2, ZERO,
                                                  [raw[:10], raw[10:]])

    def test_distinct_json_identities_are_foreign(self):
        self.chain.append(1, {})
        data = self.chain.export_tenant_range(1, 1, 1)
        with self.assertRaises(ValueError):
            self.other.verify_tenant_range_bytes("1", 1, 1, ZERO, data)

    def test_first_problem_in_physical_order_decides(self):
        # an earlier chain defect beats a later foreign record
        broken = {"tenant": "a", "seq": 1, "event": {}, "prev": "9" * 64}
        broken["hash"] = AuditChain._hash(broken)
        raw = record_bytes(broken) + self.valid_row("b", 1)
        r = self.other.verify_tenant_range_bytes("a", 1, 2, ZERO, raw)
        self.assertEqual((r["ok"], r["at"], r["reason"]),
                         (False, 1, "digest"))
        # an earlier foreign record beats a later chain defect
        raw = self.valid_row("b", 1) + record_bytes(broken)
        with self.assertRaises(ValueError):
            self.other.verify_tenant_range_bytes("a", 1, 2, ZERO, raw)

    # --- parameter boundary (ValueError before any data is read) ---

    def test_boundary_value_errors(self):
        good = self.valid_row("t", 1)
        cyclic = {}
        cyclic["self"] = cyclic
        for bad_tenant in (float("nan"), float("inf"), -float("inf"),
                           {"k": float("nan")}, {1: "x"}, cyclic, object(),
                           b"x"):
            with self.assertRaises(ValueError):
                self.other.verify_tenant_range_bytes(bad_tenant, 1, 1, ZERO,
                                                     good)
        for bad_start in (True, False, 0, -1, 1.5, "1", None, object()):
            with self.assertRaises(ValueError):
                self.other.verify_tenant_range_bytes("t", bad_start, 1, ZERO,
                                                     good)
        for bad_end in (True, False, 0.5, "2", object()):
            with self.assertRaises(ValueError):
                self.other.verify_tenant_range_bytes("t", 1, bad_end, ZERO,
                                                     good)
        with self.assertRaises(ValueError):  # end < start
            self.other.verify_tenant_range_bytes("t", 3, 2, ZERO, good)
        for bad_prev in ("0" * 63, "0" * 65, "A" * 64, "g" * 64, 0, None,
                         b"0" * 64, ["0" * 64]):
            with self.assertRaises(ValueError):
                self.other.verify_tenant_range_bytes("t", 2, 2, bad_prev,
                                                     good)
        # start_seq 1 only accepts the ZERO anchor
        with self.assertRaises(ValueError):
            self.other.verify_tenant_range_bytes("t", 1, 1, "1" * 64, good)
        # data must be exactly bytes
        for bad_data in ("x", bytearray(good), None, 1, [good], {"d": good}):
            with self.assertRaises(ValueError):
                self.other.verify_tenant_range_bytes("t", 1, 1, ZERO,
                                                     bad_data)

    def test_zero_prev_allowed_beyond_seq_one(self):
        # the ZERO anchor is only mandated at seq 1; elsewhere the content
        # decides (a record at seq 2 whose prev is ZERO verifies)
        row = self.valid_row("t", 2, ZERO)
        r = self.other.verify_tenant_range_bytes("t", 2, 2, ZERO, row)
        self.assertTrue(r["ok"])
        row = self.valid_row("t", 2, "9" * 64)
        r = self.other.verify_tenant_range_bytes("t", 2, 2, ZERO, row)
        self.assertEqual((r["ok"], r["reason"]), (False, "digest"))

    def test_parameter_boundary_precedes_any_chunk_pull(self):
        pulled = []

        def gen():
            pulled.append(1)
            yield b""

        good_prev = "8" * 64
        bad_calls = [
            (float("nan"), 1, 1, ZERO),
            ("t", True, 1, ZERO),
            ("t", 0, 1, ZERO),
            ("t", 1, 0, ZERO),
            ("t", 1, 1, "x"),
            ("t", 1, 1, "1" * 64),   # start_seq 1 requires ZERO
        ]
        for args in bad_calls:
            with self.assertRaises(ValueError):
                self.other.verify_tenant_range_chunks(*args, gen())
        self.assertEqual(pulled, [])
        # a legal call does consume the stream
        self.other.verify_tenant_range_chunks("t", 3, 3, good_prev, gen())
        self.assertEqual(pulled, [1])

    # --- chunks container boundary ---

    def test_chunks_container_boundary(self):
        good = self.valid_row("t", 1)
        for bad in (good, bytearray(good), b""):
            with self.assertRaises(ValueError):
                self.other.verify_tenant_range_chunks("t", 1, 1, ZERO, bad)
        for bad in (1, None, object()):
            with self.assertRaises(ValueError):
                self.other.verify_tenant_range_chunks("t", 1, 1, ZERO, bad)
        for bad in ([b"", "x"], [b"", 1], [b"", None], [b"", bytearray(b"")],
                    [b"", [b""]]):
            with self.assertRaises(ValueError):
                self.other.verify_tenant_range_chunks("t", 1, 1, ZERO, bad)

    def test_bad_element_stops_consumption(self):
        pulled = []

        def gen():
            pulled.append(1)
            yield b""
            pulled.append(2)
            yield "not bytes"
            pulled.append(3)  # must never be reached
            yield b""

        with self.assertRaises(ValueError):
            self.other.verify_tenant_range_chunks("t", 1, 1, ZERO, gen())
        self.assertEqual(pulled, [1, 2])

    # --- bytes/chunks equivalence and single forward consumption ---

    def test_chunks_match_bytes_field_for_field(self):
        self.seed()
        scenarios = []
        for (s, e) in [(1, 6), (3, 6), (2, 2), (4, None)]:
            data = self.chain.export_tenant_range("a", s, e)
            scenarios.append(("a", s, e, self.prev_for("a", s), data))
        good3 = self.valid_row("t", 3, "8" * 64)
        scenarios += [
            ("t", 3, 3, "8" * 64, b""),
            ("t", 3, 3, "8" * 64, b"{oops\n"),
            ("t", 3, 3, "8" * 64, b"\xff\n"),
            ("t", 3, 4, "8" * 64, good3 + b"\xff\n"),
            ("t", 3, 3, "8" * 64, self.valid_row("t", 4, "8" * 64)),
            ("t", 3, 3, "8" * 64, self.valid_row("t", 3, "9" * 64)),
            ("t", 3, 5, "8" * 64, good3),          # declared tail missing
            ("t", 3, 3, "8" * 64,                  # overshoot
             good3 + self.valid_row("t", 4, AuditChain._hash(
                 {"tenant": "t", "seq": 3, "event": {}, "prev": "8" * 64}))),
        ]
        for (tenant, s, e, prev, data) in scenarios:
            expected = self.other.verify_tenant_range_bytes(tenant, s, e,
                                                            prev, data)
            chunkings = [
                [data],
                [data[:1], data[1:]],
                [data[i:i + 1] for i in range(len(data))],
                [data[:5], b"", data[5:]],
                [b"", data, b""],
                list(data[i:i + 7] for i in range(0, len(data), 7)),
            ]
            for chunks in chunkings:
                self.assertEqual(
                    self.other.verify_tenant_range_chunks(tenant, s, e, prev,
                                                          chunks),
                    expected, (data, chunks))

    def test_chunk_may_split_multibyte_utf8(self):
        self.chain.append("t", {"msg": "héllo→世界"})
        self.chain.append("t", {"msg": "✓" * 40})
        data = self.chain.export_tenant_range("t", 1, 2)
        last = self.chain.read_tenant("t")[1]["hash"]
        for cut in range(len(data) + 1):
            r = self.other.verify_tenant_range_chunks(
                "t", 1, 2, ZERO, [data[:cut], data[cut:]])
            self.assertEqual(
                r, {"ok": True, "tenant": "t", "start_seq": 1, "end_seq": 2,
                    "count": 2, "hash": last}, cut)

    def test_first_error_stops_pulling_chunks(self):
        pulled = []

        def gen():
            pulled.append(1)
            yield self.valid_row("t", 3, "9" * 64)  # digest defect, complete
            pulled.append(2)                        # must never be reached
            yield self.valid_row("t", 4, "9" * 64)

        r = self.other.verify_tenant_range_chunks("t", 3, 4, "8" * 64, gen())
        self.assertEqual((r["ok"], r["reason"]), (False, "digest"))
        self.assertEqual(pulled, [1])

        # a foreign-tenant record also stops the pull
        pulled.clear()

        def gen2():
            pulled.append(1)
            yield self.valid_row("b", 1)
            pulled.append(2)                        # must never be reached
            yield b""

        with self.assertRaises(ValueError):
            self.other.verify_tenant_range_chunks("a", 1, 1, ZERO, gen2())
        self.assertEqual(pulled, [1])

        # a valid fragment is consumed to the end
        pulled.clear()

        def gen3():
            pulled.append(1)
            yield self.valid_row("t", 3, "8" * 64)
            pulled.append(2)
            yield b""

        r = self.other.verify_tenant_range_chunks("t", 3, 3, "8" * 64, gen3())
        self.assertTrue(r["ok"])
        self.assertEqual(pulled, [1, 2])

    def test_accepts_any_iterable_container(self):
        self.seed()
        data = self.chain.export_tenant_range("a", 2, 5)
        prev = self.prev_for("a", 2)
        halves = [data[:len(data) // 2], data[len(data) // 2:]]
        for container in (halves, tuple(halves), iter(halves),
                          (c for c in halves)):
            self.assertTrue(
                self.other.verify_tenant_range_chunks("a", 2, 5, prev,
                                                      container)["ok"])

    # --- offline purity ---

    def test_never_reads_creates_or_modifies_constructor_path(self):
        phantom = self.path.with_name("phantom.jsonl")
        chain = AuditChain(phantom)
        r = chain.verify_tenant_range_bytes("t", 1, 1, ZERO, b"")
        self.assertEqual(r["reason"], "missing")
        r = chain.verify_tenant_range_chunks("t", 1, 1, ZERO, [])
        self.assertEqual(r["reason"], "missing")
        self.assertFalse(phantom.exists())
        self.seed()
        before = self.path.read_bytes()
        data = self.chain.export_tenant_range("a", 2, 4)
        prev = self.prev_for("a", 2)
        self.assertTrue(
            chain.verify_tenant_range_bytes("a", 2, 4, prev, data)["ok"])
        self.assertTrue(
            chain.verify_tenant_range_chunks("a", 2, 4, prev,
                                             [data[:7], data[7:]])["ok"])
        self.assertEqual(self.path.read_bytes(), before)
        self.assertFalse(phantom.exists())

    def test_does_not_mutate_inputs(self):
        self.seed()
        data = self.chain.export_tenant_range("a", 2, 4)
        prev = self.prev_for("a", 2)
        chunks = [data[:5], data[5:]]
        self.other.verify_tenant_range_bytes("a", 2, 4, prev, data)
        self.other.verify_tenant_range_chunks("a", 2, 4, prev, chunks)
        self.assertEqual(b"".join(chunks), data)


if __name__ == "__main__":
    unittest.main()
