import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app import AuditChain

ZERO = "0" * 64


def record_bytes(item):
    return (json.dumps(item, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


class VerifyBytesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.path = Path(self.tmp.name) / "audit.jsonl"
        self.chain = AuditChain(self.path)
        self.other = AuditChain(self.path.with_name("never-touched.jsonl"))

    def tearDown(self):
        self.tmp.cleanup()

    def seed(self):
        self.chain.append("a", {"i": 1})
        self.chain.append("b", {"i": 1})
        self.chain.append("a", {"i": 2})
        self.chain.append("a", {"i": 3})

    def snapshot(self):
        return self.path.read_bytes()

    # --- never touches the configured path ---

    def test_never_reads_creates_or_modifies_constructor_path(self):
        # point at a nonexistent path, then only ever call verify_bytes
        chain = AuditChain(self.path.with_name("phantom.jsonl"))
        self.assertEqual(chain.verify_bytes(b"", "t"), {"ok": True, "count": 0})
        self.assertFalse(self.path.with_name("phantom.jsonl").exists())
        self.seed()
        before = self.snapshot()
        self.assertTrue(chain.verify_bytes(before, "a")["ok"])
        self.assertEqual(self.snapshot(), before)
        self.assertFalse(self.path.with_name("phantom.jsonl").exists())

    def test_does_not_mutate_data(self):
        self.seed()
        data = self.snapshot()
        self.other.verify_bytes(data, "a")
        self.other.verify_all_bytes(data)
        self.assertEqual(data, self.snapshot())

    # --- equivalence with file-backed verify ---

    def test_matches_verify_on_single_tenant_export(self):
        self.seed()
        exported = self.chain.export_tenant("a")
        self.assertEqual(self.other.verify_bytes(exported, "a"),
                         self.chain.verify("a"))
        self.assertEqual(self.other.verify_bytes(exported, "a", 3),
                         {"ok": True, "count": 3})

    def test_matches_verify_on_interleaved_full_log(self):
        self.seed()
        data = self.snapshot()
        for tenant, count in (("a", 3), ("b", 1)):
            self.assertEqual(self.other.verify_bytes(data, tenant),
                             self.chain.verify(tenant))
            self.assertEqual(self.other.verify_bytes(data, tenant),
                             {"ok": True, "count": count})
        self.assertEqual(self.other.verify_bytes(data, "zzz"),
                         {"ok": True, "count": 0})

    def test_matches_verify_expected_count_short_and_over(self):
        self.seed()
        data = self.chain.export_tenant("a")
        r = self.other.verify_bytes(data, "a", 4)
        self.assertEqual((r["ok"], r["at"], r["reason"]), (False, 4, "missing"))
        r = self.other.verify_bytes(data, "a", 2)
        self.assertEqual((r["ok"], r["at"], r["reason"]), (False, 3, "sequence"))
        r = self.other.verify_bytes(data, "a", 0)
        self.assertEqual((r["ok"], r["at"], r["reason"]), (False, 1, "sequence"))

    def test_empty_bytes_is_successful_empty_history(self):
        self.assertEqual(self.other.verify_bytes(b"", "t"), {"ok": True, "count": 0})
        self.assertEqual(self.other.verify_bytes(b"", "t", 0), {"ok": True, "count": 0})
        self.assertEqual(self.other.verify_all_bytes(b""),
                         {"ok": True, "tenants": []})

    def test_distinct_json_identities_stay_partitioned(self):
        for t in (1, 1.0, True, "1"):
            self.chain.append(t, {})
            self.chain.append(t, {})
        data = self.snapshot()
        for t in (1, 1.0, True, "1"):
            self.assertEqual(self.other.verify_bytes(data, t),
                             {"ok": True, "count": 2})

    def test_object_tenant_key_order_normalized(self):
        self.chain.append({"a": 1, "b": 2}, {})
        self.chain.append({"b": 2, "a": 1}, {})
        data = self.snapshot()
        self.assertEqual(
            self.other.verify_bytes(data, {"b": 2, "a": 1}),
            {"ok": True, "count": 2})

    # --- verdicts mirror verify exactly ---

    def write_raw(self, *chunks):
        self.path.write_bytes(b"".join(chunks))

    def valid_row(self, tenant="t", seq=1, prev=ZERO, event=None):
        row = {"tenant": tenant, "seq": seq, "event": event or {}, "prev": prev}
        row["hash"] = AuditChain._hash(row)
        return record_bytes(row)

    def test_malformed_lines_are_missing_at_physical_line(self):
        for raw, at in ((b"\n", 1), (b"   \n", 1), (b"[1,2]\n", 1),
                        (b"{not json\n", 1),
                        (self.valid_row("x") + b"{not json\n", 2)):
            r = self.other.verify_bytes(raw, "t")
            self.assertEqual((r["ok"], r["at"], r["reason"]),
                             (False, at, "missing"), raw)

    def test_missing_field_is_missing(self):
        row = {"tenant": "t", "seq": 1, "event": {}, "prev": ZERO}
        raw = (json.dumps(row) + "\n").encode()
        r = self.other.verify_bytes(raw, "t")
        self.assertEqual((r["at"], r["reason"]), (1, "missing"))

    def test_duplicate_key_non_standard_number_are_missing(self):
        for raw in (b'{"tenant":"t","tenant":"t","seq":1,"event":{},'
                    b'"prev":"' + ZERO.encode() + b'","hash":"x"}\n',
                    b'{"tenant":"t","seq":1,"event":{},"prev":"' + ZERO.encode()
                    + b'","hash":"x","x":NaN}\n',
                    b'{"tenant":"t","seq":1e999,"event":{},"prev":"'
                    + ZERO.encode() + b'","hash":"x"}\n'):
            r = self.other.verify_bytes(raw, "t")
            self.assertEqual((r["ok"], r["at"], r["reason"]),
                             (False, 1, "missing"), raw)

    def test_illegal_utf8_is_missing_at_physical_line(self):
        cases = [
            (b"\xff", 1),
            (self.valid_row() + b'{"tenant":\xff}', 2),
            (self.valid_row("x") + self.valid_row("y") + b"abc\xc2\n", 3),
        ]
        for raw, at in cases:
            r = self.other.verify_bytes(raw, "t")
            self.assertEqual((r["ok"], r["at"], r["reason"]),
                             (False, at, "missing"), raw)

    def test_sequence_and_digest_classification(self):
        good = self.valid_row("t", 1)
        gap_row = {"tenant": "t", "seq": 3, "event": {}, "prev": "x" * 64}
        gap_row["hash"] = AuditChain._hash(gap_row)
        r = self.other.verify_bytes(good + record_bytes(gap_row), "t")
        self.assertEqual((r["at"], r["reason"]), (2, "sequence"))

        bad_prev = {"tenant": "t", "seq": 2, "event": {}, "prev": "9" * 64}
        bad_prev["hash"] = AuditChain._hash(bad_prev)
        r = self.other.verify_bytes(good + record_bytes(bad_prev), "t")
        self.assertEqual((r["at"], r["reason"]), (2, "digest"))

    def test_other_tenant_bad_line_is_found_and_earlier_error_wins(self):
        # parse-broken line of another tenant is fatal to verifying t
        r = self.other.verify_bytes(b'{"tenant":"x", broken\n', "t")
        self.assertEqual((r["ok"], r["at"], r["reason"]), (False, 1, "missing"))
        # digest error for t on line 1 beats bad bytes on line 2
        tampered = {"tenant": "t", "seq": 1, "event": {"z": 9}, "prev": ZERO}
        tampered["hash"] = AuditChain._hash(
            {"tenant": "t", "seq": 1, "event": {}, "prev": ZERO})
        raw = record_bytes(tampered) + b"\xff\n"
        self.assertEqual(
            (self.other.verify_bytes(raw, "t")["at"],
             self.other.verify_bytes(raw, "t")["reason"]),
            (1, "digest"))

    # --- verify_all_bytes equivalence ---

    def test_verify_all_bytes_matches_verify_all(self):
        self.seed()
        data = self.snapshot()
        self.assertEqual(self.other.verify_all_bytes(data),
                         self.chain.verify_all())
        self.assertEqual(self.other.verify_all_bytes(data), {
            "ok": True,
            "tenants": [{"tenant": "a", "count": 3},
                        {"tenant": "b", "count": 1}],
        })

    def test_verify_all_bytes_first_error_structure(self):
        good = {"tenant": "a", "seq": 1, "event": {}, "prev": ZERO}
        good["hash"] = AuditChain._hash(good)
        bad = {"tenant": "b", "seq": 1, "event": {}, "prev": "f" * 64}
        bad["hash"] = AuditChain._hash(bad)
        raw = record_bytes(good) + record_bytes(bad)
        self.assertEqual(self.other.verify_all_bytes(raw),
                         {"ok": False, "at": 2, "tenant": "b", "reason": "digest"})
        self.assertEqual(self.other.verify_all_bytes(b"{oops\n"),
                         {"ok": False, "at": 1, "tenant": None, "reason": "missing"})
        self.assertEqual(self.other.verify_all_bytes(b"\xff"),
                         {"ok": False, "at": 1, "tenant": None, "reason": "missing"})

    # --- export written elsewhere agrees with in-memory verification ---

    def test_exported_bytes_in_memory_and_on_disk_agree(self):
        for i in range(7):
            self.chain.append("t", {"i": i})
            self.chain.append("other", {"i": i})
        exported = self.chain.export_tenant("t")
        mem = self.other.verify_bytes(exported, "t")
        out = self.path.with_name("written-back.jsonl")
        out.write_bytes(exported)
        disk = AuditChain(out).verify("t")
        self.assertEqual(mem, disk)
        self.assertEqual(mem, {"ok": True, "count": 7})
        self.assertEqual(self.other.verify_all_bytes(exported),
                         AuditChain(out).verify_all())

    # --- argument boundary: ValueError, no leaked parse exceptions ---

    def test_data_must_be_bytes(self):
        for bad in ('{"x":1}', bytearray(b""), 1, None, [b""]):
            with self.assertRaises(ValueError):
                self.other.verify_bytes(bad, "t")
            with self.assertRaises(ValueError):
                self.other.verify_all_bytes(bad)

    def test_tenant_boundary_raises_value_error(self):
        for bad in (float("nan"), float("inf"),
                    {"k": float("nan")}, {1: "x"}, object(), b"x",
                    {1, 2}):
            with self.assertRaises(ValueError):
                self.other.verify_bytes(b"", bad)

        cyc = []
        cyc.append(cyc)
        with self.assertRaises(ValueError):
            self.other.verify_bytes(b"", cyc)

    def test_expected_count_boundary_raises_value_error(self):
        for bad in (-1, 1.0, True, False, "1", 1.0, object()):
            with self.assertRaises(ValueError):
                self.other.verify_bytes(b"", "t", bad)
        # valid values do not raise
        self.assertEqual(self.other.verify_bytes(b"", "t", None),
                         {"ok": True, "count": 0})
        self.assertEqual(self.other.verify_bytes(b"", "t", 0),
                         {"ok": True, "count": 0})

    def test_value_error_beats_corrupt_snapshot(self):
        # bad inputs raise even though the buffer itself is corrupt
        with self.assertRaises(ValueError):
            self.other.verify_bytes(b"\xff", "t", -1)
        with self.assertRaises(ValueError):
            self.other.verify_bytes(b"\xff", float("nan"))
        with self.assertRaises(ValueError):
            self.other.verify_bytes("not bytes", "t")


if __name__ == "__main__":
    unittest.main()
