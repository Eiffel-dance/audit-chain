import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app import AuditChain

ZERO = "0" * 64


def record(tenant, seq, prev, event=None):
    row = {"tenant": tenant, "seq": seq, "event": {} if event is None else event,
           "prev": prev}
    row["hash"] = AuditChain._hash(row)
    return (json.dumps(row, sort_keys=True) + "\n").encode("utf-8")


class VerifyBytesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.path = Path(self.tmp.name) / "audit.jsonl"
        self.chain = AuditChain(self.path)
        self.offline = AuditChain(Path(self.tmp.name) / "never-created.jsonl")

    def tearDown(self):
        self.tmp.cleanup()

    def write_bytes(self, data):
        self.path.write_bytes(data)

    # --- basic success, equivalence with file-backed verify ---

    def test_empty_bytes_is_empty_history_and_creates_nothing(self):
        target = Path(self.tmp.name) / "absent.jsonl"
        chain = AuditChain(target)
        self.assertEqual(chain.verify_bytes(b"", "t"), {"ok": True, "count": 0})
        self.assertEqual(chain.verify_bytes(b"", "t", 0), {"ok": True, "count": 0})
        self.assertFalse(target.exists())

    def test_agrees_with_verify_on_the_same_snapshot(self):
        for i in range(4):
            self.chain.append("a", {"i": i})
        self.chain.append("b", {})
        data = self.path.read_bytes()
        self.assertEqual(self.offline.verify_bytes(data, "a"),
                         self.chain.verify("a"))
        self.assertEqual(self.offline.verify_bytes(data, "a", 4),
                         self.chain.verify("a", 4))
        self.assertEqual(self.offline.verify_bytes(data, "b"),
                         self.chain.verify("b"))
        self.assertEqual(self.offline.verify_bytes(data, "absent"),
                         {"ok": True, "count": 0})

    def test_exported_single_tenant_bytes_verify_in_memory(self):
        items = [self.chain.append("t", {"i": i}) for i in range(5)]
        exported = self.chain.export_tenant("t")
        self.assertEqual(self.offline.verify_bytes(exported, "t"),
                         {"ok": True, "count": 5})
        # written back under another AuditChain, file and memory verdicts agree
        out = Path(self.tmp.name) / "out.jsonl"
        out.write_bytes(exported)
        relocated = AuditChain(out)
        self.assertEqual(relocated.verify("t"),
                         self.offline.verify_bytes(exported, "t"))
        self.assertEqual(relocated.verify_all(),
                         self.offline.verify_all_bytes(exported))
        rows = [json.loads(l) for l in exported.decode("utf-8").splitlines()]
        self.assertEqual([r["hash"] for r in rows], [i["hash"] for i in items])

    def test_interleaved_full_log_validates_per_tenant(self):
        a1 = self.chain.append("a", {})
        self.chain.append("b", {})
        a2 = self.chain.append("a", {})
        data = self.path.read_bytes()
        self.assertEqual(self.offline.verify_bytes(data, "a"),
                         {"ok": True, "count": 2})
        self.assertEqual(self.offline.verify_bytes(data, "b"),
                         {"ok": True, "count": 1})
        rows = [json.loads(l) for l in data.decode("utf-8").splitlines()]
        self.assertEqual(rows[2]["prev"], a1["hash"])
        self.assertEqual(a2["prev"], a1["hash"])

    def test_expected_count_short_and_over(self):
        self.chain.append("t", {})
        self.chain.append("t", {})
        data = self.path.read_bytes()
        self.assertEqual(self.offline.verify_bytes(data, "t", 2),
                         {"ok": True, "count": 2})
        self.assertEqual(self.offline.verify_bytes(data, "t", 3),
                         {"ok": False, "at": 3, "reason": "missing"})
        self.assertEqual(self.offline.verify_bytes(data, "t", 1),
                         {"ok": False, "at": 2, "reason": "sequence"})
        self.assertEqual(self.offline.verify_bytes(b"", "t", 1),
                         {"ok": False, "at": 1, "reason": "missing"})

    def test_distinct_json_tenant_identities_stay_partitioned(self):
        for t in (1, 1.0, True, "1"):
            self.chain.append(t, {})
        data = self.path.read_bytes()
        for t in (1, 1.0, True, "1"):
            self.assertEqual(self.offline.verify_bytes(data, t),
                             {"ok": True, "count": 1})
        self.assertEqual(self.offline.verify_bytes(data, 1, 2),
                         {"ok": False, "at": 2, "reason": "missing"})

    def test_object_tenant_key_order_normalized(self):
        self.chain.append({"a": 1, "b": 2}, {})
        data = self.path.read_bytes()
        self.assertEqual(
            self.offline.verify_bytes(data, {"b": 2, "a": 1}),
            {"ok": True, "count": 1})

    # --- corruption classification ---

    def test_malformed_lines_are_missing_at_physical_line(self):
        cases = [
            b"\n",
            b"  \n",
            b"[1,2,3]\n",
            b'{"seq":1}\n',
            b'{"tenant":"t","seq":1,"event":{},"prev":'
            + json.dumps(ZERO).encode() + b'}\n',
            b'{"tenant":"t","seq":1,"event":{},"prev":'
            + json.dumps(ZERO).encode()
            + b',"hash":"a","hash":"b"}\n',
            b'{"tenant":"t","seq":NaN,"event":{},"prev":'
            + json.dumps(ZERO).encode() + b',"hash":"a"}\n',
            b'{"tenant":1e999}\n',
            b"{not json\n",
            b"\xff",
        ]
        for bad in cases:
            self.assertEqual(
                self.offline.verify_bytes(bad, "t"),
                {"ok": False, "at": 1, "reason": "missing"}, bad)

    def test_illegal_utf8_line_number_after_valid_records(self):
        data = record("x", 1, ZERO) + b'{"tenant":\xff}'
        self.assertEqual(self.offline.verify_bytes(data, "t"),
                         {"ok": False, "at": 2, "reason": "missing"})
        self.assertEqual(self.offline.verify_bytes(data, "x"),
                         {"ok": False, "at": 2, "reason": "missing"})

    def test_sequence_and_digest_classification(self):
        seq_bad = record("t", 2, ZERO)
        self.assertEqual(self.offline.verify_bytes(seq_bad, "t"),
                         {"ok": False, "at": 1, "reason": "sequence"})
        dig_bad = record("t", 1, "f" * 64)
        self.assertEqual(self.offline.verify_bytes(dig_bad, "t"),
                         {"ok": False, "at": 1, "reason": "digest"})
        tampered = json.loads(record("t", 1, ZERO).decode())
        tampered["event"] = {"z": 9}
        self.write_bytes((json.dumps(tampered, sort_keys=True) + "\n").encode())
        self.assertEqual(
            self.offline.verify_bytes(self.path.read_bytes(), "t"),
            {"ok": False, "at": 1, "reason": "digest"})

    def test_other_tenant_bad_lines_are_still_found(self):
        good = record("t", 1, ZERO)
        # an unparseable other-tenant line must fail even a clean t chain
        self.assertEqual(
            self.offline.verify_bytes(good + b"{oops\n", "t"),
            {"ok": False, "at": 2, "reason": "missing"})

    def test_earlier_error_takes_priority(self):
        good = json.loads(record("a", 1, ZERO).decode())
        tampered = dict(good)
        tampered["event"] = {"z": 9}
        data = ((json.dumps(tampered, sort_keys=True) + "\n").encode()
                + b"{oops\n")
        self.assertEqual(self.offline.verify_bytes(data, "a"),
                         {"ok": False, "at": 1, "reason": "digest"})
        seq_bad = record("t", 2, ZERO)
        self.assertEqual(
            self.offline.verify_bytes(seq_bad + b"\xff", "t"),
            {"ok": False, "at": 1, "reason": "sequence"})

    # --- verify_all_bytes ---

    def test_verify_all_bytes_empty_and_order(self):
        self.assertEqual(self.offline.verify_all_bytes(b""),
                         {"ok": True, "tenants": []})
        self.chain.append("b", {})
        self.chain.append("a", {})
        self.chain.append("b", {})
        self.chain.append(1, {})
        data = self.path.read_bytes()
        self.assertEqual(
            self.offline.verify_all_bytes(data),
            {"ok": True, "tenants": [
                {"tenant": "b", "count": 2},
                {"tenant": "a", "count": 1},
                {"tenant": 1, "count": 1},
            ]})
        self.assertEqual(self.offline.verify_all_bytes(data),
                         self.chain.verify_all())

    def test_verify_all_bytes_first_error_structure(self):
        good = record("a", 1, ZERO)
        bad = record("b", 1, "f" * 64)
        self.assertEqual(
            self.offline.verify_all_bytes(good + bad),
            {"ok": False, "at": 2, "tenant": "b", "reason": "digest"})
        self.assertEqual(
            self.offline.verify_all_bytes(b"{oops\n"),
            {"ok": False, "at": 1, "tenant": None, "reason": "missing"})
        missing_field = (json.dumps(
            {"tenant": "t", "seq": 1, "event": {}, "prev": ZERO}) + "\n"
        ).encode()
        self.assertEqual(
            self.offline.verify_all_bytes(missing_field),
            {"ok": False, "at": 1, "tenant": "t", "reason": "missing"})

    # --- filesystem isolation ---

    def test_constructor_path_is_never_read_or_modified(self):
        # a real, different history sits at the path; passed bytes must win
        self.chain.append("file-tenant", {})
        before = self.path.read_bytes()
        verdict_from_bytes = {"ok": False, "at": 1, "reason": "missing"}
        self.assertEqual(self.chain.verify_bytes(b"{oops\n", "file-tenant"),
                         verdict_from_bytes)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(self.chain.verify_all_bytes(b"{oops\n"),
                         {"ok": False, "at": 1, "tenant": None,
                          "reason": "missing"})
        self.assertEqual(self.path.read_bytes(), before)

    def test_input_bytes_are_not_modified(self):
        self.chain.append("t", {})
        data = self.path.read_bytes()
        snapshot = bytes(data)
        self.offline.verify_bytes(data, "t")
        self.offline.verify_bytes(data, "t", 5)
        self.offline.verify_all_bytes(data)
        self.assertEqual(data, snapshot)

    # --- ValueError boundary ---

    def test_data_must_be_bytes(self):
        for bad in ("", "x", bytearray(b"x"), None, 1, [], {}):
            with self.assertRaises(ValueError):
                self.offline.verify_bytes(bad, "t")
            with self.assertRaises(ValueError):
                self.offline.verify_all_bytes(bad)

    def test_tenant_must_cross_standard_json_boundary(self):
        cyc = []
        cyc.append(cyc)
        d = {}
        d["self"] = d
        for bad in (float("nan"), float("inf"), float("-inf"),
                    [float("nan")], {1: "x"}, object(), b"x", {1, 2},
                    ("a", 1), cyc, d):
            with self.assertRaises(ValueError):
                self.offline.verify_bytes(b"", bad)

    def test_illegal_tenant_value_error_beats_corrupt_bytes(self):
        with self.assertRaises(ValueError):
            self.offline.verify_bytes(b"\xff", float("nan"))

    def test_expected_count_must_be_non_negative_int(self):
        for bad in (-1, 1.0, True, False, "1", [], None):
            # None is allowed, handled separately below
            if bad is None:
                continue
            with self.assertRaises(ValueError):
                self.offline.verify_bytes(b"", "t", bad)
        self.assertTrue(self.offline.verify_bytes(b"", "t", None)["ok"])
        self.assertTrue(self.offline.verify_bytes(b"", "t", 0)["ok"])
        self.assertFalse(self.offline.verify_bytes(b"", "t", 3)["ok"])


if __name__ == "__main__":
    unittest.main()
