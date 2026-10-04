import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app import AuditChain, AuditChainConflictError, ZERO


def record_bytes(item):
    return (json.dumps(item, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


class VerifyHeadsTest(unittest.TestCase):
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
        self.chain.append("b", {"i": 2})

    def snapshot(self):
        return self.path.read_bytes()

    def heads_bytes(self, data=None):
        if data is None:
            data = self.snapshot()
        return self.other.heads_bytes(data)

    def expected_from_heads(self, directory):
        return [
            {"tenant": h["tenant"], "count": h["count"], "hash": h["hash"]}
            for h in directory["tenants"]
        ]

    def valid_row(self, tenant="t", seq=1, prev=ZERO, event=None):
        row = {"tenant": tenant, "seq": seq, "event": event or {}, "prev": prev}
        row["hash"] = AuditChain._hash(row)
        return record_bytes(row)

    def assertConflict(self, ctx, tenant, ec, eh, ac, ah):
        err = ctx.exception
        self.assertEqual(err.reason, "conflict")
        self.assertEqual(err.tenant, tenant)
        self.assertEqual(err.expected_count, ec)
        self.assertEqual(err.expected_hash, eh)
        self.assertEqual(err.actual_count, ac)
        self.assertEqual(err.actual_hash, ah)

    # --- success shape and equivalence with heads ---

    def test_verify_heads_file_matches_heads(self):
        self.seed()
        expected = self.expected_from_heads(self.chain.heads())
        self.assertEqual(self.chain.verify_heads(expected), self.chain.heads())
        self.assertEqual(self.chain.verify_heads(expected), {
            "ok": True,
            "tenants": [
                {"tenant": "a", "count": 3,
                 "hash": self.chain.head("a")["hash"]},
                {"tenant": "b", "count": 2,
                 "hash": self.chain.head("b")["hash"]},
            ],
        })

    def test_bytes_and_chunks_match_heads_bytes(self):
        self.seed()
        data = self.snapshot()
        expected = self.expected_from_heads(self.heads_bytes(data))
        self.assertEqual(self.other.verify_heads_bytes(data, expected),
                         self.other.heads_bytes(data))
        for chunking in ([data], [data[:1], data[1:]],
                         [data[i:i + 1] for i in range(len(data))],
                         [data[:10], b"", data[10:37], b"", data[37:]],
                         [data[i:i + 7] for i in range(0, len(data), 7)]):
            self.assertEqual(
                self.other.verify_heads_chunks(chunking, expected),
                self.other.heads_bytes(data), chunking)

    def test_returns_snapshot_first_appearance_order_not_list_order(self):
        self.seed()
        directory = self.heads_bytes()
        # Expectations supplied in reverse order still succeed and the
        # returned directory keeps the snapshot's first-appearance order.
        rev = list(reversed(self.expected_from_heads(directory)))
        result = self.other.verify_heads_bytes(self.snapshot(), rev)
        self.assertEqual([t["tenant"] for t in result["tenants"]], ["a", "b"])

    def test_expected_absent_tenant_with_empty_head_matches(self):
        # A listed tenant absent from the snapshot matches the fixed empty
        # head; the success structure still lists only actual tenants, in the
        # snapshot's first-appearance order with original values.
        for tenant, event in (({"z": 1}, 0), ("a", 0), ([1, 2], 0),
                              ({"z": 1}, 1), ("a", 1), (None, 0)):
            self.chain.append(tenant, {"i": event})
        data = self.snapshot()
        directory = self.heads_bytes(data)
        expected = self.expected_from_heads(directory)
        expected.append({"tenant": "ghost", "count": 0, "hash": ZERO})
        result = self.other.verify_heads_bytes(data, expected)
        self.assertEqual(result, directory)
        self.assertEqual(
            [t["tenant"] for t in result["tenants"]],
            [{"z": 1}, "a", [1, 2], None])

    def test_empty_snapshot_and_empty_directory(self):
        for entry in (self.other.verify_heads_bytes(b"", []),
                      self.other.verify_heads_chunks([], []),
                      self.other.verify_heads_chunks(iter([]), []),
                      self.other.verify_heads_chunks([b"", b""], [])):
            self.assertEqual(entry, {"ok": True, "tenants": []})
        # The file entry over a missing/empty log is the same empty match.
        phantom = self.path.with_name("phantom.jsonl")
        chain = AuditChain(phantom)
        self.assertEqual(chain.verify_heads([]), {"ok": True, "tenants": []})
        self.assertFalse(phantom.exists())

    def test_distinct_json_identities_match_independently(self):
        for t in (1, 1.0, True, "1"):
            self.chain.append(t, {})
            self.chain.append(t, {})
        data = self.snapshot()
        directory = self.heads_bytes(data)
        expected = self.expected_from_heads(directory)
        self.assertEqual(
            self.other.verify_heads_bytes(data, expected), directory)
        # Object key order is normalized for matching but echoed verbatim.
        self.chain.append({"a": 1, "b": 2}, {})
        self.chain.append({"b": 2, "a": 1}, {})
        data = self.snapshot()
        directory = self.heads_bytes(data)
        expected = self.expected_from_heads(directory)
        result = self.other.verify_heads_bytes(data, expected)
        self.assertEqual(result["tenants"][-1]["tenant"], {"a": 1, "b": 2})
        self.assertEqual(result["tenants"][-1]["count"], 2)

    # --- conflicts: missing/unlisted heads ---

    def test_expected_tenant_absent_from_directory_is_empty_head(self):
        self.seed()
        # The absent tenant is asserted non-empty: actual is fixed (0, ZERO).
        with self.assertRaises(AuditChainConflictError) as ctx:
            self.chain.verify_heads(
                [{"tenant": "ghost", "count": 1, "hash": "9" * 64}])
        self.assertConflict(ctx, "ghost", 1, "9" * 64, 0, ZERO)

    def test_tenant_in_directory_but_not_expected_is_expected_empty(self):
        self.seed()
        expected = [{
            "tenant": "a", "count": 3,
            "hash": self.chain.head("a")["hash"]}]
        with self.assertRaises(AuditChainConflictError) as ctx:
            self.chain.verify_heads(expected)
        # First unlisted tenant in the snapshot's first-appearance order.
        self.assertConflict(
            ctx, "b", 0, ZERO, 2, self.chain.head("b")["hash"])

    def test_unlisted_uses_snapshot_first_appearance_order(self):
        self.chain.append("x", {})
        self.chain.append("y", {})
        with self.assertRaises(AuditChainConflictError) as ctx:
            self.other.verify_heads_bytes(self.snapshot(), [])
        self.assertConflict(ctx, "x", 0, ZERO, 1,
                            self.chain.head("x")["hash"])

    def test_empty_expectation_against_nonempty_snapshot_conflicts_offline(self):
        self.seed()
        with self.assertRaises(AuditChainConflictError) as ctx:
            self.other.verify_heads_bytes(self.snapshot(), [])
        self.assertEqual(ctx.exception.tenant, "a")

    # --- conflicts: count/hash mismatch and list-order priority ---

    def test_count_mismatch_fields(self):
        self.seed()
        ha, hb = self.chain.head("a")["hash"], self.chain.head("b")["hash"]
        with self.assertRaises(AuditChainConflictError) as ctx:
            self.chain.verify_heads([
                {"tenant": "a", "count": 2, "hash": ha},
                {"tenant": "b", "count": 2, "hash": hb}])
        self.assertConflict(ctx, "a", 2, ha, 3, ha)

    def test_hash_mismatch_fields(self):
        self.seed()
        hb = self.chain.head("b")["hash"]
        with self.assertRaises(AuditChainConflictError) as ctx:
            self.chain.verify_heads([
                {"tenant": "a", "count": 3, "hash": "9" * 64},
                {"tenant": "b", "count": 2, "hash": hb}])
        self.assertConflict(
            ctx, "a", 3, "9" * 64, 3, self.chain.head("a")["hash"])

    def test_conflict_priority_is_expected_heads_list_order(self):
        self.seed()
        ha, hb = self.chain.head("a")["hash"], self.chain.head("b")["hash"]
        wrong = [
            {"tenant": "a", "count": 9, "hash": ha},
            {"tenant": "b", "count": 9, "hash": hb}]
        with self.assertRaises(AuditChainConflictError) as ctx:
            self.other.verify_heads_bytes(self.snapshot(), wrong)
        self.assertEqual(ctx.exception.tenant, "a")
        with self.assertRaises(AuditChainConflictError) as ctx:
            self.other.verify_heads_bytes(self.snapshot(), list(reversed(wrong)))
        self.assertEqual(ctx.exception.tenant, "b")

    def test_chunks_conflict_matches_bytes_conflict(self):
        self.seed()
        data = self.snapshot()
        expected = [{"tenant": "a", "count": 9, "hash": "1" * 64}]
        with self.assertRaises(AuditChainConflictError) as e1:
            self.other.verify_heads_bytes(data, expected)
        with self.assertRaises(AuditChainConflictError) as e2:
            self.other.verify_heads_chunks([data[:7], b"", data[7:]], expected)
        for attr in ("reason", "tenant", "expected_count", "expected_hash",
                     "actual_count", "actual_hash"):
            self.assertEqual(getattr(e1.exception, attr),
                             getattr(e2.exception, attr), attr)

    # --- corrupt snapshot: failure verdict, never a comparison ---

    def test_corrupt_snapshot_returns_verify_all_failure_object(self):
        good = self.valid_row("t", 1)
        row_no_hash = {"tenant": "t", "seq": 1, "event": {}, "prev": ZERO}
        cases = [
            (b"{not json\n", (1, None, "missing")),
            (b"\n", (1, None, "missing")),
            (b'{"tenant":"t","tenant":"t","seq":1,"event":{},'
             b'"prev":"' + ZERO.encode() + b'","hash":"x"}\n',
             (1, None, "missing")),
            (b"\xff", (1, None, "missing")),
            ((json.dumps(row_no_hash) + "\n").encode(), (1, "t", "missing")),
        ]
        # Expectations deliberately disagree: corruption still wins.
        expected = [{"tenant": "t", "count": 5, "hash": "f" * 64}]
        for raw, (at, tenant, reason) in cases:
            self.assertEqual(
                self.other.verify_heads_bytes(raw, expected),
                self.other.verify_all_bytes(raw), raw)
            r = self.other.verify_heads_bytes(raw, expected)
            self.assertEqual(
                (r["ok"], r["at"], r["tenant"], r["reason"]),
                (False, at, tenant, reason), raw)
            self.assertNotIn("tenants", r)

    def test_sequence_digest_corruption_beats_comparison(self):
        good = self.valid_row("t", 1)
        gap = {"tenant": "t", "seq": 3, "event": {}, "prev": "1" * 64}
        gap["hash"] = AuditChain._hash(gap)
        raw = good + record_bytes(gap)
        expected = [{"tenant": "t", "count": 3, "hash": "1" * 64}]
        self.assertEqual(
            self.other.verify_heads_bytes(raw, expected),
            {"ok": False, "at": 2, "tenant": "t", "reason": "sequence"})

        bad_prev = {"tenant": "t", "seq": 2, "event": {}, "prev": "9" * 64}
        bad_prev["hash"] = AuditChain._hash(bad_prev)
        raw = good + record_bytes(bad_prev)
        self.assertEqual(
            self.other.verify_heads_bytes(raw, expected),
            {"ok": False, "at": 2, "tenant": "t", "reason": "digest"})

    def test_chunks_corruption_matches_bytes_corruption(self):
        raw = self.valid_row("t", 1) + b"\xff\n"
        expected = [{"tenant": "t", "count": 9, "hash": "0" * 64}]
        chunks = [raw[:3], b"", raw[3:]]
        self.assertEqual(
            self.other.verify_heads_chunks(chunks, expected),
            self.other.verify_heads_bytes(raw, expected))

    def test_file_corrupt_snapshot_returns_failure_no_conflict(self):
        # The file entry reports the first corrupt point with the same
        # failure object verify_all()/heads() return on the same bytes; no
        # head comparison runs and no error is raised.
        self.path.write_bytes(b"{broken\n")
        expected = [{"tenant": "t", "count": 9, "hash": "0" * 64}]
        self.assertEqual(
            self.chain.verify_heads(expected),
            {"ok": False, "at": 1, "tenant": None, "reason": "missing"})
        self.assertEqual(self.chain.verify_heads(expected),
                         self.chain.verify_all())
        self.assertEqual(self.snapshot(), b"{broken\n")

    # --- expected_heads boundary: ValueError before reading/parsing ---

    def test_expected_heads_must_be_list(self):
        for bad in (None, 1, "x", (), {}, object()):
            with self.assertRaises(ValueError):
                self.other.verify_heads_bytes(b"", bad)
            with self.assertRaises(ValueError):
                self.chain.verify_heads(bad)
            with self.assertRaises(ValueError):
                self.other.verify_heads_chunks([], bad)

    def test_member_must_be_exact_three_key_object(self):
        good = {"tenant": "t", "count": 0, "hash": ZERO}
        for bad in (None, 1, "x", [], [good],
                    {"tenant": "t"},
                    {"tenant": "t", "count": 0},
                    {"tenant": "t", "count": 0, "hash": ZERO, "x": 1},
                    {"tenant": "t", "count": 0,
                     "expected_hash": ZERO}):
            with self.assertRaises(ValueError):
                self.other.verify_heads_bytes(b"", [bad])

    def test_tenant_boundary(self):
        for bad in (float("nan"), float("inf"), {"k": float("nan")},
                    {1: "x"}, object(), b"x", {1, 2}):
            with self.assertRaises(ValueError):
                self.other.verify_heads_bytes(
                    b"", [{"tenant": bad, "count": 0, "hash": ZERO}])

    def test_count_boundary(self):
        for bad in (-1, 1.0, True, False, "1", None, 1.5, [1]):
            with self.assertRaises(ValueError):
                self.other.verify_heads_bytes(
                    b"", [{"tenant": "t", "count": bad, "hash": ZERO}])

    def test_hash_boundary(self):
        for bad in ("", "x", "A" * 64, "0" * 63, "0" * 65, 1, None,
                    bytes(64), ZERO.encode()):
            with self.assertRaises(ValueError):
                self.other.verify_heads_bytes(
                    b"", [{"tenant": "t", "count": 0, "hash": bad}])

    def test_duplicate_canonical_tenant_rejected(self):
        head = {"count": 0, "hash": ZERO}
        for dup in (
            [{"tenant": "a", **head}, {"tenant": "a", **head}],
            [{"tenant": {"a": 1, "b": 2}, **head},
             {"tenant": {"b": 2, "a": 1}, **head}],
            [{"tenant": 1, **head}, {"tenant": 1, **head}],
        ):
            with self.assertRaises(ValueError):
                self.other.verify_heads_bytes(b"", dup)
        # Distinct JSON identities are not duplicates even if loosely equal.
        for t in (1, 1.0, True, "1"):
            self.chain.append(t, {})
            self.chain.append(t, {})
        ok = [{"tenant": t, "count": 2, "hash": self.chain.head(t)["hash"]}
              for t in (1, 1.0, True, "1")]
        self.assertTrue(
            self.other.verify_heads_bytes(self.snapshot(), ok)["ok"])

    def test_boundary_value_error_beats_corrupt_snapshot(self):
        # A malformed expectation ends as ValueError even though the snapshot
        # is corrupt, and no byte is read/parsed for the comparison.
        corrupt = b"{not json\n"
        bad_lists = (
            [1], "x", None,
            [{"tenant": "t", "count": -1, "hash": ZERO}],
            [{"tenant": "t", "count": 1.0, "hash": ZERO}],
            [{"tenant": "t", "count": True, "hash": ZERO}],
            [{"tenant": "t", "count": 0, "hash": "Z" * 64}],
            [{"tenant": float("nan"), "count": 0, "hash": ZERO}],
            [{"tenant": "a", "count": 0, "hash": ZERO},
             {"tenant": "a", "count": 0, "hash": ZERO}],
        )
        for bad in bad_lists:
            with self.assertRaises(ValueError):
                self.other.verify_heads_bytes(corrupt, bad)
            with self.assertRaises(ValueError):
                self.other.verify_heads_chunks([corrupt], bad)

    # --- data / chunks container boundary ---

    def test_data_must_be_bytes(self):
        for bad in ("", "{}", bytearray(b""), bytearray(b"{}"), 1, None,
                    [b""]):
            with self.assertRaises(ValueError):
                self.other.verify_heads_bytes(bad, [])

    def test_bare_bytes_container_is_value_error(self):
        for bad in (b"", b"{}", bytearray(b""), bytearray(b"{}")):
            with self.assertRaises(ValueError):
                self.other.verify_heads_chunks(bad, [])

    def test_non_iterable_container_is_value_error(self):
        for bad in (1, None, object()):
            with self.assertRaises(ValueError):
                self.other.verify_heads_chunks(bad, [])

    def test_first_non_bytes_member_is_value_error(self):
        for bad in ([b"", "x"], [b"", 1], [b"", None], [b"", bytearray(b"")],
                    [b"", [b""]], ["x"], [bytearray(b"")]):
            with self.assertRaises(ValueError):
                self.other.verify_heads_chunks(bad, [])

    def test_bad_member_stops_consumption_immediately(self):
        pulled = []

        def gen():
            pulled.append(1)
            yield b""
            pulled.append(2)
            yield "not bytes"
            pulled.append(3)  # must never be reached
            yield b""

        with self.assertRaises(ValueError):
            self.other.verify_heads_chunks(gen(), [])
        self.assertEqual(pulled, [1, 2])

    def test_expected_heads_validated_before_chunks_consumed(self):
        pulled = []

        def gen():
            pulled.append(1)
            yield b""

        with self.assertRaises(ValueError):
            self.other.verify_heads_chunks(gen(), [{"tenant": 1}])
        self.assertEqual(pulled, [])

    # --- offline purity ---

    def test_offline_entries_never_touch_constructor_path(self):
        phantom = self.path.with_name("phantom.jsonl")
        chain = AuditChain(phantom)
        self.assertEqual(chain.verify_heads_bytes(b"", []),
                         {"ok": True, "tenants": []})
        self.assertEqual(chain.verify_heads_chunks([], []),
                         {"ok": True, "tenants": []})
        self.assertFalse(phantom.exists())
        self.seed()
        before = self.snapshot()
        expected = self.expected_from_heads(chain.heads_bytes(before))
        self.assertTrue(chain.verify_heads_bytes(before, expected)["ok"])
        self.assertTrue(
            chain.verify_heads_chunks([before[:7], b"", before[7:]], expected)
            ["ok"])
        self.assertEqual(self.snapshot(), before)
        self.assertFalse(phantom.exists())

    def test_file_entry_is_read_only(self):
        self.seed()
        before = self.snapshot()
        expected = self.expected_from_heads(self.chain.heads())
        self.chain.verify_heads(expected)
        with self.assertRaises(AuditChainConflictError):
            self.chain.verify_heads([])
        with self.assertRaises(AuditChainConflictError):
            self.chain.verify_heads(
                [{"tenant": "a", "count": 9, "hash": "0" * 64}])
        self.assertEqual(self.snapshot(), before)

    def test_inputs_are_not_mutated(self):
        self.seed()
        data = self.snapshot()
        chunks = [data[:9], b"", data[9:]]
        expected = self.expected_from_heads(self.heads_bytes(data))
        import copy
        expected_copy = copy.deepcopy(expected)
        self.other.verify_heads_bytes(data, expected)
        self.other.verify_heads_chunks(chunks, expected)
        self.assertEqual(data, self.snapshot())
        self.assertEqual(b"".join(chunks), data)
        self.assertEqual(expected, expected_copy)

    def test_chunk_split_at_every_byte_offset(self):
        self.chain.append("t", {"msg": "héllo→世界"})
        self.chain.append("t", {"msg": "✓" * 40})
        data = self.snapshot()
        expected = self.expected_from_heads(self.heads_bytes(data))
        for cut in range(len(data) + 1):
            self.assertEqual(
                self.other.verify_heads_chunks([data[:cut], data[cut:]],
                                               expected),
                self.other.verify_heads_bytes(data, expected), cut)


if __name__ == "__main__":
    unittest.main()
