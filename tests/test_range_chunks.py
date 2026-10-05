import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app import (AuditChain, AuditChainConflictError, AuditChainRangeError,
                 AuditChainStateError)

ZERO = "0" * 64


def record_bytes(item):
    return (json.dumps(item, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


class ExportTenantRangeChunksTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.path = Path(self.tmp.name) / "audit.jsonl"
        self.chain = AuditChain(self.path)
        for i in range(6):
            self.chain.append("a", {"i": i, "s": "审计-世界"})
            self.chain.append("b", {"i": i})

    def tearDown(self):
        self.tmp.cleanup()

    # --- byte-for-byte equivalence with export_tenant_range ---

    def test_concatenation_equals_export_tenant_range_for_many_sizes(self):
        expected = self.chain.export_tenant_range("a", 2, 5)
        self.assertTrue(expected)
        for size in range(1, len(expected) + 2):
            chunks = list(self.chain.export_tenant_range_chunks("a", 2, 5, size))
            self.assertEqual(b"".join(chunks), expected, size)
            self.assertTrue(all(isinstance(c, bytes) for c in chunks))
            self.assertTrue(all(len(c) <= size for c in chunks), size)

    def test_open_tail_and_single_record_ranges(self):
        tail = self.chain.export_tenant_range("a", 3)
        self.assertEqual(
            b"".join(self.chain.export_tenant_range_chunks("a", 3, None, 7)),
            tail)
        self.assertEqual(
            b"".join(self.chain.export_tenant_range_chunks("a", 3, chunk_size=7)),
            tail)
        one = self.chain.export_tenant_range("b", 6, 6)
        self.assertEqual(
            b"".join(self.chain.export_tenant_range_chunks("b", 6, 6, 1)), one)

    def test_single_byte_chunks_cut_anywhere_round_trip(self):
        # a from-1 range is a complete chain: single-byte chunks (cutting
        # inside UTF-8 sequences, JSON and newlines) still concatenate to a
        # history a fresh log verifies offline.
        expected = self.chain.export_tenant_range("a", 1, 6)
        one_by_one = list(self.chain.export_tenant_range_chunks("a", 1, 6, 1))
        self.assertEqual(one_by_one, [bytes([b]) for b in expected])
        out = Path(self.tmp.name) / "out.jsonl"
        out.write_bytes(b"".join(one_by_one))
        self.assertEqual(AuditChain(out).verify("a"), {"ok": True, "count": 6})

    # --- boundary: ValueError before any read, no file touched ---

    def test_boundary_errors_raise_before_reading_and_create_nothing(self):
        missing = AuditChain(Path(self.tmp.name) / "nope.jsonl")
        cases = [
            ("a", True, None, 5), ("a", 0, None, 5), ("a", -2, None, 5),
            ("a", 1.0, None, 5), ("a", "1", None, 5), ("a", None, None, 5),
            ("a", 1, True, 5), ("a", 1, 0, 5), ("a", 2, 1, 5),
            ("a", 1, 1.5, 5), ("a", 1, "x", 5),
            ("a", 1, None, 0), ("a", 1, None, -3), ("a", 1, None, True),
            ("a", 1, None, 1.0), ("a", 1, None, "5"), ("a", 1, None, None),
            (float("nan"), 1, None, 5), ({1: "x"}, 1, None, 5),
        ]
        for t, s, e, cs in cases:
            with self.assertRaises(ValueError, msg=(t, s, e, cs)):
                missing.export_tenant_range_chunks(t, s, e, cs)
        self.assertFalse((Path(self.tmp.name) / "nope.jsonl").exists())

    def test_value_error_beats_corrupt_history(self):
        row = {"tenant": "a", "seq": 1, "event": {}, "prev": ZERO}
        row["hash"] = AuditChain._hash(row)
        row["event"] = {"tampered": True}
        self.path.write_text(json.dumps(row, sort_keys=True) + "\n")
        before = self.path.read_bytes()
        with self.assertRaises(ValueError):
            self.chain.export_tenant_range_chunks(float("nan"), 1, None, 5)
        with self.assertRaises(ValueError):
            self.chain.export_tenant_range_chunks("a", 0, None, 5)
        with self.assertRaises(ValueError):
            self.chain.export_tenant_range_chunks("a", 1, None, 0)
        self.assertEqual(self.path.read_bytes(), before)

    # --- range errors and corruption: same fields as the non-chunked entry ---

    def test_range_errors_carry_tenant_range_and_verified_count(self):
        with self.assertRaises(AuditChainRangeError) as cm:
            self.chain.export_tenant_range_chunks("a", 7, None, 4)
        self.assertEqual((cm.exception.tenant, cm.exception.start_seq,
                          cm.exception.end_seq, cm.exception.count,
                          cm.exception.reason),
                         ("a", 7, None, 6, "range"))
        with self.assertRaises(AuditChainRangeError) as cm:
            self.chain.export_tenant_range_chunks("a", 2, 9, 4)
        self.assertEqual((cm.exception.start_seq, cm.exception.end_seq,
                          cm.exception.count), (2, 9, 6))
        with self.assertRaises(AuditChainRangeError) as cm:
            self.chain.export_tenant_range_chunks("zzz", 1, None, 4)
        self.assertEqual(cm.exception.count, 0)

    def test_corrupt_history_raises_state_error_before_first_chunk(self):
        rows = [json.loads(l) for l in self.path.read_text().splitlines()]
        rows[0]["event"] = {"i": 99}  # break the first hash of tenant a
        with self.path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, sort_keys=True) + "\n")
        before = self.path.read_bytes()
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.export_tenant_range_chunks("a", 1, 2, 1)
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("a", 1, "digest", 1))
        self.assertEqual(self.path.read_bytes(), before)

    def test_never_creates_or_modifies_any_file(self):
        before = self.path.read_bytes()
        entries = sorted(p.name for p in Path(self.tmp.name).iterdir())
        list(self.chain.export_tenant_range_chunks("a", 2, 4, 3))
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(sorted(p.name for p in Path(self.tmp.name).iterdir()),
                         entries)


class ImportTenantRangeChunksTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)
        self.src = AuditChain(self.tmp_path / "src.jsonl")
        for i in range(6):
            self.src.append("a", {"i": i, "s": "审计€"})
        self.chain = AuditChain(self.tmp_path / "audit.jsonl")
        for i in range(2):
            self.chain.append("a", {"i": i, "s": "审计€"})
        self.head = self.chain.head("a")
        self.seg = self.src.export_tenant_range("a", 3, 5)

    def tearDown(self):
        self.tmp.cleanup()

    def graft(self, chunks):
        return self.chain.import_tenant_range_chunks(
            "a", chunks, self.head["count"], self.head["hash"])

    # --- equivalence with the byte entry ---

    def test_segment_grafts_from_known_head_at_any_cut(self):
        for cut in range(len(self.seg) + 1):
            chain = AuditChain(self.tmp_path / f"d{cut}.jsonl")
            for i in range(2):
                chain.append("a", {"i": i, "s": "审计€"})
            records = chain.import_tenant_range_chunks(
                "a", [self.seg[:cut], self.seg[cut:]],
                self.head["count"], self.head["hash"])
            self.assertEqual([r["seq"] for r in records], [3, 4, 5])
            self.assertEqual(chain.verify("a"), {"ok": True, "count": 5})

    def test_empty_chunks_and_odd_boundaries_are_allowed(self):
        chunks = [self.seg[:3], b"", self.seg[3:40], self.seg[40:41],
                  b"", self.seg[41:]]
        records = self.graft(iter(chunks))
        self.assertEqual([r["seq"] for r in records], [3, 4, 5])
        self.assertEqual(self.chain.verify("a"), {"ok": True, "count": 5})

    def test_result_records_and_bytes_match_non_chunked_entry(self):
        other = AuditChain(self.tmp_path / "other.jsonl")
        for i in range(2):
            other.append("a", {"i": i, "s": "审计€"})
        chunked = self.graft([self.seg])
        plain = other.import_tenant_range(
            "a", self.seg, self.head["count"], self.head["hash"])
        self.assertEqual(chunked, plain)
        self.assertEqual((self.tmp_path / "audit.jsonl").read_bytes(),
                         (self.tmp_path / "other.jsonl").read_bytes())

    def test_empty_concatenation_is_noop(self):
        self.assertEqual(self.graft([b"", b""]), [])
        self.assertFalse((self.tmp_path / "fresh.jsonl").exists())
        fresh = AuditChain(self.tmp_path / "fresh.jsonl")
        self.assertEqual(
            fresh.import_tenant_range_chunks("a", [], 0, ZERO), [])
        self.assertFalse((self.tmp_path / "fresh.jsonl").exists())

    # --- chunk container boundary: ValueError before the target is read ---

    def test_container_boundary_errors_precede_target_state(self):
        # the target chain is corrupt; a boundary error must still win
        self.chain.append("b", {"i": 0})
        rows = self.chain.path.read_bytes()
        i = rows.index(b'"hash": "') + len(b'"hash": "')
        self.chain.path.write_bytes(rows[:i] + b"0" + rows[i + 1:])
        for bad in (self.seg, bytearray(self.seg), 42, object(),
                    [self.seg, "x"], [b"a", bytearray(b"b")], [None]):
            with self.assertRaises(ValueError, msg=repr(bad)):
                self.graft(bad)

    def test_head_assertion_validated_before_chunks_are_consumed(self):
        consumed = []

        def gen():
            consumed.append(1)
            yield self.seg

        for args in [(float("nan"), 2, self.head["hash"]),
                     ("a", -1, self.head["hash"]),
                     ("a", True, self.head["hash"]),
                     ("a", 1.0, self.head["hash"]),
                     ("a", 2, "zz"), ("a", 2, None), ("a", 2, 0)]:
            with self.assertRaises(ValueError, msg=args):
                self.chain.import_tenant_range_chunks(
                    args[0], gen(), args[1], args[2])
        self.assertEqual(consumed, [])

    # --- input defects and conflicts: same fields as the byte entry ---

    def test_foreign_tenant_is_value_error(self):
        other = {"tenant": "b", "seq": 1, "event": {}, "prev": ZERO}
        other["hash"] = AuditChain._hash(other)
        data = self.seg + record_bytes(other)
        before = self.chain.path.read_bytes()
        with self.assertRaises(ValueError):
            self.graft([data])
        self.assertEqual(self.chain.path.read_bytes(), before)

    def test_truncated_input_is_state_error_with_line(self):
        before = self.chain.path.read_bytes()
        with self.assertRaises(AuditChainStateError) as cm:
            self.graft([self.seg[:-10]])
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("a", 5, "missing", 3))
        self.assertEqual(self.chain.path.read_bytes(), before)

    def test_stale_head_is_conflict_with_both_heads_and_writes_nothing(self):
        self.graft([self.seg])  # chain now at count 5
        before = self.chain.path.read_bytes()
        with self.assertRaises(AuditChainConflictError) as cm:
            self.graft([self.seg])  # same segment, same stale assertion
        self.assertEqual(cm.exception.reason, "conflict")
        self.assertEqual((cm.exception.tenant, cm.exception.expected_count,
                          cm.exception.expected_hash, cm.exception.actual_count),
                         ("a", 2, self.head["hash"], 5))
        self.assertNotEqual(cm.exception.actual_hash, self.head["hash"])
        self.assertEqual(self.chain.path.read_bytes(), before)


class ImportAllRangeChunksTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)
        self.src = AuditChain(self.tmp_path / "src.jsonl")
        for i in range(5):
            self.src.append("x", {"i": i})
            self.src.append("y", {"i": i})
        self.chain = AuditChain(self.tmp_path / "audit.jsonl")
        for i in range(2):
            self.chain.append("x", {"i": i})
            self.chain.append("y", {"i": i})
        hx, hy = self.chain.head("x"), self.chain.head("y")
        self.heads = [
            {"tenant": "x", "expected_count": hx["count"],
             "expected_hash": hx["hash"]},
            {"tenant": "y", "expected_count": hy["count"],
             "expected_hash": hy["hash"]},
        ]
        sx = self.src.export_tenant_range("x", 3).splitlines(keepends=True)
        sy = self.src.export_tenant_range("y", 3).splitlines(keepends=True)
        self.stream = b"".join(x for pair in zip(sx, sy) for x in pair)

    def tearDown(self):
        self.tmp.cleanup()

    def graft(self, chunks, heads=None):
        return self.chain.import_all_range_chunks(
            chunks, self.heads if heads is None else heads)

    def test_interleaved_segments_graft_as_one_block(self):
        chunks = [self.stream[i:i + 11]
                  for i in range(0, len(self.stream), 11)]
        records = self.graft(chunks)
        self.assertEqual([r["seq"] for r in records], [3, 3, 4, 4, 5, 5])
        self.assertEqual([r["tenant"] for r in records],
                         ["x", "y", "x", "y", "x", "y"])
        self.assertTrue(self.chain.verify_all()["ok"])
        self.assertEqual(self.chain.head("x")["count"], 5)
        self.assertEqual(self.chain.head("y")["count"], 5)

    def test_result_records_and_bytes_match_non_chunked_entry(self):
        other = AuditChain(self.tmp_path / "other.jsonl")
        for i in range(2):
            other.append("x", {"i": i})
            other.append("y", {"i": i})
        chunked = self.graft([self.stream])
        plain = other.import_all_range(self.stream, self.heads)
        self.assertEqual(chunked, plain)
        self.assertEqual((self.tmp_path / "audit.jsonl").read_bytes(),
                         (self.tmp_path / "other.jsonl").read_bytes())

    def test_empty_concatenation_is_noop(self):
        self.assertEqual(self.graft([]), [])
        self.assertEqual(self.graft([b"", b""]), [])
        fresh = AuditChain(self.tmp_path / "fresh.jsonl")
        self.assertEqual(fresh.import_all_range_chunks([], []), [])
        self.assertFalse((self.tmp_path / "fresh.jsonl").exists())

    def test_expected_heads_validated_before_chunks_are_consumed(self):
        consumed = []

        def gen():
            consumed.append(1)
            yield self.stream

        hx = self.heads[0]
        bad_heads = [
            None, "x", 42,
            [{"tenant": "x", "count": 1, "hash": ZERO}],
            [{"tenant": "x", "expected_count": -1, "expected_hash": ZERO}],
            [{"tenant": "x", "expected_count": True, "expected_hash": ZERO}],
            [{"tenant": "x", "expected_count": 2, "expected_hash": "zz"}],
            [hx, hx],
        ]
        for heads in bad_heads:
            with self.assertRaises(ValueError, msg=heads):
                self.chain.import_all_range_chunks(gen(), heads)
        self.assertEqual(consumed, [])

    def test_container_boundary_errors(self):
        for bad in (self.stream, bytearray(self.stream), 42,
                    [self.stream, "x"], [b"a", bytearray(b"b")]):
            with self.assertRaises(ValueError, msg=type(bad)):
                self.graft(bad)

    def test_unlisted_tenant_is_value_error(self):
        before = self.chain.path.read_bytes()
        with self.assertRaises(ValueError):
            self.graft([self.stream], [self.heads[0]])  # y records unlisted
        self.assertEqual(self.chain.path.read_bytes(), before)

    def test_listed_tenant_without_records_is_missing(self):
        sx_only = self.src.export_tenant_range("x", 3)
        with self.assertRaises(AuditChainStateError) as cm:
            self.graft([sx_only])
        self.assertEqual((cm.exception.tenant, cm.exception.reason),
                         ("y", "missing"))

    def test_replay_is_conflict_with_actual_heads_and_writes_nothing(self):
        self.graft([self.stream])
        before = self.chain.path.read_bytes()
        with self.assertRaises(AuditChainConflictError) as cm:
            self.graft([self.stream])
        self.assertEqual(cm.exception.reason, "conflict")
        self.assertEqual((cm.exception.tenant, cm.exception.expected_count,
                          cm.exception.actual_count), ("x", 2, 5))
        self.assertEqual(self.chain.path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
