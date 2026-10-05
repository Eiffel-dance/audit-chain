import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app import (
    AuditChain,
    AuditChainConflictError,
    AuditChainRangeError,
    AuditChainStateError,
    ZERO,
)


def record_bytes(item):
    return (json.dumps(item, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


class ExportTenantRangeChunksTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.path = Path(self.tmp.name) / "audit.jsonl"
        self.chain = AuditChain(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def seed(self, n=8):
        for i in range(n):
            self.chain.append("a", {"i": i, "s": "审计-世界"})
            self.chain.append("b", {"i": i})

    # --- result shape and byte-equivalence with export_tenant_range ---

    def test_returns_iterator_of_bytes_chunks_matching_range_export(self):
        self.seed()
        for args in [(3, 6), (1, 8), (5, None), (2, 2)]:
            start, end = args
            expected = self.chain.export_tenant_range("a", start, end)
            for size in (1, 7, 64, 10_000):
                it = self.chain.export_tenant_range_chunks(
                    "a", start, end, size)
                self.assertEqual(iter(it), it)
                chunks = list(it)
                self.assertTrue(chunks)
                self.assertTrue(all(isinstance(c, bytes) for c in chunks))
                self.assertTrue(all(len(c) <= size for c in chunks))
                self.assertEqual(b"".join(chunks), expected)

    def test_open_tail_default_end(self):
        self.seed()
        expected = self.chain.export_tenant_range("a", 4)
        got = b"".join(self.chain.export_tenant_range_chunks("a", 4, chunk_size=5))
        self.assertEqual(got, expected)

    def test_keyword_and_positional_chunk_size(self):
        self.seed()
        a = b"".join(self.chain.export_tenant_range_chunks("a", 2, 5, 9))
        b = b"".join(
            self.chain.export_tenant_range_chunks("a", 2, 5, chunk_size=9))
        self.assertEqual(a, b)

    # --- boundary: ValueError before any read ---

    def test_boundary_errors_raise_before_reading(self):
        self.seed()
        before = self.path.read_bytes()
        bad_calls = [
            (object(), 1, 2, 4),                # illegal tenant
            ("a", True, 2, 4),                  # bool start
            ("a", 0, 2, 4),                     # non-positive start
            ("a", -3, 2, 4),                    # negative start
            ("a", 1.5, 2, 4),                   # float start
            ("a", "1", 2, 4),                   # str start
            ("a", 3, 2, 4),                     # end < start
            ("a", 1, True, 4),                  # bool end
            ("a", 1, 2.0, 4),                   # float end
            ("a", 1, 2, True),                  # bool chunk_size
            ("a", 1, 2, 0),                     # zero chunk_size
            ("a", 1, 2, -8),                    # negative chunk_size
            ("a", 1, 2, 2.5),                   # float chunk_size
            ("a", 1, 2, "8"),                   # str chunk_size
            ("a", 1, 2, None),                  # missing chunk_size
        ]
        for args in bad_calls:
            with self.assertRaises(ValueError, msg=args):
                self.chain.export_tenant_range_chunks(*args)
        self.assertEqual(self.path.read_bytes(), before)

    def test_range_errors_match_non_chunked_entry(self):
        self.seed(3)
        for args in [(5, None), (1, 4), (2, 9)]:
            start, end = args
            with self.assertRaises(AuditChainRangeError) as ctx:
                self.chain.export_tenant_range_chunks("a", start, end, 4)
            err = ctx.exception
            self.assertEqual(err.tenant, "a")
            self.assertEqual(err.start_seq, start)
            self.assertEqual(err.end_seq, end)
            self.assertEqual(err.count, 3)
            self.assertEqual(err.reason, "range")

    def test_empty_chain_is_range_error_not_chunks(self):
        with self.assertRaises(AuditChainRangeError):
            self.chain.export_tenant_range_chunks("a", 1, 1, 4)
        self.assertFalse(self.path.exists())

    def test_corrupt_chain_raises_state_error_before_any_chunk(self):
        self.seed(4)
        rows = self.path.read_text().splitlines(keepends=True)
        rows[2] = rows[2].replace('"i": 1', '"i": 100')
        self.path.write_text("".join(rows))
        with self.assertRaises(AuditChainStateError) as ctx:
            self.chain.export_tenant_range_chunks("a", 1, 2, 16)
        self.assertEqual(ctx.exception.tenant, "a")
        self.assertEqual(ctx.exception.reason, "digest")

    def test_snapshot_is_atomic_across_chunks(self):
        self.seed(6)
        it = self.chain.export_tenant_range_chunks("a", 1, 6, 3)
        first = next(it)
        self.chain.append("a", {"i": "late"})
        rest = b"".join(it)
        self.assertEqual(
            first + rest, self.chain.export_tenant_range("a", 1, 6))


class ImportTenantRangeChunksTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.path = Path(self.tmp.name) / "audit.jsonl"
        self.chain = AuditChain(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def make_source(self, n=6, name="source.jsonl"):
        src = AuditChain(self.path.with_name(name))
        for i in range(n):
            src.append("a", {"i": i, "s": "审计"})
            src.append("b", {"i": i})
        return src

    def head_of(self, src, tenant, n):
        if n == 0:
            return 0, ZERO
        seg = src.export_tenant_range(tenant, 1, n)
        rows = [json.loads(l) for l in seg.decode().splitlines()]
        return n, rows[-1]["hash"]

    def seed_target_prefix(self, src, tenant, n):
        # Give self.chain the first n records of src's tenant chain.
        seg = src.export_tenant_range(tenant, 1, n)
        self.chain.import_tenant(tenant, seg)

    # --- round trips over many chunkings ---

    def test_round_trip_matches_non_chunked_import(self):
        src = self.make_source()
        count, tail = self.head_of(src, "a", 2)
        seg = src.export_tenant_range("a", 3, 5)
        chunkings = [
            [seg],
            [seg[:1], seg[1:]],
            [seg[i:i + 1] for i in range(len(seg))],
            [seg[:7], b"", seg[7:40], b"", seg[40:]],
            [b"", b"", seg, b""],
            tuple(seg[i:i + 13] for i in range(0, len(seg), 13)),
            (seg[i:i + 5] for i in range(0, len(seg), 5)),
        ]
        expected_records = None
        for i, chunks in enumerate(chunkings):
            target = AuditChain(self.path.with_name(f"t{i}.jsonl"))
            target.import_tenant("a", src.export_tenant_range("a", 1, 2))
            got = target.import_tenant_range_chunks("a", chunks, count, tail)
            if expected_records is None:
                expected_records = got
            self.assertEqual(got, expected_records)
            self.assertEqual([r["seq"] for r in got], [3, 4, 5])
            self.assertEqual(target.verify("a"), {"ok": True, "count": 5})

    def test_empty_concatenation_is_noop(self):
        src = self.make_source()
        count, tail = self.head_of(src, "a", 2)
        target = AuditChain(self.path.with_name("t.jsonl"))
        self.assertEqual(
            target.import_tenant_range_chunks("a", [b"", b""], count, tail),
            [],
        )
        self.assertFalse(target.path.exists())

    # --- boundary: container and header validation before consuming ---

    def test_container_boundary_errors(self):
        seg_src = self.make_source()
        count, tail = self.head_of(seg_src, "a", 2)
        seg = seg_src.export_tenant_range("a", 3)
        for bad in (seg, bytearray(seg), 42, None):
            with self.assertRaises(ValueError):
                self.chain.import_tenant_range_chunks("a", bad, count, tail)
        with self.assertRaises(ValueError):
            self.chain.import_tenant_range_chunks(
                "a", [seg[:5], "nope"], count, tail)
        with self.assertRaises(ValueError):
            self.chain.import_tenant_range_chunks(
                "a", [seg[:5], bytearray(seg[5:])], count, tail)

    def test_header_validated_before_chunks_consumed(self):
        pulled = []

        def gen():
            pulled.append(True)
            yield b"x"

        with self.assertRaises(ValueError):
            self.chain.import_tenant_range_chunks(
                object(), gen(), 0, ZERO)      # bad tenant
        with self.assertRaises(ValueError):
            self.chain.import_tenant_range_chunks(
                "a", gen(), -1, ZERO)          # bad count
        with self.assertRaises(ValueError):
            self.chain.import_tenant_range_chunks(
                "a", gen(), True, ZERO)        # bool count
        with self.assertRaises(ValueError):
            self.chain.import_tenant_range_chunks(
                "a", gen(), 0, "zz")           # bad hash
        self.assertEqual(pulled, [])

    def test_expected_heads_validated_before_chunks_consumed_multi(self):
        pulled = []

        def gen():
            pulled.append(True)
            yield b"x"

        bad_heads = [
            "not a list",
            [{"tenant": "a", "expected_count": 0}],                 # keys
            [{"tenant": "a", "expected_count": -1,
              "expected_hash": ZERO}],                              # count
            [{"tenant": "a", "expected_count": 0,
              "expected_hash": "ZZ"}],                              # hash
            [{"tenant": "a", "expected_count": 0, "expected_hash": ZERO},
             {"tenant": "a", "expected_count": 0,
              "expected_hash": ZERO}],                              # dup
        ]
        for heads in bad_heads:
            with self.assertRaises(ValueError):
                self.chain.import_all_range_chunks(gen(), heads)
        self.assertEqual(pulled, [])

    # --- content and conflict semantics identical to the byte entry ---

    def test_foreign_tenant_is_value_error(self):
        src = self.make_source()
        count, tail = self.head_of(src, "a", 2)
        seg = src.export_tenant_range("b", 3)
        with self.assertRaises(ValueError):
            self.chain.import_tenant_range_chunks("a", [seg], count, tail)

    def test_corrupt_input_is_state_error_with_line(self):
        src = self.make_source()
        count, tail = self.head_of(src, "a", 2)
        seg = src.export_tenant_range("a", 3, 4)
        rows = seg.decode().splitlines(keepends=True)
        bad = (rows[0] + rows[1].replace('"seq": 4', '"seq": 5')).encode()
        with self.assertRaises(AuditChainStateError) as ctx:
            self.chain.import_tenant_range_chunks("a", [bad], count, tail)
        self.assertEqual(ctx.exception.tenant, "a")
        self.assertEqual(ctx.exception.reason, "sequence")
        self.assertEqual(ctx.exception.line, 2)

    def test_head_mismatch_conflict_carries_both_heads(self):
        src = self.make_source()
        self.seed_target_prefix(src, "a", 3)
        count, tail = self.head_of(src, "a", 2)
        seg = src.export_tenant_range("a", 3)
        with self.assertRaises(AuditChainConflictError) as ctx:
            self.chain.import_tenant_range_chunks("a", [seg], count, tail)
        err = ctx.exception
        self.assertEqual(err.reason, "conflict")
        self.assertEqual(err.expected_count, count)
        self.assertEqual(err.expected_hash, tail)
        self.assertEqual(err.actual_count, 3)
        self.assertNotEqual(err.actual_hash, tail)
        # nothing written
        self.assertEqual(
            self.chain.read_tenant("a"),
            self.chain.read_tenant("a"),
        )
        self.assertEqual(len(self.chain.read_tenant("a")), 3)

    def test_failure_writes_no_partial_bytes(self):
        src = self.make_source()
        count, tail = self.head_of(src, "a", 2)
        self.seed_target_prefix(src, "a", 2)
        before = self.path.read_bytes()
        seg = src.export_tenant_range("a", 3)
        bad = seg[:-2] + b"xx"  # corrupt the tail
        with self.assertRaises(AuditChainStateError):
            self.chain.import_tenant_range_chunks("a", [bad], count, tail)
        self.assertEqual(self.path.read_bytes(), before)


class ImportAllRangeChunksTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.path = Path(self.tmp.name) / "audit.jsonl"
        self.chain = AuditChain(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def make_source(self, n=6, name="source.jsonl"):
        src = AuditChain(self.path.with_name(name))
        for i in range(n):
            src.append("a", {"i": i, "s": "审计"})
            src.append("b", {"i": i})
        return src

    def head_of(self, src, tenant, n):
        if n == 0:
            return 0, ZERO
        seg = src.export_tenant_range(tenant, 1, n)
        rows = [json.loads(l) for l in seg.decode().splitlines()]
        return n, rows[-1]["hash"]

    def interleaved_segments(self, src, prefixes):
        # Build the interleaved multi-tenant byte stream import_all_range
        # accepts, plus its expected_heads, from open-tail segments.
        parts, heads = [], []
        for tenant, n in prefixes.items():
            seg = src.export_tenant_range(tenant, n + 1)
            parts.append(seg)
            count, tail = self.head_of(src, tenant, n)
            heads.append({"tenant": tenant, "expected_count": count,
                          "expected_hash": tail})
        # interleave line by line
        lines = [p.decode().splitlines(keepends=True) for p in parts]
        out = []
        for i in range(max(len(x) for x in lines)):
            for chunk in lines:
                if i < len(chunk):
                    out.append(chunk[i])
        return "".join(out).encode(), heads

    def test_round_trip_interleaved(self):
        src = self.make_source()
        # target holds prefixes a:2, b:1
        self.chain.import_tenant("a", src.export_tenant_range("a", 1, 2))
        self.chain.import_tenant("b", src.export_tenant_range("b", 1, 1))
        data, heads = self.interleaved_segments(src, {"a": 2, "b": 1})
        chunks = [data[i:i + 11] for i in range(0, len(data), 11)]
        got = self.chain.import_all_range_chunks(chunks, heads)
        self.assertEqual(len(got), 4 + 5)
        self.assertEqual(self.chain.verify("a"), {"ok": True, "count": 6})
        self.assertEqual(self.chain.verify("b"), {"ok": True, "count": 6})
        # returned records identical to the byte entry's
        other = AuditChain(self.path.with_name("other.jsonl"))
        other.import_tenant("a", src.export_tenant_range("a", 1, 2))
        other.import_tenant("b", src.export_tenant_range("b", 1, 1))
        self.assertEqual(got, other.import_all_range(data, heads))

    def test_empty_concatenation_is_noop(self):
        heads = [{"tenant": "a", "expected_count": 0, "expected_hash": ZERO}]
        self.assertEqual(
            self.chain.import_all_range_chunks([b"", b""], heads), [])
        self.assertFalse(self.path.exists())

    def test_unlisted_tenant_is_value_error(self):
        src = self.make_source()
        data = src.export_all()
        heads = [{"tenant": "a", "expected_count": 0, "expected_hash": ZERO}]
        with self.assertRaises(ValueError):
            self.chain.import_all_range_chunks([data], heads)

    def test_conflict_and_no_partial_write(self):
        src = self.make_source()
        self.chain.import_tenant("a", src.export_tenant_range("a", 1, 3))
        data, heads = self.interleaved_segments(src, {"a": 2, "b": 0})
        before = self.path.read_bytes()
        with self.assertRaises(AuditChainConflictError) as ctx:
            self.chain.import_all_range_chunks([data], heads)
        self.assertEqual(ctx.exception.reason, "conflict")
        self.assertEqual(ctx.exception.tenant, "a")
        self.assertEqual(self.path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
