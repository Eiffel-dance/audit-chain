import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app import AuditChain, AuditChainStateError

ZERO = "0" * 64


def record_bytes(item):
    return (json.dumps(item, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


class ExportChunksTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.path = Path(self.tmp.name) / "audit.jsonl"
        self.chain = AuditChain(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def seed(self):
        self.chain.append("a", {"i": 1, "s": "审计→✓"})
        self.chain.append("b", {"i": 1})
        self.chain.append("a", {"i": 2})
        self.chain.append("a", {"i": 3, "s": "x" * 50})
        self.chain.append("b", {"i": 2})

    # --- equivalence with export_tenant / export_all ---

    def test_concatenation_matches_byte_exports_for_many_sizes(self):
        self.seed()
        for tenant, get_full, get_chunks in (
            ("a", self.chain.export_tenant, self.chain.export_tenant_chunks),
            ("b", self.chain.export_tenant, self.chain.export_tenant_chunks),
            ("zzz", self.chain.export_tenant, self.chain.export_tenant_chunks),
        ):
            full = get_full(tenant)
            sizes = [1, 2, 3, 7, 64, max(1, len(full)), len(full) + 1, 10**6]
            for size in sizes:
                chunks = get_chunks(tenant, size)
                self.assertEqual(b"".join(get_chunks(tenant, size)), full,
                                 (tenant, size))
                self.assertEqual(list(chunks), list(get_chunks(tenant, size)))
        full_all = self.chain.export_all()
        for size in (1, 5, 17, len(full_all), len(full_all) + 1, 10**6):
            self.assertEqual(b"".join(self.chain.export_all_chunks(size)),
                             full_all, size)

    def test_chunk_shapes_and_element_types(self):
        self.seed()
        full = self.chain.export_tenant("a")
        for size in (1, 7, len(full), len(full) + 5):
            chunks = list(self.chain.export_tenant_chunks("a", size))
            self.assertTrue(all(isinstance(c, bytes) for c in chunks))
            self.assertTrue(all(len(c) == size for c in chunks[:-1]), size)
            if full:
                self.assertTrue(1 <= len(chunks[-1]) <= size, size)
            self.assertEqual(b"".join(chunks), full, size)
        chunks = list(self.chain.export_all_chunks(3))
        self.assertTrue(all(isinstance(c, bytes) for c in chunks))
        self.assertEqual(b"".join(chunks), self.chain.export_all())

    def test_chunk_may_split_multibyte_utf8_and_newlines(self):
        self.chain.append("t", {"msg": "héllo→世界"})
        self.chain.append("t", {"msg": "✓" * 40})
        full = self.chain.export_tenant("t")
        # size 1 forces cuts inside multibyte sequences and around every \n
        self.assertEqual(b"".join(self.chain.export_tenant_chunks("t", 1)),
                         full)
        self.assertEqual(b"".join(self.chain.export_all_chunks(1)),
                         self.chain.export_all())

    def test_empty_results(self):
        self.assertEqual(list(self.chain.export_tenant_chunks("t", 4)), [])
        self.assertEqual(list(self.chain.export_all_chunks(4)), [])
        self.assertFalse(self.path.exists())
        self.seed()
        self.assertEqual(list(self.chain.export_tenant_chunks("zzz", 4)), [])

    def test_export_is_pinned_to_one_snapshot(self):
        self.seed()
        before_t = self.chain.export_tenant("a")
        before_all = self.chain.export_all()
        it_t = self.chain.export_tenant_chunks("a", 5)
        it_all = self.chain.export_all_chunks(5)
        # appends after the call must not leak into the in-flight export
        self.chain.append("a", {"i": 99})
        self.chain.append("c", {"i": 1})
        self.assertEqual(b"".join(it_t), before_t)
        self.assertEqual(b"".join(it_all), before_all)

    # --- boundary ---

    def test_chunk_size_boundary(self):
        self.seed()
        for bad in (0, -1, -100, True, False, 1.0, 2.5, "1", None, object(),
                    b"x", [1]):
            with self.assertRaises(ValueError):
                self.chain.export_tenant_chunks("a", bad)
            with self.assertRaises(ValueError):
                self.chain.export_all_chunks(bad)

    def test_tenant_boundary(self):
        self.seed()
        for bad in (float("nan"), float("inf"), {"k": float("nan")},
                    {1: "x"}, object(), b"x", {1, 2}):
            with self.assertRaises(ValueError):
                self.chain.export_tenant_chunks(bad, 4)

    def test_boundary_checked_without_reading_or_creating(self):
        phantom = self.path.with_name("phantom.jsonl")
        chain = AuditChain(phantom)
        for bad in (0, True, 1.0, "x", None):
            with self.assertRaises(ValueError):
                chain.export_all_chunks(bad)
            with self.assertRaises(ValueError):
                chain.export_tenant_chunks("t", bad)
        with self.assertRaises(ValueError):
            chain.export_tenant_chunks(float("nan"), 4)
        self.assertFalse(phantom.exists())

    # --- corruption: state error before any chunk, no partial export ---

    def test_corruption_raises_state_error_with_fields(self):
        self.seed()
        data = bytearray(self.path.read_bytes())
        # tamper with a hash byte inside the first record (tenant "a")
        idx = data.find(b'"hash"')
        data[idx + 9] = ord("0") if data[idx + 9] != ord("0") else ord("1")
        self.path.write_bytes(bytes(data))
        for size in (1, 16, 10**6):
            with self.assertRaises(AuditChainStateError) as ctx:
                self.chain.export_tenant_chunks("a", size)
            err = ctx.exception
            self.assertEqual(err.tenant, "a")
            self.assertEqual(err.reason, "digest")
            self.assertIsNotNone(err.seq)
            self.assertIsNotNone(err.line)
            # byte entry point reports the identical error
            with self.assertRaises(AuditChainStateError) as ctx2:
                self.chain.export_tenant("a")
            self.assertEqual((err.tenant, err.seq, err.reason, err.line),
                             (ctx2.exception.tenant, ctx2.exception.seq,
                              ctx2.exception.reason, ctx2.exception.line))
        with self.assertRaises(AuditChainStateError):
            self.chain.export_all_chunks(8)
        # an untainted tenant still exports fine
        self.assertEqual(b"".join(self.chain.export_tenant_chunks("b", 3)),
                         self.chain.export_tenant("b"))

    def test_unparseable_log_raises_before_any_chunk(self):
        self.path.write_bytes(b'{"tenant": "t", "seq": 1, "event": {}, '
                              b'"prev": "' + ZERO.encode() + b'", "hash": "'
                              + b"x" * 64 + b'"}\n\xff\n')
        for size in (1, 4):
            with self.assertRaises(AuditChainStateError):
                self.chain.export_tenant_chunks("t", size)
            with self.assertRaises(AuditChainStateError):
                self.chain.export_all_chunks(size)

    # --- offline purity ---

    def test_never_creates_or_modifies_the_log(self):
        phantom = self.path.with_name("phantom.jsonl")
        chain = AuditChain(phantom)
        self.assertEqual(list(chain.export_tenant_chunks("t", 2)), [])
        self.assertEqual(list(chain.export_all_chunks(2)), [])
        self.assertFalse(phantom.exists())
        self.seed()
        before = self.path.read_bytes()
        self.assertTrue(list(self.chain.export_tenant_chunks("a", 3)))
        self.assertTrue(list(self.chain.export_all_chunks(3)))
        self.assertEqual(self.path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
