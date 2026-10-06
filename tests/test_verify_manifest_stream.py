import copy
import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app import AuditChain, ZERO


def record_bytes(item):
    return (json.dumps(item, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


class VerifyManifestStreamTest(unittest.TestCase):
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
        self.chain.append("tëα", {"v": "世界→✓"})
        self.chain.append("a", {"i": 2})
        self.chain.append("b", {"i": 2})
        self.chain.append("a", {"i": 3})

    def snapshot(self):
        return self.path.read_bytes()

    def row(self, tenant="t", seq=1, prev=ZERO, event=None):
        item = {"tenant": tenant, "seq": seq, "event": event or {},
                "prev": prev}
        item["hash"] = AuditChain._hash(item)
        return record_bytes(item)

    def valid_pair(self):
        first = {"tenant": "a", "seq": 1, "event": {}, "prev": ZERO}
        first["hash"] = AuditChain._hash(first)
        second = {"tenant": "a", "seq": 2, "event": {}, "prev": first["hash"]}
        second["hash"] = AuditChain._hash(second)
        return record_bytes(first) + record_bytes(second)

    def chunkings(self, data):
        yield [data]
        if data:
            yield [data[i:i + 1] for i in range(len(data))]
        for cut in range(len(data) + 1):
            yield [data[:cut], data[cut:]]
        yield [data[:7], b"", data[7:19], b"", data[19:]]
        yield [b"", b"", data, b""]
        yield list(data[i:i + 5] for i in range(0, len(data), 5))

    # --- success: field-for-field identical to the bytes/chunks entries ---

    def test_matches_bytes_and_chunks_for_many_chunkings(self):
        self.seed()
        data = self.snapshot()
        manifest = self.other.manifest_bytes(data)
        for chunks in self.chunkings(data):
            rs = self.other.verify_manifest_stream(iter(chunks),
                                                   copy.deepcopy(manifest))
            rb = self.other.verify_manifest_bytes(data, copy.deepcopy(manifest))
            rc = self.other.verify_manifest_chunks(list(chunks),
                                                   copy.deepcopy(manifest))
            self.assertEqual(rs, rb, chunks)
            self.assertEqual(rs, rc, chunks)
            self.assertTrue(rs["ok"])
            self.assertEqual(rs["manifest"], manifest)

    def test_matches_with_split_multibyte_utf8(self):
        self.seed()
        data = self.snapshot()
        manifest = self.other.manifest_bytes(data)
        for cut in range(len(data) + 1):
            chunks = [data[:cut], data[cut:]]
            rs = self.other.verify_manifest_stream(iter(chunks),
                                                   copy.deepcopy(manifest))
            rb = self.other.verify_manifest_bytes(data, copy.deepcopy(manifest))
            self.assertEqual(rs, rb, cut)

    def test_empty_stream_matches_empty_manifest(self):
        empty = self.other.manifest_bytes(b"")
        for chunks in ([], iter([]), [b""], [b"", b""], (b"",)):
            rs = self.other.verify_manifest_stream(iter(list(chunks)),
                                                   copy.deepcopy(empty))
            self.assertEqual(rs, {"ok": True, "manifest": empty}, chunks)

    def test_byte_fields_describe_the_exact_original_bytes(self):
        row = self.row("a", 1)
        for data in (row, row.rstrip(b"\n")):
            manifest = self.other.manifest_bytes(data)
            rs = self.other.verify_manifest_stream(iter([data[:3], data[3:]]),
                                                   copy.deepcopy(manifest))
            self.assertTrue(rs["ok"])
            self.assertEqual(rs["manifest"]["byte_length"], len(data))
            self.assertEqual(rs["manifest"]["byte_sha256"],
                             hashlib.sha256(data).hexdigest())

    def test_manifest_argument_is_not_mutated_and_echo_is_computed(self):
        self.seed()
        data = self.snapshot()
        manifest = self.other.manifest_bytes(data)
        sent = copy.deepcopy(manifest)
        rs = self.other.verify_manifest_stream(iter([data]), sent)
        self.assertTrue(rs["ok"])
        self.assertEqual(sent, manifest)
        self.assertIsNot(rs["manifest"], sent)

    # --- fixed comparison order: byte_length, byte_sha256, tenant_order,
    #     tenant_count, tenant_hash, first difference only ---

    def test_manifest_mismatches_match_bytes_entry(self):
        self.seed()
        data = self.snapshot()
        good = self.other.manifest_bytes(data)
        variants = []

        bad = copy.deepcopy(good); bad["byte_length"] += 1
        variants.append(("byte_length", bad))
        bad = copy.deepcopy(good); bad["byte_sha256"] = "f" * 64
        variants.append(("byte_sha256", bad))
        bad = copy.deepcopy(good); bad["tenants"] = list(reversed(bad["tenants"]))
        variants.append(("tenant_order", bad))
        bad = copy.deepcopy(good); bad["tenants"] = bad["tenants"][:-1]
        variants.append(("tenant_order", bad))
        bad = copy.deepcopy(good)
        bad["tenants"].append({"tenant": "zzz", "count": 0, "hash": ZERO})
        variants.append(("tenant_order", bad))
        bad = copy.deepcopy(good); bad["tenants"][1]["count"] += 1
        variants.append(("tenant_count", bad))
        bad = copy.deepcopy(good); bad["tenants"][0]["hash"] = "e" * 64
        variants.append(("tenant_hash", bad))

        for field, bad in variants:
            chunks = [data[:11], data[11:]]
            rs = self.other.verify_manifest_stream(iter(chunks),
                                                   copy.deepcopy(bad))
            rb = self.other.verify_manifest_bytes(data, copy.deepcopy(bad))
            rc = self.other.verify_manifest_chunks([data], copy.deepcopy(bad))
            self.assertEqual(rs, rb, field)
            self.assertEqual(rs, rc, field)
            self.assertFalse(rs["ok"])
            self.assertEqual(rs["reason"], "manifest")
            self.assertEqual(rs["field"], field)

    def test_byte_length_wins_over_later_fields_in_fixed_order(self):
        # Non-empty data against the empty manifest: byte_length differs and
        # is reported before tenant_order.
        self.seed()
        data = self.snapshot()
        empty = self.other.manifest_bytes(b"")
        rs = self.other.verify_manifest_stream(iter([data]),
                                               copy.deepcopy(empty))
        self.assertEqual(rs, self.other.verify_manifest_bytes(data, empty))
        self.assertEqual(rs["field"], "byte_length")

    def test_tenant_order_locator_values(self):
        self.seed()
        data = self.snapshot()
        good = self.other.manifest_bytes(data)
        # Drop the last manifest tenant: at that index expected is None and
        # actual carries the snapshot tenant.
        bad = copy.deepcopy(good)
        dropped = bad["tenants"].pop()
        rs = self.other.verify_manifest_stream(iter([data]), bad)
        index = len(good["tenants"]) - 1
        self.assertEqual(rs["field"], "tenant_order")
        self.assertEqual(rs["index"], index)
        self.assertIsNone(rs["expected"])
        self.assertEqual(rs["actual"], dropped["tenant"])

    # --- chain integrity wins over the manifest comparison ---

    def test_corrupt_chain_matches_verify_all_bytes_across_cuts(self):
        good = self.valid_pair()
        first = {"tenant": "a", "seq": 1, "event": {}, "prev": ZERO}
        first["hash"] = AuditChain._hash(first)
        second_ok = {"tenant": "a", "seq": 2, "event": {},
                     "prev": first["hash"]}
        second_ok["hash"] = AuditChain._hash(second_ok)
        bad_seq = {"tenant": "a", "seq": 3, "event": {},
                   "prev": first["hash"]}
        bad_seq["hash"] = AuditChain._hash(bad_seq)
        bad_prev = {"tenant": "a", "seq": 2, "event": {}, "prev": ZERO}
        bad_prev["hash"] = AuditChain._hash(bad_prev)
        bad_hash = dict(second_ok, hash="0" * 64)
        corrupt = {
            "parse": good.replace(b'{"event"', b'{event', 1),
            "not_object": b"42\n",
            "missing_field": record_bytes(
                {"tenant": "a", "seq": 1, "event": {}, "prev": ZERO}),
            "sequence": record_bytes(first) + record_bytes(bad_seq),
            "digest_prev": record_bytes(first) + record_bytes(bad_prev),
            "digest_hash": record_bytes(first) + record_bytes(bad_hash),
            "utf8": record_bytes(first) + b"\xff\xfe\n",
        }
        manifest = self.other.manifest_bytes(good)
        for name, data in corrupt.items():
            for chunks in self.chunkings(data):
                rs = self.other.verify_manifest_stream(
                    iter(chunks), copy.deepcopy(manifest))
                rb = self.other.verify_manifest_bytes(
                    data, copy.deepcopy(manifest))
                self.assertEqual(rs, rb, (name, chunks))
                self.assertFalse(rs["ok"])
                self.assertIn(rs["reason"], ("missing", "sequence", "digest"))
                self.assertNotIn("field", rs)

    def test_first_bad_line_stops_pull_before_manifest_compare(self):
        first_line = self.row("a", 1)
        pulled = []

        def chunks():
            pulled.append(1); yield first_line
            pulled.append(2); yield b'{"tenant": "a", "seq": 2}\n'
            pulled.append(3); yield b'never-requested'

        good = self.other.manifest_bytes(self.valid_pair())
        bad_manifest = copy.deepcopy(good)
        bad_manifest["byte_sha256"] = "0" * 64
        rs = self.other.verify_manifest_stream(chunks(), bad_manifest)
        self.assertEqual((rs["ok"], rs["reason"], rs["at"]),
                         (False, "missing", 2))
        self.assertEqual(rs["tenant"], "a")
        self.assertEqual(pulled, [1, 2])

    def test_unterminated_bad_tail_reported_when_line_completes(self):
        first_line = self.row("a", 1)
        pulled = []

        def chunks():
            pulled.append(1); yield first_line
            pulled.append(2); yield b'{broken'   # no LF yet: undecidable
            pulled.append(3); yield b'}\n'       # line 2 now fails to parse
            pulled.append(4); yield b'never'

        manifest = self.other.manifest_bytes(first_line)
        rs = self.other.verify_manifest_stream(chunks(),
                                               copy.deepcopy(manifest))
        self.assertEqual(pulled, [1, 2, 3])
        self.assertEqual((rs["ok"], rs["reason"], rs["at"]),
                         (False, "missing", 2))

    def test_valid_history_is_consumed_to_the_end(self):
        data = self.valid_pair()
        manifest = self.other.manifest_bytes(data)
        pulled = []

        def chunks():
            for i in range(0, len(data), 7):
                pulled.append(i)
                yield data[i:i + 7]

        rs = self.other.verify_manifest_stream(chunks(),
                                               copy.deepcopy(manifest))
        self.assertTrue(rs["ok"])
        self.assertEqual(len(pulled), (len(data) + 6) // 7)

    # --- container boundary ---

    def test_bare_bytes_or_bytearray_is_value_error(self):
        manifest = self.other.manifest_bytes(b"")
        for container in (b"abc", bytearray(b"abc")):
            with self.assertRaises(ValueError):
                self.other.verify_manifest_stream(container,
                                                  copy.deepcopy(manifest))

    def test_non_iterable_container_is_value_error(self):
        manifest = self.other.manifest_bytes(b"")
        for container in (1, None, 1.5, object()):
            with self.assertRaises(ValueError):
                self.other.verify_manifest_stream(container,
                                                  copy.deepcopy(manifest))

    def test_first_non_bytes_member_is_value_error(self):
        manifest = self.other.manifest_bytes(b"")
        for members in ((b"", bytearray(b"x")), ("x",), (42, b"ok"),
                        (b"ok", None)):
            with self.assertRaises(ValueError):
                self.other.verify_manifest_stream(iter(members),
                                                  copy.deepcopy(manifest))

    def test_bad_member_stops_consumption(self):
        manifest = self.other.manifest_bytes(b"")
        pulled = []

        def chunks():
            pulled.append(1); yield b"abc"
            pulled.append(2); yield 7
            pulled.append(3); yield b"def"

        with self.assertRaises(ValueError):
            self.other.verify_manifest_stream(chunks(),
                                              copy.deepcopy(manifest))
        self.assertEqual(pulled, [1, 2])

    def test_iterator_exception_passes_through(self):
        manifest = self.other.manifest_bytes(b"")

        class Boom(Exception):
            pass

        def chunks():
            yield b"abc"
            raise Boom("from the caller")

        with self.assertRaises(Boom):
            self.other.verify_manifest_stream(chunks(),
                                              copy.deepcopy(manifest))

    # --- manifest boundary: before any chunk is pulled ---

    def test_bad_manifest_raises_without_pulling_a_chunk(self):
        sha = hashlib.sha256(b"").hexdigest()
        bad_manifests = [
            {},
            {"version": 1, "byte_length": 0, "byte_sha256": sha},
            {"version": 2, "byte_length": 0, "byte_sha256": sha,
             "tenants": []},
            {"version": True, "byte_length": 0, "byte_sha256": sha,
             "tenants": []},
            {"version": 1, "byte_length": -1, "byte_sha256": sha,
             "tenants": []},
            {"version": 1, "byte_length": 1.0, "byte_sha256": sha,
             "tenants": []},
            {"version": 1, "byte_length": 0, "byte_sha256": "A" * 64,
             "tenants": []},
            {"version": 1, "byte_length": 0, "byte_sha256": sha,
             "tenants": {}},
            {"version": 1, "byte_length": 0, "byte_sha256": sha,
             "tenants": [{"tenant": "a", "count": 0}]},
            {"version": 1, "byte_length": 0, "byte_sha256": sha,
             "tenants": [{"tenant": "a", "count": True, "hash": ZERO}]},
            {"version": 1, "byte_length": 0, "byte_sha256": sha,
             "tenants": [{"tenant": "a", "count": 0, "hash": "z" * 64}]},
            {"version": 1, "byte_length": 0, "byte_sha256": sha,
             "tenants": [
                 {"tenant": {"x": 1}, "count": 0, "hash": ZERO},
                 {"tenant": {"x": 1}, "count": 0, "hash": ZERO},
             ]},
        ]
        for bad in bad_manifests:
            pulled = []

            def chunks():
                pulled.append(1)
                yield b""

            with self.assertRaises(ValueError):
                self.other.verify_manifest_stream(chunks(), bad)
            self.assertEqual(pulled, [], bad)

    def test_manifest_boundary_precedes_container_boundary(self):
        with self.assertRaises(ValueError):
            self.other.verify_manifest_stream(123, {"version": 2})
        with self.assertRaises(ValueError):
            self.other.verify_manifest_stream(b"x",
                                              self.other.manifest_bytes(b""))

    # --- offline purity ---

    def test_never_reads_creates_or_modifies_constructor_path(self):
        self.assertFalse(self.other.path.exists())
        data = self.valid_pair()
        manifest = self.other.manifest_bytes(data)
        rs = self.other.verify_manifest_stream(
            iter([data[:9], b"", data[9:]]), copy.deepcopy(manifest))
        self.assertTrue(rs["ok"])
        self.assertFalse(self.other.path.exists())

    def test_does_not_mutate_chunks(self):
        data = self.valid_pair()
        pieces = [bytearray(data[i:i + 6]) for i in range(0, len(data), 6)]
        before = [bytes(p) for p in pieces]
        manifest = self.other.manifest_bytes(data)
        rs = self.other.verify_manifest_stream(
            (bytes(p) for p in pieces), copy.deepcopy(manifest))
        self.assertTrue(rs["ok"])
        self.assertEqual([bytes(p) for p in pieces], before)


if __name__ == "__main__":
    unittest.main()
