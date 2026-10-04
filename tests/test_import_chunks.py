import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app import (
    AuditChain,
    AuditChainConflictError,
    AuditChainStateError,
    ZERO,
)


def record_bytes(item):
    return (json.dumps(item, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def chunkings(data):
    return [
        [data],
        [data[:1], data[1:]],
        [data[i:i + 1] for i in range(len(data))],
        [data[:10], b"", data[10:37], b"", data[37:]],
        [b"", b"", data, b""],
        list(data[i:i + 7] for i in range(0, len(data), 7)),
    ]


class ImportTenantChunksTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.path = Path(self.tmp.name) / "audit.jsonl"
        self.chain = AuditChain(self.path)
        self.source = AuditChain(self.path.with_name("source.jsonl"))

    def tearDown(self):
        self.tmp.cleanup()

    def make_export(self, tenant="t", n=5):
        for i in range(n):
            self.source.append(tenant, {"i": i, "s": "审计→✓"})
        return self.source.export_tenant(tenant)

    def valid_row(self, tenant="t", seq=1, prev=ZERO, event=None):
        row = {"tenant": tenant, "seq": seq, "event": event or {}, "prev": prev}
        row["hash"] = AuditChain._hash(row)
        return row

    # --- equivalence with import_tenant ---

    def test_matches_import_tenant_for_many_chunkings(self):
        data = self.make_export()
        for chunks in chunkings(data):
            target = AuditChain(self.path.with_name("target.jsonl"))
            records = target.import_tenant_chunks("t", chunks)
            self.assertEqual(records, self.source.read_tenant("t"))
            self.assertEqual(self.path.with_name("target.jsonl").read_bytes(),
                             data)
            self.assertEqual(target.verify("t"), {"ok": True, "count": 5})
            self.path.with_name("target.jsonl").unlink()

    def test_returned_records_match_byte_entry(self):
        data = self.make_export("t", 3)
        via_bytes = self.chain.import_tenant("t", data)
        other = AuditChain(self.path.with_name("other.jsonl"))
        via_chunks = other.import_tenant_chunks("t", [data[:11], data[11:]])
        self.assertEqual(via_chunks, via_bytes)
        self.assertEqual([r["seq"] for r in via_chunks], [1, 2, 3])

    def test_accepts_any_iterable_container(self):
        data = self.make_export("t", 2)
        halves = [data[:len(data) // 2], data[len(data) // 2:]]
        for container in (halves, tuple(halves), iter(halves),
                          (c for c in halves)):
            target = AuditChain(self.path.with_name("target.jsonl"))
            self.assertEqual(len(target.import_tenant_chunks("t", container)),
                             2)
            self.path.with_name("target.jsonl").unlink()

    def test_chunk_may_split_multibyte_utf8(self):
        data = self.make_export("t", 3)
        for cut in range(len(data) + 1):
            target = AuditChain(self.path.with_name("target.jsonl"))
            records = target.import_tenant_chunks(
                "t", [data[:cut], data[cut:]])
            self.assertEqual(len(records), 3, cut)
            self.assertEqual(target.verify("t"), {"ok": True, "count": 3})
            self.path.with_name("target.jsonl").unlink()

    def test_empty_chunks_are_noop(self):
        for chunks in ([], (), iter([]), [b""], [b"", b""]):
            self.assertEqual(self.chain.import_tenant_chunks("t", chunks), [])
        self.assertFalse(self.path.exists())

    # --- container and tenant boundary ---

    def test_bare_bytes_container_is_value_error(self):
        for bad in (b"", b"{}", bytearray(b""), bytearray(b"x")):
            with self.assertRaises(ValueError):
                self.chain.import_tenant_chunks("t", bad)
            with self.assertRaises(ValueError):
                self.chain.import_all_chunks(bad)
        self.assertFalse(self.path.exists())

    def test_non_iterable_container_is_value_error(self):
        for bad in (1, None, object()):
            with self.assertRaises(ValueError):
                self.chain.import_tenant_chunks("t", bad)
            with self.assertRaises(ValueError):
                self.chain.import_all_chunks(bad)
        self.assertFalse(self.path.exists())

    def test_non_bytes_element_is_value_error(self):
        for bad in ([b"", "x"], [b"", 1], [b"", None], [b"", bytearray(b"")],
                    [b"", [b""]]):
            with self.assertRaises(ValueError):
                self.chain.import_tenant_chunks("t", bad)
            with self.assertRaises(ValueError):
                self.chain.import_all_chunks(bad)
        self.assertFalse(self.path.exists())

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
            self.chain.import_tenant_chunks("t", gen())
        self.assertEqual(pulled, [1, 2])

        pulled.clear()
        with self.assertRaises(ValueError):
            self.chain.import_all_chunks(gen())
        self.assertEqual(pulled, [1, 2])

    def test_illegal_tenant_is_value_error(self):
        data = self.make_export("t", 1)
        for bad in (float("nan"), float("inf"), {"k": float("nan")},
                    {1: "x"}, object(), b"x", {1, 2}):
            with self.assertRaises(ValueError):
                self.chain.import_tenant_chunks(bad, [data])
        self.assertFalse(self.path.exists())

    def test_tenant_checked_before_container_is_consumed(self):
        pulled = []

        def gen():
            pulled.append(1)
            yield b""

        with self.assertRaises(ValueError):
            self.chain.import_tenant_chunks(float("nan"), gen())
        self.assertEqual(pulled, [])

    # --- input failures: same classes/fields as the byte entry points ---

    def test_foreign_tenant_is_value_error(self):
        self.source.append("a", {})
        self.source.append("b", {})
        data = self.source.export_all()
        chunks = [data[:13], data[13:]]
        with self.assertRaises(ValueError):
            self.chain.import_tenant_chunks("a", chunks)
        with self.assertRaises(ValueError):
            self.chain.import_tenant("a", data)
        self.assertFalse(self.path.exists())

    def test_invalid_jsonl_is_state_error_with_matching_fields(self):
        good = record_bytes(self.valid_row("t", 1))
        cases = [
            b"{not json\n",
            b"\n",
            b"[1,2]\n",
            good + b"\xff\n",
            good + b'{"tenant":"t","seq":2',  # truncated tail record
        ]
        for raw in cases:
            chunks = [raw[:3], b"", raw[3:]]
            with self.assertRaises(AuditChainStateError) as ctx:
                self.chain.import_tenant_chunks("t", chunks)
            with self.assertRaises(AuditChainStateError) as ctx2:
                self.chain.import_tenant("t", raw)
            self.assertEqual(
                (ctx.exception.tenant, ctx.exception.seq,
                 ctx.exception.reason, ctx.exception.line),
                (ctx2.exception.tenant, ctx2.exception.seq,
                 ctx2.exception.reason, ctx2.exception.line), raw)
        self.assertFalse(self.path.exists())

    def test_sequence_and_digest_defects(self):
        good = record_bytes(self.valid_row("t", 1))
        gap = record_bytes(self.valid_row("t", 3, prev="x" * 64))
        raw = good + gap
        with self.assertRaises(AuditChainStateError) as ctx:
            self.chain.import_tenant_chunks("t", [raw[:20], raw[20:]])
        self.assertEqual((ctx.exception.tenant, ctx.exception.seq,
                          ctx.exception.reason, ctx.exception.line),
                         ("t", 2, "sequence", 2))
        bad_prev = record_bytes(self.valid_row("t", 2, prev="9" * 64))
        raw = good + bad_prev
        with self.assertRaises(AuditChainStateError) as ctx:
            self.chain.import_tenant_chunks("t", [raw[:20], raw[20:]])
        self.assertEqual((ctx.exception.reason, ctx.exception.line),
                         ("digest", 2))
        self.assertFalse(self.path.exists())

    # --- target state: conflict, priority, atomicity ---

    def test_occupied_target_chain_is_conflict(self):
        data = self.make_export("t", 3)
        self.chain.append("t", {"local": True})
        chunks = [data[:9], data[9:]]
        with self.assertRaises(AuditChainConflictError) as ctx:
            self.chain.import_tenant_chunks("t", chunks)
        err = ctx.exception
        self.assertEqual(err.tenant, "t")
        self.assertEqual((err.expected_count, err.expected_hash), (0, ZERO))
        self.assertEqual(err.actual_count, 1)
        # identical to the byte entry point's conflict
        with self.assertRaises(AuditChainConflictError) as ctx2:
            self.chain.import_tenant("t", data)
        self.assertEqual((err.actual_count, err.actual_hash),
                         (ctx2.exception.actual_count,
                          ctx2.exception.actual_hash))

    def test_input_problem_beats_target_state(self):
        # corrupt input + conflicting target: the input error wins
        self.chain.append("t", {"local": True})
        raw = b"{not json\n"
        with self.assertRaises(AuditChainStateError):
            self.chain.import_tenant_chunks("t", [raw])
        # foreign tenant + conflicting target: the input ValueError wins
        self.source.append("other", {})
        foreign = self.source.export_tenant("other")
        with self.assertRaises(ValueError):
            self.chain.import_tenant_chunks("t", [foreign[:5], foreign[5:]])

    def test_failure_writes_no_partial_bytes(self):
        self.chain.append("keep", {"i": 1})
        before = self.path.read_bytes()
        data = self.make_export("t", 4)
        tampered = bytearray(data)
        tampered[-3] = ord("0") if tampered[-3] != ord("0") else ord("1")
        for chunks in ([bytes(tampered)], [bytes(tampered[:15]),
                                           bytes(tampered[15:])]):
            with self.assertRaises(AuditChainStateError):
                self.chain.import_tenant_chunks("t", chunks)
        keep_export = self.make_export("keep", 2)
        with self.assertRaises(AuditChainConflictError):
            self.chain.import_tenant_chunks("keep", [keep_export])
        with self.assertRaises(ValueError):
            self.chain.import_tenant_chunks("t", [data, b"", 1])
        self.assertEqual(self.path.read_bytes(), before)

    def test_commit_is_one_indivisible_block(self):
        # other tenants already on disk stay in place; the graft appends whole
        self.chain.append("x", {"i": 1})
        data = self.make_export("t", 3)
        before = self.path.read_bytes()
        records = self.chain.import_tenant_chunks("t", [data[:8], data[8:]])
        self.assertEqual(len(records), 3)
        after = self.path.read_bytes()
        self.assertTrue(after.startswith(before))
        self.assertEqual(after[len(before):], data)
        self.assertEqual(self.chain.verify_all()["ok"], True)


class ImportAllChunksTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.path = Path(self.tmp.name) / "audit.jsonl"
        self.chain = AuditChain(self.path)
        self.source = AuditChain(self.path.with_name("source.jsonl"))

    def tearDown(self):
        self.tmp.cleanup()

    def seed_source(self):
        self.source.append("a", {"i": 1, "s": "审计"})
        self.source.append("b", {"i": 1})
        self.source.append("a", {"i": 2})
        self.source.append("b", {"i": 2, "s": "→✓"})
        return self.source.export_all()

    def test_matches_import_all_for_many_chunkings(self):
        data = self.seed_source()
        expected = [json.loads(line)
                    for line in data.decode("utf-8").splitlines()]
        for chunks in chunkings(data):
            target = AuditChain(self.path.with_name("target.jsonl"))
            records = target.import_all_chunks(chunks)
            self.assertEqual(records, expected)
            self.assertEqual(self.path.with_name("target.jsonl").read_bytes(),
                             data)
            self.assertEqual(target.verify_all(),
                             {"ok": True, "tenants": [
                                 {"tenant": "a", "count": 2},
                                 {"tenant": "b", "count": 2}]})
            self.path.with_name("target.jsonl").unlink()

    def test_returned_records_in_physical_order(self):
        data = self.seed_source()
        records = self.chain.import_all_chunks([data[:25], b"", data[25:]])
        self.assertEqual([(r["tenant"], r["seq"]) for r in records],
                         [("a", 1), ("b", 1), ("a", 2), ("b", 2)])

    def test_empty_chunks_are_noop(self):
        for chunks in ([], [b""], [b"", b""]):
            self.assertEqual(self.chain.import_all_chunks(chunks), [])
        self.assertFalse(self.path.exists())

    def test_invalid_input_is_state_error_matching_byte_entry(self):
        good = self.seed_source()
        raw = bytearray(good)
        # corrupt the last record's hash
        idx = raw.rfind(b'"hash"')
        raw[idx + 9] = ord("0") if raw[idx + 9] != ord("0") else ord("1")
        raw = bytes(raw)
        chunks = [raw[:17], raw[17:40], raw[40:]]
        with self.assertRaises(AuditChainStateError) as ctx:
            self.chain.import_all_chunks(chunks)
        with self.assertRaises(AuditChainStateError) as ctx2:
            self.chain.import_all(raw)
        self.assertEqual(
            (ctx.exception.tenant, ctx.exception.seq,
             ctx.exception.reason, ctx.exception.line),
            (ctx2.exception.tenant, ctx2.exception.seq,
             ctx2.exception.reason, ctx2.exception.line))
        self.assertFalse(self.path.exists())

    def test_occupied_involved_chain_is_conflict(self):
        data = self.seed_source()
        self.chain.append("b", {"local": True})
        with self.assertRaises(AuditChainConflictError) as ctx:
            self.chain.import_all_chunks([data[:31], data[31:]])
        err = ctx.exception
        self.assertEqual(err.tenant, "b")
        self.assertEqual((err.expected_count, err.expected_hash), (0, ZERO))
        self.assertEqual(err.actual_count, 1)
        with self.assertRaises(AuditChainConflictError) as ctx2:
            self.chain.import_all(data)
        self.assertEqual((err.tenant, err.actual_count, err.actual_hash),
                         (ctx2.exception.tenant, ctx2.exception.actual_count,
                          ctx2.exception.actual_hash))

    def test_uninvolved_tenant_may_already_exist(self):
        data = self.seed_source()
        self.chain.append("zzz", {"local": True})
        records = self.chain.import_all_chunks([data])
        self.assertEqual(len(records), 4)
        self.assertEqual(self.chain.verify_all(),
                         {"ok": True, "tenants": [
                             {"tenant": "zzz", "count": 1},
                             {"tenant": "a", "count": 2},
                             {"tenant": "b", "count": 2}]})

    def test_failure_writes_no_partial_bytes(self):
        self.chain.append("keep", {"i": 1})
        before = self.path.read_bytes()
        data = self.seed_source()
        with self.assertRaises(AuditChainStateError):
            self.chain.import_all_chunks([data + b"\xff\n"])
        keep_source = AuditChain(self.path.with_name("keep-source.jsonl"))
        keep_source.append("keep", {"i": 99})
        with self.assertRaises(AuditChainConflictError):
            self.chain.import_all_chunks([keep_source.export_tenant("keep")])
        self.assertEqual(self.path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
