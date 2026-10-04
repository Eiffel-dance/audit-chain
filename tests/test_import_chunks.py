import json
import threading
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


class ImportChunksTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.path = Path(self.tmp.name) / "audit.jsonl"
        self.chain = AuditChain(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def write_rows(self, rows, path=None):
        path = path or self.path
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(r if isinstance(r, str) else json.dumps(r, sort_keys=True))
                f.write("\n")

    def seed_source(self, name="source.jsonl", spec=None):
        src = AuditChain(self.path.with_name(name))
        spec = spec or [("a", {"i": i, "s": "审计"}) for i in range(5)] \
            + [("b", {"i": i}) for i in range(3)]
        for tenant, event in spec:
            src.append(tenant, event)
        return src

    def all_chunkings(self, data):
        return [
            [data],
            [data[:1], data[1:]],
            [data[i:i + 1] for i in range(len(data))],
            [data[:7], b"", data[7:40], b"", data[40:]],
            [b"", b"", data, b""],
            [data[i:i + 13] for i in range(0, len(data), 13)],
            tuple(data[i:i + 3] for i in range(0, len(data), 3)),
            (data[i:i + 5] for i in range(0, len(data), 5)),
        ]

    # --- equivalence with the byte entries ---

    def test_import_tenant_chunks_matches_import_tenant_for_many_cuts(self):
        src = self.seed_source()
        data = src.export_tenant("a")
        baseline = AuditChain(self.path.with_name("base.jsonl"))
        expected_records = baseline.import_tenant("a", data)
        for idx, chunks in enumerate(self.all_chunkings(data)):
            dst = AuditChain(self.path.with_name(f"t-{idx}.jsonl"))
            records = dst.import_tenant_chunks("a", chunks)
            self.assertEqual(records, expected_records, idx)
            self.assertEqual(dst.export_tenant("a"), data, idx)
            self.assertEqual(dst.verify("a"), {"ok": True, "count": 5}, idx)

    def test_import_all_chunks_matches_import_all_for_many_cuts(self):
        src = self.seed_source()
        data = src.export_all()
        baseline = AuditChain(self.path.with_name("base.jsonl"))
        expected_records = baseline.import_all(data)
        self.assertEqual(baseline.export_all(), data)
        for idx, chunks in enumerate(self.all_chunkings(data)):
            dst = AuditChain(self.path.with_name(f"a-{idx}.jsonl"))
            records = dst.import_all_chunks(chunks)
            self.assertEqual(records, expected_records, idx)
            self.assertEqual(dst.export_all(), data, idx)

    def test_records_returned_in_physical_order_with_verbatim_values(self):
        src = self.seed_source()
        data = src.export_all()
        records = self.chain.import_all_chunks([data[:3], b"", data[3:]])
        self.assertEqual([r["tenant"] for r in records],
                         [json.loads(l)["tenant"]
                          for l in data.decode().splitlines()])
        self.assertTrue(all(
            set(r) == {"tenant", "seq", "event", "prev", "hash"}
            for r in records))

    def test_split_inside_multibyte_utf8_at_every_byte(self):
        src = AuditChain(self.path.with_name("u.jsonl"))
        src.append("t", {"msg": "héllo→世界"})
        src.append("t", {"msg": "✓" * 30})
        src.append("u", {"msg": "αβγδε"})
        tenant_data = src.export_tenant("t")
        all_data = src.export_all()
        for cut in range(len(tenant_data) + 1):
            dst = AuditChain(self.path.with_name(f"u1-{cut}.jsonl"))
            dst.import_tenant_chunks("t", [tenant_data[:cut], tenant_data[cut:]])
            self.assertEqual(dst.export_tenant("t"), tenant_data, cut)
        for cut in range(0, len(all_data) + 1, 7):
            dst = AuditChain(self.path.with_name(f"u2-{cut}.jsonl"))
            dst.import_all_chunks([all_data[:cut], all_data[cut:]])
            self.assertEqual(dst.export_all(), all_data, cut)

    # --- container / element boundary ---

    def test_bare_bytes_or_bytearray_container_is_value_error(self):
        for bad in (b"", b"{}", bytearray(b""), bytearray(b"abc")):
            with self.assertRaises(ValueError):
                self.chain.import_tenant_chunks("t", bad)
            with self.assertRaises(ValueError):
                self.chain.import_all_chunks(bad)
        self.assertFalse(self.path.exists())

    def test_non_iterable_container_is_value_error(self):
        for bad in (1, None, True, object()):
            with self.assertRaises(ValueError):
                self.chain.import_tenant_chunks("t", bad)
            with self.assertRaises(ValueError):
                self.chain.import_all_chunks(bad)

    def test_non_bytes_element_is_value_error(self):
        bad_cases = (
            [b"", "x"], [b"", 1], [b"", None], [b"", bytearray(b"")],
            [b"", [b""]], [b"{}", 3.0],
        )
        for bad in bad_cases:
            with self.assertRaises(ValueError):
                self.chain.import_tenant_chunks("t", bad)
            with self.assertRaises(ValueError):
                self.chain.import_all_chunks(bad)

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

    def test_tenant_boundary_is_value_error(self):
        src = self.seed_source()
        data = src.export_tenant("a")
        for bad in (float("nan"), float("inf"), {1: "x"}, object(),
                    b"bytes", {1, 2}):
            with self.assertRaises(ValueError):
                self.chain.import_tenant_chunks(bad, [data])

    def test_boundary_errors_raise_before_target_is_read(self):
        # corrupt target must never be consulted when the input is malformed
        row = {"tenant": "t", "seq": 1, "event": {}, "prev": ZERO}
        row["hash"] = AuditChain._hash(row)
        row["event"] = {"tampered": True}
        self.write_rows([row])
        before = self.path.read_bytes()
        with self.assertRaises(ValueError):
            self.chain.import_tenant_chunks(float("nan"), [b"x"])
        with self.assertRaises(ValueError):
            self.chain.import_tenant_chunks("t", b"not-a-container")
        with self.assertRaises(ValueError):
            self.chain.import_tenant_chunks("t", [1])
        with self.assertRaises(ValueError):
            self.chain.import_all_chunks(b"not-a-container")
        self.assertEqual(self.path.read_bytes(), before)
        self.assertFalse(
            self.path.with_name("never.jsonl").exists())

    # --- empty input: no-op, nothing created or read ---

    def test_empty_chunk_sequence_is_noop(self):
        self.assertFalse(self.path.exists())
        for empty in ([], (), iter([]), [b""], [b"", b""], (b"",)):
            self.assertEqual(self.chain.import_tenant_chunks("t", empty), [])
            self.assertEqual(self.chain.import_all_chunks(empty), [])
        self.assertFalse(self.path.exists())
        # even against a corrupt target no byte is read or changed
        self.path.write_bytes(b"\xff")
        corrupt = self.path.read_bytes()
        self.assertEqual(
            self.chain.import_tenant_chunks("t", iter([b"", b""])), [])
        self.assertEqual(self.chain.import_all_chunks([b""]), [])
        self.assertEqual(self.path.read_bytes(), corrupt)

    # --- input state errors: same fields and priority as byte entries ---

    def test_invalid_jsonl_raises_state_error_with_physical_line(self):
        cases = [
            ([b"{not json\n"], "t", 1, "missing", 1),
            ([b"\n"], "t", 1, "missing", 1),
            ([b"[1,2]\n"], "t", 1, "missing", 1),
            ([b"\xff"], "t", 1, "missing", 1),
        ]
        for chunks, tenant, seq, reason, line in cases:
            dst = AuditChain(self.path.with_name(f"e-{line}.jsonl"))
            with self.assertRaises(AuditChainStateError) as cm:
                dst.import_tenant_chunks(tenant, chunks)
            self.assertEqual((cm.exception.tenant, cm.exception.seq,
                              cm.exception.reason, cm.exception.line),
                             (tenant, seq, reason, line))
            with self.assertRaises(AuditChainStateError) as cm:
                dst.import_all_chunks(chunks)
            self.assertEqual((cm.exception.tenant, cm.exception.seq,
                              cm.exception.reason, cm.exception.line),
                             (None, None, reason, line))

    def test_sequence_and_digest_errors_across_chunk_boundary(self):
        good = {"tenant": "t", "seq": 1, "event": {}, "prev": ZERO}
        good["hash"] = AuditChain._hash(good)
        gap = {"tenant": "t", "seq": 3, "event": {}, "prev": "x" * 64}
        gap["hash"] = AuditChain._hash(gap)
        raw = record_bytes(good) + record_bytes(gap)
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.import_tenant_chunks("t", [raw[:3], b"", raw[3:]])
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("t", 2, "sequence", 2))
        bad_prev = {"tenant": "t", "seq": 2, "event": {}, "prev": "9" * 64}
        bad_prev["hash"] = AuditChain._hash(bad_prev)
        raw = record_bytes(good) + record_bytes(bad_prev)
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.import_tenant_chunks(
                "t", (raw[i:i + 1] for i in range(len(raw))))
        self.assertEqual((cm.exception.seq, cm.exception.reason, cm.exception.line),
                         (2, "digest", 2))

    def test_error_in_a_late_chunk_reports_its_line_and_writes_nothing(self):
        # a long valid prefix delivered in earlier chunks, damage at the end
        src = self.seed_source(spec=[("t", {}) for _ in range(6)])
        good = src.export_tenant("t")
        damaged = good + b"{oops\n"
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.import_tenant_chunks(
                "t", [good[:50], good[50:], b"{oops\n"])
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("t", 7, "missing", 7))
        self.assertFalse(self.path.exists())

    def test_foreign_tenant_in_single_tenant_input_is_value_error(self):
        src = self.seed_source()
        data = src.export_tenant("a")
        foreign = src.export_tenant("b")
        # a complete, valid prefix of the declared tenant, then a foreign
        # record: the foreign identity violates the single-tenant contract
        with self.assertRaises(ValueError):
            self.chain.import_tenant_chunks("a", [data[:1], data[1:], foreign])
        with self.assertRaises(ValueError):
            self.chain.import_tenant_chunks("b", [foreign, data])
        # an earlier malformed line still outranks a later foreign record
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.import_tenant_chunks("a", [data[:5], foreign])
        self.assertEqual(cm.exception.reason, "missing")
        # 1 and "1" are distinct canonical identities
        s2 = AuditChain(self.path.with_name("ids.jsonl"))
        s2.append(1, {}); s2.append("1", {})
        with self.assertRaises(ValueError):
            self.chain.import_tenant_chunks(
                1, list(s2.export_tenant_chunks("1", 10)))
        self.assertFalse(self.path.exists())

    def test_input_error_takes_priority_over_corrupt_target(self):
        good = {"tenant": "t", "seq": 1, "event": {}, "prev": ZERO}
        good["hash"] = AuditChain._hash(good)
        tampered = dict(good)
        tampered["event"] = {"z": 9}
        self.write_rows([tampered])  # target corrupt for t
        before = self.path.read_bytes()
        # digest-defective input AND corrupt target: input wins
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.import_tenant_chunks("t", [record_bytes(tampered)])
        self.assertEqual(cm.exception.reason, "digest")
        self.assertEqual(self.path.read_bytes(), before)

    def test_input_error_takes_priority_over_target_conflict(self):
        src = self.seed_source(spec=[("t", {}), ("t", {})])
        self.chain.import_tenant_chunks("t", [src.export_tenant("t")])
        before = self.path.read_bytes()
        with self.assertRaises(AuditChainStateError):
            self.chain.import_tenant_chunks("t", [b"{bad\n"])
        self.assertEqual(self.path.read_bytes(), before)

    # --- target conflict ---

    def test_existing_target_chain_is_conflict_with_actual_tail(self):
        src = self.seed_source(spec=[("t", {"i": i}) for i in range(3)])
        data = src.export_tenant("t")
        self.chain.import_tenant_chunks("t", [data])
        with self.assertRaises(AuditChainConflictError) as cm:
            self.chain.import_tenant_chunks("t", [data[:1], data[1:]])
        self.assertEqual((cm.exception.tenant, cm.exception.expected_count,
                          cm.exception.expected_hash, cm.exception.actual_count),
                         ("t", 0, ZERO, 3))
        self.assertEqual(cm.exception.reason, "conflict")

    def test_import_all_conflict_in_first_appearance_order(self):
        src = self.seed_source()
        data = src.export_all()
        self.chain.import_all_chunks([data])
        before = self.path.read_bytes()
        with self.assertRaises(AuditChainConflictError) as cm:
            self.chain.import_all_chunks([data[:2], b"", data[2:]])
        # first tenant to appear in the interleaved input is "a"
        first_tenant = json.loads(data.decode().splitlines()[0])["tenant"]
        self.assertEqual(cm.exception.tenant, first_tenant)
        self.assertEqual(cm.exception.reason, "conflict")
        self.assertEqual(self.path.read_bytes(), before)

    def test_conflict_writes_no_byte(self):
        src = self.seed_source(spec=[("t", {})])
        data = src.export_tenant("t")
        self.chain.append("other", {})
        self.chain.import_tenant_chunks("t", [data])
        before = self.path.read_bytes()
        with self.assertRaises(AuditChainConflictError):
            self.chain.import_tenant_chunks("t", [data])
        self.assertEqual(self.path.read_bytes(), before)

    # --- success: atomic commit, interleaved tenants, newline repair ---

    def test_commit_appends_after_other_tenant_with_block_atomicity(self):
        self.chain.append("other", {"k": "v"})
        src = self.seed_source()
        data = src.export_all()
        records = self.chain.import_all_chunks(
            (data[i:i + 11] for i in range(0, len(data), 11)))
        self.assertEqual(len(records), 8)
        self.assertEqual(self.chain.verify_all()["ok"], True)
        self.assertEqual(self.chain.export_all(), self.path.read_bytes())

    def test_imports_onto_target_without_trailing_newline(self):
        row = {"tenant": "o", "seq": 1, "event": {}, "prev": ZERO}
        row["hash"] = AuditChain._hash(row)
        self.path.write_bytes(json.dumps(row, sort_keys=True).encode())
        src = self.seed_source(spec=[("t", {}) for _ in range(2)])
        data = src.export_tenant("t")
        self.chain.import_tenant_chunks(
            "t", [data[:1], data[1:len(data) // 2], data[len(data) // 2:]])
        self.assertEqual(self.chain.verify("o"), {"ok": True, "count": 1})
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": 2})

    def test_content_committed_once_as_single_indivisible_block(self):
        src = self.seed_source(spec=[("t", {"i": i}) for i in range(10)])
        data = src.export_tenant("t")
        records = self.chain.import_tenant_chunks(
            "t", [data[i:i + 1] for i in range(len(data))])
        # exactly the source records, once each, in order
        self.assertEqual([r["seq"] for r in records], list(range(1, 11)))
        self.assertEqual(self.path.read_bytes(), data)

    # --- concurrency: readers see only pre/post state, one winning import ---

    def test_concurrent_readers_see_only_complete_pre_or_post_state(self):
        src = self.seed_source(spec=[("t", {"i": i}) for i in range(50)])
        data = src.export_tenant("t")
        chunks = [data[i:i + 17] for i in range(0, len(data), 17)]

        observed_counts = set()
        box = threading.Lock()
        stop = threading.Event()

        def reader():
            while not stop.is_set():
                r = self.chain.verify("t")
                with box:
                    observed_counts.add(
                        r["count"] if r["ok"] else ("bad", r["at"]))

        threads = [threading.Thread(target=reader) for _ in range(4)]
        for t in threads:
            t.start()
        self.chain.import_tenant_chunks("t", list(chunks))
        stop.set()
        for t in threads:
            t.join(timeout=2)

        # readers only ever observed the complete empty chain (0) or the
        # complete imported chain (50), never a partial count or a failure
        self.assertTrue(observed_counts.issubset({0, 50}))
        self.assertIn(50, observed_counts)
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": 50})

    def test_concurrent_conflicting_imports_one_winner_no_partial_bytes(self):
        src = self.seed_source(spec=[("t", {"i": i}) for i in range(20)])
        data = src.export_tenant("t")
        outcomes = []

        def importer():
            try:
                self.chain.import_tenant_chunks(
                    "t", (data[i:i + 9] for i in range(0, len(data), 9)))
                outcomes.append("ok")
            except AuditChainConflictError:
                outcomes.append("conflict")
            except AuditChainStateError:
                outcomes.append("state")

        threads = [threading.Thread(target=importer) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        self.assertEqual(outcomes.count("ok"), 1)
        self.assertEqual(outcomes.count("conflict"), 5)
        self.assertEqual(self.path.read_bytes(), data)
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": 20})


if __name__ == "__main__":
    unittest.main()
