import hashlib
import json
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app import AuditChain, AuditChainStateError

ZERO = "0" * 64


def record_bytes(item):
    return (json.dumps(item, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


class ExportTenantTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.path = Path(self.tmp.name) / "audit.jsonl"
        self.chain = AuditChain(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def write_rows(self, rows):
        with self.path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(r if isinstance(r, str) else json.dumps(r, sort_keys=True))
                f.write("\n")

    def export_to_new_chain(self, tenant, name="exported.jsonl"):
        data = self.chain.export_tenant(tenant)
        out = Path(self.tmp.name) / name
        out.write_bytes(data)
        return AuditChain(out), data

    # --- success: bytes, offline verification and verbatim semantics ---

    def test_export_returns_utf8_bytes_matching_source_records(self):
        items = [self.chain.append("t", {"i": i, "s": "审计"}) for i in range(4)]
        data = self.chain.export_tenant("t")
        self.assertIsInstance(data, bytes)
        self.assertEqual(data, b"".join(record_bytes(it) for it in items))
        # every physical line is one standalone JSON object ending in \n
        for raw in data.splitlines():
            row = json.loads(raw.decode("utf-8"))
            self.assertEqual(set(row), {"tenant", "seq", "event", "prev", "hash"})
        self.assertTrue(data.endswith(b"\n"))

    def test_exported_file_verifies_offline_with_unchained_chain(self):
        for i in range(5):
            self.chain.append("t", {"i": i})
        new_chain, _ = self.export_to_new_chain("t")
        self.assertEqual(new_chain.verify("t"), {"ok": True, "count": 5})
        r = new_chain.verify_all()
        self.assertEqual(r, {"ok": True, "tenants": [{"tenant": "t", "count": 5}]})

    def test_export_preserves_seq_prev_hash_chain(self):
        items = [self.chain.append("t", {"i": i}) for i in range(3)]
        new_chain, data = self.export_to_new_chain("t")
        rows = [json.loads(l) for l in data.decode("utf-8").splitlines()]
        self.assertEqual([r["seq"] for r in rows], [1, 2, 3])
        self.assertEqual(rows[0]["prev"], ZERO)
        self.assertEqual(rows[1]["prev"], items[0]["hash"])
        self.assertEqual(rows[2]["prev"], items[1]["hash"])
        # independently recompute each hash from the exported bytes
        for r in rows:
            payload = json.dumps(
                {k: r[k] for k in ("tenant", "seq", "event", "prev")},
                sort_keys=True, separators=(",", ":"),
            ).encode("utf-8")
            self.assertEqual(hashlib.sha256(payload).hexdigest(), r["hash"])
        self.assertEqual(new_chain.verify("t", 3), {"ok": True, "count": 3})

    def test_export_does_not_create_or_modify_any_file(self):
        self.chain.append("t", {})
        self.chain.append("u", {})
        before = self.path.read_bytes()
        entries = sorted(p.name for p in Path(self.tmp.name).iterdir())
        data = self.chain.export_tenant("t")
        self.assertTrue(data)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(sorted(p.name for p in Path(self.tmp.name).iterdir()),
                         entries)

    # --- interleaving ---

    def test_export_filters_interleaved_tenants_keeps_physical_order(self):
        a1 = self.chain.append("a", {"i": 1})
        b1 = self.chain.append("b", {"i": 1})
        a2 = self.chain.append("a", {"i": 2})
        b2 = self.chain.append("b", {"i": 1})
        a3 = self.chain.append("a", {"i": 3})
        data = self.chain.export_tenant("a")
        rows = [json.loads(l) for l in data.decode("utf-8").splitlines()]
        self.assertEqual([r["event"] for r in rows], [{"i": 1}, {"i": 2}, {"i": 3}])
        self.assertTrue(all(r["tenant"] == "a" for r in rows))
        self.assertEqual(
            data, record_bytes(a1) + record_bytes(a2) + record_bytes(a3))
        new_a, _ = self.export_to_new_chain("a", "a.jsonl")
        self.assertEqual(new_a.verify("a"), {"ok": True, "count": 3})
        self.assertEqual(new_a.verify_all(),
                         {"ok": True, "tenants": [{"tenant": "a", "count": 3}]})
        new_b, data_b = self.export_to_new_chain("b", "b.jsonl")
        self.assertEqual(data_b, record_bytes(b1) + record_bytes(b2))
        self.assertEqual(new_b.verify("b"), {"ok": True, "count": 2})

    def test_export_unknown_tenant_is_empty_but_still_validates_file(self):
        self.chain.append("a", {})
        self.chain.append("a", {})
        self.assertEqual(self.chain.export_tenant("zzz"), b"")
        out = Path(self.tmp.name) / "empty.jsonl"
        out.write_bytes(self.chain.export_tenant("zzz"))
        fresh = AuditChain(out)
        self.assertEqual(fresh.verify("zzz"), {"ok": True, "count": 0})
        self.assertEqual(fresh.verify_all(), {"ok": True, "tenants": []})

    def test_export_rejects_distinct_json_identities_mixing(self):
        # 1, 1.0, true, "1" are four unrelated chains: each export contains
        # only its own identity even though the records interleave on disk.
        for t in (1, 1.0, True, "1"):
            self.chain.append(t, {})
            self.chain.append(t, {})
        for t in (1, 1.0, True, "1"):
            data = self.chain.export_tenant(t)
            rows = [json.loads(l) for l in data.decode("utf-8").splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(
                AuditChain._tenant_key(r["tenant"]) == AuditChain._tenant_key(t)
                for r in rows))
            out = Path(self.tmp.name) / f"tenant_{type(t).__name__}.jsonl"
            out.write_bytes(data)
            self.assertEqual(AuditChain(out).verify(t), {"ok": True, "count": 2})

    # --- complex JSON tenants and events ---

    def test_export_complex_object_tenant_key_order_normalized(self):
        t1 = {"a": [1, {"x": 1, "y": 2}], "b": {"p": True, "q": None}}
        t2 = {"b": {"q": None, "p": True}, "a": [1, {"y": 2, "x": 1}]}
        self.chain.append(t1, {"k": "v", "n": [1, 2, {"z": None}]})
        self.chain.append(t2, {"k": "w"})
        self.chain.append("other", {})
        data = self.chain.export_tenant({"a": [1, {"x": 1, "y": 2}],
                                         "b": {"p": True, "q": None}})
        rows = [json.loads(l) for l in data.decode("utf-8").splitlines()]
        self.assertEqual(len(rows), 2)
        self.assertEqual([r["seq"] for r in rows], [1, 2])
        self.assertEqual(rows[1]["prev"], rows[0]["hash"])
        out = Path(self.tmp.name) / "complex.jsonl"
        out.write_bytes(data)
        chain = AuditChain(out)
        self.assertEqual(chain.verify(t1), {"ok": True, "count": 2})
        self.assertTrue(chain.verify_all()["ok"])

    def test_export_unicode_and_deep_event_payloads_round_trip(self):
        events = [{"s": "审计-日志"}, {"f": 1.5, "b": True, "0": None},
                  [[], {" nested ": ["a", "b"]}]]
        for e in events:
            self.chain.append("t", e)
        data = self.chain.export_tenant("t")
        rows = [json.loads(l) for l in data.decode("utf-8").splitlines()]
        self.assertEqual([r["event"] for r in rows], events)
        out = Path(self.tmp.name) / "u.jsonl"
        out.write_bytes(data)
        self.assertEqual(AuditChain(out).verify("t"), {"ok": True, "count": 3})

    # --- empty history ---

    def test_export_missing_file_returns_empty_bytes_creates_nothing(self):
        self.assertFalse(self.path.exists())
        self.assertEqual(self.chain.export_tenant("t"), b"")
        self.assertFalse(self.path.exists())

    def test_export_empty_file_returns_empty_bytes(self):
        self.path.write_bytes(b"")
        self.assertEqual(self.chain.export_tenant("t"), b"")

    def test_export_empty_written_file_is_a_valid_empty_snapshot(self):
        out = Path(self.tmp.name) / "out.jsonl"
        out.write_bytes(self.chain.export_tenant("t"))
        chain = AuditChain(out)
        self.assertEqual(chain.verify("t"), {"ok": True, "count": 0})
        self.assertEqual(chain.verify_all(), {"ok": True, "tenants": []})

    # --- determinism ---

    def test_repeated_exports_are_byte_identical(self):
        for i in range(6):
            self.chain.append("t", {"i": i})
            self.chain.append("other", {"i": i})
        first = self.chain.export_tenant("t")
        for _ in range(5):
            self.assertEqual(self.chain.export_tenant("t"), first)

    # --- tenant input boundary: ValueError before any read ---

    def test_export_rejects_illegal_tenant_with_value_error(self):
        bad_tenants = [
            float("nan"), float("inf"), float("-inf"),
            {"k": float("nan")}, [1, [float("-inf")]],
            {1: "x"}, {"a": {2: 3}}, object(), b"bytes", {1, 2}, ("a", 1),
        ]
        for bad in bad_tenants:
            with self.assertRaises(ValueError):
                self.chain.export_tenant(bad)
        self.assertFalse(self.path.exists())

    def test_export_rejects_cyclic_tenant_with_value_error(self):
        cyc = []
        cyc.append(cyc)
        with self.assertRaises(ValueError):
            self.chain.export_tenant(cyc)
        d = {}
        d["self"] = d
        with self.assertRaises(ValueError):
            self.chain.export_tenant(d)

    def test_export_value_error_beats_corrupt_history_and_keeps_bytes(self):
        row = {"tenant": "t", "seq": 1, "event": {}, "prev": ZERO}
        row["hash"] = AuditChain._hash(row)
        row["event"] = {"tampered": True}  # digest broken at line 1
        self.write_rows([row])
        before = self.path.read_bytes()
        for bad in (float("nan"), {"k": float("inf")}, {1: "x"}):
            with self.assertRaises(ValueError):
                self.chain.export_tenant(bad)
        self.assertEqual(self.path.read_bytes(), before)

    # --- source state errors: same first broken point as append/verify ---

    def test_export_unparseable_line_raises_missing_at_line(self):
        other = {"tenant": "x", "seq": 1, "event": {}, "prev": ZERO}
        other["hash"] = AuditChain._hash(other)
        self.write_rows([other, "{not json"])
        before = self.path.read_bytes()
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.export_tenant("t")
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("t", 1, "missing", 2))
        self.assertEqual(self.path.read_bytes(), before)

    def test_export_missing_field_raises_missing_at_own_seq(self):
        self.chain.append("t", {})
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(
                {"tenant": "t", "seq": 2, "event": {}, "prev": "x"}) + "\n")
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.export_tenant("t")
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("t", 2, "missing", 2))

    def test_export_sequence_error_raises_sequence_at_expected(self):
        good = {"tenant": "t", "seq": 1, "event": {}, "prev": ZERO}
        good["hash"] = AuditChain._hash(good)
        gap = {"tenant": "t", "seq": 3, "event": {}, "prev": good["hash"]}
        gap["hash"] = AuditChain._hash(gap)
        self.write_rows([good, gap])
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.export_tenant("t")
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("t", 2, "sequence", 2))

    def test_export_digest_error_raises_digest_and_no_partial_prefix(self):
        a1 = self.chain.append("t", {"i": 1})
        a2 = self.chain.append("t", {"i": 2})
        rows = [json.loads(l) for l in self.path.read_text().splitlines()]
        rows[0]["event"] = {"i": 99}  # break the first hash
        self.write_rows(rows)
        before = self.path.read_bytes()
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.export_tenant("t")
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("t", 1, "digest", 1))
        # failure must not surface the verified prefix and must not write
        self.assertEqual(self.path.read_bytes(), before)

    def test_export_illegal_utf8_raises_missing_with_physical_line(self):
        valid = {"tenant": "t", "seq": 1, "event": {}, "prev": ZERO}
        valid["hash"] = AuditChain._hash(valid)
        before = (json.dumps(valid, sort_keys=True) + "\n").encode("utf-8") + b"\xff"
        self.path.write_bytes(before)
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.export_tenant("t")
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("t", 2, "missing", 2))
        self.assertEqual(self.path.read_bytes(), before)

    def test_export_error_priority_digest_before_later_missing(self):
        good = {"tenant": "t", "seq": 1, "event": {}, "prev": ZERO}
        good["hash"] = AuditChain._hash(good)
        tampered = dict(good)
        tampered["event"] = {"z": 9}
        self.path.write_bytes(
            (json.dumps(tampered, sort_keys=True) + "\n").encode("utf-8")
            + b"{oops\n")
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.export_tenant("t")
        self.assertEqual((cm.exception.seq, cm.exception.reason, cm.exception.line),
                         (1, "digest", 1))

    def test_export_error_priority_sequence_before_later_bad_bytes(self):
        row = {"tenant": "t", "seq": 2, "event": {}, "prev": ZERO}
        row["hash"] = AuditChain._hash(row)
        self.path.write_bytes(
            (json.dumps(row, sort_keys=True) + "\n").encode("utf-8") + b"\xff")
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.export_tenant("t")
        self.assertEqual((cm.exception.seq, cm.exception.reason, cm.exception.line),
                         (1, "sequence", 1))

    def test_export_error_priority_missing_before_later_sequence(self):
        self.path.write_bytes(b"{oops\n" + json.dumps(
            {"tenant": "t", "seq": 5, "event": {}, "prev": ZERO,
             "hash": "x"}).encode("utf-8") + b"\n")
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.export_tenant("t")
        self.assertEqual((cm.exception.seq, cm.exception.reason, cm.exception.line),
                         (1, "missing", 1))

    def test_export_other_tenant_corruption_does_not_block_export(self):
        self.chain.append("b", {})
        rows = [json.loads(l) for l in self.path.read_text().splitlines()]
        rows[0]["event"] = {"x": 1}  # b's digest is broken; t unaffected
        self.write_rows(rows)
        self.chain.append("t", {"ok": True})
        data = self.chain.export_tenant("t")
        rows = [json.loads(l) for l in data.decode("utf-8").splitlines()]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["tenant"], "t")

    def test_export_target_corruption_after_interleaving_reports_target(self):
        # interleaving valid b record does not move the target's error line
        good = {"tenant": "a", "seq": 1, "event": {}, "prev": ZERO}
        good["hash"] = AuditChain._hash(good)
        b = {"tenant": "b", "seq": 1, "event": {}, "prev": ZERO}
        b["hash"] = AuditChain._hash(b)
        bad = {"tenant": "a", "seq": 3, "event": {}, "prev": good["hash"]}
        bad["hash"] = AuditChain._hash(bad)
        self.write_rows([good, b, bad])
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.export_tenant("a")
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("a", 2, "sequence", 3))


class ExportConcurrencyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.path = Path(self.tmp.name) / "audit.jsonl"
        self.chain = AuditChain(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_export_sees_only_complete_pre_or_post_append_snapshot(self):
        # Seed a baseline; while appends pour in, every export must be a
        # prefix-complete snapshot: exported seqs are exactly 1..k, hashes
        # link, and the bytes verify offline on a fresh AuditChain.
        for i in range(10):
            self.chain.append("t", {"i": i})
        seen_counts = set()
        problems = []
        box = threading.Lock()
        stop = threading.Event()

        def exporter():
            while not stop.is_set():
                data = self.chain.export_tenant("t")
                rows = [json.loads(l) for l in data.decode("utf-8").splitlines()]
                seqs = [r["seq"] for r in rows]
                if seqs != list(range(1, len(rows) + 1)):
                    with box:
                        problems.append(("seqs", seqs))
                    return
                prev = ZERO
                for r in rows:
                    if r["prev"] != prev:
                        with box:
                            problems.append(("prev", r["seq"]))
                        return
                    if r["hash"] != AuditChain._hash(r):
                        with box:
                            problems.append(("hash", r["seq"]))
                        return
                    prev = r["hash"]
                with box:
                    seen_counts.add(len(rows))

        def other_exporter():
            # distinct tenant must always export exactly zero records here
            while not stop.is_set():
                if self.chain.export_tenant("never-used") != b"":
                    with box:
                        problems.append(("foreign",))
                    return

        exporters = [threading.Thread(target=exporter) for _ in range(4)]
        exporters += [threading.Thread(target=other_exporter) for _ in range(2)]
        for t in exporters:
            t.start()

        def writer(i):
            for j in range(40):
                self.chain.append("t", {"w": i, "j": j})

        writers = [threading.Thread(target=writer, args=(i,)) for i in range(6)]
        for t in writers:
            t.start()
        for t in writers:
            t.join()
        stop.set()
        for t in exporters:
            t.join(timeout=2)

        self.assertEqual(problems, [])
        total = 10 + 6 * 40
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": total})
        # final export is complete and verifies offline on a fresh path
        final = self.chain.export_tenant("t")
        out = Path(self.tmp.name) / "final.jsonl"
        out.write_bytes(final)
        self.assertEqual(AuditChain(out).verify("t"), {"ok": True, "count": total})
        self.assertIn(total, seen_counts)
        # observed counts must be a staircase with no torn prefixes: every
        # count between min and max observed was itself a valid snapshot
        self.assertTrue(seen_counts, "no exports observed")
        lo, hi = min(seen_counts), max(seen_counts)
        self.assertTrue(all(10 <= k <= total for k in seen_counts))
        self.assertEqual(hi, total)

    def test_concurrent_appends_to_other_tenants_keep_export_stable(self):
        for i in range(5):
            self.chain.append("a", {"i": i})
        stop = threading.Event()
        captured = set()
        box = threading.Lock()

        def exporter():
            while not stop.is_set():
                data = self.chain.export_tenant("a")
                with box:
                    captured.add(data)

        t = threading.Thread(target=exporter)
        t.start()

        def writer():
            for i in range(100):
                self.chain.append("b", {"i": i})

        ws = [threading.Thread(target=writer) for _ in range(4)]
        for w in ws:
            w.start()
        for w in ws:
            w.join()
        stop.set()
        t.join(timeout=2)

        # a never changed: every export is byte-identical, with no b records
        self.assertEqual(len(captured), 1)
        rows = [json.loads(x) for x in next(iter(captured)).decode().splitlines()]
        self.assertEqual(len(rows), 5)
        self.assertTrue(all(r["tenant"] == "a" for r in rows))

    def test_export_under_concurrent_corruption_raises_or_succeeds_cleanly(self):
        # The source is swapped for a corrupt one inside an exclusive lease, so
        # lock-holding exporters observe either the whole intact snapshot or
        # the whole corrupt one: they must never receive a partial prefix.
        import fcntl
        self.chain.append("t", {"v": 1})
        outcomes = []
        box = threading.Lock()
        stop = threading.Event()

        def exporter():
            while not stop.is_set():
                try:
                    data = self.chain.export_tenant("t")
                    rows = [json.loads(l) for l in data.decode().splitlines()]
                    with box:
                        outcomes.append(("ok", len(rows)))
                except AuditChainStateError as e:
                    with box:
                        outcomes.append(("err", e.seq, e.reason, e.line))
                except Exception as e:  # noqa: BLE001
                    with box:
                        outcomes.append(("leak", type(e).__name__))

        threads = [threading.Thread(target=exporter) for _ in range(4)]
        for t in threads:
            t.start()
        # give exporters a chance to observe the intact snapshot
        while not outcomes:
            pass
        row = json.loads(self.path.read_text())
        row["event"] = {"v": 2}
        corrupt = (json.dumps(row, sort_keys=True) + "\n").encode("utf-8")
        with open(self.path, "r+b") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            f.seek(0)
            f.truncate()
            f.write(corrupt)
            f.flush()
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        stop.set()
        for t in threads:
            t.join(timeout=2)

        kinds = {o[0] for o in outcomes}
        self.assertTrue(kinds <= {"ok", "err"})
        self.assertIn("err", kinds)
        for o in outcomes:
            if o[0] == "ok":
                self.assertEqual(o[1], 1)  # the intact pre-corruption snapshot
            else:
                self.assertEqual(tuple(o[1:]), (1, "digest", 1))


if __name__ == "__main__":
    unittest.main()
