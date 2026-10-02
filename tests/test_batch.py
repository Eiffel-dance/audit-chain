import hashlib
import json
import multiprocessing as mp
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app import AuditChain, AuditChainStateError

ZERO = "0" * 64


def _record_bytes(item):
    return (json.dumps(item, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _tenant_rows(path):
    rows = [json.loads(l) for l in Path(path).read_text().splitlines()]
    by = {}
    for r in rows:
        by.setdefault(json.dumps(r["tenant"], sort_keys=True), []).append(r)
    return rows, by


class AppendBatchTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.path = Path(self.tmp.name) / "audit.jsonl"
        self.chain = AuditChain(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, rows):
        with self.path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(r if isinstance(r, str) else json.dumps(r, sort_keys=True))
                f.write("\n")

    # --- empty batch: pure no-op ---

    def test_empty_batch_returns_empty_list_and_creates_nothing(self):
        self.assertEqual(self.chain.append_batch("t", []), [])
        self.assertFalse(self.path.exists())

    def test_empty_batch_does_not_touch_existing_bytes(self):
        self.chain.append("a", {})
        self.chain.append("b", {})
        before = self.path.read_bytes()
        self.assertEqual(self.chain.append_batch("t", []), [])
        self.assertEqual(self.path.read_bytes(), before)

    # --- basic chaining and return value ---

    def test_batch_assigns_consecutive_seqs_and_chained_prevs(self):
        items = self.chain.append_batch("t", [{"i": 1}, {"i": 2}, {"i": 3}])
        self.assertEqual([it["seq"] for it in items], [1, 2, 3])
        self.assertEqual(items[0]["prev"], ZERO)
        self.assertEqual(items[1]["prev"], items[0]["hash"])
        self.assertEqual(items[2]["prev"], items[1]["hash"])
        for it in items:
            self.assertEqual(it["tenant"], "t")
            self.assertEqual(len(it["hash"]), 64)

    def test_returned_records_match_disk_fields_and_values_in_order(self):
        events = [{"i": 1}, {"s": "x"}, [1, 2], None, True, "e"]
        items = self.chain.append_batch("t", events)
        rows = [json.loads(l) for l in self.path.read_text().splitlines()]
        self.assertEqual(len(rows), len(items))
        for item, row in zip(items, rows):
            self.assertEqual(row, item)
            self.assertEqual(set(row), {"tenant", "seq", "event", "prev", "hash"})
            # independently recompute the digest from the on-disk row
            payload = json.dumps(
                {k: row[k] for k in ("tenant", "seq", "event", "prev")},
                sort_keys=True, separators=(",", ":"),
            ).encode("utf-8")
            self.assertEqual(hashlib.sha256(payload).hexdigest(), row["hash"])
        self.assertEqual([r["event"] for r in rows], events)

    def test_one_record_per_line_machine_readable_jsonl(self):
        self.chain.append_batch("t", [{"i": i} for i in range(5)])
        raw = self.path.read_bytes()
        self.assertTrue(raw.endswith(b"\n"))
        lines = raw.splitlines()
        self.assertEqual(len(lines), 5)
        for line in lines:
            json.loads(line)  # each physical line is one complete JSON object

    def test_batch_first_seq_and_prev_match_existing_chain_tip(self):
        a = self.chain.append("t", {"i": 0})
        items = self.chain.append_batch("t", [{"i": 1}, {"i": 2}])
        self.assertEqual(items[0]["seq"], 2)
        self.assertEqual(items[0]["prev"], a["hash"])
        self.assertEqual(items[1]["seq"], 3)
        self.assertEqual(items[1]["prev"], items[0]["hash"])
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": 3})

    # --- equivalence with sequential single appends ---

    def test_batch_bytes_identical_to_sequential_appends(self):
        events = [{"i": 1}, {"k": "v"}, [None, True], {"nested": {"a": 1}}]
        batch_chain = AuditChain(self.path)
        returned = batch_chain.append_batch("t", events)
        batch_bytes = self.path.read_bytes()

        seq_path = self.path.with_name("seq.jsonl")
        seq_chain = AuditChain(seq_path)
        seq_items = [seq_chain.append("t", e) for e in events]
        seq_bytes = seq_path.read_bytes()

        self.assertEqual(batch_bytes, seq_bytes)
        self.assertEqual(returned, seq_items)

    def test_mixed_batch_and_single_appends_verify_like_sequential(self):
        self.chain.append("a", 0)
        self.chain.append_batch("a", [1, 2])
        self.chain.append("b", 0)
        self.chain.append("a", 3)
        self.chain.append_batch("b", [1, 2, 3])
        r = self.chain.verify_all()
        self.assertEqual(r, {"ok": True, "tenants": [
            {"tenant": "a", "count": 4},
            {"tenant": "b", "count": 4},
        ]})

    def test_legacy_file_without_trailing_newline_batch_appends_safely(self):
        self.chain.append("t", {"v": 1})
        self.path.write_bytes(self.path.read_bytes().rstrip(b"\n"))
        items = self.chain.append_batch("t", [{"v": 2}, {"v": 3}])
        lines = self.path.read_text().splitlines()
        self.assertEqual(len(lines), 3)  # no glued-together line
        for raw in lines:
            json.loads(raw)
        self.assertEqual([it["seq"] for it in items], [2, 3])
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": 3})

    # --- tenant interleaving ---

    def test_different_tenants_each_count_from_one(self):
        a = self.chain.append_batch("a", [1, 2, 3])
        b = self.chain.append_batch("b", [1, 2])
        self.assertEqual([it["seq"] for it in a], [1, 2, 3])
        self.assertEqual([it["seq"] for it in b], [1, 2])
        self.assertEqual(a[0]["prev"], ZERO)
        self.assertEqual(b[0]["prev"], ZERO)
        self.assertEqual(self.chain.verify("a"), {"ok": True, "count": 3})
        self.assertEqual(self.chain.verify("b"), {"ok": True, "count": 2})

    def test_tenant_type_boundaries_match_append(self):
        for t in (1, 1.0, True, "1"):
            items = self.chain.append_batch(t, [{}, {}])
            self.assertEqual([it["seq"] for it in items], [1, 2])
            self.assertEqual(items[0]["prev"], ZERO)
        r = self.chain.verify_all()
        self.assertEqual([(x["tenant"], x["count"]) for x in r["tenants"]],
                         [(1, 2), (1.0, 2), (True, 2), ("1", 2)])

    # --- ValueError: input boundary ---

    def test_events_must_be_a_list(self):
        for bad in ((1, 2), {1: 2}, None, "ab", 1, 1.5, True, {1, 2}):
            with self.assertRaises(ValueError):
                self.chain.append_batch("t", bad)
        self.assertFalse(self.path.exists())

    def test_bad_event_value_raises_value_error_without_bytes(self):
        for bad in (float("nan"), float("inf"), float("-inf"),
                    {1: 2}, object(), b"x"):
            with self.assertRaises(ValueError):
                self.chain.append_batch("t", [{}, bad, {}])
            self.assertFalse(self.path.exists(), repr(bad))

    def test_cyclic_event_list_rejected(self):
        cyc = []
        cyc.append(cyc)
        with self.assertRaises(ValueError):
            self.chain.append_batch("t", cyc)
        self.assertFalse(self.path.exists())

    def test_bad_tenant_rejected_even_with_valid_events(self):
        with self.assertRaises(ValueError):
            self.chain.append_batch(float("nan"), [1])
        with self.assertRaises(ValueError):
            self.chain.append_batch({1: 2}, [1])
        self.assertFalse(self.path.exists())

    def test_value_error_takes_priority_over_corrupt_history(self):
        good = {"tenant": "t", "seq": 1, "event": {}, "prev": ZERO}
        good["hash"] = AuditChain._hash(good)
        tampered = dict(good)
        tampered["event"] = {"x": 9}
        self.write([tampered])
        before = self.path.read_bytes()
        # events not a list: ValueError, not AuditChainStateError
        with self.assertRaises(ValueError):
            self.chain.append_batch("t", None)
        # illegal event: ValueError wins over the digest corruption
        with self.assertRaises(ValueError):
            self.chain.append_batch("t", [float("nan")])
        self.assertEqual(self.path.read_bytes(), before)

    def test_failed_validation_changes_nothing_after_prior_batches(self):
        self.chain.append_batch("t", [1, 2])
        before = self.path.read_bytes()
        with self.assertRaises(ValueError):
            self.chain.append_batch("t", [3, float("inf")])
        self.assertEqual(self.path.read_bytes(), before)

    # --- corrupt history: same first broken point as append ---

    def test_corrupt_history_raises_state_error_and_keeps_bytes(self):
        self.chain.append("t", {})
        rows = [json.loads(l) for l in self.path.read_text().splitlines()]
        rows[0]["event"] = {"x": 9}
        self.write(rows)
        before = self.path.read_bytes()
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.append_batch("t", [{}, {}])
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("t", 1, "digest", 1))
        self.assertEqual(self.path.read_bytes(), before)

    def test_state_error_matches_append_for_each_reason(self):
        # missing: unparseable line
        self.write(["{oops"])
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.append_batch("t", [1])
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("t", 1, "missing", 1))
        # sequence: first t record claims seq 2
        row = {"tenant": "t", "seq": 2, "event": {}, "prev": ZERO}
        row["hash"] = AuditChain._hash(row)
        self.write([row])
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.append_batch("t", [1])
        self.assertEqual((cm.exception.seq, cm.exception.reason, cm.exception.line),
                         (1, "sequence", 1))

    def test_illegal_utf8_reported_like_append_and_bytes_untouched(self):
        good = {"tenant": "t", "seq": 1, "event": {}, "prev": ZERO}
        good["hash"] = AuditChain._hash(good)
        before = (json.dumps(good, sort_keys=True) + "\n").encode("utf-8") + b"\xff"
        self.path.write_bytes(before)
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.append_batch("t", [1, 2])
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("t", 2, "missing", 2))
        self.assertEqual(self.path.read_bytes(), before)

    def test_other_tenant_corruption_does_not_block_batch(self):
        self.chain.append("b", {})
        rows = [json.loads(l) for l in self.path.read_text().splitlines()]
        rows[0]["event"] = {"x": 1}
        self.write(rows)
        items = self.chain.append_batch("t", [1, 2])
        self.assertEqual([it["seq"] for it in items], [1, 2])
        self.assertEqual(items[0]["prev"], ZERO)
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": 2})

    def test_failure_leaves_no_partial_records(self):
        # digest break at existing seq 1: nothing from the 10-event batch may land
        self.chain.append_batch("t", [{"v": 0}])
        rows = [json.loads(l) for l in self.path.read_text().splitlines()]
        rows[0]["event"] = {"v": 99}
        self.write(rows)
        before = self.path.read_bytes()
        with self.assertRaises(AuditChainStateError):
            self.chain.append_batch("t", [{"v": i} for i in range(1, 11)])
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(len(self.path.read_text().splitlines()), 1)


def _mp_batch_writer(args):
    path, tenant_key, size, worker_id = args
    chain = AuditChain(path)
    tenant = json.loads(tenant_key)
    chain.append_batch(tenant, [{"w": worker_id, "j": j} for j in range(size)])


def _mp_mixed_writer(args):
    path, tenant_key, size, worker_id, mode = args
    chain = AuditChain(path)
    tenant = json.loads(tenant_key)
    events = [{"w": worker_id, "j": j} for j in range(size)]
    if mode == "batch":
        chain.append_batch(tenant, events)
    else:
        for e in events:
            chain.append(tenant, e)


class BatchConcurrencyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.path = Path(self.tmp.name) / "audit.jsonl"
        self.chain = AuditChain(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_threads_batches_same_tenant_form_one_chain(self):
        n_thread, size = 16, 25
        total = n_thread * size
        results, errors = [], []
        box = threading.Lock()

        def worker(i):
            try:
                items = self.chain.append_batch(
                    "t", [{"w": i, "j": j} for j in range(size)])
            except Exception as e:  # noqa: BLE001
                with box:
                    errors.append(repr(e))
                return
            with box:
                results.append(items)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_thread)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [])
        # every batch got a contiguous, disjoint seq interval
        intervals = sorted((it[0]["seq"], it[-1]["seq"]) for it in results)
        self.assertEqual(intervals, [(i * size + 1, (i + 1) * size)
                                     for i in range(n_thread)])
        flat = [it for batch in results for it in batch]
        self.assertEqual(sorted(it["seq"] for it in flat), list(range(1, total + 1)))
        # on-disk rows are exactly the returned records, one row each
        disk = self.path.read_bytes().splitlines(keepends=True)
        self.assertEqual(sorted(disk),
                         sorted(_record_bytes(it) for it in flat))
        # in the tenant's physical record order, no other t-append landed
        # inside a batch: each batch's rows are consecutive on disk too
        rows = [json.loads(l) for l in disk]
        seen = set()
        for r in rows:
            seen.add(r["seq"])
        positions = {r["seq"]: i for i, r in enumerate(rows)}
        for batch in results:
            seqs = [it["seq"] for it in batch]
            ps = [positions[s] for s in seqs]
            self.assertEqual(ps, list(range(min(ps), min(ps) + size)))
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": total})

    def test_mixed_batch_and_single_threads_never_splice_a_batch(self):
        n_batch, n_single, size = 8, 8, 20
        batches, singles, errors = [], [], []
        box = threading.Lock()

        def batch_worker(i):
            try:
                items = self.chain.append_batch(
                    "t", [{"w": i, "j": j} for j in range(size)])
                with box:
                    batches.append(items)
            except Exception as e:  # noqa: BLE001
                with box:
                    errors.append(repr(e))

        def single_worker(i):
            try:
                local = [self.chain.append("t", {"w": 100 + i, "j": j})
                         for j in range(size)]
                with box:
                    singles.extend(local)
            except Exception as e:  # noqa: BLE001
                with box:
                    errors.append(repr(e))

        threads = [threading.Thread(target=batch_worker, args=(i,))
                   for i in range(n_batch)]
        threads += [threading.Thread(target=single_worker, args=(i,))
                    for i in range(n_single)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [])
        total = (n_batch + n_single) * size
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": total})
        rows = [json.loads(l) for l in self.path.read_text().splitlines()]
        positions = {r["seq"]: i for i, r in enumerate(rows)}
        # each batch must occupy consecutive physical positions with no
        # interleaving single append inside it
        for batch in batches:
            ps = sorted(positions[it["seq"]] for it in batch)
            self.assertEqual(ps, list(range(ps[0], ps[0] + size)))
        # every seq 1..total present exactly once, links intact on verify
        self.assertEqual(sorted(r["seq"] for r in rows), list(range(1, total + 1)))

    def test_concurrent_readers_never_observe_a_half_batch(self):
        n_writer, size, n_reader = 10, 30, 6
        total = n_writer * size
        stop = threading.Event()
        snapshots = []
        box = threading.Lock()

        def reader():
            while not stop.is_set():
                r = self.chain.verify("t")
                with box:
                    snapshots.append(("verify", r))
                r2 = self.chain.verify_all()
                with box:
                    snapshots.append(("verify_all", r2))

        readers = [threading.Thread(target=reader, daemon=True) for _ in range(n_reader)]
        for t in readers:
            t.start()

        def writer(i):
            self.chain.append_batch("t", [{"w": i, "j": j} for j in range(size)])

        writers = [threading.Thread(target=writer, args=(i,)) for i in range(n_writer)]
        for t in writers:
            t.start()
        for t in writers:
            t.join()
        stop.set()
        for t in readers:
            t.join(timeout=2)

        for kind, r in snapshots:
            self.assertTrue(r["ok"], (kind, r))
            if kind == "verify":
                # only whole batches are visible, so counts are batch-aligned
                self.assertIn(r["count"], [k * size for k in range(n_writer + 1)])
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": total})

    def test_threads_interleaved_tenants_in_batches(self):
        tenants = ["a", "b", 1, 1.0, True, "1", {"k": "v"}]
        size = 20
        results = []
        box = threading.Lock()

        def worker(tenant):
            items = self.chain.append_batch(tenant, [{"i": j} for j in range(size)])
            with box:
                results.append((tenant, items))

        threads = [threading.Thread(target=worker, args=(t,)) for t in tenants]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        for tenant, items in results:
            self.assertEqual([it["seq"] for it in items], list(range(1, size + 1)))
            self.assertEqual(items[0]["prev"], ZERO)
        r = self.chain.verify_all()
        self.assertTrue(r["ok"], r)
        self.assertEqual(
            {json.dumps(x["tenant"], sort_keys=True): x["count"] for x in r["tenants"]},
            {json.dumps(t, sort_keys=True): size for t in tenants},
        )

    def test_processes_batches_same_tenant_one_contiguous_history(self):
        n_proc, size = 6, 15
        ctx = mp.get_context("fork")
        procs = [
            ctx.Process(target=_mp_batch_writer,
                        args=((str(self.path), json.dumps("t"), size, i),))
            for i in range(n_proc)
        ]
        for pr in procs:
            pr.start()
        for pr in procs:
            pr.join()
        self.assertTrue(all(pr.exitcode == 0 for pr in procs))
        rows, by = _tenant_rows(self.path)
        self.assertEqual(len(rows), n_proc * size)
        t_rows = by[json.dumps("t")]
        self.assertEqual(sorted(r["seq"] for r in t_rows),
                         list(range(1, n_proc * size + 1)))
        self.assertEqual(self.chain.verify("t"),
                         {"ok": True, "count": n_proc * size})
        self.assertTrue(self.chain.verify_all()["ok"])

    def test_processes_mixed_batch_and_single_writers(self):
        n_proc, size = 8, 12
        ctx = mp.get_context("fork")
        procs = [
            ctx.Process(target=_mp_mixed_writer,
                        args=((str(self.path), json.dumps("t"), size, i,
                               "batch" if i % 2 == 0 else "single"),))
            for i in range(n_proc)
        ]
        for pr in procs:
            pr.start()
        for pr in procs:
            pr.join()
        self.assertTrue(all(pr.exitcode == 0 for pr in procs))
        self.assertEqual(self.chain.verify("t"),
                         {"ok": True, "count": n_proc * size})
        self.assertTrue(self.chain.verify_all()["ok"])

    def test_value_error_under_contention_with_batches(self):
        outcomes = []
        box = threading.Lock()

        def good(i):
            self.chain.append_batch("t", [{"i": i}, {"i": i + 0.5}])
            with box:
                outcomes.append("ok")

        def bad(i):
            try:
                payload = [{"i": i}, float("nan")] if i % 2 == 0 else (1, 2)
                self.chain.append_batch("t", payload)
                with box:
                    outcomes.append("no-raise")
            except ValueError:
                with box:
                    outcomes.append("value")
            except Exception as e:  # noqa: BLE001
                with box:
                    outcomes.append(type(e).__name__)

        threads = []
        for i in range(30):
            threads.append(threading.Thread(target=good, args=(i,)))
            threads.append(threading.Thread(target=bad, args=(i,)))
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(outcomes.count("ok"), 30)
        self.assertEqual(outcomes.count("value"), 30)
        self.assertEqual(set(outcomes), {"ok", "value"})
        self.assertEqual(self.chain.verify("t")["count"], 60)


if __name__ == "__main__":
    unittest.main()
