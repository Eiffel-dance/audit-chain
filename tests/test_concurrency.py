import json
import multiprocessing as mp
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app import AuditChain, AuditChainStateError

ZERO = "0" * 64


def _mp_writer(args):
    # Module-level target for forked workers: append `per` records.
    path, tenant_key, per, worker_id = args
    chain = AuditChain(path)
    tenant = json.loads(tenant_key)
    for j in range(per):
        chain.append(tenant, {"w": worker_id, "j": j})


def _mp_reader(path, stop_event, bad_q, done_q):
    # Every snapshot must be a fully self-consistent pre/post-append history.
    chain = AuditChain(path)
    while not stop_event.is_set():
        r = chain.verify("t")
        if not r["ok"]:
            bad_q.put(("verify", r))
            break
        r2 = chain.verify_all()
        if not r2["ok"]:
            bad_q.put(("verify_all", r2))
            break
    done_q.put(1)


class ConcurrencyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.path = Path(self.tmp.name) / "audit.jsonl"
        self.chain = AuditChain(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def _record_bytes(self, item):
        return (json.dumps(item, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")

    def test_threads_same_tenant_one_contiguous_history(self):
        n_thread, per = 16, 30
        results, errors = [], []
        box = threading.Lock()

        def worker(i):
            local = []
            try:
                for j in range(per):
                    local.append(self.chain.append("t", {"w": i, "j": j}))
            except Exception as e:  # noqa: BLE001
                with box:
                    errors.append(repr(e))
            with box:
                results.extend(local)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_thread)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [])
        # every successful call maps 1:1 to one physical row
        self.assertEqual(len(results), n_thread * per)
        self.assertEqual(sorted(it["seq"] for it in results),
                         list(range(1, n_thread * per + 1)))
        rows = self.path.read_bytes().splitlines(keepends=True)
        self.assertEqual(len(rows), len(results))
        self.assertEqual(sorted(rows),
                         sorted(self._record_bytes(it) for it in results))
        self.assertEqual(self.chain.verify("t"),
                         {"ok": True, "count": n_thread * per})

    def test_threads_interleaved_tenants_each_count_from_one(self):
        tenants = ["a", "b", 1, 1.0, True, "1", {"k": "v"}]
        per = 20
        results = []
        box = threading.Lock()

        def worker(tenant):
            local = [self.chain.append(tenant, {"i": j}) for j in range(per)]
            with box:
                results.extend((tenant, it) for it in local)

        threads = [threading.Thread(target=worker, args=(t,)) for t in tenants]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        by = {}
        for tenant, it in results:
            by.setdefault(json.dumps(tenant, sort_keys=True), []).append(it["seq"])
        for key, seqs in by.items():
            self.assertEqual(sorted(seqs), list(range(1, per + 1)), key)
        r = self.chain.verify_all()
        self.assertTrue(r["ok"], r)
        self.assertEqual(
            {json.dumps(x["tenant"], sort_keys=True): x["count"] for x in r["tenants"]},
            {json.dumps(t, sort_keys=True): per for t in tenants},
        )

    def test_concurrent_readers_never_observe_a_half_record(self):
        n_writer, per, n_reader = 10, 30, 6
        total = n_writer * per
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
            for j in range(per):
                self.chain.append("t", {"w": i, "j": j})

        writers = [threading.Thread(target=writer, args=(i,)) for i in range(n_writer)]
        for t in writers:
            t.start()
        for t in writers:
            t.join()
        stop.set()
        for t in readers:
            t.join(timeout=2)

        # a torn/partial record would surface as missing/sequence/digest
        for kind, r in snapshots:
            self.assertTrue(r["ok"], (kind, r))
            if kind == "verify":
                self.assertIn(r["count"], range(total + 1))
            else:
                self.assertLessEqual(sum(x["count"] for x in r["tenants"]), total)
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": total})

    def test_processes_same_tenant_one_contiguous_history(self):
        n_proc, per = 6, 15
        ctx = mp.get_context("fork")
        procs = [
            ctx.Process(target=_mp_writer,
                        args=((str(self.path), json.dumps("t"), per, i),))
            for i in range(n_proc)
        ]
        for pr in procs:
            pr.start()
        for pr in procs:
            pr.join()
        self.assertTrue(all(pr.exitcode == 0 for pr in procs))
        rows = [json.loads(l) for l in self.path.read_text().splitlines()]
        self.assertEqual(len(rows), n_proc * per)
        self.assertEqual(sorted(r["seq"] for r in rows),
                         list(range(1, n_proc * per + 1)))
        self.assertEqual(self.chain.verify("t"),
                         {"ok": True, "count": n_proc * per})
        self.assertTrue(self.chain.verify_all()["ok"])

    def test_processes_readers_and_writers_agree(self):
        n_writer, per, n_reader = 4, 20, 3
        ctx = mp.get_context("fork")
        stop = ctx.Event()
        bad_q, done_q = ctx.Queue(), ctx.Queue()
        readers = [
            ctx.Process(target=_mp_reader,
                        args=(str(self.path), stop, bad_q, done_q))
            for _ in range(n_reader)
        ]
        writers = [
            ctx.Process(target=_mp_writer,
                        args=((str(self.path), json.dumps("t"), per, i),))
            for i in range(n_writer)
        ]
        for pr in readers:
            pr.start()
        for pr in writers:
            pr.start()
        for pr in writers:
            pr.join()
            self.assertEqual(pr.exitcode, 0)
        stop.set()
        for pr in readers:
            pr.join()
            self.assertEqual(pr.exitcode, 0)
        problems = []
        while not bad_q.empty():
            problems.append(bad_q.get())
        self.assertEqual(problems, [], "a reader saw an inconsistent snapshot")
        self.assertEqual(self.chain.verify("t"),
                         {"ok": True, "count": n_writer * per})

    def test_value_error_takes_priority_under_contention(self):
        outcomes = []
        box = threading.Lock()

        def good(i):
            self.chain.append("t", {"i": i})
            with box:
                outcomes.append("ok")

        def bad(i, payload):
            try:
                self.chain.append("t", payload)
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
            threads.append(threading.Thread(
                target=bad, args=(i, float("nan") if i % 2 == 0 else {"x": float("inf")})))
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(outcomes.count("ok"), 30)
        self.assertEqual(outcomes.count("value"), 30)
        self.assertEqual(set(outcomes), {"ok", "value"})
        self.assertEqual(self.chain.verify("t")["count"], 30)

    def test_corrupt_history_rejected_under_contention_file_untouched(self):
        self.chain.append("t", {"v": 1})
        row = json.loads(self.path.read_text())
        row["event"] = {"v": 2}  # breaks digest at line 1
        self.path.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
        before = self.path.read_bytes()
        details = []
        box = threading.Lock()

        def attempt(i):
            try:
                self.chain.append("t", {"i": i})
                with box:
                    details.append(("success",))
            except AuditChainStateError as e:
                with box:
                    details.append((e.tenant, e.seq, e.reason, e.line))

        threads = [threading.Thread(target=attempt, args=(i,)) for i in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(details, [("t", 1, "digest", 1)] * 16)
        self.assertEqual(self.path.read_bytes(), before)
        # an intact tenant is still able to append on its own chain
        item = self.chain.append("other", {})
        self.assertEqual((item["seq"], item["prev"]), (1, ZERO))

    def test_verify_never_creates_the_data_file(self):
        self.assertFalse(self.path.exists())
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": 0})
        self.assertEqual(self.chain.verify_all(), {"ok": True, "tenants": []})
        self.assertFalse(self.path.exists())

    def test_verify_boundary_holds_during_concurrent_appends(self):
        # Repeated verify calls racing appends: illegal tenants must raise
        # ValueError every time (never TypeError/RecursionError, never a
        # verdict), and legal verifies must only see consistent snapshots.
        stop = threading.Event()
        problems = []
        box = threading.Lock()

        def reader():
            while not stop.is_set():
                for bad in (float("nan"), {"k": float("inf")}, {1: "x"}):
                    try:
                        self.chain.verify(bad)
                        with box:
                            problems.append(("accepted", repr(bad)))
                    except ValueError:
                        pass
                    except Exception as e:  # noqa: BLE001
                        with box:
                            problems.append(("leaked", repr(e)))
                r = self.chain.verify("t")
                if not r["ok"]:
                    with box:
                        problems.append(("verify", r))

        readers = [threading.Thread(target=reader, daemon=True)
                   for _ in range(4)]
        for t in readers:
            t.start()

        def writer(i):
            for j in range(25):
                self.chain.append("t", {"w": i, "j": j})

        writers = [threading.Thread(target=writer, args=(i,)) for i in range(8)]
        for t in writers:
            t.start()
        for t in writers:
            t.join()
        stop.set()
        for t in readers:
            t.join(timeout=2)

        self.assertEqual(problems, [])
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": 200})

    def test_legacy_file_without_trailing_newline_appends_safely(self):
        self.chain.append("t", {"v": 1})
        self.path.write_bytes(self.path.read_bytes().rstrip(b"\n"))
        results = []
        box = threading.Lock()

        def worker(i):
            local = [self.chain.append("t", {"w": i, "j": j}) for j in range(10)]
            with box:
                results.extend(local)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        lines = self.path.read_text().splitlines()
        for raw in lines:
            json.loads(raw)  # two records must never be glued into one line
        self.assertEqual(len(lines), 81)
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": 81})


if __name__ == "__main__":
    unittest.main()
