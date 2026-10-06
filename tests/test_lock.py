import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import app
from app import (
    AuditChain,
    AuditChainConflictError,
    AuditChainLockError,
    AuditChainStateError,
)

ZERO = "0" * 64


class _FakeMsvcrt:
    # Minimal stand-in for the Windows msvcrt lock backend, exercised on
    # any platform: validates the offset-0 one-byte region contract and
    # can be jammed to simulate an acquisition failure.
    LK_LOCK = 1
    LK_UNLCK = 0

    def __init__(self):
        self.jammed = False
        self.calls = []

    def locking(self, fd, mode, nbytes):
        assert os.lseek(fd, 0, os.SEEK_CUR) == 0
        assert nbytes == 1
        if self.jammed and mode == self.LK_LOCK:
            raise OSError("simulated lock failure")
        self.calls.append(mode)


class _BackendGuard:
    # Context manager swapping the lock backends module-wide, restored on
    # exit no matter how the body ends.
    def __init__(self, fcntl_value, msvcrt_value):
        self._new = (fcntl_value, msvcrt_value)

    def __enter__(self):
        self._saved = (app.fcntl, app.msvcrt)
        app.fcntl, app.msvcrt = self._new

    def __exit__(self, *exc):
        app.fcntl, app.msvcrt = self._saved
        return False


class LockErrorTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.path = Path(self.tmp.name) / "audit.jsonl"
        self.chain = AuditChain(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_construction_and_import_need_no_lock_backend(self):
        # A platform without any lock module still imports the module and
        # constructs chains; only lock-taking operations may fail.
        with _BackendGuard(None, None):
            chain = AuditChain(self.path)
            self.assertEqual(chain.path, self.path)

    def test_no_backend_raises_lock_error_with_fixed_fields(self):
        with _BackendGuard(None, None):
            for call, operation in [
                (lambda: self.chain.append("t", {"x": 1}), "append"),
                (lambda: self.chain.append_batch("t", [{"x": 1}]),
                 "append_batch"),
                (lambda: self.chain.append_batch_stream("t", iter([{"x": 1}])),
                 "append_batch_stream"),
                (lambda: self.chain.append_if_head("t", {}, 0, ZERO),
                 "append_if_head"),
                (lambda: self.chain.append_batch_if_head("t", [{}], 0, ZERO),
                 "append_batch_if_head"),
                (lambda: self.chain.append_many(
                    [{"tenant": "t", "event": {}}]), "append_many"),
                (lambda: self.chain.append_many_stream(
                    iter([{"tenant": "t", "event": {}}])),
                 "append_many_stream"),
                (lambda: self.chain.append_many_if_heads(
                    [{"tenant": "t", "events": [{}], "expected_count": 0,
                      "expected_hash": ZERO}]), "append_many_if_heads"),
                (lambda: self.chain.compact(), "compact"),
            ]:
                with self.assertRaises(AuditChainLockError) as ctx:
                    call()
                err = ctx.exception
                self.assertEqual(err.reason, "lock")
                self.assertEqual(err.operation, operation)
                self.assertEqual(err.path, str(self.path))
            # no failed operation created or touched the log
            self.assertFalse(self.path.exists())

    def test_no_backend_read_entries_raise_on_existing_log(self):
        self.chain.append("t", {"v": 1})
        before = self.path.read_bytes()
        head_hash = self.chain.head("t")["hash"]
        with _BackendGuard(None, None):
            for call, operation in [
                (lambda: self.chain.verify("t"), "verify"),
                (lambda: self.chain.verify_all(), "verify_all"),
                (lambda: self.chain.head("t"), "head"),
                (lambda: self.chain.heads(), "heads"),
                (lambda: self.chain.manifest(), "manifest"),
                (lambda: self.chain.read_tenant("t"), "read_tenant"),
                (lambda: self.chain.export_tenant("t"), "export_tenant"),
                (lambda: self.chain.export_all(), "export_all"),
                (lambda: self.chain.verify_heads(
                    [{"tenant": "t", "count": 1, "hash": head_hash}]),
                 "verify_heads"),
            ]:
                with self.assertRaises(AuditChainLockError) as ctx:
                    call()
                self.assertEqual(ctx.exception.reason, "lock")
                self.assertEqual(ctx.exception.operation, operation)
                self.assertEqual(ctx.exception.path, str(self.path))
        self.assertEqual(self.path.read_bytes(), before)

    def test_no_backend_keeps_missing_log_empty_history_semantics(self):
        # A missing log is the legitimate empty history and needs no lock.
        with _BackendGuard(None, None):
            self.assertEqual(self.chain.verify("t"), {"ok": True, "count": 0})
            self.assertEqual(self.chain.verify_all(),
                             {"ok": True, "tenants": []})
        self.assertFalse(self.path.exists())

    def test_value_error_still_precedes_lock_failure(self):
        with _BackendGuard(None, None):
            with self.assertRaises(ValueError):
                self.chain.append("t", float("nan"))
            with self.assertRaises(ValueError):
                self.chain.verify(float("inf"))
        self.assertFalse(self.path.exists())

    def test_lock_acquisition_failure_leaves_file_and_state_untouched(self):
        self.chain.append("t", {"v": 1})
        before = self.path.read_bytes()
        real_flock = app.fcntl.flock if app.fcntl is not None else None
        if real_flock is None:
            self.skipTest("fcntl backend not available on this platform")

        def boom(fd, op):
            raise OSError("simulated lock failure")

        app.fcntl.flock = boom
        try:
            with self.assertRaises(AuditChainLockError) as ctx:
                self.chain.append("t", {"v": 2})
            self.assertEqual(ctx.exception.reason, "lock")
            self.assertEqual(ctx.exception.operation, "append")
            self.assertEqual(ctx.exception.path, str(self.path))
            with self.assertRaises(AuditChainLockError):
                self.chain.verify("t")
        finally:
            app.fcntl.flock = real_flock
        # nothing appended, nothing truncated, nothing partially returned
        self.assertEqual(self.path.read_bytes(), before)
        # the failed lease left no lock state behind: normal use resumes
        item = self.chain.append("t", {"v": 2})
        self.assertEqual(item["seq"], 2)
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": 2})

    def test_lock_released_after_exception_inside_lease(self):
        # A corrupt history raises inside the exclusive lease; the lease
        # must be released so later calls are not blocked.
        self.chain.append("t", {"v": 1})
        row = json.loads(self.path.read_text())
        row["event"] = {"v": 2}
        self.path.write_text(json.dumps(row, sort_keys=True) + "\n",
                             encoding="utf-8")
        with self.assertRaises(AuditChainStateError):
            self.chain.append("t", {"v": 3})
        # an intact tenant can still append immediately: no stuck lock
        item = self.chain.append("other", {})
        self.assertEqual((item["seq"], item["prev"]), (1, ZERO))
        # the corrupt chain still reports its state error inside the lease
        with self.assertRaises(AuditChainStateError):
            self.chain.append_if_head("t", {}, 0, ZERO)
        item = self.chain.append_if_head("other", {"v": 2}, 1,
                                         self.chain.head("other")["hash"])
        self.assertEqual(item["seq"], 2)


class MsvcrtBackendTest(unittest.TestCase):
    # The Windows backend path, exercised through a fake msvcrt so the
    # branch is covered on any platform.
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.path = Path(self.tmp.name) / "audit.jsonl"
        self.chain = AuditChain(self.path)
        self.fake = _FakeMsvcrt()
        self._guard = _BackendGuard(None, self.fake)
        self._guard.__enter__()

    def tearDown(self):
        self._guard.__exit__()
        self.tmp.cleanup()

    def test_full_workflow_through_the_msvcrt_branch(self):
        self.chain.append("a", {"i": 1})
        self.chain.append_batch("a", [{"i": 2}, {"i": 3}])
        head = self.chain.head("a")
        self.chain.append_if_head("a", {"i": 4}, 3, head["hash"])
        self.chain.append_many([{"tenant": "b", "event": {"i": 1}}])
        self.assertEqual(self.chain.verify("a"), {"ok": True, "count": 4})
        self.assertTrue(self.chain.verify_all()["ok"])
        manifest = self.chain.manifest()
        self.assertTrue(self.chain.verify_manifest(manifest)["ok"])
        exported = self.chain.export_all()
        other = AuditChain(Path(self.tmp.name) / "other.jsonl")
        other.import_all(exported)
        self.assertTrue(other.compare(self.chain)["equal"])
        self.assertEqual(self.chain.compact(), self.chain.manifest())
        self.assertEqual(self.chain.verify("a"), {"ok": True, "count": 4})
        # shared reads released their lease explicitly
        self.assertIn(_FakeMsvcrt.LK_UNLCK, self.fake.calls)

    def test_losing_conditional_append_creates_nothing(self):
        try:
            self.chain.append_if_head("z", {}, 5, ZERO)
        except AuditChainConflictError as e:
            self.assertEqual((e.actual_count, e.actual_hash), (0, ZERO))
        else:
            self.fail("expected AuditChainConflictError")
        self.assertFalse(self.path.exists())

    def test_jammed_backend_raises_lock_error_without_partial_data(self):
        self.chain.append("a", {"i": 1})
        before = self.path.read_bytes()
        self.fake.jammed = True
        with self.assertRaises(AuditChainLockError) as ctx:
            self.chain.append("a", {"i": 2})
        self.assertEqual(ctx.exception.reason, "lock")
        self.assertEqual(ctx.exception.operation, "append")
        self.assertEqual(ctx.exception.path, str(self.path))
        with self.assertRaises(AuditChainLockError):
            self.chain.verify("a")
        self.assertEqual(self.path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
