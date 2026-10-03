import contextlib
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path

ZERO = "0" * 64
FIELDS = ("tenant", "seq", "event", "prev", "hash")


def _validate_json_value(value, _stack=()):
    # Standard-JSON input boundary: only null, bool, finite int, finite
    # float, str, array (list) and object (dict with str keys) may cross
    # it, recursively. NaN/Infinity/-Infinity, non-string object keys and
    # anything without an unambiguous standard JSON encoding (including
    # cyclic containers) are rejected with ValueError.
    if value is None or isinstance(value, (bool, int, str)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite number is not standard JSON: {value!r}")
        return
    if isinstance(value, (list, dict)):
        if id(value) in _stack:
            raise ValueError("cyclic container cannot be encoded as standard JSON")
        stack = _stack + (id(value),)
        if isinstance(value, list):
            for item in value:
                _validate_json_value(item, stack)
        else:
            for k, v in value.items():
                if not isinstance(k, str):
                    raise ValueError(f"non-string object key is not standard JSON: {k!r}")
                _validate_json_value(v, stack)
        return
    raise ValueError(f"value has no standard JSON encoding: {value!r}")


def _strict_loads(raw):
    # Parse one physical line as standard JSON only. Python's json is
    # otherwise lenient: it accepts NaN/Infinity/-Infinity literals,
    # silently keeps the last of duplicate object keys, and parses
    # overflowing numbers like 1e999 as inf. All of these are rejected
    # here so a non-standard line surfaces as a parse failure.
    def reject_constant(name):
        raise ValueError(f"non-standard JSON constant: {name}")

    def reject_non_finite(text):
        value = float(text)
        if not math.isfinite(value):
            raise ValueError(f"non-finite JSON number: {text}")
        return value

    def object_no_duplicate_keys(pairs):
        obj = {}
        for k, v in pairs:
            if k in obj:
                raise ValueError(f"duplicate object key: {k!r}")
            obj[k] = v
        return obj

    return json.loads(
        raw,
        parse_constant=reject_constant,
        parse_float=reject_non_finite,
        object_pairs_hook=object_no_duplicate_keys,
    )


def _validate_expected_count(expected_count):
    # expected_count is an optional exact-length expectation: when given it
    # must be a non-negative plain int. bool is rejected even though it is an
    # int subclass, floats and other types are too, so the comparison below is
    # always against an unambiguous integer.
    if expected_count is None:
        return
    if isinstance(expected_count, bool) or not isinstance(expected_count, int) \
            or expected_count < 0:
        raise ValueError(
            "expected_count must be a non-negative integer or None, got "
            f"{expected_count!r}"
        )


def _validate_required_count(expected_count):
    # append_if_head's head assertion carries a mandatory count: a
    # non-negative plain int, never None. bool is rejected even though it
    # subclasses int; floats, strings and other types are too.
    if isinstance(expected_count, bool) or not isinstance(expected_count, int) \
            or expected_count < 0:
        raise ValueError(
            "expected_count must be a non-negative integer, got "
            f"{expected_count!r}"
        )


def _validate_expected_hash(expected_hash):
    # The asserted chain-head digest is exactly 64 lowercase hex characters;
    # an empty chain is asserted with the module ZERO. Uppercase, the wrong
    # length or any other type/shape are caller errors, not chain conflicts.
    if not isinstance(expected_hash, str) or len(expected_hash) != 64 \
            or any(c not in "0123456789abcdef" for c in expected_hash):
        raise ValueError(
            "expected_hash must be 64 lowercase hexadecimal characters, got "
            f"{expected_hash!r}"
        )


class AuditChainStateError(Exception):
    def __init__(self, tenant, seq, reason=None, line=None):
        self.tenant = tenant
        self.seq = seq
        self.reason = reason
        self.line = line
        detail = f" (line {line})" if line is not None else ""
        super().__init__(f"invalid audit chain state for tenant {tenant!r} at seq {seq}{detail}")


class AuditChainConflictError(Exception):
    # A syntactically valid head assertion that does not match the chain
    # tail observed inside the lease. reason is fixed; the five fields let
    # a caller distinguish a stale/wrong-count expectation from a
    # stale/wrong-hash one without re-reading the log.
    def __init__(self, tenant, expected_count, expected_hash,
                 actual_count, actual_hash):
        self.tenant = tenant
        self.expected_count = expected_count
        self.expected_hash = expected_hash
        self.actual_count = actual_count
        self.actual_hash = actual_hash
        self.reason = "conflict"
        super().__init__(
            f"audit chain head conflict for tenant {tenant!r}: "
            f"expected count={expected_count} hash={expected_hash}, "
            f"actual count={actual_count} hash={actual_hash}"
        )


class _Broken(Exception):
    # Internal: first broken point found while scanning the file.
    # at     -- tenant seq when it can be determined, else None
    # line   -- 1-based file line number when at cannot be determined
    # expect -- tenant seq that was expected at this point
    def __init__(self, reason, at, line, expect):
        self.reason, self.at, self.line, self.expect = reason, at, line, expect


class AuditChain:
    def __init__(self, path):
        self.path = Path(path)

    @contextlib.contextmanager
    def _dir_lease(self):
        # Auxiliary exclusive lease on the containing directory itself. It
        # never carries data bytes; its only job is to make the moment a data
        # file first appears indivisible relative to a conditional append
        # that is allowed to create the file only on its winning branch.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        dfd = os.open(self.path.parent, os.O_RDONLY)
        try:
            fcntl.flock(dfd, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(dfd, fcntl.LOCK_UN)
            os.close(dfd)

    @contextlib.contextmanager
    def _write_lease(self):
        # Exclusive lease for one read-modify-append. The same open file
        # description carries the lock, the history read and the final append:
        # flock is tied to that description, so the lease cannot be silently
        # lost to a second open of the path, and the bytes appended are the
        # ones computed from the bytes just read. A fresh fd per call makes
        # flock genuinely exclude both other processes and other threads.
        # The creating open happens inside the directory lease, so a
        # conditional append probing a not-yet-existing path cannot race the
        # file's first appearance (see append_if_head); the lease is released
        # again before blocking on the data file, so it never serializes
        # unrelated chains in the same directory for a write's duration.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._dir_lease():
            f = open(self.path, "a+b")
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            f.seek(0)
            yield f
            # Flush while the lease is still held, so the lock is released
            # only after the full record has reached the kernel file.
            f.flush()
        finally:
            # Closing the description also releases the flock.
            f.close()

    def _read_snapshot(self):
        # One complete, self-consistent read-only view, protected by a shared
        # lease on the data file itself. The file is opened O_RDONLY, so a
        # verify never creates the path: when it does not exist yet that is a
        # legitimate pre-append view and yields empty bytes. Under the shared
        # lock no append can be mid-flight, so the snapshot is always the full
        # state either before or after some append, never a partial record.
        try:
            f = open(self.path, "rb")
        except FileNotFoundError:
            return b""
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            return f.read()
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            f.close()

    @staticmethod
    def _decode_lines(data):
        # Split the raw JSONL bytes into physical lines while treating the
        # first illegal UTF-8 byte as a hard, locatable stop point.
        # Returns (lines, bad_line): lines are the complete physical lines
        # preceding any corruption (all lines when the file is valid UTF-8),
        # bad_line is the 1-based line number of the first undecodable line
        # or None. A bad line is never yielded, even if the bytes before the
        # illegal byte looked like complete JSON: callers scan the preceding
        # complete lines first, so an earlier JSON/sequence/digest error
        # still takes priority.
        try:
            return data.decode("utf-8").splitlines(), None
        except UnicodeDecodeError as exc:
            # start is the offset of the first byte that cannot be decoded;
            # everything before it is a valid UTF-8 prefix.
            start = exc.start
        bad_line = data.count(b"\n", 0, start) + 1
        cut = data.rfind(b"\n", 0, start) + 1  # start of the bad line
        return data[:cut].decode("utf-8").splitlines(), bad_line

    @staticmethod
    def _tenant_key(tenant):
        # Canonical JSON data identity of a tenant value. Object key order is
        # normalized away, while distinct JSON types or number spellings stay
        # distinct partitions: 1, 1.0, true and "1" are four different
        # tenants regardless of host-language loose equality.
        return json.dumps(tenant, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _hash(item):
        payload = json.dumps(
            {k: item[k] for k in ("tenant", "seq", "event", "prev")},
            sort_keys=True, separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _scan(self, tenant, lines, bad_line=None):
        # Validate the target tenant's chain over the decoded JSONL lines.
        # Returns (verified_count, last_hash). Raises _Broken at the first
        # problem; records of other tenants may interleave and are skipped.
        # Tenant matching uses the same canonical JSON identity as
        # verify_all, never host-language loose equality. If bad_line is
        # given, an undecodable physical line follows the lines provided;
        # it is reported as missing only when no earlier problem was found.
        key = self._tenant_key(tenant)
        expected, prev, count = 1, ZERO, 0
        for line, raw in enumerate(lines, 1):
            try:
                item = _strict_loads(raw)
            except Exception:
                raise _Broken("missing", None, line, expected)
            if not isinstance(item, dict) or "tenant" not in item:
                raise _Broken("missing", None, line, expected)
            if self._tenant_key(item["tenant"]) != key:
                continue
            if any(k not in item for k in FIELDS):
                seq = item.get("seq")
                at = seq if isinstance(seq, int) and not isinstance(seq, bool) else None
                raise _Broken("missing", at, line, expected)
            if isinstance(item["seq"], bool) or item["seq"] != expected:
                raise _Broken("sequence", expected, line, expected)
            if item["prev"] != prev:
                raise _Broken("digest", expected, line, expected)
            if item["hash"] != self._hash(item):
                raise _Broken("digest", expected, line, expected)
            count += 1
            prev = item["hash"]
            expected += 1
        if bad_line is not None:
            raise _Broken("missing", None, bad_line, expected)
        return count, prev

    def append(self, tenant, event):
        # Validate the new input against the standard-JSON boundary before
        # anything else: an illegal tenant/event raises ValueError without
        # reading history or writing a single byte, and takes priority over
        # any append action (including corrupt-history rejection). It is
        # deliberately also evaluated before taking the lock, so a malformed
        # caller can never be turned into a different outcome by contention.
        _validate_json_value(tenant)
        _validate_json_value(event)
        with self._write_lease() as f:
            # The scan and the write happen under one exclusive lease on the
            # very file description used for both, so the record is always
            # computed from the latest complete state right before its own
            # bytes hit disk: competing appends serialize here and each
            # observes the previous winner's record already on disk.
            data = f.read()
            lines, bad_line = self._decode_lines(data)
            try:
                count, prev = self._scan(tenant, lines, bad_line)
            except _Broken as b:
                seq = b.at if b.at is not None else b.expect
                raise AuditChainStateError(tenant, seq, b.reason, b.line) from None
            item = {"tenant": tenant, "seq": count + 1, "event": event, "prev": prev}
            item["hash"] = self._hash(item)
            # One O_APPEND write places the whole record atomically at the
            # current end of the file; under the exclusive lease no other
            # writer moves that end, and no existing byte can be overwritten.
            prefix = b"" if (not data or data.endswith(b"\n")) else b"\n"
            record = prefix + (
                json.dumps(item, sort_keys=True, allow_nan=False) + "\n"
            ).encode("utf-8")
            f.write(record)
            return item

    def append_batch(self, tenant, events):
        # Append a group of same-tenant events as one verifiable interval.
        # events must be a list (a bare JSON value, including a string, is
        # not a batch); the tenant and every event cross the same standard-
        # JSON boundary as in append. All of this is checked before taking
        # the lease, so a malformed call raises ValueError without reading
        # history or creating the file, under any contention. An empty list
        # is a no-op: nothing is created and no existing byte is touched.
        if not isinstance(events, list):
            raise ValueError(
                f"events must be a list, got {type(events).__name__}"
            )
        _validate_json_value(tenant)
        for event in events:
            _validate_json_value(event)
        if not events:
            return []
        with self._write_lease() as f:
            # One lease covers the whole read-build-append interval, so the
            # batch is an indivisible result for competing writers: their
            # records serialize wholly before or wholly after these, never
            # between two of its records. The records are emitted as one
            # byte block at the current end of file, so readers under the
            # shared lease likewise see either the whole batch or none of
            # it. The first record continues the tenant chain found on disk;
            # each later record links to the previous record of the batch.
            data = f.read()
            lines, bad_line = self._decode_lines(data)
            try:
                count, prev = self._scan(tenant, lines, bad_line)
            except _Broken as b:
                seq = b.at if b.at is not None else b.expect
                raise AuditChainStateError(tenant, seq, b.reason, b.line) from None
            items = []
            chunks = []
            for event in events:
                item = {"tenant": tenant, "seq": count + 1,
                        "event": event, "prev": prev}
                item["hash"] = self._hash(item)
                items.append(item)
                chunks.append(
                    json.dumps(item, sort_keys=True, allow_nan=False) + "\n"
                )
                count += 1
                prev = item["hash"]
            prefix = b"" if (not data or data.endswith(b"\n")) else b"\n"
            block = prefix + "".join(chunks).encode("utf-8")
            f.write(block)
            return items

    def _append_if_head_locked(self, f, tenant, event,
                               expected_count, expected_hash):
        # Scan, compare and append on a description already holding the
        # exclusive data-file lease. A corrupt history is reported exactly as
        # append reports it and the assertion is never evaluated against an
        # unverifiable chain; a well-formed but mismatching head is a conflict.
        # Nothing is written on either failure path.
        data = f.read()
        lines, bad_line = self._decode_lines(data)
        try:
            count, head = self._scan(tenant, lines, bad_line)
        except _Broken as b:
            seq = b.at if b.at is not None else b.expect
            raise AuditChainStateError(tenant, seq, b.reason, b.line) from None
        # _scan returns ZERO for an empty chain, so an empty history is
        # compared by the exact same rule as a non-empty one: the call
        # proceeds only when both the length and the tail digest match.
        if count != expected_count or head != expected_hash:
            raise AuditChainConflictError(
                tenant, expected_count, expected_hash, count, head
            )
        # Head confirmed. The record uses the identical fields, seq numbering,
        # prev digest and hash algorithm as append.
        item = {"tenant": tenant, "seq": count + 1, "event": event, "prev": head}
        item["hash"] = self._hash(item)
        # One O_APPEND write places the whole record atomically at the
        # current end of file; under the exclusive lease no other writer
        # moves that end, and no existing byte can be overwritten.
        prefix = b"" if (not data or data.endswith(b"\n")) else b"\n"
        record = prefix + (
            json.dumps(item, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
        f.write(record)
        return item

    def _append_batch_if_head_locked(self, f, tenant, events,
                                     expected_count, expected_hash):
        # Batch twin of _append_if_head_locked: scan, compare the asserted
        # head and build-plus-append the whole batch on one description that
        # already holds the exclusive data-file lease. A corrupt history is
        # reported exactly as append_batch reports it and the assertion is
        # never evaluated against an unverifiable chain; a well-formed but
        # mismatching head is a conflict. Nothing is written on either
        # failure path, so a rejected batch leaves no partial records.
        data = f.read()
        lines, bad_line = self._decode_lines(data)
        try:
            count, head = self._scan(tenant, lines, bad_line)
        except _Broken as b:
            seq = b.at if b.at is not None else b.expect
            raise AuditChainStateError(tenant, seq, b.reason, b.line) from None
        # _scan returns ZERO for an empty chain, so an empty history is
        # compared by the exact same rule as a non-empty one: the batch
        # proceeds only when both the length and the tail digest match.
        if count != expected_count or head != expected_hash:
            raise AuditChainConflictError(
                tenant, expected_count, expected_hash, count, head
            )
        # Head confirmed. Build the batch exactly as append_batch would
        # against the same history: the first record continues the confirmed
        # head, every later record links to the previous record inside the
        # batch, seqs are contiguous from expected_count + 1, and the whole
        # batch is one byte block -- byte-for-byte identical to that many
        # plain appends made against the same head.
        items = []
        chunks = []
        for event in events:
            item = {"tenant": tenant, "seq": count + 1,
                    "event": event, "prev": head}
            item["hash"] = self._hash(item)
            items.append(item)
            chunks.append(
                json.dumps(item, sort_keys=True, allow_nan=False) + "\n"
            )
            count += 1
            head = item["hash"]
        prefix = b"" if (not data or data.endswith(b"\n")) else b"\n"
        block = prefix + "".join(chunks).encode("utf-8")
        f.write(block)
        return items

    def append_if_head(self, tenant, event, expected_count, expected_hash):
        # Conditional append behind an optimistic head assertion. Every input
        # crosses its boundary before a lease is taken, a path is probed or a
        # single byte is read: tenant/event use the same standard-JSON
        # boundary as append, expected_count must be a non-negative plain int
        # (bool, negative, float and other types are ValueError), and
        # expected_hash must be exactly the 64 lowercase hex characters of a
        # sha256 digest (the empty-chain head is asserted with ZERO). A
        # malformed call ends here, creates nothing and never depends on
        # contention or history.
        _validate_json_value(tenant)
        _validate_json_value(event)
        _validate_required_count(expected_count)
        _validate_expected_hash(expected_hash)
        # One exclusive lease covers validation-read, comparison and the
        # write. On an existing log that is the ordinary data-file lease;
        # before the log exists, the data-file lock cannot be taken without
        # creating the path (and a losing assertion must leave no file), so
        # the directory lease -- the same one ordinary appends hold while
        # creating the file -- guards the whole missing-file branch. The
        # existence probe, the empty-chain comparison and a winning creation
        # are therefore indivisible relative to every other writer. Lock
        # order is always directory-then-data, so the nesting cannot
        # deadlock against a plain append. _dir_lease also ensures the
        # parent directory exists.
        with self._dir_lease():
            try:
                f = open(self.path, "r+b")
            except FileNotFoundError:
                # Definitive empty chain for every tenant: no other writer
                # can create the path while this lease is held. A non-empty
                # assertion is a deterministic conflict and must not create
                # the log; only an exact (0, ZERO) assertion may.
                if expected_count != 0 or expected_hash != ZERO:
                    raise AuditChainConflictError(
                        tenant, expected_count, expected_hash, 0, ZERO
                    )
                f = open(self.path, "a+b")
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                f.seek(0)
                # When the file existed, assertions against the same head
                # serialize here: one appends and observes (count, head),
                # every other observes the winner's new tail and gets the
                # conflict with the actual values, no byte written by a
                # losing call. When this call just created the file the scan
                # confirms the empty head it was allowed to create for.
                return self._append_if_head_locked(
                    f, tenant, event, expected_count, expected_hash
                )
            finally:
                f.flush()
                f.close()

    def append_batch_if_head(self, tenant, events,
                             expected_count, expected_hash):
        # Atomic batch append behind an optimistic head assertion: the batch
        # validation-read, head comparison and the whole batch commit are one
        # indivisible lease. Every input crosses its boundary before a lease
        # is taken, a path is probed or a single byte is read: events must be
        # a list (a bare JSON value, including a string, is not a batch) whose
        # elements and the tenant use the same standard-JSON boundary as
        # append_batch/append_if_head; expected_count must be a non-negative
        # plain int (bool, negative, float and other types are ValueError);
        # expected_hash must be exactly the 64 lowercase hex characters of a
        # sha256 digest (the empty-chain head is asserted with ZERO). An empty
        # list is a no-op that returns [] and, like every rejected call,
        # creates no file and touches no byte -- the parameters are still
        # validated first, so an empty batch with malformed parameters raises
        # ValueError exactly as a non-empty one would.
        if not isinstance(events, list):
            raise ValueError(
                f"events must be a list, got {type(events).__name__}"
            )
        _validate_json_value(tenant)
        for event in events:
            _validate_json_value(event)
        _validate_required_count(expected_count)
        _validate_expected_hash(expected_hash)
        if not events:
            return []
        # Leases follow append_if_head exactly. On an existing log the
        # ordinary data-file lease covers scan, compare and the block write;
        # before the log exists, the directory lease -- the same one ordinary
        # appends hold while creating the file -- guards the whole
        # missing-file branch, so the existence probe, the empty-chain
        # comparison and a winning file creation are indivisible relative to
        # every other writer. Lock order stays directory-then-data.
        with self._dir_lease():
            try:
                f = open(self.path, "r+b")
            except FileNotFoundError:
                # Definitive empty chain for every tenant: no other writer
                # can create the path while this lease is held. Only an exact
                # (0, ZERO) assertion may create the log for a non-empty
                # batch; any other head is a deterministic conflict and must
                # leave no file behind.
                if expected_count != 0 or expected_hash != ZERO:
                    raise AuditChainConflictError(
                        tenant, expected_count, expected_hash, 0, ZERO
                    )
                f = open(self.path, "a+b")
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                f.seek(0)
                # Assertions against the same head serialize here: one batch
                # appends and observes (count, head), every other observes the
                # winner's new tail and gets the conflict carrying the actual
                # post-win values, no byte written by a losing call. The
                # winning block is itself indivisible: other-tenant appends
                # land wholly before or after it, and shared-lock readers see
                # only the pre- or post-commit snapshot.
                return self._append_batch_if_head_locked(
                    f, tenant, events, expected_count, expected_hash
                )
            finally:
                f.flush()
                f.close()

    def _verify_all_snapshot(self, data):
        # Core of verify_all over an exact in-memory snapshot. Touches no path:
        # the caller owns how the bytes were obtained (a shared-lease file read
        # or a caller-supplied buffer), so the same logic backs both
        # verify_all and the offline verify_all_bytes entry point.
        lines, bad_line = self._decode_lines(data)
        states = {}  # serialized tenant -> [expected_seq, prev_hash, count]
        order = []   # (tenant_value, state) in first-appearance order
        for line, raw in enumerate(lines, 1):
            try:
                item = _strict_loads(raw)
            except Exception:
                return {"ok": False, "at": line, "tenant": None, "reason": "missing"}
            if not isinstance(item, dict) or "tenant" not in item:
                return {"ok": False, "at": line, "tenant": None, "reason": "missing"}
            tenant = item["tenant"]
            if any(k not in item for k in FIELDS):
                return {"ok": False, "at": line, "tenant": tenant, "reason": "missing"}
            key = self._tenant_key(tenant)
            state = states.get(key)
            if state is None:
                state = [1, ZERO, 0]
                states[key] = state
                order.append((tenant, state))
            expected, prev, _ = state
            if isinstance(item["seq"], bool) or item["seq"] != expected:
                return {"ok": False, "at": line, "tenant": tenant, "reason": "sequence"}
            if item["prev"] != prev:
                return {"ok": False, "at": line, "tenant": tenant, "reason": "digest"}
            if item["hash"] != self._hash(item):
                return {"ok": False, "at": line, "tenant": tenant, "reason": "digest"}
            state[0] += 1
            state[1] = item["hash"]
            state[2] += 1
        if bad_line is not None:
            # First undecodable byte: tenant cannot be parsed, so None.
            return {"ok": False, "at": bad_line, "tenant": None, "reason": "missing"}
        return {"ok": True,
                "tenants": [{"tenant": t, "count": s[2]} for t, s in order]}

    def verify_all(self):
        # Validate every tenant chain in one read-only pass over the file, in
        # physical line order. Each tenant gets an independent expected seq and
        # prev digest starting at (1, ZERO) on first appearance; records of
        # different tenants may interleave. The serialized tenant is only an
        # internal key so distinct types (1 vs "1") stay separate chains while
        # the original value is reported back unchanged. The shared lease
        # pins the snapshot to a state wholly before or after any append.
        return self._verify_all_snapshot(self._read_snapshot())

    def export_tenant(self, tenant):
        # Offline, single-tenant migration export. Returns UTF-8 bytes only:
        # it never creates, mutates or deletes any path and never touches the
        # network. The tenant crosses the same standard-JSON input boundary as
        # append/verify, and, like verify, it is checked before any history is
        # read, so an illegal value raises ValueError without a single byte
        # being read or created.
        _validate_json_value(tenant)
        # One shared-lease snapshot pins the whole export to a state wholly
        # before or after some append: a concurrent writer can never land in
        # the middle of it. Nothing is assembled into a result until the exact
        # scan verify(tenant) runs has cleared the complete snapshot, so a
        # corrupt source raises AuditChainStateError instead of returning the
        # verified prefix. A missing or empty file is a legitimate empty
        # snapshot and yields b"".
        data = self._read_snapshot()
        lines, bad_line = self._decode_lines(data)
        try:
            self._scan(tenant, lines, bad_line)
        except _Broken as b:
            seq = b.at if b.at is not None else b.expect
            raise AuditChainStateError(tenant, seq, b.reason, b.line) from None
        # The full snapshot is valid for this tenant; only now collect its
        # records in physical order, dropping every interleaved tenant. Each
        # record is re-emitted with the exact serialization and newline rule
        # append uses, so the bytes a caller drops at a new JSONL path verify
        # offline with unchanged seq/prev/hash semantics; distinct JSON tenant
        # identities never share an export.
        key = self._tenant_key(tenant)
        chunks = []
        for raw in lines:
            item = _strict_loads(raw)
            if self._tenant_key(item["tenant"]) == key:
                chunks.append(
                    (json.dumps(item, sort_keys=True, allow_nan=False) + "\n")
                    .encode("utf-8")
                )
        return b"".join(chunks)

    def read_tenant(self, tenant, start_seq=1, page_size=None):
        # Read-only, single-tenant paged view. Returns plain record dicts
        # (each carrying exactly the tenant/seq/event/prev/hash fields with
        # their on-disk values) ordered by ascending seq; records of other
        # tenants may interleave on disk but never appear here or move the
        # page boundaries, since seqs are the target tenant's own 1..count.
        #
        # Every input crosses its boundary before any byte is read: the tenant
        # uses the same standard-JSON rule as append/verify/export, and
        # start_seq/page_size must be unambiguous positive integers (bool is
        # rejected even though it subclasses int; floats and other types are
        # too), with page_size additionally allowed to be None. A rejected
        # call reads nothing and, via the read-only snapshot below, can never
        # create or alter the log.
        _validate_json_value(tenant)
        if isinstance(start_seq, bool) or not isinstance(start_seq, int) \
                or start_seq < 1:
            raise ValueError(
                f"start_seq must be a positive integer, got {start_seq!r}"
            )
        if page_size is not None and (
            isinstance(page_size, bool) or not isinstance(page_size, int)
            or page_size < 1
        ):
            raise ValueError(
                "page_size must be a positive integer or None, got "
                f"{page_size!r}"
            )
        # One shared-lease snapshot, exactly as verify/export use: the result
        # is the full tenant history either wholly before or wholly after any
        # concurrent append, never a torn page. A missing path or an unknown
        # tenant is simply an empty chain and yields [].
        data = self._read_snapshot()
        lines, bad_line = self._decode_lines(data)
        try:
            # The whole tenant history -- first record through chain tail --
            # must verify with the exact JSON identity, sequence and digest
            # rules of verify before a single record is surfaced. Slicing
            # happens only after this, so a broken chain raises
            # AuditChainStateError (same tenant/seq/reason/line location as
            # verify/append/export) even when the requested page lies entirely
            # before the damage; no verified prefix is ever returned.
            self._scan(tenant, lines, bad_line)
        except _Broken as b:
            seq = b.at if b.at is not None else b.expect
            raise AuditChainStateError(tenant, seq, b.reason, b.line) from None
        # Snapshot proven valid: collect the target tenant in physical order
        # (identical to seq order for that tenant), then take the page purely
        # in memory. A start past the current count or a short tail page just
        # yields the available suffix, possibly [], rather than an error.
        key = self._tenant_key(tenant)
        records = []
        for raw in lines:
            item = _strict_loads(raw)
            if self._tenant_key(item["tenant"]) == key:
                records.append(item)
        lo = start_seq - 1
        if page_size is None:
            return records[lo:]
        return records[lo:lo + page_size]

    def _verify_snapshot(self, tenant, data, expected_count=None):
        # Core of verify over an exact in-memory snapshot: the same JSON
        # canonicalization, tenant identity and digest rules as the file path,
        # but the bytes are given and never touched on disk. Backs both verify
        # (file snapshot) and verify_bytes (caller snapshot).
        lines, bad_line = self._decode_lines(data)
        try:
            count, _ = self._scan(tenant, lines, bad_line)
        except _Broken as b:
            return {"ok": False, "at": b.at if b.at is not None else b.line, "reason": b.reason}
        if expected_count is not None:
            if count < expected_count:
                return {"ok": False, "at": count + 1, "reason": "missing"}
            if count > expected_count:
                return {"ok": False, "at": expected_count + 1, "reason": "sequence"}
        return {"ok": True, "count": count}

    def verify(self, tenant, expected_count=None):
        # Both arguments cross their boundary before any history is read, in
        # the exact order verify_bytes uses: the tenant first (same standard-
        # JSON input boundary as in append/append_batch: NaN/Infinity/
        # -Infinity, non-string object keys, cyclic containers and values
        # without a standard JSON encoding raise ValueError here, never
        # TypeError/RecursionError from a downstream json.dumps), then
        # expected_count, which must be None or a non-negative plain int
        # (bool is rejected even though it subclasses int; floats, strings
        # and other types are too). Both raise before the file is opened, so
        # a rejected call never sees a verdict computed against a corrupt or
        # missing file, leaves every byte untouched and creates nothing --
        # the file below is opened read-only regardless.
        _validate_json_value(tenant)
        _validate_expected_count(expected_count)
        return self._verify_snapshot(
            tenant, self._read_snapshot(), expected_count
        )

    def verify_bytes(self, data, tenant, expected_count=None):
        # Pure in-memory, offline entry point. The caller hands over the raw
        # bytes of a JSONL history (a single-tenant export, or a full log with
        # interleaved tenants); this never reads, creates or modifies the path
        # this AuditChain points at and never touches the network. Results are
        # byte-for-byte the same verdicts verify would give for identical file
        # contents, including physical line numbers and first-error priority.
        # Inputs cross their boundary before any parsing, and a malformed
        # buffer surfaces only as a missing/sequence/digest verdict -- the
        # underlying parse exception is never leaked.
        if not isinstance(data, bytes):
            raise ValueError(f"data must be bytes, got {type(data).__name__}")
        _validate_json_value(tenant)
        _validate_expected_count(expected_count)
        return self._verify_snapshot(tenant, data, expected_count)

    def verify_all_bytes(self, data):
        # In-memory counterpart of verify_all over a caller-supplied snapshot:
        # same tenant first-appearance order, counts and first-error structure.
        # Empty bytes are a legitimate successful empty history, and like
        # verify_bytes this neither reads nor creates the configured path.
        if not isinstance(data, bytes):
            raise ValueError(f"data must be bytes, got {type(data).__name__}")
        return self._verify_all_snapshot(data)
