import contextlib
import fcntl
import hashlib
import json
import math
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


class AuditChainStateError(Exception):
    def __init__(self, tenant, seq, reason=None, line=None):
        self.tenant = tenant
        self.seq = seq
        self.reason = reason
        self.line = line
        detail = f" (line {line})" if line is not None else ""
        super().__init__(f"invalid audit chain state for tenant {tenant!r} at seq {seq}{detail}")


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
    def _write_lease(self):
        # Exclusive lease for one read-modify-append. The same open file
        # description carries the lock, the history read and the final append:
        # flock is tied to that description, so the lease cannot be silently
        # lost to a second open of the path, and the bytes appended are the
        # ones computed from the bytes just read. A fresh fd per call makes
        # flock genuinely exclude both other processes and other threads.
        self.path.parent.mkdir(parents=True, exist_ok=True)
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

    @staticmethod
    def _scan(tenant, lines, bad_line=None):
        # Validate the target tenant's chain over the decoded JSONL lines.
        # Returns (verified_count, last_hash). Raises _Broken at the first
        # problem; records of other tenants may interleave and are skipped.
        # Tenant matching uses the same canonical JSON identity as
        # verify_all, never host-language loose equality. If bad_line is
        # given, an undecodable physical line follows the lines provided;
        # it is reported as missing only when no earlier problem was found.
        # Pure with respect to the filesystem: callers supply the lines, so
        # the file-backed methods and the in-memory verify_bytes entry share
        # this one implementation.
        key = AuditChain._tenant_key(tenant)
        expected, prev, count = 1, ZERO, 0
        for line, raw in enumerate(lines, 1):
            try:
                item = _strict_loads(raw)
            except Exception:
                raise _Broken("missing", None, line, expected)
            if not isinstance(item, dict) or "tenant" not in item:
                raise _Broken("missing", None, line, expected)
            if AuditChain._tenant_key(item["tenant"]) != key:
                continue
            if any(k not in item for k in FIELDS):
                seq = item.get("seq")
                at = seq if isinstance(seq, int) and not isinstance(seq, bool) else None
                raise _Broken("missing", at, line, expected)
            if isinstance(item["seq"], bool) or item["seq"] != expected:
                raise _Broken("sequence", expected, line, expected)
            if item["prev"] != prev:
                raise _Broken("digest", expected, line, expected)
            if item["hash"] != AuditChain._hash(item):
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

    @staticmethod
    def _verify_all_snapshot(data):
        # Pure all-tenant verification of one JSONL byte snapshot: no path,
        # lock or other I/O. validate each tenant chain in one pass, in
        # physical line order. Each tenant gets an independent expected seq
        # and prev digest starting at (1, ZERO) on first appearance; records
        # of different tenants may interleave. The serialized tenant is only
        # an internal key so distinct types (1 vs "1") stay separate chains
        # while the original value is reported back unchanged.
        lines, bad_line = AuditChain._decode_lines(data)
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
            key = AuditChain._tenant_key(tenant)
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
            if item["hash"] != AuditChain._hash(item):
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
        # Validate every tenant chain in one read-only pass over the file. The
        # scan itself is the pure snapshot core shared with verify_all_bytes;
        # the shared lease only pins which bytes constitute the snapshot, so
        # the view is wholly before or after any append, never a partial one.
        data = self._read_snapshot()
        return self._verify_all_snapshot(data)

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

    @staticmethod
    def _expected_count_verdict(count, expected_count):
        # Apply the expected_count gate to an already verified chain:
        # a shortfall is a missing tail, an overshoot a sequence error.
        if expected_count is not None:
            if count < expected_count:
                return {"ok": False, "at": count + 1, "reason": "missing"}
            if count > expected_count:
                return {"ok": False, "at": expected_count + 1, "reason": "sequence"}
        return {"ok": True, "count": count}

    @staticmethod
    def _verify_tenant_snapshot(data, tenant):
        # Pure single-tenant verification of one JSONL byte snapshot: no
        # path, lock or other I/O, and data is only read. The decode/scan
        # rules are exactly verify's: standard-JSON lines, canonical tenant
        # identity, sha256 links, first broken point wins (missing / sequence
        # / digest, at a tenant seq when knowable else the physical line).
        lines, bad_line = AuditChain._decode_lines(data)
        try:
            count, _ = AuditChain._scan(tenant, lines, bad_line)
        except _Broken as b:
            return {"ok": False,
                    "at": b.at if b.at is not None else b.line,
                    "reason": b.reason}
        return {"ok": True, "count": count}

    def verify_bytes(self, data, tenant, expected_count=None):
        # Purely in-memory, offline single-tenant verification. The caller
        # hands over the raw JSONL bytes directly; the path this chain was
        # constructed with is never read, created or modified, and data is
        # not mutated. A single-tenant export and a full log with interleaved
        # tenants both validate, under the same standard-JSON, tenant
        # identity and digest rules as verify, whose verdict shape this
        # returns exactly.
        if not isinstance(data, bytes):
            raise ValueError(
                f"data must be bytes, got {type(data).__name__}"
            )
        # Tenant crosses the standard-JSON boundary exactly as in verify;
        # ValueError is raised before the snapshot is inspected, so no
        # underlying json/recursion error can leak.
        _validate_json_value(tenant)
        if expected_count is not None and (
            isinstance(expected_count, bool)
            or not isinstance(expected_count, int)
            or expected_count < 0
        ):
            raise ValueError(
                "expected_count must be a non-negative integer or None"
            )
        result = self._verify_tenant_snapshot(data, tenant)
        if not result["ok"]:
            return result
        return self._expected_count_verdict(result["count"], expected_count)

    def verify_all_bytes(self, data):
        # Purely in-memory, offline all-tenant verification. The bytes are the
        # whole snapshot: tenants, counts, order and the first-error structure
        # are exactly verify_all's. Empty bytes are a successful empty history
        # and create no file. The constructor path is never touched and data
        # is only read.
        if not isinstance(data, bytes):
            raise ValueError(
                f"data must be bytes, got {type(data).__name__}"
            )
        return self._verify_all_snapshot(data)

    def verify(self, tenant, expected_count=None):
        # The tenant crosses the same standard-JSON input boundary as in
        # append/append_batch, and it is checked before any history is read:
        # NaN/Infinity/-Infinity, non-string object keys, cyclic containers
        # and values without a standard JSON encoding raise ValueError here,
        # never TypeError/RecursionError from a downstream json.dumps, and
        # never a verdict computed against a corrupt or missing file. The
        # file itself is opened read-only below, so a rejected call leaves
        # every byte untouched and creates nothing.
        _validate_json_value(tenant)
        data = self._read_snapshot()
        result = self._verify_tenant_snapshot(data, tenant)
        if not result["ok"]:
            return result
        return self._expected_count_verdict(result["count"], expected_count)
