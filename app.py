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


def _validate_expected_heads(expected_heads):
    # expected_heads is an ordered list of compact head expectations, one per
    # canonical JSON tenant identity; list order is the conflict-priority
    # order. Each member must be an object carrying exactly the keys tenant,
    # count and hash: tenant crosses the standard-JSON boundary, count must be
    # a non-negative plain int (bool rejected even though it subclasses int),
    # hash exactly 64 lowercase hex characters (the empty-chain head is ZERO).
    # A canonical tenant identity may appear at most once. Every one of these
    # checks, like every other entry point's input boundary, is a caller error
    # and ends as ValueError before any snapshot byte is read or parsed, so it
    # takes strict priority over a corrupt-history verdict and a head
    # comparison; an empty list is the valid expectation of an empty
    # directory. The returned tuples keep the list order and the original
    # tenant value verbatim.
    if not isinstance(expected_heads, list):
        raise ValueError(
            "expected_heads must be a list, got "
            f"{type(expected_heads).__name__}"
        )
    assertions = []
    seen = set()
    for head in expected_heads:
        if not isinstance(head, dict) or set(head) != {"tenant", "count", "hash"}:
            raise ValueError(
                "each expected head must be an object containing exactly the "
                f"keys 'tenant', 'count' and 'hash', got {head!r}"
            )
        tenant = head["tenant"]
        count = head["count"]
        head_hash = head["hash"]
        _validate_json_value(tenant)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(
                f"count must be a non-negative integer, got {count!r}"
            )
        _validate_expected_hash(head_hash)
        key = AuditChain._tenant_key(tenant)
        if key in seen:
            raise ValueError(
                "duplicate tenant in expected_heads (same canonical JSON "
                f"identity): {tenant!r}"
            )
        seen.add(key)
        assertions.append((tenant, count, head_hash))
    return assertions


def _validate_expected_range_heads(expected_heads):
    # expected_heads of a segmented multi-tenant import: an ordered list of
    # compact head assertions, one per canonical JSON tenant identity; list
    # order is the conflict-priority order. Each member must be an object
    # carrying exactly the keys tenant, expected_count and expected_hash:
    # tenant crosses the standard-JSON boundary, expected_count must be a
    # non-negative plain int (bool rejected even though it subclasses int),
    # expected_hash exactly 64 lowercase hex characters (the empty-chain head
    # is ZERO). A canonical tenant identity may appear at most once. Every
    # one of these checks is a caller error and ends as ValueError before any
    # lease is taken, any history is read or the path is probed, so it takes
    # strict priority over a corrupt-history verdict and a head comparison.
    # The returned tuples keep the list order and the original tenant value
    # verbatim.
    if not isinstance(expected_heads, list):
        raise ValueError(
            "expected_heads must be a list, got "
            f"{type(expected_heads).__name__}"
        )
    assertions = []
    seen = set()
    for head in expected_heads:
        if not isinstance(head, dict) or set(head) != {
            "tenant", "expected_count", "expected_hash"
        }:
            raise ValueError(
                "each expected head must be an object containing exactly "
                "the keys 'tenant', 'expected_count' and "
                f"'expected_hash', got {head!r}"
            )
        tenant = head["tenant"]
        expected_count = head["expected_count"]
        expected_hash = head["expected_hash"]
        _validate_json_value(tenant)
        _validate_required_count(expected_count)
        _validate_expected_hash(expected_hash)
        key = AuditChain._tenant_key(tenant)
        if key in seen:
            raise ValueError(
                "duplicate tenant head assertion (same canonical JSON "
                f"identity): {tenant!r}"
            )
        seen.add(key)
        assertions.append((tenant, expected_count, expected_hash))
    return assertions


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


class AuditChainRangeError(Exception):
    # A syntactically valid range request that the verified chain cannot
    # satisfy: the chain is empty, or the requested closed interval reaches
    # past the chain tail. reason is fixed; tenant/start_seq/end_seq echo the
    # request (end_seq None when the caller asked for the open tail) and
    # count carries the verified chain length observed inside the snapshot,
    # so a caller can re-plan the segment without re-reading the log.
    def __init__(self, tenant, start_seq, end_seq, count):
        self.tenant = tenant
        self.start_seq = start_seq
        self.end_seq = end_seq
        self.count = count
        self.reason = "range"
        super().__init__(
            f"audit chain range error for tenant {tenant!r}: "
            f"requested [{start_seq}, {end_seq}] of {count} records"
        )


class _Broken(Exception):
    # Internal: first broken point found while scanning the file.
    # at     -- tenant seq when it can be determined, else None
    # line   -- 1-based file line number when at cannot be determined
    # expect -- tenant seq that was expected at this point
    def __init__(self, reason, at, line, expect):
        self.reason, self.at, self.line, self.expect = reason, at, line, expect


class _ForeignTenant(Exception):
    # Internal: an import stream that must contain exactly one tenant's
    # history carried a record of another canonical JSON identity.
    def __init__(self, line):
        self.line = line


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
    def _split_lf(text):
        # Physical lines are bounded by exactly one character: the byte 0x0A
        # (LF), so a CRLF sequence is still one line -- the trailing CR stays
        # attached to its line and strict JSON accepts it as line-ending
        # whitespace. No other character ends a record: U+0085, U+2028,
        # U+2029 and a bare CR are valid UTF-8 content of the current line
        # and are handed to strict JSON (which keeps the three Unicode
        # separators as ordinary string content and rejects a raw CR inside
        # a JSON string). str.splitlines() must never be used here: it treats
        # all of those as line boundaries. The input is non-empty and, for a
        # trailing-LF text, the single empty segment after the final LF is
        # dropped (matching a physical record count: every LF terminates one
        # line, it does not begin an extra empty one).
        lines = text.split("\n")
        if text.endswith("\n"):
            lines.pop()
        return lines

    @classmethod
    def _decode_lines(cls, data):
        # Split the raw JSONL bytes into LF-bounded physical lines while
        # treating the first illegal UTF-8 byte as a hard, locatable stop
        # point. Returns (lines, bad_line): lines are the complete physical
        # lines preceding any corruption (all lines when the bytes are valid
        # UTF-8), bad_line is the 1-based LF physical line number of the
        # first undecodable line or None. A bad line is never yielded, even
        # if the bytes before the illegal byte looked like complete JSON:
        # callers scan the preceding complete lines first, so an earlier
        # JSON/sequence/digest error still takes priority. Empty bytes are
        # zero physical lines, not one blank one.
        if not data:
            return [], None
        try:
            return cls._split_lf(data.decode("utf-8")), None
        except UnicodeDecodeError as exc:
            # start is the offset of the first byte that cannot be decoded;
            # everything before it is a valid UTF-8 prefix. Physical lines
            # count LF bytes only, so the bad line's number is one plus the
            # number of LFs before that offset.
            start = exc.start
            bad_line = data.count(b"\n", 0, start) + 1
            cut = data.rfind(b"\n", 0, start) + 1  # start of the bad line
            prefix = data[:cut]
            lines = cls._split_lf(prefix.decode("utf-8")) if prefix else []
            return lines, bad_line

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
            seq = item["seq"]
            # The seq of a stored record must be a JSON integer exactly equal
            # to the next expected one: append only ever writes plain ints,
            # so a bool, a float spelling (1.0, 1e0 -- host-language loose
            # equality must not admit them), a string, null, an array/object
            # or any mismatching integer is a sequence defect, never a valid
            # record, and is never renumbered, skipped or rewritten.
            if isinstance(seq, bool) or not isinstance(seq, int) \
                    or seq != expected:
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

    def _scan_import(self, tenant, lines, bad_line=None):
        # Validate an offline single-tenant export before it is grafted onto
        # another log. Unlike _scan, no foreign records are tolerated: every
        # physical line must be a well-formed record carrying the exact same
        # canonical JSON tenant identity, seqs must run 1..n from a ZERO prev,
        # and each digest must verify. Returns the parsed records in physical
        # (== seq) order. A chain defect raises _Broken with the same
        # reason/at/line/expect filling rules _scan uses, so the resulting
        # AuditChainStateError is shaped exactly like append's; a record of
        # another tenant identity violates the single-tenant input contract
        # and raises _ForeignTenant (surfaced as ValueError by import_tenant),
        # with the first problem in physical order taking priority.
        key = self._tenant_key(tenant)
        records = []
        expected, prev = 1, ZERO
        for line, raw in enumerate(lines, 1):
            try:
                item = _strict_loads(raw)
            except Exception:
                raise _Broken("missing", None, line, expected)
            if not isinstance(item, dict) or "tenant" not in item:
                raise _Broken("missing", None, line, expected)
            if self._tenant_key(item["tenant"]) != key:
                raise _ForeignTenant(line)
            if any(k not in item for k in FIELDS):
                seq = item.get("seq")
                at = seq if isinstance(seq, int) and not isinstance(seq, bool) else None
                raise _Broken("missing", at, line, expected)
            seq = item["seq"]
            # Same strict JSON-integer seq rule as _scan: a float spelling
            # (1.0, 1e0), bool, string, null, array/object or a mismatching
            # integer is a sequence defect, never renumbered or skipped.
            if isinstance(seq, bool) or not isinstance(seq, int) \
                    or seq != expected:
                raise _Broken("sequence", expected, line, expected)
            if item["prev"] != prev:
                raise _Broken("digest", expected, line, expected)
            if item["hash"] != self._hash(item):
                raise _Broken("digest", expected, line, expected)
            records.append(item)
            prev = item["hash"]
            expected += 1
        if bad_line is not None:
            raise _Broken("missing", None, bad_line, expected)
        return records

    def _scan_import_range(self, tenant, lines, bad_line, first_seq, first_prev):
        # Validate an offline single-tenant chain segment before it is
        # grafted onto a non-empty target chain. Same rules as _scan_import
        # with two differences: the segment continues a known head instead of
        # starting a chain, so seqs must run first_seq, first_seq+1, ... from
        # a first prev equal to first_prev (the asserted target tail); and
        # every record must carry exactly the five FIELDS keys (the segmented
        # input contract), so an extra key is a missing-class defect exactly
        # as import_all's exact_fields scan reports it. Returns the parsed
        # records in physical (== seq) order. Chain defects raise _Broken
        # with the same reason/at/line/expect filling rules _scan_import
        # uses; a record of another canonical tenant identity raises
        # _ForeignTenant; the first problem in physical order takes priority.
        key = self._tenant_key(tenant)
        records = []
        expected, prev = first_seq, first_prev
        for line, raw in enumerate(lines, 1):
            try:
                item = _strict_loads(raw)
            except Exception:
                raise _Broken("missing", None, line, expected)
            if not isinstance(item, dict) or "tenant" not in item:
                raise _Broken("missing", None, line, expected)
            if self._tenant_key(item["tenant"]) != key:
                raise _ForeignTenant(line)
            if set(item) != set(FIELDS):
                seq = item.get("seq")
                at = seq if isinstance(seq, int) and not isinstance(seq, bool) else None
                raise _Broken("missing", at, line, expected)
            seq = item["seq"]
            # Same strict JSON-integer seq rule as _scan_import: a float
            # spelling (1.0, 1e0), bool, string, null, array/object or a
            # mismatching integer is a sequence defect, never renumbered.
            if isinstance(seq, bool) or not isinstance(seq, int) \
                    or seq != expected:
                raise _Broken("sequence", expected, line, expected)
            if item["prev"] != prev:
                raise _Broken("digest", expected, line, expected)
            if item["hash"] != self._hash(item):
                raise _Broken("digest", expected, line, expected)
            records.append(item)
            prev = item["hash"]
            expected += 1
        if bad_line is not None:
            raise _Broken("missing", None, bad_line, expected)
        return records

    def _scan_all_chains(self, lines, bad_line=None, exact_fields=False):
        # Validate every tenant chain over decoded JSONL lines in one
        # physical-order pass, records of different tenants interleaving
        # freely. Each tenant gets an independent expected seq and prev
        # digest starting at (1, ZERO) on first appearance -- the per-tenant
        # rules _scan applies, evaluated for all tenants at once the way
        # verify_all does. Returns the parsed records in physical order.
        # The first defective physical line raises AuditChainStateError
        # directly, with tenant/seq/reason/line filled exactly the way the
        # _scan/_Broken path fills them for a single tenant; a line whose
        # tenant cannot be determined at all (unparseable, non-object, no
        # "tenant" key) reports tenant None and seq None. With
        # exact_fields=True a record must carry exactly the five FIELDS keys
        # (the import_all input contract); otherwise only missing keys are
        # defects, matching the file-scan rule every other entry point uses.
        # If bad_line is given, an undecodable physical line follows the
        # lines provided; it is reported as missing only when no earlier
        # problem was found.
        states = {}  # serialized tenant -> [expected_seq, prev_hash]
        records = []
        for line, raw in enumerate(lines, 1):
            try:
                item = _strict_loads(raw)
            except Exception:
                raise AuditChainStateError(None, None, "missing", line) from None
            if not isinstance(item, dict) or "tenant" not in item:
                raise AuditChainStateError(None, None, "missing", line) from None
            tenant = item["tenant"]
            key = self._tenant_key(tenant)
            state = states.get(key)
            if state is None:
                state = [1, ZERO]
                states[key] = state
            expected, prev = state
            if exact_fields:
                malformed = set(item) != set(FIELDS)
            else:
                malformed = any(k not in item for k in FIELDS)
            if malformed:
                seq = item.get("seq")
                at = seq if isinstance(seq, int) \
                    and not isinstance(seq, bool) else expected
                raise AuditChainStateError(tenant, at, "missing", line) from None
            seq = item["seq"]
            # Same strict JSON-integer seq rule as _scan: 1.0/1e0, bools,
            # strings, null, arrays/objects and mismatching integers are all
            # sequence defects, so this scan agrees with every other entry.
            if isinstance(seq, bool) or not isinstance(seq, int) \
                    or seq != expected:
                raise AuditChainStateError(
                    tenant, expected, "sequence", line) from None
            if item["prev"] != prev:
                raise AuditChainStateError(
                    tenant, expected, "digest", line) from None
            if item["hash"] != self._hash(item):
                raise AuditChainStateError(
                    tenant, expected, "digest", line) from None
            state[0] += 1
            state[1] = item["hash"]
            records.append(item)
        if bad_line is not None:
            # First undecodable byte: tenant cannot be parsed, so None.
            raise AuditChainStateError(None, None, "missing", bad_line) from None
        return records

    def _scan_import_ranges(self, lines, bad_line, assertions):
        # Validate a multi-tenant segmented import byte stream before the
        # target is touched. assertions is the fully validated list of
        # (tenant, expected_count, expected_hash) tuples in expected_heads
        # order; each asserted tenant must appear in the stream at least
        # once. Tenants may interleave freely, but every physical line must be
        # an object carrying exactly the five FIELDS keys of a tenant named in
        # assertions; a record of an unlisted canonical tenant identity
        # violates the segmented input contract and raises _ForeignTenant
        # (surfaced as ValueError by import_all_range), exactly the rule
        # _scan_import applies. Each tenant's seqs run expected_count+1,
        # expected_count+2, ... continuously and the first prev equals the
        # asserted hash, each digest recomputed. Returns the parsed records in
        # physical order. The first defective physical line raises
        # AuditChainStateError directly, filled exactly the way
        # _scan_all_chains fills it; a line whose tenant cannot be determined
        # (unparseable, non-object, no "tenant" key) reports tenant/seq None.
        # After every line has validated, a listed tenant with no record is
        # still a defect: the first such assertion in assertions order is
        # reported as missing with line None. If bad_line is given, an
        # undecodable physical line follows the lines provided and is
        # reported as missing only when no earlier problem was found.
        states = {}  # serialized tenant -> [tenant, expected, prev, seen]
        for tenant, count, head in assertions:
            states[self._tenant_key(tenant)] = [
                tenant, count + 1, head, False]
        records = []
        for line, raw in enumerate(lines, 1):
            try:
                item = _strict_loads(raw)
            except Exception:
                raise AuditChainStateError(None, None, "missing", line) from None
            if not isinstance(item, dict) or "tenant" not in item:
                raise AuditChainStateError(None, None, "missing", line) from None
            tenant = item["tenant"]
            key = self._tenant_key(tenant)
            state = states.get(key)
            if state is None:
                # A canonical tenant identity no head assertion lists: this
                # violates the segmented input contract (surfaced as
                # ValueError by import_all_range), the same rule
                # _scan_import applies to a foreign single-tenant record. It
                # is raised in physical-line order, so an earlier chain
                # defect on a prior line still takes priority.
                raise _ForeignTenant(line)
            expected, prev = state[1], state[2]
            if set(item) != set(FIELDS):
                seq = item.get("seq")
                at = seq if isinstance(seq, int) \
                    and not isinstance(seq, bool) else expected
                raise AuditChainStateError(tenant, at, "missing", line) from None
            seq = item["seq"]
            # Same strict JSON-integer seq rule as every other scan: 1.0/1e0,
            # bools, strings, null, arrays/objects and mismatching integers are
            # all sequence defects.
            if isinstance(seq, bool) or not isinstance(seq, int) \
                    or seq != expected:
                raise AuditChainStateError(
                    tenant, expected, "sequence", line) from None
            if item["prev"] != prev:
                raise AuditChainStateError(
                    tenant, expected, "digest", line) from None
            if item["hash"] != self._hash(item):
                raise AuditChainStateError(
                    tenant, expected, "digest", line) from None
            state[1] += 1
            state[2] = item["hash"]
            state[3] = True
            records.append(item)
        if bad_line is not None:
            # First undecodable byte: tenant cannot be parsed, so None.
            raise AuditChainStateError(None, None, "missing", bad_line) from None
        # Every listed tenant must have at least one record in the stream.
        for tenant, count, _head in assertions:
            state = states[self._tenant_key(tenant)]
            if not state[3]:
                raise AuditChainStateError(
                    tenant, count + 1, "missing", None) from None
        return records

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

    def _append_batch_if_head_locked(self, f, tenant, events,
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
        # Head confirmed. The records use the identical fields, seq numbering,
        # prev digest and hash algorithm as append: the first continues the
        # asserted chain head, each later one links to the previous record of
        # the batch, exactly as append_batch links its records.
        items = []
        chunks = []
        prev = head
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
        # One O_APPEND write places the whole batch atomically at the
        # current end of file; under the exclusive lease no other writer
        # moves that end, and no existing byte can be overwritten.
        prefix = b"" if (not data or data.endswith(b"\n")) else b"\n"
        block = prefix + "".join(chunks).encode("utf-8")
        f.write(block)
        return items

    def _append_if_head_locked(self, f, tenant, event,
                               expected_count, expected_hash):
        # Single-event special case of the batch conditional append; the
        # shared helper keeps the scan, comparison, serialization and write
        # rules of both entry points byte-for-byte identical.
        return self._append_batch_if_head_locked(
            f, tenant, [event], expected_count, expected_hash
        )[0]

    @contextlib.contextmanager
    def _head_lease(self, tenant, expected_count, expected_hash):
        # One exclusive lease covering validation-read, head comparison and
        # the write of a conditional append. On an existing log that is the
        # ordinary data-file lease; before the log exists, the data-file lock
        # cannot be taken without creating the path (and a losing assertion
        # must leave no file), so the directory lease -- the same one ordinary
        # appends hold while creating the file -- guards the whole
        # missing-file branch. The existence probe, the empty-chain
        # comparison and a winning creation are therefore indivisible
        # relative to every other writer. Lock order is always
        # directory-then-data, so the nesting cannot deadlock against a plain
        # append. _dir_lease also ensures the parent directory exists.
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
                yield f
            finally:
                f.flush()
                f.close()

    def append_if_head(self, tenant, event, expected_count, expected_hash):
        # Conditional append behind an optimistic head assertion. Every input
        # crosses its boundary before a lease is taken, a path is probed or a
        # single byte is read: tenant/event use the same standard-JSON
        # boundary as append, expected_count must be a non-negative plain int
        # (bool, negative, float and other types are ValueError), and
        # expected_hash must be exactly the 64 lowercase hex characters of a
        # sha256 digest (the empty-chain head is asserted with ZERO). A
        # malformed call ends here, creates nothing and never depends on
        # contention or history. The lease itself (and its missing-file
        # branch) is shared with append_batch_if_head; see _head_lease.
        _validate_json_value(tenant)
        _validate_json_value(event)
        _validate_required_count(expected_count)
        _validate_expected_hash(expected_hash)
        with self._head_lease(tenant, expected_count, expected_hash) as f:
            return self._append_if_head_locked(
                f, tenant, event, expected_count, expected_hash
            )

    def append_batch_if_head(self, tenant, events,
                             expected_count, expected_hash):
        # Conditional batch append: append_batch's atomic group commit fused
        # with append_if_head's optimistic head assertion. events must be a
        # list (a bare JSON value, including a string, is not a batch); the
        # tenant and every event cross the same standard-JSON boundary as
        # append; expected_count must be a non-negative plain int (bool,
        # negative, float and other types are ValueError) and expected_hash
        # must be exactly the 64 lowercase hex characters of a sha256 digest
        # (the empty-chain head is asserted with ZERO). Every input crosses
        # its boundary before a lease is taken, a path is probed or a single
        # byte is read, so a malformed call raises ValueError without
        # creating or touching anything, under any contention. An empty list
        # is a no-op after validation: nothing is created, no existing byte
        # is touched and no assertion is evaluated.
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
        # One exclusive lease covers the scan of the tenant's full chain, the
        # head comparison and the single block write, so the batch commits as
        # one indivisible byte interval: competing writers and shared-lease
        # readers only ever observe the state wholly before or wholly after
        # it. A corrupt history raises AuditChainStateError (same
        # tenant/seq/reason/line as append) and takes priority over the
        # conflict check; a well-formed but mismatching head raises
        # AuditChainConflictError with the actual tail; neither failure path
        # writes a byte. On a not-yet-existing log only an exact (0, ZERO)
        # assertion may create the file (see _head_lease). The returned
        # records are numbered from expected_count + 1 with the first prev
        # equal to expected_hash, and the committed bytes are identical to
        # calling append for each event in order from the same head.
        with self._head_lease(tenant, expected_count, expected_hash) as f:
            return self._append_batch_if_head_locked(
                f, tenant, events, expected_count, expected_hash
            )

    def append_many(self, entries):
        # Cross-tenant atomic append. entries must be a list of objects whose
        # key set is exactly {"tenant", "event"}; list order is the write
        # order. All of that, plus the standard-JSON boundary on every tenant
        # and event, is checked before a lease is taken, a path is probed or a
        # single byte is read, so a malformed call raises ValueError without
        # creating or touching anything, under any contention. An empty list
        # is a no-op after validation: no file is created, no history is read
        # and no byte changes.
        if not isinstance(entries, list):
            raise ValueError(
                f"entries must be a list, got {type(entries).__name__}"
            )
        norm = []
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != {"tenant", "event"}:
                raise ValueError(
                    "each entry must be an object containing exactly the "
                    f"keys 'tenant' and 'event', got {entry!r}"
                )
            tenant, event = entry["tenant"], entry["event"]
            _validate_json_value(tenant)
            _validate_json_value(event)
            norm.append((tenant, event))
        if not norm:
            return []
        with self._write_lease() as f:
            # One lease and one consistent snapshot cover the validation of
            # every affected chain and the single block write, so the whole
            # group commits as one indivisible byte interval: competing
            # writers serialize wholly before or after it and shared-lease
            # readers observe either the complete pre-commit or the complete
            # post-commit state, never a partial group.
            data = f.read()
            lines, bad_line = self._decode_lines(data)
            # Each involved tenant gets the exact full-chain scan append
            # runs (seqs 1..n from a ZERO prev, each digest recomputed);
            # foreign records interleave freely. A defect is recorded, not
            # raised immediately: when several affected chains are broken,
            # the first physical line in the log decides the result, and an
            # equal line is broken by input order.
            heads = {}   # serialized tenant -> [count, prev]
            broken = []  # (line, first_input_index, AuditChainStateError)
            for idx, (tenant, _event) in enumerate(norm):
                key = self._tenant_key(tenant)
                if key in heads:
                    continue
                try:
                    count, prev = self._scan(tenant, lines, bad_line)
                except _Broken as b:
                    seq = b.at if b.at is not None else b.expect
                    broken.append((
                        b.line, idx,
                        AuditChainStateError(tenant, seq, b.reason, b.line),
                    ))
                else:
                    heads[key] = [count, prev]
            if broken:
                raise min(broken, key=lambda x: (x[0], x[1]))[2] from None
            # Every affected pre-state is valid. Build the records in input
            # order; a tenant appearing several times gets contiguous seq
            # numbering for its occurrences, each record linking to the
            # previous one (on disk or earlier in this group), using the
            # identical fields, seq numbering, prev digest and hash rule as
            # append.
            items = []
            chunks = []
            progress = {key: [count, prev] for key, (count, prev) in heads.items()}
            for tenant, event in norm:
                key = self._tenant_key(tenant)
                count, prev = progress[key]
                item = {"tenant": tenant, "seq": count + 1,
                        "event": event, "prev": prev}
                item["hash"] = self._hash(item)
                items.append(item)
                chunks.append(
                    json.dumps(item, sort_keys=True, allow_nan=False) + "\n"
                )
                progress[key] = [count + 1, item["hash"]]
            # One O_APPEND write places the whole group atomically at the
            # current end of file; under the exclusive lease no other writer
            # moves that end, and no existing byte can be overwritten. The
            # physical JSONL order is the input order, while other tenants'
            # legitimate interleaved records already on disk stay in place.
            prefix = b"" if (not data or data.endswith(b"\n")) else b"\n"
            f.write(prefix + "".join(chunks).encode("utf-8"))
            return items

    @contextlib.contextmanager
    def _heads_lease(self, norm):
        # Exclusive lease for one conditional multi-tenant commit shared by
        # append_many_if_heads and import_all_range: validation snapshot, every
        # head comparison and the single block write must be indivisible
        # relative to all other writers. norm carries the asserted heads as
        # (tenant, expected_count, expected_hash) tuples (plus an entry's
        # events for append_many_if_heads); the lease itself only reads the
        # count and hash. On an existing log that is the ordinary data-file
        # lease -- a fresh "r+b" open fails instead of creating the path, so a
        # losing call leaves no file. Before the log exists, the data-file
        # lock cannot be taken without creating it (and a losing assertion
        # must leave no trace), so the directory lease -- the same one
        # ordinary appends hold while creating the file -- guards the whole
        # missing-file branch; the existence probe, the empty-chain
        # comparisons and a winning creation are therefore indivisible relative
        # to every other writer. The file is created ("a+b") only when every
        # assertion is an exact (0, ZERO) empty-chain assertion; otherwise the
        # first mismatching entry in input order is a deterministic conflict
        # against the (0, ZERO) actual head and is raised before any open that
        # could create the path. Lock order is always directory-then-data,
        # never deadlocking against a plain append. _dir_lease also makes the
        # parent directory.
        with self._dir_lease():
            try:
                f = open(self.path, "r+b")
            except FileNotFoundError:
                # Definitive empty chain for every tenant: no other writer can
                # create the path while this lease is held.
                mismatch = next(
                    ((t, ec, eh) for t, _e, ec, eh in norm
                     if ec != 0 or eh != ZERO),
                    None,
                )
                if mismatch is not None:
                    tenant, expected_count, expected_hash = mismatch
                    raise AuditChainConflictError(
                        tenant, expected_count, expected_hash, 0, ZERO
                    )
                f = open(self.path, "a+b")
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                f.seek(0)
                yield f
            finally:
                f.flush()
                f.close()

    def _append_many_if_heads_locked(self, f, norm):
        # Scan, compare and append on a description already holding the
        # exclusive data-file lease. norm is the fully validated entry list as
        # tuples (tenant, events(list, non-empty), expected_count,
        # expected_hash), canonical-tenant unique, in input order.
        data = f.read()
        lines, bad_line = self._decode_lines(data)
        # Each involved tenant gets the exact full-chain scan append runs
        # (seqs 1..n from a ZERO prev, each digest recomputed); foreign
        # records interleave freely. A defect is recorded, not raised
        # immediately: when several involved chains are broken, the first
        # physical line in the log decides the result, an equal line is broken
        # by the entry's first-appearance input index -- exactly the rule
        # append_many follows -- and damage to a tenant no entry involves is
        # ignored. The state is never asserted against an unverifiable chain.
        heads = {}   # serialized tenant -> [count, prev]
        broken = []  # (line, first_input_index, AuditChainStateError)
        for idx, (tenant, _events, _ec, _eh) in enumerate(norm):
            key = self._tenant_key(tenant)
            # norm is validated unique per canonical tenant identity, so each
            # key is scanned once; idx is its first (only) input position.
            try:
                count, prev = self._scan(tenant, lines, bad_line)
            except _Broken as b:
                seq = b.at if b.at is not None else b.expect
                broken.append((
                    b.line, idx,
                    AuditChainStateError(tenant, seq, b.reason, b.line),
                ))
            else:
                heads[key] = [count, prev]
        if broken:
            raise min(broken, key=lambda x: (x[0], x[1]))[2] from None
        # Every involved pre-state is valid. Compare every assertion in entry
        # order; the first mismatching one, regardless of tenant, raises with
        # that entry's expectation and the chain tail observed inside the
        # lease. State errors above take strict priority; a conflict writes no
        # byte.
        for tenant, _events, expected_count, expected_hash in norm:
            count, head = heads[self._tenant_key(tenant)]
            if count != expected_count or head != expected_hash:
                raise AuditChainConflictError(
                    tenant, expected_count, expected_hash, count, head
                )
        # All heads confirmed. Emit the records entry by entry, event by event:
        # each tenant is numbered continuously from its asserted tail, with
        # the first record's prev equal to the asserted hash and each later
        # record linking to the previous record of that tenant (on disk, or
        # earlier in this block). The physical JSONL order is the entry/event
        # input order. Fields, hashing, canonical serialization
        # (sort_keys=True, allow_nan=False) and the missing-newline prefix are
        # byte-for-byte the rules append/append_many follow.
        items = []
        chunks = []
        progress = {key: [count, prev] for key, (count, prev) in heads.items()}
        for tenant, events, _ec, _eh in norm:
            key = self._tenant_key(tenant)
            count, prev = progress[key]
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
            progress[key] = [count, prev]
        # One O_APPEND write places the whole group atomically at the current
        # end of file; under the exclusive lease no other writer moves that
        # end, and no existing byte can be overwritten.
        prefix = b"" if (not data or data.endswith(b"\n")) else b"\n"
        f.write(prefix + "".join(chunks).encode("utf-8"))
        return items

    def append_many_if_heads(self, entries):
        # Cross-tenant conditional atomic append: append_many's indivisible
        # multi-tenant group commit fused with one head assertion per tenant,
        # generalizing append_batch_if_head to several chains at once. entries
        # must be a list; every member must be an object whose key set is
        # exactly {"tenant", "events", "expected_count", "expected_hash"}.
        # events must be a non-empty list, expected_count a non-negative plain
        # int (bool rejected even though it subclasses int), expected_hash
        # exactly 64 lowercase hex characters (the empty-chain head is ZERO);
        # a canonical JSON tenant identity may appear at most once. The tenant
        # and every event cross the same standard-JSON boundary as append.
        # All of that is checked before a lease is taken, a path is probed or a
        # single byte is read, so a malformed call raises ValueError without
        # creating or touching anything, under any contention. An empty list
        # is a no-op after validation: no file is created, no history is read
        # and no byte changes.
        if not isinstance(entries, list):
            raise ValueError(
                f"entries must be a list, got {type(entries).__name__}"
            )
        norm = []
        seen = set()
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != {
                "tenant", "events", "expected_count", "expected_hash"
            }:
                raise ValueError(
                    "each entry must be an object containing exactly the keys "
                    "'tenant', 'events', 'expected_count' and "
                    f"'expected_hash', got {entry!r}"
                )
            tenant = entry["tenant"]
            events = entry["events"]
            expected_count = entry["expected_count"]
            expected_hash = entry["expected_hash"]
            if not isinstance(events, list) or not events:
                raise ValueError(
                    "events must be a non-empty list, got "
                    f"{events!r}"
                )
            _validate_json_value(tenant)
            for event in events:
                _validate_json_value(event)
            _validate_required_count(expected_count)
            _validate_expected_hash(expected_hash)
            key = self._tenant_key(tenant)
            if key in seen:
                raise ValueError(
                    f"duplicate tenant entry (same canonical JSON identity): "
                    f"{tenant!r}"
                )
            seen.add(key)
            norm.append((tenant, events, expected_count, expected_hash))
        if not norm:
            return []
        # One exclusive lease covers every involved chain's full scan, all
        # head comparisons and the single block write, so the group commits as
        # one indivisible byte interval: competing writers and shared-lease
        # readers only ever observe the state wholly before or wholly after
        # it. A not-yet-existing log may be created only when every assertion
        # is the exact empty-chain head (0, ZERO); any other assertion is a
        # deterministic conflict (actual (0, ZERO)) that leaves no file -- the
        # lease derives that rule from norm before opening the path. A corrupt
        # involved chain raises AuditChainStateError (same
        # tenant/seq/reason/line as append, first physical line then input
        # order deciding across chains) and takes priority over the conflict
        # checks; a well-formed but mismatching head raises
        # AuditChainConflictError at the first mismatching entry in entries
        # order, with the actual tail; neither failure path writes a byte.
        # Concurrent commits on overlapping heads therefore have at most one
        # winner; every loser observes the winner's new tail inside the lease
        # and conflicts with that actual head.
        with self._heads_lease(norm) as f:
            return self._append_many_if_heads_locked(f, norm)

    def _verify_all_snapshot(self, data, with_hash=False):
        # Core of verify_all over an exact in-memory snapshot. Touches no path:
        # the caller owns how the bytes were obtained (a shared-lease file read
        # or a caller-supplied buffer), so the same logic backs both
        # verify_all and the offline verify_all_bytes entry point. with_hash
        # additionally attaches each tenant's verified tail digest, backing
        # the heads() entry point without changing verify_all's result shape.
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
            seq = item["seq"]
            # Same strict JSON-integer seq rule as _scan: 1.0/1e0, bools,
            # strings, null, arrays/objects and mismatching integers are all
            # sequence defects, so file and memory scans agree exactly.
            if isinstance(seq, bool) or not isinstance(seq, int) \
                    or seq != expected:
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
        # state[1] is the running prev hash, which after a tenant's last
        # record is exactly that tenant's verified tail digest.
        tenants = []
        for t, s in order:
            entry = {"tenant": t, "count": s[2]}
            if with_hash:
                entry["hash"] = s[1]
            tenants.append(entry)
        return {"ok": True, "tenants": tenants}

    def _parse_all_snapshot(self, data):
        # Parse an already-verified in-memory snapshot into per-tenant records
        # without touching any path. The bytes are assumed to have passed
        # _verify_all_snapshot(data, with_hash=True), so every physical line is
        # strict standard JSON and the chains are internally consistent; the
        # parsing here never re-validates. Returns a dict mapping the serialized
        # canonical tenant key to [tenant_value, records_in_seq_order], inserted
        # in first-appearance order (so iteration preserves it); the tenant
        # value keeps its original first-appearance spelling and each record
        # keeps its parsed five-field values verbatim. Logical equality between
        # two snapshots is then plain Python value equality on these records,
        # aligned by canonical key and seq.
        lines, _bad_line = self._decode_lines(data)
        chains = {}
        for raw in lines:
            item = _strict_loads(raw)
            tenant = item["tenant"]
            key = self._tenant_key(tenant)
            chain = chains.get(key)
            if chain is None:
                chain = [tenant, []]
                chains[key] = chain
            chain[1].append(item)
        return chains

    def compare_bytes(self, left_data, right_data):
        # Pure in-memory reconciliation of two independent backup snapshots:
        # it consumes only the two given bytes buffers, never reads, creates or
        # modifies the path this AuditChain points at, never touches the
        # network and never mutates either buffer. The snapshots may interleave
        # the same tenants in different physical orders: comparison aligns
        # records by canonical JSON tenant identity (the same identity append/
        # verify_all use) and per-tenant seq, so interleaving order is never a
        # difference and a tenant's reported value is the first one that names
        # it on each side.
        #
        # The type boundary crosses first, before either snapshot is parsed:
        # each argument must be exactly bytes -- bytearray, str and every other
        # type are rejected with ValueError immediately, so a non-bytes value
        # can never reach the decoder. Both buffers then run the exact
        # verify_all_bytes rules (strict UTF-8, LF-only physical lines,
        # standard JSON with duplicate keys and non-standard numbers rejected,
        # field set, canonical tenant identity, per-tenant seq, prev and hash).
        # A corrupt snapshot is not compared: the result reports that side with
        # verify_all_bytes' exact failure fields and "side" naming it, and when
        # both are corrupt the left side is always reported first. Empty
        # snapshots equal one another and a tenant absent from a side is its
        # empty chain.
        #
        # Once both are valid, tenants are compared in left first-appearance
        # order followed by tenants appearing only on the right, each chain
        # record by record from seq 1 on the logical values. Full agreement
        # returns {"ok": True, "equal": True, "tenants": [...]} with one entry
        # per tenant (original tenant, count and tail hash) in that order. The
        # first difference returns
        # {"ok": True, "equal": False, "tenant", "seq", "reason", "left",
        # "right"}: a record present on both sides but differing in event, prev
        # or hash is "different" (both records attached); a chain longer on
        # either side reports the first absent record as "missing_left" or
        # "missing_right" with the existing record attached and the missing
        # side None. Exactly one position -- the first the ordering selects --
        # is reported, never a cascade of follow-on differences.
        if not isinstance(left_data, bytes):
            raise ValueError(
                "left_data must be bytes, got "
                f"{type(left_data).__name__}"
            )
        if not isinstance(right_data, bytes):
            raise ValueError(
                "right_data must be bytes, got "
                f"{type(right_data).__name__}"
            )
        left_heads = self._verify_all_snapshot(left_data, with_hash=True)
        if not left_heads["ok"]:
            return {
                "ok": False,
                "side": "left",
                "at": left_heads["at"],
                "tenant": left_heads["tenant"],
                "reason": left_heads["reason"],
            }
        right_heads = self._verify_all_snapshot(right_data, with_hash=True)
        if not right_heads["ok"]:
            return {
                "ok": False,
                "side": "right",
                "at": right_heads["at"],
                "tenant": right_heads["tenant"],
                "reason": right_heads["reason"],
            }
        left_chains = self._parse_all_snapshot(left_data)
        right_chains = self._parse_all_snapshot(right_data)
        # Tenant comparison order: the left side's first-appearance order, then
        # any tenant only the right side names (in its first-appearance order).
        order = list(left_chains)
        for key in right_chains:
            if key not in left_chains:
                order.append(key)
        for key in order:
            left_chain = left_chains.get(key)
            right_chain = right_chains.get(key)
            left_records = left_chain[1] if left_chain is not None else []
            right_records = right_chain[1] if right_chain is not None else []
            # The tenant value reported for a difference is the first one that
            # names this canonical identity, preferring the left side because
            # the ordering starts there.
            if left_chain is not None:
                tenant = left_chain[0]
            else:
                tenant = right_chain[0]
            shared = min(len(left_records), len(right_records))
            for i in range(shared):
                left_item = left_records[i]
                right_item = right_records[i]
                # Records aligned by canonical tenant identity and seq: only
                # event, prev and hash can differ logically, and any one of
                # them makes the position "different".
                if left_item["event"] != right_item["event"] \
                        or left_item["prev"] != right_item["prev"] \
                        or left_item["hash"] != right_item["hash"]:
                    return {
                        "ok": True,
                        "equal": False,
                        "tenant": tenant,
                        "seq": i + 1,
                        "reason": "different",
                        "left": left_item,
                        "right": right_item,
                    }
            if len(left_records) > len(right_records):
                return {
                    "ok": True,
                    "equal": False,
                    "tenant": tenant,
                    "seq": shared + 1,
                    "reason": "missing_right",
                    "left": left_records[shared],
                    "right": None,
                }
            if len(right_records) > len(left_records):
                return {
                    "ok": True,
                    "equal": False,
                    "tenant": tenant,
                    "seq": shared + 1,
                    "reason": "missing_left",
                    "left": None,
                    "right": right_records[shared],
                }
        # Every aligned chain agrees record for record. The summary follows the
        # comparison order above, carrying each tenant's original value, count
        # and tail hash from the side that first names it.
        tenants = []
        for key in order:
            chain = left_chains.get(key)
            if chain is None:
                chain = right_chains[key]
            records = chain[1]
            tenants.append({
                "tenant": chain[0],
                "count": len(records),
                "hash": records[-1]["hash"] if records else ZERO,
            })
        return {"ok": True, "equal": True, "tenants": tenants}

    def compare_chunks(self, left_chunks, right_chunks):
        # Chunked offline counterpart of compare_bytes: each side's JSONL
        # snapshot arrives as an iterable of bytes chunks in file order rather
        # than one contiguous buffer, so a caller reconciling two streamed
        # backups never has to assemble either side itself. Chunks may be cut
        # at any byte boundary -- mid-UTF-8-sequence, mid-line or mid-record --
        # empty chunks are allowed, and an empty iterable (or one holding only
        # empty chunks) is the empty snapshot; only each side's in-order
        # concatenation is ever decoded. Like compare_bytes this consumes only
        # caller data: it never reads, creates or modifies the path this
        # AuditChain points at, never touches the network, keeps no cache or
        # on-disk index and never mutates either input chunk.
        #
        # The container boundary crosses first, exactly the boundary
        # verify_all_chunks/heads_chunks use on each side: a bare bytes or
        # bytearray object is not a chunk container (its iteration would yield
        # ints), a non-iterable container and a non-bytes (bytearray included)
        # element are all ValueError, and the first bad element ends
        # consumption of that side without pulling any further element. The
        # left side is checked first and consumed to completion before the
        # right side is even probed, so a left boundary error wins over a
        # right one; every boundary error on either side wins over content
        # validation, so a bad right chunk raises even though the left's
        # concatenation is corrupt. Only after both sides have crossed the
        # boundary are the two concatenations handed to compare_bytes, making
        # the result field-for-field compare_bytes' verdict on the same bytes:
        # strict UTF-8, LF-only physical lines, standard JSON, canonical
        # tenant identity, per-tenant seq/prev/hash validation with left
        # reported first, then tenant-by-tenant comparison from seq 1 in left
        # first-appearance order followed by right-only tenants, returning the
        # same equal/tenants summary or exactly one first difference
        # (different/missing_left/missing_right with the original records or
        # None), never a cascade.
        left_data = self._join_chunks(left_chunks)
        right_data = self._join_chunks(right_chunks)
        return self.compare_bytes(left_data, right_data)

    def verify_all(self):
        # Validate every tenant chain in one read-only pass over the file, in
        # physical line order. Each tenant gets an independent expected seq and
        # prev digest starting at (1, ZERO) on first appearance; records of
        # different tenants may interleave. The serialized tenant is only an
        # internal key so distinct types (1 vs "1") stay separate chains while
        # the original value is reported back unchanged. The shared lease
        # pins the snapshot to a state wholly before or after any append.
        return self._verify_all_snapshot(self._read_snapshot())

    def heads(self):
        # Read-only chain-head directory for every tenant in the log: the
        # multi-tenant counterpart of head(), fusing verify_all's validation
        # pass with the tail digest each tenant's head() would report, so a
        # caller can prepare conditional appends for several tenants from one
        # consistent view. Takes no arguments and no input boundary applies.
        #
        # Success returns {"ok": True, "tenants": [...]} with one entry per
        # tenant in first-appearance physical order, each carrying the
        # original tenant value verbatim plus its verified count and tail
        # hash -- the exact (expected_count, expected_hash) pair an
        # append_if_head/append_batch_if_head call for that tenant asserts.
        # A missing or empty log is a legitimate empty history and yields
        # {"ok": True, "tenants": []}. Every chain is validated over one
        # shared-lease snapshot with the same identity, sequence and digest
        # rules as verify_all, so the first defect in physical line order is
        # reported with verify_all's exact failure fields
        # ({"ok": False, "at": line, "tenant": ..., "reason": ...}) and no
        # partial head list is ever returned. Like every read entry point it
        # creates nothing, modifies no byte, writes no cache and never
        # touches the network; concurrent appends only choose which complete
        # pre- or post-append snapshot the result corresponds to.
        return self._verify_all_snapshot(self._read_snapshot(), with_hash=True)

    def _export_tenant_bytes(self, tenant):
        # Shared body of export_tenant and the chunked exporter: the caller
        # has already validated the tenant across the standard-JSON boundary.
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

    def export_tenant(self, tenant):
        # Offline, single-tenant migration export. Returns UTF-8 bytes only:
        # it never creates, mutates or deletes any path and never touches the
        # network. The tenant crosses the same standard-JSON input boundary as
        # append/verify, and, like verify, it is checked before any history is
        # read, so an illegal value raises ValueError without a single byte
        # being read or created.
        _validate_json_value(tenant)
        return self._export_tenant_bytes(tenant)

    def export_tenant_range(self, tenant, start_seq, end_seq=None):
        # Segmented offline migration export: the closed interval
        # [start_seq, end_seq] of one tenant's verified chain, so an archive
        # or restore can continue from a known chain head. Returns UTF-8
        # bytes only: like export_tenant it never creates, mutates or deletes
        # any path and never touches the network.
        #
        # Every input crosses its boundary before a single byte is read: the
        # tenant uses the same standard-JSON rule as append/verify/export,
        # start_seq must be a positive plain int (bool is rejected even
        # though it subclasses int; floats and other types are too), and
        # end_seq must be None (open tail) or a plain int not smaller than
        # start_seq. A rejected call raises ValueError without reading or
        # creating anything, under any contention.
        _validate_json_value(tenant)
        if isinstance(start_seq, bool) or not isinstance(start_seq, int) \
                or start_seq < 1:
            raise ValueError(
                f"start_seq must be a positive integer, got {start_seq!r}"
            )
        if end_seq is not None and (
            isinstance(end_seq, bool) or not isinstance(end_seq, int)
            or end_seq < start_seq
        ):
            raise ValueError(
                "end_seq must be None or an integer not smaller than "
                f"start_seq, got {end_seq!r}"
            )
        # One shared-lease snapshot, exactly as verify/export_tenant use: the
        # segment is cut from the full state wholly before or wholly after
        # any concurrent append, never a torn read. The tenant's complete
        # chain -- first record through tail -- must verify with the exact
        # JSON identity, sequence and digest rules of verify before any
        # record is surfaced: a corrupt source raises AuditChainStateError
        # (same tenant/seq/reason/line as append/verify/export_tenant) even
        # when the requested interval lies entirely before the damage, and no
        # verified prefix is ever returned.
        data = self._read_snapshot()
        lines, bad_line = self._decode_lines(data)
        try:
            count, _ = self._scan(tenant, lines, bad_line)
        except _Broken as b:
            seq = b.at if b.at is not None else b.expect
            raise AuditChainStateError(tenant, seq, b.reason, b.line) from None
        # Snapshot proven valid. The closed interval must be fully available:
        # an empty chain (count 0 fails start_seq >= 1), a start past the
        # tail, or an explicit end past the tail all raise
        # AuditChainRangeError with the verified count, instead of returning
        # a short or empty segment. end_seq=None means the open tail and can
        # never exceed it.
        if start_seq > count or (end_seq is not None and end_seq > count):
            raise AuditChainRangeError(tenant, start_seq, end_seq, count)
        end = count if end_seq is None else end_seq
        # Only now collect the target tenant's records in physical order
        # (identical to seq order for that tenant) and keep exactly the
        # requested interval. Records are re-emitted verbatim with the exact
        # serialization and newline rule append/export_tenant use: seq, prev
        # and hash keep their source values, so the first record's prev still
        # names the omitted predecessor chain head and import_tenant_range
        # can re-anchor the segment on the matching target tail.
        key = self._tenant_key(tenant)
        chunks = []
        for raw in lines:
            item = _strict_loads(raw)
            if self._tenant_key(item["tenant"]) == key \
                    and start_seq <= item["seq"] <= end:
                chunks.append(
                    (json.dumps(item, sort_keys=True, allow_nan=False) + "\n")
                    .encode("utf-8")
                )
        return b"".join(chunks)

    def export_all(self):
        # Whole-log offline migration export: the multi-tenant counterpart
        # of export_tenant. Returns the log's raw UTF-8 bytes only: it never
        # creates, mutates or deletes any path and never touches the network.
        # A missing or empty log is a legitimate empty snapshot and yields
        # b"". One shared-lease snapshot pins the result to a state wholly
        # before or after some append, so a concurrent writer can never land
        # in the middle of it and the file is never modified.
        #
        # Before any byte is returned, every tenant chain in the snapshot is
        # validated in one physical-order pass with the same identity,
        # sequence and digest rules verify_all applies: a physical line that
        # is not strict UTF-8 / standard JSON, carries duplicate keys or
        # non-standard numbers, or any tenant's missing field, sequence gap
        # or prev/hash mismatch raises AuditChainStateError -- reason only
        # missing/sequence/digest, located by tenant, seq and the physical
        # line of the first defect -- instead of returning a partial result.
        # The verified bytes are returned verbatim (the exact snapshot,
        # interleaved tenant order included), so import_all can graft the
        # whole history onto an empty log with no re-serialization ambiguity.
        data = self._read_snapshot()
        lines, bad_line = self._decode_lines(data)
        self._scan_all_chains(lines, bad_line)
        return data

    @staticmethod
    def _validate_chunk_size(chunk_size):
        # chunk_size names the maximum byte length of one produced export
        # chunk: it must be a positive plain int. bool is rejected even though
        # it subclasses int (so True/False are never a size of one/zero), as
        # are floats, strings and every other type, and zero/negative ints.
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) \
                or chunk_size <= 0:
            raise ValueError(
                f"chunk_size must be a positive integer, got {chunk_size!r}"
            )

    @staticmethod
    def _iter_fixed_chunks(data, chunk_size):
        # Yield the verified snapshot bytes as successive slices of at most
        # chunk_size bytes, splitting purely on byte offsets with no regard
        # for UTF-8, JSON or newline boundaries, so the concatenation of the
        # yielded chunks is byte-for-byte the input. Empty data yields no
        # chunk at all; every non-empty input yields at least one chunk, the
        # last possibly short. chunk_size is already validated positive.
        for i in range(0, len(data), chunk_size):
            yield data[i:i + chunk_size]

    def export_tenant_chunks(self, tenant, chunk_size):
        # Chunked offline, single-tenant migration export for large histories:
        # the large-history counterpart of export_tenant. Returns an iterator
        # of bytes chunks whose concatenation is byte-for-byte exactly what
        # export_tenant(tenant) returns for the same snapshot; the iterator
        # itself is the result, so callers stream the history instead of
        # holding one contiguous copy. Chunks are cut on plain byte offsets and
        # may split a UTF-8 sequence, a JSON object or a newline. Like
        # export_tenant this never creates, mutates or deletes any path and
        # never touches the network.
        #
        # The tenant crosses the same standard-JSON boundary as
        # append/verify/export_tenant and chunk_size must be a positive plain
        # int (bool rejected even though it subclasses int; floats, strings,
        # zero and negatives too); both are checked before the chunk iterator
        # is produced or any history is read, so an illegal call raises
        # ValueError immediately rather than from a half-consumed iterator.
        _validate_json_value(tenant)
        self._validate_chunk_size(chunk_size)
        # The whole export is fixed to one complete snapshot: the full chain
        # verification -- the exact scan export_tenant runs -- completes over
        # one shared-lease snapshot before the first chunk is yielded, so a
        # corrupt source raises AuditChainStateError (same
        # tenant/seq/reason/line as export_tenant) instead of yielding a
        # partial prefix, and a concurrent append can never land between
        # chunks. A missing tenant or an empty/missing log is a legitimate
        # empty export and yields no chunks.
        data = self._export_tenant_bytes(tenant)
        return self._iter_fixed_chunks(data, chunk_size)

    def export_all_chunks(self, chunk_size):
        # Chunked whole-log offline migration export: the large-history
        # counterpart of export_all. Returns an iterator of bytes chunks whose
        # concatenation is byte-for-byte exactly what export_all() returns for
        # the same snapshot (the verified raw bytes, interleaved tenant order
        # included). Chunks are cut on plain byte offsets and may split a
        # UTF-8 sequence, a JSON object or a newline. Like export_all this
        # never creates, mutates or deletes any path and never touches the
        # network.
        #
        # chunk_size must be a positive plain int (bool rejected even though
        # it subclasses int; floats, strings, zero and negatives too), checked
        # before the iterator is produced or any history is read, so an
        # illegal call raises ValueError immediately. The whole export is
        # fixed to one complete snapshot: every tenant chain verifies over one
        # shared-lease snapshot before the first chunk is yielded (the exact
        # pass export_all runs), so a defect raises AuditChainStateError --
        # reason only missing/sequence/digest, located by tenant, seq and
        # physical line -- instead of yielding a partial result. A missing or
        # empty log is a legitimate empty export and yields no chunks.
        self._validate_chunk_size(chunk_size)
        data = self.export_all()
        return self._iter_fixed_chunks(data, chunk_size)

    def export_tenant_range_chunks(self, tenant, start_seq, end_seq=None,
                                   chunk_size=None):
        # Chunked segmented offline migration export: the large-history
        # counterpart of export_tenant_range. Returns an iterator of bytes
        # chunks whose concatenation is byte-for-byte exactly what
        # export_tenant_range(tenant, start_seq, end_seq) returns for the
        # same snapshot -- the closed interval [start_seq, end_seq] of one
        # tenant's verified chain (end_seq=None the open tail), records
        # keeping their source seq/prev/hash values verbatim. Chunks are cut
        # on plain byte offsets and may split a UTF-8 sequence, a JSON object
        # or a newline. Like export_tenant_range this never creates, mutates
        # or deletes any path and never touches the network.
        #
        # Every input crosses its boundary before the chunk iterator is
        # produced or a single byte is read: the tenant uses the same
        # standard-JSON rule as append/verify/export, start_seq must be a
        # positive plain int, end_seq must be None or a plain int not smaller
        # than start_seq, and chunk_size must be a positive plain int (bool
        # rejected even though it subclasses int; floats, strings, zero and
        # negatives too). A rejected call raises ValueError immediately
        # rather than from a half-consumed iterator, under any contention.
        _validate_json_value(tenant)
        if isinstance(start_seq, bool) or not isinstance(start_seq, int) \
                or start_seq < 1:
            raise ValueError(
                f"start_seq must be a positive integer, got {start_seq!r}"
            )
        if end_seq is not None and (
            isinstance(end_seq, bool) or not isinstance(end_seq, int)
            or end_seq < start_seq
        ):
            raise ValueError(
                "end_seq must be None or an integer not smaller than "
                f"start_seq, got {end_seq!r}"
            )
        self._validate_chunk_size(chunk_size)
        # The whole export is fixed to one complete snapshot: the full-chain
        # verification and the range check export_tenant_range runs complete
        # over one shared-lease snapshot before the first chunk is yielded,
        # so a corrupt source raises AuditChainStateError and an
        # unsatisfiable interval AuditChainRangeError (same fields as
        # export_tenant_range) instead of yielding a partial prefix, and a
        # concurrent append can never land between chunks.
        data = self.export_tenant_range(tenant, start_seq, end_seq)
        return self._iter_fixed_chunks(data, chunk_size)

    def import_tenant(self, tenant, data):
        # Offline counterpart of export_tenant: graft a single-tenant export
        # (or any strictly valid single-tenant JSONL history) onto this log
        # without a network or remote anchor. The records keep their original
        # tenant/seq/event/prev/hash values verbatim -- nothing is renumbered
        # or recomputed -- so verify/verify_all/verify_bytes/export_tenant
        # validate the merged log by the exact existing rules.
        #
        # Boundary first, exactly as every other entry point orders it: the
        # tenant crosses the standard-JSON boundary and data must be bytes,
        # both before any lease is taken, any history is read or the path is
        # probed, so an illegal call raises ValueError regardless of target
        # state or contention and creates nothing. Empty bytes are a no-op
        # after that validation: no file is created, no history is read and
        # no byte changes.
        _validate_json_value(tenant)
        if not isinstance(data, bytes):
            raise ValueError(f"data must be bytes, got {type(data).__name__}")
        if not data:
            return []
        # The complete input chain is verified before the target is ever
        # touched: strict UTF-8 JSONL, this exact canonical tenant identity on
        # every physical line, seq starting at 1 with prev ZERO and each
        # digest matching. A chain defect raises AuditChainStateError --
        # reason missing/sequence/digest, with tenant, seq and physical line
        # filled as append fills them; a record of any other tenant identity
        # violates the single-tenant input contract and raises ValueError.
        # Either kind reaches the caller before a target-side check can.
        lines, bad_line = self._decode_lines(data)
        try:
            records = self._scan_import(tenant, lines, bad_line)
        except _ForeignTenant as ft:
            raise ValueError(
                f"import data must contain only tenant {tenant!r}; "
                f"a record of another tenant appears at line {ft.line}"
            ) from None
        except _Broken as b:
            seq = b.at if b.at is not None else b.expect
            raise AuditChainStateError(tenant, seq, b.reason, b.line) from None
        with self._write_lease() as f:
            # Target validation and the append share one exclusive lease on
            # the description used for both, so the graft is indivisible:
            # competing writers serialize wholly before or after it. A corrupt
            # target is reported exactly as append reports it and takes
            # priority over the emptiness assertion; a target that already
            # holds a record for this tenant is a conflict -- an import can
            # only land on an empty chain for that tenant -- with the fixed
            # empty-head expectation (0, ZERO) and the observed tail as
            # actual_*. Other tenants' valid records may already be present.
            target = f.read()
            t_lines, t_bad = self._decode_lines(target)
            try:
                count, head = self._scan(tenant, t_lines, t_bad)
            except _Broken as b:
                seq = b.at if b.at is not None else b.expect
                raise AuditChainStateError(tenant, seq, b.reason, b.line) from None
            if count != 0:
                raise AuditChainConflictError(
                    tenant, 0, ZERO, count, head
                )
            # One O_APPEND write commits the whole history as a single block,
            # re-serialized with the exact rule append/export use. The first
            # imported record has prev ZERO, so it links directly into the
            # empty target chain regardless of the other tenants interleaved
            # on disk; seqs and hashes stay exactly the validated ones.
            chunks = [
                json.dumps(item, sort_keys=True, allow_nan=False) + "\n"
                for item in records
            ]
            prefix = b"" if (not target or target.endswith(b"\n")) else b"\n"
            f.write(prefix + "".join(chunks).encode("utf-8"))
            return records

    def import_tenant_range(self, tenant, data, expected_count, expected_hash):
        # Segmented offline counterpart of export_tenant_range: graft a
        # single-tenant chain segment (typically produced by
        # export_tenant_range) onto this log so the target chain continues
        # from a known, asserted head. The records keep their original
        # tenant/seq/event/prev/hash values verbatim -- nothing is renumbered
        # or recomputed -- so verify/verify_all/verify_bytes/export_tenant
        # validate the merged log by the exact existing rules.
        #
        # Boundary first, exactly as every other entry point orders it: the
        # tenant crosses the standard-JSON boundary, data must be bytes,
        # expected_count must be a non-negative plain int (bool rejected even
        # though it subclasses int; floats, strings and other types are too)
        # and expected_hash must be exactly the 64 lowercase hex characters
        # of a sha256 digest (an empty target chain is asserted with ZERO).
        # All of that is checked before any lease is taken, any history is
        # read or the path is probed, so an illegal call raises ValueError
        # regardless of target state or contention and creates nothing.
        # Empty bytes are a no-op after that validation: no file is created,
        # no history is read and no byte changes.
        _validate_json_value(tenant)
        if not isinstance(data, bytes):
            raise ValueError(f"data must be bytes, got {type(data).__name__}")
        _validate_required_count(expected_count)
        _validate_expected_hash(expected_hash)
        if not data:
            return []
        # The complete input segment is verified before the target is ever
        # touched: strict UTF-8 JSONL, this exact canonical tenant identity
        # on every physical line, every record carrying exactly the five
        # existing fields, seqs running expected_count+1, expected_count+2,
        # ... with the first prev equal to expected_hash and each digest
        # recomputed. A chain defect raises AuditChainStateError -- reason
        # missing/sequence/digest, with tenant, seq and the input's physical
        # line filled as append fills them; a record of any other tenant
        # identity violates the single-tenant input contract and raises
        # ValueError. Either kind reaches the caller before a target-side
        # check can.
        lines, bad_line = self._decode_lines(data)
        try:
            records = self._scan_import_range(
                tenant, lines, bad_line, expected_count + 1, expected_hash
            )
        except _ForeignTenant as ft:
            raise ValueError(
                f"import data must contain only tenant {tenant!r}; "
                f"a record of another tenant appears at line {ft.line}"
            ) from None
        except _Broken as b:
            seq = b.at if b.at is not None else b.expect
            raise AuditChainStateError(tenant, seq, b.reason, b.line) from None
        # Target validation, the head assertion and the append share one
        # exclusive lease (the same one append_if_head uses: the data-file
        # lease on an existing log, the directory lease guarding the
        # missing-file branch), so check-and-commit is a single atomic
        # operation. A not-yet-existing log may be created only by an exact
        # (0, ZERO) assertion; any other assertion is a deterministic
        # conflict against the (0, ZERO) actual head that leaves no file.
        with self._head_lease(tenant, expected_count, expected_hash) as f:
            # A corrupt target is reported exactly as append reports it and
            # takes priority over the conflict check; a well-formed target
            # whose count and tail hash do not both match the assertion is a
            # conflict carrying the observed tail as actual_*. Neither
            # failure path writes a byte. Concurrent commits on the same head
            # serialize here: at most one wins, and every loser observes the
            # winner's new tail inside the lease and conflicts with those
            # actual values.
            target = f.read()
            t_lines, t_bad = self._decode_lines(target)
            try:
                count, head = self._scan(tenant, t_lines, t_bad)
            except _Broken as b:
                seq = b.at if b.at is not None else b.expect
                raise AuditChainStateError(tenant, seq, b.reason, b.line) from None
            if count != expected_count or head != expected_hash:
                raise AuditChainConflictError(
                    tenant, expected_count, expected_hash, count, head
                )
            # Head confirmed: the segment's first prev names exactly this
            # tail, so the graft links seamlessly. One O_APPEND write commits
            # the whole segment as a single block, re-serialized with the
            # exact rule append/export use; other tenants' interleaved
            # records already on disk stay in place.
            chunks = [
                json.dumps(item, sort_keys=True, allow_nan=False) + "\n"
                for item in records
            ]
            prefix = b"" if (not target or target.endswith(b"\n")) else b"\n"
            f.write(prefix + "".join(chunks).encode("utf-8"))
            return records

    def import_all(self, data):
        # Whole-log offline counterpart of export_all: graft a multi-tenant
        # interleaved JSONL snapshot (typically produced by export_all) onto
        # this log as one indivisible block, without a network or remote
        # anchor. Records keep their original tenant/seq/event/prev/hash
        # values verbatim -- nothing is renumbered or recomputed -- so
        # verify/verify_all/verify_bytes/export_tenant/export_all validate
        # the merged log by the exact existing rules.
        #
        # Boundary first, exactly as import_tenant orders it: data must be
        # bytes, checked before any lease is taken, any history is read or
        # the path is probed, so an illegal call raises ValueError regardless
        # of target state or contention and creates nothing. Empty bytes are
        # a no-op after that check: no file is created, no history is read
        # and no byte changes.
        if not isinstance(data, bytes):
            raise ValueError(f"data must be bytes, got {type(data).__name__}")
        if not data:
            return []
        # The complete input is verified in memory before the target is ever
        # touched: strict UTF-8 JSONL, every physical line an object carrying
        # exactly the five existing fields, tenants allowed to interleave but
        # each one's chain running 1..n from a ZERO prev with every digest
        # recomputed. Parse failures, missing or extra fields, duplicate
        # keys, non-standard numbers and illegal UTF-8 are missing, sequence
        # defects are sequence, prev/hash mismatches are digest -- all raised
        # as AuditChainStateError located by tenant, seq and the input's
        # physical line number, the first defective line deciding.
        lines, bad_line = self._decode_lines(data)
        records = self._scan_all_chains(lines, bad_line, exact_fields=True)
        # Tenants involved in the input, in first-appearance order: only
        # their chains are checked on the target below.
        involved = []
        seen = set()
        for item in records:
            key = self._tenant_key(item["tenant"])
            if key not in seen:
                seen.add(key)
                involved.append(item["tenant"])
        with self._write_lease() as f:
            # Target validation and the append share one exclusive lease on
            # the description used for both, so check-and-commit is a single
            # atomic operation: competing writers and shared-lease readers
            # only ever observe the state wholly before or wholly after the
            # graft. Each involved tenant gets the exact full-chain scan
            # append runs; a corrupt target chain raises
            # AuditChainStateError (same tenant/seq/reason/line as append,
            # first physical line then input first-appearance order deciding
            # across chains) and takes priority over the emptiness assertion.
            # A target that already holds records for an involved tenant is a
            # conflict, raised in input first-appearance order with the fixed
            # empty-head expectation (0, ZERO) and the observed tail as
            # actual_*. Neither failure path writes a byte.
            target = f.read()
            t_lines, t_bad = self._decode_lines(target)
            heads = {}   # serialized tenant -> (count, head)
            broken = []  # (line, first_input_index, AuditChainStateError)
            for idx, tenant in enumerate(involved):
                try:
                    count, head = self._scan(tenant, t_lines, t_bad)
                except _Broken as b:
                    seq = b.at if b.at is not None else b.expect
                    broken.append((
                        b.line, idx,
                        AuditChainStateError(tenant, seq, b.reason, b.line),
                    ))
                else:
                    heads[self._tenant_key(tenant)] = (count, head)
            if broken:
                raise min(broken, key=lambda x: (x[0], x[1]))[2] from None
            for tenant in involved:
                count, head = heads[self._tenant_key(tenant)]
                if count != 0:
                    raise AuditChainConflictError(tenant, 0, ZERO, count, head)
            # Every involved chain is empty. One O_APPEND write commits the
            # whole snapshot as a single block, re-serialized with the exact
            # rule append/export use; each tenant's first record has prev
            # ZERO, so it links directly into its empty target chain
            # regardless of the other tenants interleaved on disk. The file
            # is created by the lease only on this winning path: a missing
            # target cannot fail any check above.
            chunks = [
                json.dumps(item, sort_keys=True, allow_nan=False) + "\n"
                for item in records
            ]
            prefix = b"" if (not target or target.endswith(b"\n")) else b"\n"
            f.write(prefix + "".join(chunks).encode("utf-8"))
            return records

    def import_tenant_chunks(self, tenant, chunks):
        # Chunked offline counterpart of import_tenant for large histories:
        # the caller streams a single-tenant JSONL export as an ordered
        # iterable of bytes chunks instead of one contiguous buffer. Chunks
        # may be cut at any byte boundary -- mid-UTF-8-sequence, mid-line or
        # mid-record -- and empty chunks are allowed; only their in-order
        # concatenation is the history. Records keep their original
        # tenant/seq/event/prev/hash values verbatim -- nothing is renumbered
        # or recomputed.
        #
        # Boundary first, exactly as import_tenant orders it: the tenant
        # crosses the standard-JSON boundary and chunks must be an iterable
        # of bytes -- a bare bytes or bytearray object is not a chunk
        # container (its iteration would yield ints), a non-iterable
        # container, or a non-bytes element are all ValueError, a bad element
        # ending consumption at once -- all before any lease is taken, any
        # history is read or the path is probed. The complete input is joined
        # and verified before the target is touched, so field-for-field this
        # accepts exactly the complete contents import_tenant accepts and
        # raises exactly what import_tenant raises: chain defects are
        # AuditChainStateError (reason missing/sequence/digest, with tenant,
        # seq and the input's physical line), a record of another canonical
        # tenant identity is ValueError, an existing target chain for this
        # tenant is AuditChainConflictError, and input problems take strict
        # priority over target state. An empty concatenation is a no-op after
        # validation returning [], creating and reading nothing. On success
        # the whole content lands as one indivisible commit and the parsed
        # records are returned in physical order, identical objects to what
        # import_tenant returns for the same bytes; no failure path writes a
        # single partial byte, and shared-lease readers observe only the
        # complete pre-commit or post-commit state.
        _validate_json_value(tenant)
        data = self._join_chunks(chunks)
        return self.import_tenant(tenant, data)

    def import_all_chunks(self, chunks):
        # Chunked whole-log offline counterpart of import_all for large
        # histories: the caller streams a multi-tenant interleaved JSONL
        # snapshot (typically produced by export_all/export_all_chunks) as an
        # ordered iterable of bytes chunks. Chunks may be cut at any byte
        # boundary -- mid-UTF-8-sequence, mid-line or mid-record -- and empty
        # chunks are allowed; only their in-order concatenation is the
        # snapshot. Records keep their original values verbatim.
        #
        # Boundary first, exactly as import_all orders it: chunks must be an
        # iterable of bytes -- a bare bytes or bytearray object is not a
        # chunk container, a non-iterable container or a non-bytes element
        # are all ValueError, a bad element ending consumption at once --
        # checked before any lease is taken, any history is read or the path
        # is probed. The complete input is joined and verified in memory
        # before the target is touched, so field-for-field this accepts
        # exactly the complete contents import_all accepts and raises exactly
        # what import_all raises: malformed lines, fields, sequences and
        # digests are AuditChainStateError located by tenant, seq and the
        # input's physical line, a target that already holds an involved
        # tenant is AuditChainConflictError in input first-appearance order,
        # and input problems take strict priority over target state. An empty
        # concatenation is a no-op after validation returning [], creating
        # and reading nothing. On success the whole snapshot lands as one
        # indivisible block and the parsed records are returned in physical
        # order, identical objects to what import_all returns for the same
        # bytes; no failure path writes a single partial byte, and
        # shared-lease readers observe only the complete pre-commit or
        # post-commit state.
        data = self._join_chunks(chunks)
        return self.import_all(data)

    def import_all_range(self, data, expected_heads):
        # Multi-tenant segmented offline migration: the cross-tenant
        # counterpart of import_tenant_range. Several tenants' JSONL segments
        # (typically produced by export_tenant_range calls against a source
        # log) may interleave in one byte stream and are grafted onto this
        # log as one indivisible block, each tenant continuing from its own
        # known chain head named in expected_heads, without a network or
        # remote anchor. Records keep their original
        # tenant/seq/event/prev/hash values verbatim -- nothing is renumbered
        # or recomputed -- so verify/verify_all/verify_bytes/export_* validate
        # the merged log by the exact existing rules.
        #
        # Boundary first, exactly as every other entry point orders it: data
        # must be bytes and expected_heads must be a list whose members are
        # objects carrying exactly the keys tenant, expected_count and
        # expected_hash; the tenant crosses the standard-JSON boundary,
        # expected_count must be a non-negative plain int (bool rejected even
        # though it subclasses int; floats, strings and other types are too)
        # and expected_hash must be exactly the 64 lowercase hex characters
        # of a sha256 digest (an empty target chain is asserted with ZERO); a
        # canonical JSON tenant identity may appear at most once. All of that
        # is checked before any lease is taken, any history is read or the
        # path is probed, so an illegal call raises ValueError regardless of
        # target state or contention and creates nothing. Empty bytes are a
        # no-op after that validation: no file is created, no history is read
        # and no byte changes; the assertions themselves are not evaluated.
        if not isinstance(data, bytes):
            raise ValueError(f"data must be bytes, got {type(data).__name__}")
        assertions = _validate_expected_range_heads(expected_heads)
        if not data:
            return []
        # The complete input is verified in memory, before the target is ever
        # touched: strict UTF-8 JSONL, every physical line an object carrying
        # exactly the five existing fields of a listed tenant, each asserted
        # tenant present at least once (tenants may interleave), seqs running
        # expected_count+1, expected_count+2, ... from a first prev equal to
        # the asserted hash and every digest recomputed. Parse failures,
        # missing/extra fields, duplicate keys, non-standard numbers, blank
        # lines, non-objects, illegal UTF-8 or a listed tenant without
        # records are missing; sequence defects are sequence; prev/hash
        # mismatches are digest -- all raised as AuditChainStateError located
        # by tenant, seq and the input's physical line number (tenant/seq
        # None when they cannot be determined), the first defective physical
        # line deciding; input problems take priority over every target-side
        # state. A record of a canonical tenant identity no assertion lists
        # violates the input contract and is a ValueError, also decided by
        # first physical line against the chain defects above. Nothing is
        # read from the target and nothing is written on any input failure.
        lines, bad_line = self._decode_lines(data)
        try:
            records = self._scan_import_ranges(lines, bad_line, assertions)
        except _ForeignTenant as ft:
            raise ValueError(
                "import data must contain only tenants named by "
                f"expected_heads; a record of another tenant appears at "
                f"line {ft.line}"
            ) from None
        # Target scan, the head comparisons and the append share one
        # exclusive lease (the same one append_many_if_heads uses: the
        # data-file lease on an existing log, the directory lease guarding
        # the missing-file branch), so check-and-commit is a single atomic
        # operation. A not-yet-existing log may be created only when every
        # assertion is the exact empty-chain head (0, ZERO); any other
        # assertion is a deterministic conflict against the (0, ZERO) actual
        # head that leaves no file -- the lease derives that rule from
        # assertions before opening the path.
        norm = [
            (tenant, None, expected_count, expected_hash)
            for tenant, expected_count, expected_hash in assertions
        ]
        with self._heads_lease(norm) as f:
            # Only the chains this input involves are scanned. A corrupt
            # involved chain raises AuditChainStateError (same
            # tenant/seq/reason/line as append, first physical line then
            # expected_heads order deciding across chains) and takes strict
            # priority over the head comparisons; damage to an uninvolved
            # tenant is ignored. The state is never compared against an
            # unverifiable chain.
            target = f.read()
            t_lines, t_bad = self._decode_lines(target)
            heads = {}   # serialized tenant -> (count, head)
            broken = []  # (line, assertion_index, AuditChainStateError)
            for idx, (tenant, _ec, _eh) in enumerate(assertions):
                try:
                    count, head = self._scan(tenant, t_lines, t_bad)
                except _Broken as b:
                    seq = b.at if b.at is not None else b.expect
                    broken.append((
                        b.line, idx,
                        AuditChainStateError(tenant, seq, b.reason, b.line),
                    ))
                else:
                    heads[self._tenant_key(tenant)] = (count, head)
            if broken:
                raise min(broken, key=lambda x: (x[0], x[1]))[2] from None
            # Every involved chain is valid. Compare the assertions in
            # expected_heads order; the first mismatching one raises with its
            # expectation and the chain tail observed inside the lease. A
            # conflict writes no byte.
            for tenant, expected_count, expected_hash in assertions:
                count, head = heads[self._tenant_key(tenant)]
                if count != expected_count or head != expected_hash:
                    raise AuditChainConflictError(
                        tenant, expected_count, expected_hash, count, head
                    )
            # All heads confirmed: each segment's first prev names exactly
            # the matching target tail, so the graft links seamlessly. One
            # O_APPEND write commits the whole stream as a single block,
            # re-serialized verbatim with the exact canonical rule
            # append/export use (sort_keys=True, allow_nan=False); other
            # tenants' interleaved records already on disk stay in place and
            # shared-lease readers observe only the complete pre-commit or
            # post-commit state. The records are returned in the input's
            # physical order.
            chunks = [
                json.dumps(item, sort_keys=True, allow_nan=False) + "\n"
                for item in records
            ]
            prefix = b"" if (not target or target.endswith(b"\n")) else b"\n"
            f.write(prefix + "".join(chunks).encode("utf-8"))
            return records

    def import_tenant_range_chunks(self, tenant, chunks, expected_count,
                                   expected_hash):
        # Chunked segmented offline counterpart of export_tenant_range_chunks
        # for large histories: the caller streams a single-tenant chain
        # segment as an ordered iterable of bytes chunks instead of one
        # contiguous buffer, and the segment is grafted onto this log so the
        # target chain continues from the asserted head. Chunks may be cut at
        # any byte boundary -- mid-UTF-8-sequence, mid-line or mid-record --
        # and empty chunks are allowed; only their in-order concatenation is
        # the segment. Records keep their original tenant/seq/event/prev/hash
        # values verbatim -- nothing is renumbered or recomputed.
        #
        # Boundary first, exactly as import_tenant_range orders it: the
        # tenant crosses the standard-JSON boundary, expected_count must be a
        # non-negative plain int (bool rejected even though it subclasses
        # int; floats, strings and other types are too) and expected_hash must
        # be exactly the 64 lowercase hex characters of a sha256 digest (an
        # empty target chain is asserted with ZERO) -- all three before a
        # single chunk is consumed. The chunk container follows the exact
        # boundary import_tenant_chunks uses: a bare bytes or bytearray
        # object is not a chunk container (its iteration would yield ints), a
        # non-iterable container, or a non-bytes element are all ValueError,
        # a bad element ending consumption at once -- all before any lease is
        # taken, any history is read or the path is probed. The complete
        # input is joined and verified before the target is touched, so
        # field-for-field this accepts exactly the complete contents
        # import_tenant_range accepts and raises exactly what
        # import_tenant_range raises: chain defects are
        # AuditChainStateError (reason missing/sequence/digest, with tenant,
        # seq and the input's physical line), a record of another canonical
        # tenant identity is ValueError, a mismatching target head is
        # AuditChainConflictError with the observed tail, and input problems
        # take strict priority over target state. An empty concatenation is a
        # no-op after validation returning [], creating and reading nothing.
        # On success the whole segment lands as one indivisible commit and
        # the parsed records are returned in physical (== seq) order,
        # identical objects to what import_tenant_range returns for the same
        # bytes; no failure path writes a single partial byte, and
        # shared-lease readers observe only the complete pre-commit or
        # post-commit state.
        _validate_json_value(tenant)
        _validate_required_count(expected_count)
        _validate_expected_hash(expected_hash)
        data = self._join_chunks(chunks)
        return self.import_tenant_range(
            tenant, data, expected_count, expected_hash
        )

    def import_all_range_chunks(self, chunks, expected_heads):
        # Chunked multi-tenant segmented offline migration: the large-history
        # counterpart of import_all_range. The caller streams several
        # tenants' interleaved JSONL segments (typically produced by
        # export_tenant_range_chunks calls against a source log) as an
        # ordered iterable of bytes chunks, and the whole stream is grafted
        # onto this log as one indivisible block, each tenant continuing from
        # its own head named in expected_heads. Chunks may be cut at any byte
        # boundary -- mid-UTF-8-sequence, mid-line or mid-record -- and empty
        # chunks are allowed; only their in-order concatenation is the
        # stream. Records keep their original values verbatim.
        #
        # Boundary first, exactly as import_all_range orders it:
        # expected_heads must satisfy the exact boundary import_all_range
        # documents (a list of objects carrying exactly the keys tenant,
        # expected_count and expected_hash; standard-JSON tenant;
        # non-negative plain int expected_count, bool rejected; 64 lowercase
        # hex expected_hash; no duplicate canonical tenant identity) and
        # crosses before a single chunk is consumed. The chunk container
        # follows the exact boundary import_all_chunks uses: a bare bytes or
        # bytearray object is not a chunk container, a non-iterable container
        # or a non-bytes element are all ValueError, a bad element ending
        # consumption at once -- all before any lease is taken, any history
        # is read or the path is probed. The complete input is joined and
        # verified in memory before the target is touched, so field-for-field
        # this accepts exactly the complete contents import_all_range accepts
        # and raises exactly what import_all_range raises: malformed lines,
        # fields, sequences, digests and listed tenants without records are
        # AuditChainStateError located by tenant, seq and the input's
        # physical line, a record of a canonical tenant identity no assertion
        # lists is ValueError, a mismatching target head is
        # AuditChainConflictError in expected_heads order with the observed
        # tail, and input problems take strict priority over target state. An
        # empty concatenation is a no-op after validation returning [],
        # creating and reading nothing. On success the whole stream lands as
        # one indivisible block and the parsed records are returned in
        # physical order, identical objects to what import_all_range returns
        # for the same bytes; no failure path writes a single partial byte,
        # and shared-lease readers observe only the complete pre-commit or
        # post-commit state.
        _validate_expected_range_heads(expected_heads)
        data = self._join_chunks(chunks)
        return self.import_all_range(data, expected_heads)

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

    def head(self, tenant):
        # Read-only chain-head query: the exact (count, hash) pair a caller
        # can feed straight into append_if_head/append_batch_if_head as
        # (expected_count, expected_hash), without reading the log itself.
        # Returns {"tenant": tenant, "count": count, "hash": hash} with the
        # tenant value echoed back exactly as passed.
        #
        # The tenant crosses the same standard-JSON input boundary as
        # append/verify/export_tenant, checked before any byte is read, so an
        # illegal value raises ValueError without touching the log and takes
        # priority over any history state. A missing or empty log, or a log
        # without any record of this tenant, is a legitimate empty chain and
        # yields count=0, hash=ZERO; other tenants' records never count.
        _validate_json_value(tenant)
        # One shared-lease snapshot, exactly as verify/export/read use: the
        # result corresponds to the full state wholly before or wholly after
        # any concurrent append, never a torn read. Even though only the
        # chain tail is reported, the tenant's complete chain -- first record
        # through tail -- must verify with the exact JSON identity, sequence
        # and digest rules of verify: a corrupt prefix or suffix raises
        # AuditChainStateError (same tenant/seq/reason/line as
        # append/export_tenant, reason only missing/sequence/digest) instead
        # of surfacing a head computed over a broken history. The scan never
        # creates or modifies the log and writes no cache.
        data = self._read_snapshot()
        lines, bad_line = self._decode_lines(data)
        try:
            count, last = self._scan(tenant, lines, bad_line)
        except _Broken as b:
            seq = b.at if b.at is not None else b.expect
            raise AuditChainStateError(tenant, seq, b.reason, b.line) from None
        # _scan yields prev=ZERO for an empty chain, so the empty case falls
        # out of the same rule as a non-empty one. A head observed here can
        # still lose a later race; that is exactly what the conditional
        # append's conflict semantics are for.
        return {"tenant": tenant, "count": count, "hash": last}

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

    def heads_bytes(self, data):
        # Pure in-memory, offline chain-head directory: the offline
        # counterpart of heads(), validating a caller-supplied JSONL snapshot
        # exactly the way heads() validates its shared-lease file snapshot.
        # Like verify_all_bytes this consumes only the given memory: it never
        # reads, creates or modifies the path this AuditChain points at, never
        # touches the network and keeps no cache or on-disk index, and the
        # buffer itself is never mutated.
        #
        # data must be bytes exactly as verify_all_bytes requires: bytearray,
        # str and every other type are rejected with ValueError before any
        # parsing, so no underlying decode/parse exception is ever leaked. The
        # verification then reuses verify_all_bytes' exact strict UTF-8,
        # LF-only physical-line, standard-JSON (duplicate keys and
        # non-standard numbers rejected) rules and per-tenant chain rules from
        # seq=1, prev=ZERO, so success has the exact shape heads() returns --
        # {"ok": True, "tenants": [...]} with one entry per tenant in
        # first-appearance physical order carrying the original tenant value,
        # its verified count and its tail hash, the exact
        # (expected_count, expected_hash) pair an offline conditional append
        # needs; empty bytes are a successful empty history and yield
        # {"ok": True, "tenants": []}. The first defect is reported with
        # verify_all_bytes' exact failure object --
        # {"ok": False, "at": line, "tenant": ..., "reason": ...}, tenant None
        # when the line cannot name one -- reason missing for strict-decode/
        # parse/object/required-field failures, sequence for a seq that is not
        # that tenant's next one, digest for a prev/hash mismatch, and no
        # partial tenant list is ever returned.
        if not isinstance(data, bytes):
            raise ValueError(f"data must be bytes, got {type(data).__name__}")
        return self._verify_all_snapshot(data, with_hash=True)

    @staticmethod
    def _join_chunks(chunks):
        # Validate and concatenate a chunked byte history. chunks must be an
        # iterable whose every element is bytes; a bare bytes or bytearray
        # object is not a chunk container (its iteration would yield ints)
        # and raises ValueError up front, as does a non-iterable container.
        # Element types are checked lazily while consuming: the first
        # non-bytes element raises ValueError immediately, without pulling
        # any further element from the iterable. Empty chunks are allowed and
        # an empty iterable is the empty history. Chunks may be cut at any
        # byte boundary -- mid-UTF-8-sequence, mid-line or mid-record --
        # since only the concatenation is ever decoded.
        if isinstance(chunks, (bytes, bytearray)):
            raise ValueError(
                "chunks must be an iterable of bytes, got a single "
                f"{type(chunks).__name__} object"
            )
        try:
            iterator = iter(chunks)
        except TypeError:
            raise ValueError(
                "chunks must be an iterable of bytes, got "
                f"{type(chunks).__name__}"
            ) from None
        parts = []
        for chunk in iterator:
            if not isinstance(chunk, bytes):
                raise ValueError(
                    f"each chunk must be bytes, got {type(chunk).__name__}"
                )
            parts.append(chunk)
        return b"".join(parts)

    def verify_chunks(self, chunks, tenant, expected_count=None):
        # Chunked offline counterpart of verify_bytes for large histories:
        # the caller hands over the JSONL history as an iterable of bytes
        # chunks in file order instead of one contiguous buffer, so the
        # caller never has to assemble the full bytes itself. Chunks may be
        # cut at any byte boundary (mid-UTF-8-sequence, mid-line, mid-record)
        # and empty chunks are allowed; only their concatenation is the
        # history. Like verify_bytes this consumes only caller data: it never
        # reads, creates or modifies the configured path, never touches the
        # network, and keeps no cache or on-disk index.
        #
        # Boundary first, in the exact order verify_bytes uses: the chunk
        # container must be an iterable of bytes (a bare bytes or bytearray
        # object, a non-iterable container or a non-bytes element are all
        # ValueError, a bad element ending consumption at once), then the
        # tenant crosses the same standard-JSON boundary as append/verify,
        # then expected_count must be None or a non-negative plain int (bool
        # rejected even though it subclasses int). The verdict over the
        # concatenated bytes is field-for-field the verdict verify_bytes
        # would return for the same bytes, including physical line numbers,
        # first-error priority and the expected_count check.
        if isinstance(chunks, (bytes, bytearray)):
            raise ValueError(
                "chunks must be an iterable of bytes, got a single "
                f"{type(chunks).__name__} object"
            )
        _validate_json_value(tenant)
        _validate_expected_count(expected_count)
        return self._verify_snapshot(
            tenant, self._join_chunks(chunks), expected_count
        )

    def verify_all_chunks(self, chunks):
        # Chunked offline counterpart of verify_all_bytes: every tenant chain
        # in the history is validated over the concatenation of the given
        # bytes chunks, supplied in file order. Chunks may be cut at any byte
        # boundary and empty chunks are allowed; an empty iterable is a
        # legitimate successful empty history. The result -- ok, at, tenant,
        # reason on failure, or the tenant list with counts and
        # first-appearance order on success -- is field-for-field the verdict
        # verify_all_bytes would return for the concatenated bytes. Like
        # verify_all_bytes this consumes only caller data: it never reads,
        # creates or modifies the configured path, never touches the network,
        # and keeps no cache or on-disk index. The container must be an
        # iterable of bytes; a bare bytes or bytearray object, a non-iterable
        # container or a non-bytes element are all ValueError, a bad element
        # ending consumption at once.
        return self._verify_all_snapshot(self._join_chunks(chunks))

    def heads_chunks(self, chunks):
        # Chunked offline counterpart of heads_bytes: the caller supplies the
        # JSONL snapshot whose chain heads are wanted as an ordered iterable of
        # bytes chunks in file order instead of one contiguous buffer. Chunks
        # may be cut at any byte boundary -- mid-UTF-8-sequence, mid-JSON,
        # mid-line or mid-record -- and empty chunks are allowed; an empty
        # iterable, or one holding only empty chunks, is the empty history.
        # Only the in-order concatenation is ever decoded, so a split across a
        # UTF-8 sequence, a JSON object or an LF changes no semantics: the
        # result is field-for-field exactly what heads_bytes returns for the
        # same concatenated bytes, and both are field-for-field what heads()
        # returns for identical file contents. Like heads_bytes this consumes
        # only caller data: it never reads, creates or modifies the configured
        # path, never touches the network and keeps no cache or on-disk index.
        # The container boundary is exactly verify_all_chunks' boundary: a
        # bare bytes or bytearray object is not a chunk container, a
        # non-iterable container or a non-bytes (bytearray included) element
        # are all ValueError, the first bad element ending consumption at
        # once. Success carries each tenant's original value, verified count
        # and tail hash -- the (expected_count, expected_hash) pair an offline
        # conditional append asserts -- in first-appearance physical order;
        # the first defect is reported with verify_all_bytes' exact failure
        # object and no partial tenant list is returned.
        return self._verify_all_snapshot(
            self._join_chunks(chunks), with_hash=True
        )

    def _verify_heads_snapshot(self, data, assertions):
        # Core of verify_heads over an exact in-memory snapshot. Touches no
        # path: the caller owns how the bytes were obtained (a shared-lease
        # file read or a caller-supplied buffer), so the same logic backs
        # verify_heads, verify_heads_bytes and the chunked entry point.
        #
        # The complete snapshot is validated first, exactly the way
        # heads_bytes/verify_all_bytes validate theirs (strict UTF-8, LF-only
        # physical lines, standard JSON, per-tenant seq/prev/hash from
        # seq=1, prev=ZERO), and the first corrupt point is returned as that
        # exact failure object -- {"ok": False, "at", "tenant", "reason"} --
        # with no head comparison attempted. Only a fully self-consistent
        # snapshot reaches the directory comparison.
        result = self._verify_all_snapshot(data, with_hash=True)
        if not result["ok"]:
            return result
        actual = {}   # serialized tenant -> [count, hash]
        order = []    # tenants present in the snapshot, first-appearance
        for entry in result["tenants"]:
            tenant = entry["tenant"]
            key = self._tenant_key(tenant)
            actual[key] = [entry["count"], entry["hash"]]
            order.append(tenant)
        # First compare the expectations in expected_heads order, so list
        # order alone decides the conflict when several tenants disagree: a
        # tenant missing from the directory has the fixed empty head
        # (0, ZERO), exactly the head()/heads() empty-chain convention.
        for tenant, expected_count, expected_hash in assertions:
            state = actual.get(self._tenant_key(tenant))
            if state is None:
                count, head = 0, ZERO
            else:
                count, head = state
            if count != expected_count or head != expected_hash:
                raise AuditChainConflictError(
                    tenant, expected_count, expected_hash, count, head
                )
        # Every listed head matched. A tenant present in the directory but
        # absent from the expectation list is an unlisted head: it is treated
        # as an expected empty chain (0, ZERO), the first such tenant in the
        # snapshot's first-appearance order raising, so a backup missing a
        # tenant can never verify as complete.
        expected_keys = {self._tenant_key(t) for t, _c, _h in assertions}
        for tenant in order:
            if self._tenant_key(tenant) not in expected_keys:
                count, head = actual[self._tenant_key(tenant)]
                raise AuditChainConflictError(tenant, 0, ZERO, count, head)
        # Directory and expectation agree exactly. Return heads_bytes'
        # success structure (verified counts and tail hashes, first-
        # appearance order, original tenant values) so a caller gets the
        # confirmed directory back.
        return result

    def verify_heads(self, expected_heads):
        # Read-only expected-head directory check: prove that the tenant-head
        # directory held by this log agrees with a caller-supplied complete
        # history expectation, not merely that the snapshot is internally
        # self-consistent -- the extra assurance verify_all cannot give for a
        # service migration, restore or offline backup reconciliation. Takes
        # one shared-lease file snapshot (the same view verify_all/heads use)
        # and never creates, modifies or caches a file.
        #
        # The whole expected_heads boundary crosses before any byte is read:
        # it must be a list of objects carrying exactly the keys tenant,
        # count and hash, the tenant crosses the standard-JSON boundary,
        # count is a non-negative plain int (bool, negative, float and other
        # types are ValueError), hash exactly 64 lowercase hex characters,
        # and a canonical JSON tenant identity may appear at most once. A
        # malformed call raises ValueError without reading history or probing
        # the path, under any contention, and takes priority over a corrupt
        # snapshot. List order is the conflict-priority order.
        #
        # The snapshot is fully validated with the exact strict UTF-8, LF
        # physical-line, standard-JSON and missing/sequence/digest rules of
        # verify_all/heads; the first defect returns their exact failure
        # object ({"ok": False, "at", "tenant", "reason"}) with no head
        # comparison. Only afterwards are heads compared in expected_heads
        # order: a tenant missing from the directory is the fixed empty head
        # (0, ZERO); a tenant present in the directory but absent from the
        # expectation is an expected (0, ZERO) in the snapshot's
        # first-appearance order. The first unequal head raises
        # AuditChainConflictError with reason fixed "conflict" and
        # tenant/expected_count/expected_hash/actual_count/actual_hash
        # filled; a read-only check never writes a byte. Full agreement
        # returns heads_bytes' success structure
        # {"ok": True, "tenants": [...]} in the snapshot's first-appearance
        # order carrying the original tenant, count and tail hash; an empty
        # snapshot matching an empty expectation yields the empty directory.
        # No network or remote anchor is involved.
        assertions = _validate_expected_heads(expected_heads)
        return self._verify_heads_snapshot(self._read_snapshot(), assertions)

    def verify_heads_bytes(self, data, expected_heads):
        # Pure in-memory, offline expected-head directory check: the offline
        # counterpart of verify_heads over a caller-supplied snapshot. Like
        # heads_bytes/verify_all_bytes it consumes only the given memory: it
        # never reads, creates or modifies the path this AuditChain points
        # at, never touches the network, keeps no cache or on-disk index and
        # never mutates the buffer.
        #
        # Boundary first, before any parsing: data must be exactly bytes
        # (bytearray, str and every other type are ValueError, no underlying
        # decode/parse exception leaked) and expected_heads must satisfy the
        # exact boundary verify_heads documents (list of {tenant, count,
        # hash} objects; standard-JSON tenant; non-negative plain int count,
        # bool rejected; 64 lowercase hex hash; no duplicate canonical
        # tenant). Both raise before a snapshot byte is read or parsed, so a
        # malformed call ends as ValueError regardless of a corrupt buffer.
        # The snapshot then validates with heads_bytes' exact strict rules
        # and the first defect returns verify_all_bytes' exact failure
        # object, with no head comparison. On a self-consistent snapshot the
        # heads are compared exactly as verify_heads compares them --
        # expected_heads order first, missing directory tenants fixed at
        # (0, ZERO), unlisted directory tenants expected (0, ZERO) in
        # first-appearance order -- and the first inequality raises
        # AuditChainConflictError (reason "conflict", the five head fields
        # filled). Full agreement returns the heads_bytes success structure;
        # empty bytes matching an empty expectation give
        # {"ok": True, "tenants": []}.
        if not isinstance(data, bytes):
            raise ValueError(f"data must be bytes, got {type(data).__name__}")
        assertions = _validate_expected_heads(expected_heads)
        return self._verify_heads_snapshot(data, assertions)

    def verify_heads_chunks(self, chunks, expected_heads):
        # Chunked offline counterpart of verify_heads_bytes: the caller
        # supplies the JSONL snapshot as an ordered iterable of bytes chunks
        # instead of one contiguous buffer. Chunks may be cut at any byte
        # boundary -- mid-UTF-8-sequence, mid-JSON, mid-line or mid-record --
        # and empty chunks are allowed; an empty iterable, or one holding
        # only empty chunks, is the empty history. Only the in-order
        # concatenation is ever decoded, so the result is field-for-field
        # exactly what verify_heads_bytes returns for the same concatenated
        # bytes, and both are what verify_heads returns for identical file
        # contents. Like the bytes entry this consumes only caller data: it
        # never reads, creates or modifies the configured path, never
        # touches the network and keeps no cache or on-disk index.
        #
        # The expected_heads boundary is exactly verify_heads' boundary and
        # crosses before the chunk container is consumed, so a malformed
        # expectation never pulls a chunk. The container boundary is exactly
        # verify_all_chunks'/heads_chunks': a bare bytes or bytearray object
        # is not a chunk container, a non-iterable container or a non-bytes
        # (bytearray included) element are all ValueError, the first bad
        # element ending consumption at once. After joining, the snapshot
        # validates and the heads compare exactly as
        # verify_heads_bytes/verify_heads do: a corrupt snapshot returns
        # verify_all_bytes' failure object before any comparison, the first
        # unequal head raises AuditChainConflictError (reason "conflict"),
        # and full agreement returns the heads_bytes success structure.
        assertions = _validate_expected_heads(expected_heads)
        return self._verify_heads_snapshot(
            self._join_chunks(chunks), assertions
        )
