import hashlib, json, math
from pathlib import Path

ZERO = "0" * 64
FIELDS = ("tenant", "seq", "event", "prev", "hash")


def _reject_constant(name):
    # The bare tokens NaN, Infinity and -Infinity are JavaScript extensions,
    # not standard JSON, so a line containing one is unreadable corruption.
    raise ValueError(f"non-standard JSON token: {name}")


def _parse_finite_float(token):
    # Same conversion as the default decoder, but an overflowed literal such
    # as 1e309 (which parses to inf without ever hitting parse_constant) must
    # be rejected instead of silently becoming Infinity.
    value = float(token)
    if not math.isfinite(value):
        raise ValueError(f"non-finite JSON number: {token}")
    return value


def _reject_duplicate_keys(pairs):
    # Fires bottom-up for every object, so duplicate keys in nested objects
    # are caught too. A line with a repeated key has no unambiguous standard
    # JSON meaning and is treated as missing-class corruption.
    seen = set()
    for key, _ in pairs:
        if key in seen:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        seen.add(key)
    return dict(pairs)


_STRICT_DECODER = json.JSONDecoder(
    parse_constant=_reject_constant,
    parse_float=_parse_finite_float,
    object_pairs_hook=_reject_duplicate_keys,
)


def _loads_strict(raw):
    # Parse exactly one standard JSON value. Trailing data and syntactically
    # invalid content raise ValueError (JSONDecodeError is a ValueError
    # subclass), as do duplicate object keys and any spelling that would
    # decode to a non-finite number.
    return _STRICT_DECODER.decode(raw)


def _validate_json_value(value):
    # Input boundary for append: null, booleans, finite integers and floats,
    # strings, arrays and objects with string keys, recursively. NaN,
    # Infinity, -Infinity, non-string object keys, tuples, bytes, sets and
    # any other value without an unambiguous standard JSON encoding fail.
    stack = [value]
    while stack:
        v = stack.pop()
        if v is None or isinstance(v, (bool, int, str)):
            continue
        if isinstance(v, float):
            if not math.isfinite(v):
                raise ValueError(
                    "tenant and event must be standard JSON values: "
                    "non-finite number is not allowed")
        elif isinstance(v, list):
            stack.extend(v)
        elif isinstance(v, dict):
            for key, sub in v.items():
                if not isinstance(key, str):
                    raise ValueError(
                        "tenant and event must be standard JSON values: "
                        "object keys must be strings")
                stack.append(sub)
        else:
            raise ValueError(
                "tenant and event must be standard JSON values: "
                f"{type(v).__name__} cannot be encoded unambiguously")


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

    def _read_bytes(self):
        return self.path.read_bytes() if self.path.exists() else b""

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
                item = _loads_strict(raw)
            except ValueError:
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
        # Validate the new inputs against the standard JSON boundary before
        # touching the file at all: on a nonexistent, empty or non-empty file
        # an invalid append must never write a byte, and this ValueError takes
        # priority over any history scan or append action.
        _validate_json_value(tenant)
        _validate_json_value(event)
        data = self._read_bytes()
        lines, bad_line = self._decode_lines(data)
        try:
            count, prev = self._scan(tenant, lines, bad_line)
        except _Broken as b:
            seq = b.at if b.at is not None else b.expect
            raise AuditChainStateError(tenant, seq, b.reason, b.line) from None
        item = {"tenant": tenant, "seq": count + 1, "event": event, "prev": prev}
        item["hash"] = self._hash(item)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        prefix = b"" if not data or data.endswith(b"\n") else b"\n"
        with self.path.open("ab") as f:
            f.write(prefix + (json.dumps(item, sort_keys=True) + "\n").encode("utf-8"))
        return item

    def verify_all(self):
        # Validate every tenant chain in one read-only pass over the file, in
        # physical line order. Each tenant gets an independent expected seq and
        # prev digest starting at (1, ZERO) on first appearance; records of
        # different tenants may interleave. The serialized tenant is only an
        # internal key so distinct types (1 vs "1") stay separate chains while
        # the original value is reported back unchanged.
        states = {}  # serialized tenant -> [expected_seq, prev_hash, count]
        order = []   # (tenant_value, state) in first-appearance order
        lines, bad_line = self._decode_lines(self._read_bytes())
        for line, raw in enumerate(lines, 1):
            try:
                item = _loads_strict(raw)
            except ValueError:
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

    def verify(self, tenant, expected_count=None):
        lines, bad_line = self._decode_lines(self._read_bytes())
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
