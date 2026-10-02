import hashlib, json
from pathlib import Path

ZERO = "0" * 64
FIELDS = ("tenant", "seq", "event", "prev", "hash")


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

    def _read(self):
        # Returns (text, bad_line). text is the decodable UTF-8 prefix of the
        # file; bad_line is the 1-based physical line of the first byte that
        # is not valid UTF-8, or None when the whole file decodes. The line
        # is the number of LF bytes before that byte plus one, so it stays
        # stable even when the bad byte sits on the last line or at EOF.
        if not self.path.exists():
            return "", None
        data = self.path.read_bytes()
        try:
            return data.decode("utf-8"), None
        except UnicodeDecodeError as e:
            return (data[:e.start].decode("utf-8"),
                    data.count(b"\n", 0, e.start) + 1)

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

    def _scan(self, tenant, text, bad_line=None):
        # Validate the target tenant's chain over the whole UTF-8 JSONL file.
        # Returns (verified_count, last_hash). Raises _Broken at the first
        # problem; records of other tenants may interleave and are skipped.
        # Tenant matching uses the same canonical JSON identity as
        # verify_all, never host-language loose equality. bad_line, when
        # given, is the physical line where undecodable bytes begin: any
        # error found earlier in text still wins, otherwise the decode
        # failure itself is reported as missing at that line.
        key = self._tenant_key(tenant)
        expected, prev, count = 1, ZERO, 0
        for line, raw in enumerate(text.splitlines(), 1):
            try:
                item = json.loads(raw)
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
        text, bad_line = self._read()
        try:
            count, prev = self._scan(tenant, text, bad_line)
        except _Broken as b:
            seq = b.at if b.at is not None else b.expect
            raise AuditChainStateError(tenant, seq, b.reason, b.line) from None
        item = {"tenant": tenant, "seq": count + 1, "event": event, "prev": prev}
        item["hash"] = self._hash(item)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        prefix = "" if not text or text.endswith("\n") else "\n"
        with self.path.open("a", encoding="utf-8", newline="") as f:
            f.write(prefix + json.dumps(item, sort_keys=True) + "\n")
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
        text, bad_line = self._read()
        for line, raw in enumerate(text.splitlines(), 1):
            try:
                item = json.loads(raw)
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
            # Undecodable bytes with no earlier problem: the tenant at that
            # physical line cannot be determined, so report None.
            return {"ok": False, "at": bad_line, "tenant": None, "reason": "missing"}
        return {"ok": True,
                "tenants": [{"tenant": t, "count": s[2]} for t, s in order]}

    def verify(self, tenant, expected_count=None):
        text, bad_line = self._read()
        try:
            count, _ = self._scan(tenant, text, bad_line)
        except _Broken as b:
            return {"ok": False, "at": b.at if b.at is not None else b.line, "reason": b.reason}
        if expected_count is not None:
            if count < expected_count:
                return {"ok": False, "at": count + 1, "reason": "missing"}
            if count > expected_count:
                return {"ok": False, "at": expected_count + 1, "reason": "sequence"}
        return {"ok": True, "count": count}
