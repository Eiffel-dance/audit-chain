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
        return self.path.read_text(encoding="utf-8") if self.path.exists() else ""

    @staticmethod
    def _hash(item):
        payload = json.dumps(
            {k: item[k] for k in ("tenant", "seq", "event", "prev")},
            sort_keys=True, separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _scan(self, tenant, text):
        # Validate the target tenant's chain over the whole UTF-8 JSONL file.
        # Returns (verified_count, last_hash). Raises _Broken at the first
        # problem; records of other tenants may interleave and are skipped.
        expected, prev, count = 1, ZERO, 0
        for line, raw in enumerate(text.splitlines(), 1):
            try:
                item = json.loads(raw)
            except Exception:
                raise _Broken("missing", None, line, expected)
            if not isinstance(item, dict) or "tenant" not in item:
                raise _Broken("missing", None, line, expected)
            if item["tenant"] != tenant:
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
        return count, prev

    def append(self, tenant, event):
        text = self._read()
        try:
            count, prev = self._scan(tenant, text)
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

    def verify(self, tenant, expected_count=None):
        try:
            count, _ = self._scan(tenant, self._read())
        except _Broken as b:
            return {"ok": False, "at": b.at if b.at is not None else b.line, "reason": b.reason}
        if expected_count is not None:
            if count < expected_count:
                return {"ok": False, "at": count + 1, "reason": "missing"}
            if count > expected_count:
                return {"ok": False, "at": expected_count + 1, "reason": "sequence"}
        return {"ok": True, "count": count}
