import hashlib, json
from pathlib import Path

_ZERO = "0" * 64
_FIELDS = ("tenant", "seq", "event", "prev", "hash")


class AuditChainStateError(Exception):
    """Raised by append when the stored history for a tenant is invalid.

    Carries the tenant and the first affected tenant sequence number.
    """

    def __init__(self, tenant, seq, reason):
        self.tenant = tenant
        self.seq = seq
        self.reason = reason
        super().__init__(
            "invalid audit history for tenant %r at seq %r: %s" % (tenant, seq, reason)
        )


class AuditChain:
    def __init__(self, path):
        self.path = Path(path)

    def _lines(self):
        if not self.path.exists():
            return []
        return self.path.read_text(encoding="utf-8").splitlines()

    @staticmethod
    def _hash(item):
        payload = {k: item[k] for k in ("tenant", "seq", "event", "prev")}
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def _scan(self, tenant):
        """Yield ("record", item, line_no) for the tenant's records in file
        order, or ("error", line_no) for lines that cannot be attributed to
        any tenant. Records of other tenants are skipped."""
        for line_no, raw in enumerate(self._lines(), 1):
            try:
                item = json.loads(raw)
            except ValueError:
                yield ("error", line_no)
                continue
            if not isinstance(item, dict) or "tenant" not in item:
                yield ("error", line_no)
                continue
            if item["tenant"] == tenant:
                yield ("record", item, line_no)

    def _history(self, tenant):
        """Return the tenant's validated records, raising
        AuditChainStateError on the first inconsistency."""
        records = []
        prev = _ZERO
        for kind, *payload in self._scan(tenant):
            expected = len(records) + 1
            if kind == "error":
                raise AuditChainStateError(
                    tenant, expected, "unreadable line %d" % payload[0]
                )
            item = payload[0]
            if any(k not in item for k in _FIELDS):
                raise AuditChainStateError(tenant, expected, "missing field")
            if item["seq"] != expected:
                raise AuditChainStateError(tenant, expected, "broken sequence")
            if item["prev"] != prev:
                raise AuditChainStateError(tenant, expected, "broken prev link")
            if item["hash"] != self._hash(item):
                raise AuditChainStateError(tenant, expected, "digest mismatch")
            records.append(item)
            prev = item["hash"]
        return records

    def append(self, tenant, event):
        own = self._history(tenant)
        item = {
            "tenant": tenant,
            "seq": len(own) + 1,
            "event": event,
            "prev": own[-1]["hash"] if own else _ZERO,
        }
        item["hash"] = self._hash(item)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(item, sort_keys=True) + "\n")
        return item

    def verify(self, tenant, expected_count=None):
        expected = 1
        prev = _ZERO
        for kind, *payload in self._scan(tenant):
            if kind == "error":
                return {"ok": False, "at": payload[0], "reason": "missing"}
            item, line_no = payload
            if expected_count is not None and expected > expected_count:
                return {"ok": False, "at": expected, "reason": "sequence"}
            if any(k not in item for k in _FIELDS):
                at = item["seq"] if isinstance(item.get("seq"), int) else line_no
                return {"ok": False, "at": at, "reason": "missing"}
            if item["seq"] != expected:
                return {"ok": False, "at": expected, "reason": "sequence"}
            if item["prev"] != prev or item["hash"] != self._hash(item):
                return {"ok": False, "at": expected, "reason": "digest"}
            expected += 1
            prev = item["hash"]
        count = expected - 1
        if expected_count is not None and count < expected_count:
            return {"ok": False, "at": count + 1, "reason": "missing"}
        return {"ok": True, "count": count}
