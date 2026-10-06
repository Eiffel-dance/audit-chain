import hashlib
import json
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app import (
    AuditChain,
    AuditChainStateError,
    MANIFEST_VERSION,
    ZERO,
)


def record_bytes(item):
    return (json.dumps(item, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def empty_manifest():
    return {
        "version": MANIFEST_VERSION,
        "byte_length": 0,
        "byte_sha256": hashlib.sha256(b"").hexdigest(),
        "tenants": [],
    }


def make_record(tenant, seq, event, prev):
    item = {"tenant": tenant, "seq": seq, "event": event, "prev": prev}
    item["hash"] = AuditChain._hash(item)
    return item


def chain_records(*specs):
    # specs: (tenant, event) pairs in physical order; returns the record
    # dicts with correct per-tenant seq/prev/hash linkage.
    records = []
    tails = {}
    for tenant, event in specs:
        key = json.dumps(tenant, sort_keys=True, separators=(",", ":"))
        seq, prev = tails.get(key, (1, ZERO))
        item = make_record(tenant, seq, event, prev)
        records.append(item)
        tails[key] = (seq + 1, item["hash"])
    return records


class CompactTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.path = Path(self.tmp.name) / "audit.jsonl"
        self.chain = AuditChain(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def snapshot(self):
        return self.path.read_bytes()

    def seed(self):
        self.chain.append("a", {"i": 1})
        self.chain.append("b", {"i": 1})
        self.chain.append("a", {"i": 2})
        self.chain.append("a", {"i": 3})
        self.chain.append("b", {"i": 2})

    # --- empty history ---

    def test_missing_path_yields_empty_manifest_and_creates_nothing(self):
        m = self.chain.compact()
        self.assertEqual(m, empty_manifest())
        self.assertFalse(self.path.exists())

    def test_empty_file_yields_empty_manifest_and_stays_empty(self):
        self.path.write_bytes(b"")
        m = self.chain.compact()
        self.assertEqual(m, empty_manifest())
        self.assertEqual(self.snapshot(), b"")

    # --- canonical input is a byte-identical no-op ---

    def test_canonical_log_is_unchanged(self):
        self.seed()
        before = self.snapshot()
        m = self.chain.compact()
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(m, self.chain.manifest())
        self.assertEqual(m["byte_length"], len(before))
        self.assertEqual(m["byte_sha256"], hashlib.sha256(before).hexdigest())
        self.assertEqual(
            [(t["tenant"], t["count"]) for t in m["tenants"]],
            [("a", 3), ("b", 2)])

    def test_repeated_compact_is_idempotent(self):
        self.seed()
        m1 = self.chain.compact()
        data1 = self.snapshot()
        m2 = self.chain.compact()
        self.assertEqual(m1, m2)
        self.assertEqual(self.snapshot(), data1)
        # Record digests, tenant heads and the tenant directory are stable.
        self.assertEqual(self.chain.heads()["tenants"], [
            {"tenant": t["tenant"], "count": t["count"], "hash": t["hash"]}
            for t in m1["tenants"]
        ])

    # --- redundancy is removed, meaning preserved ---

    def test_crlf_whitespace_key_order_and_final_lf_normalized(self):
        records = chain_records(
            ("a", {"x": 1}), ("b", [1, 2]), ("a", "héllo→世界"))
        raw = (
            b'{ "tenant": "a", "seq": 1, "event": {"x": 1}, '
            + f'"prev": "{ZERO}", "hash": "{records[0]["hash"]}"'.encode()
            + b" }\r\n"
            # unsorted keys, no spaces
            + b'{"hash":"%s","event":[1,2],"prev":"%s","seq":1,"tenant":"b"}\n'
            % (records[1]["hash"].encode(), ZERO.encode())
            # raw UTF-8, no trailing LF on the last line
            + ('{"tenant":"a","seq":2,"event":"héllo→世界","prev":"%s","hash":"%s"}'
               % (records[0]["hash"], records[2]["hash"])).encode("utf-8")
        )
        self.path.write_bytes(raw)
        m = self.chain.compact()
        expected = b"".join(record_bytes(r) for r in records)
        self.assertEqual(self.snapshot(), expected)
        # The manifest describes the new bytes; tenant directory unchanged.
        self.assertEqual(m["byte_length"], len(expected))
        self.assertEqual(m["byte_sha256"],
                         hashlib.sha256(expected).hexdigest())
        self.assertEqual(
            m["tenants"],
            [{"tenant": "a", "count": 2, "hash": records[2]["hash"]},
             {"tenant": "b", "count": 1, "hash": records[1]["hash"]}])
        # All existing entries validate the compacted log.
        self.assertEqual(self.chain.verify_all(),
                         {"ok": True, "tenants": [
                             {"tenant": "a", "count": 2},
                             {"tenant": "b", "count": 1}]})
        self.assertEqual(self.chain.export_all(), expected)
        self.assertEqual(self.chain.manifest(), m)
        self.assertEqual(self.chain.verify_manifest(m),
                         {"ok": True, "manifest": m})

    def test_physical_interleaving_and_event_values_preserved(self):
        records = chain_records(
            ({"z": 1}, 0), ("a", {"nested": [1, {"k": None}, True]}),
            ([1, 2], "evt"), ({"z": 1}, 1.5), ("a", -0.0), (None, [1e0]))
        self.path.write_bytes(b"".join(
            (json.dumps(r) + "\n").encode("utf-8") for r in records))
        m = self.chain.compact()
        lines = self.snapshot().decode("utf-8").split("\n")[:-1]
        parsed = [json.loads(line) for line in lines]
        # Five fields, same values, same physical order.
        for got, want in zip(parsed, records):
            self.assertEqual(set(got), {"tenant", "seq", "event", "prev",
                                        "hash"})
            self.assertEqual(got, want)
        self.assertEqual([t["tenant"] for t in m["tenants"]],
                         [{"z": 1}, "a", [1, 2], None])
        # Number spellings normalize to the canonical JSON encoding of the
        # same value (1e0 -> 1.0, -0.0 -> -0.0).
        self.assertIn('"event": [1.0]', lines[-1])
        self.assertEqual(self.chain.verify_all()["ok"], True)

    def test_distinct_json_identities_stay_partitioned(self):
        records = chain_records((1, "x"), (1.0, "x"), (True, "x"), ("1", "x"))
        self.path.write_bytes(b"".join(record_bytes(r) for r in records))
        m = self.chain.compact()
        self.assertEqual(
            [(t["tenant"], t["count"]) for t in m["tenants"]],
            [(1, 1), (1.0, 1), (True, 1), ("1", 1)])
        self.assertEqual(self.snapshot(),
                         b"".join(record_bytes(r) for r in records))

    def test_chains_continue_after_compact(self):
        self.seed()
        heads_before = self.chain.heads()
        self.chain.compact()
        # Appends after compact link onto the unchanged tails.
        item = self.chain.append("a", {"i": 4})
        self.assertEqual(item["seq"], 4)
        self.assertEqual(
            item["prev"],
            next(t["hash"] for t in heads_before["tenants"]
                 if t["tenant"] == "a"))
        self.assertTrue(self.chain.verify_all()["ok"])
        self.assertEqual(self.chain.head("a")["count"], 4)

    # --- corruption: AuditChainStateError, original bytes untouched ---

    def assert_corrupt(self, raw, tenant, seq, reason, line):
        self.path.write_bytes(raw)
        with self.assertRaises(AuditChainStateError) as ctx:
            self.chain.compact()
        err = ctx.exception
        self.assertEqual(
            (err.tenant, err.seq, err.reason, err.line),
            (tenant, seq, reason, line))
        # No partial replacement: the original bytes are exactly preserved.
        self.assertEqual(self.snapshot(), raw)

    def test_unparseable_line_is_missing(self):
        self.assert_corrupt(b"{not json\n", None, None, "missing", 1)

    def test_blank_line_is_missing(self):
        self.assert_corrupt(b"\n", None, None, "missing", 1)

    def test_bad_utf8_is_missing(self):
        good = record_bytes(make_record("t", 1, {}, ZERO))
        self.assert_corrupt(good + b"\xff\n", None, None, "missing", 2)

    def test_missing_field_is_missing(self):
        row = {"tenant": "t", "seq": 1, "event": {}, "prev": ZERO}
        raw = (json.dumps(row) + "\n").encode()
        self.assert_corrupt(raw, "t", 1, "missing", 1)

    def test_extra_field_is_missing(self):
        row = make_record("t", 1, {}, ZERO)
        row["extra"] = 1
        raw = (json.dumps(row) + "\n").encode()
        self.assert_corrupt(raw, "t", 1, "missing", 1)

    def test_nonstandard_number_is_missing(self):
        self.assert_corrupt(b'{"tenant":"t","seq":1,"event":NaN,'
                            b'"prev":"' + ZERO.encode() + b'","hash":"x"}\n',
                            None, None, "missing", 1)

    def test_sequence_gap_is_sequence(self):
        r1 = make_record("t", 1, {}, ZERO)
        r3 = make_record("t", 3, {}, r1["hash"])
        raw = record_bytes(r1) + record_bytes(r3)
        self.assert_corrupt(raw, "t", 2, "sequence", 2)

    def test_float_seq_spelling_is_sequence(self):
        r1 = make_record("t", 1, {}, ZERO)
        raw = record_bytes(r1) + (
            b'{"tenant":"t","seq":2.0,"event":{},"prev":"'
            + r1["hash"].encode() + b'","hash":"' + b"0" * 64 + b'"}\n')
        self.assert_corrupt(raw, "t", 2, "sequence", 2)

    def test_bad_prev_is_digest(self):
        r1 = make_record("t", 1, {}, ZERO)
        r2 = make_record("t", 2, {}, "9" * 64)
        raw = record_bytes(r1) + record_bytes(r2)
        self.assert_corrupt(raw, "t", 2, "digest", 2)

    def test_bad_hash_is_digest(self):
        r1 = make_record("t", 1, {}, ZERO)
        r1["hash"] = "0" * 64
        self.assert_corrupt(record_bytes(r1), "t", 1, "digest", 1)

    def test_error_fields_match_export_all(self):
        r1 = make_record("a", 1, {}, ZERO)
        r2 = make_record("b", 1, {}, ZERO)
        bad = make_record("a", 9, {}, r1["hash"])
        raw = record_bytes(r1) + record_bytes(r2) + record_bytes(bad)
        self.path.write_bytes(raw)
        with self.assertRaises(AuditChainStateError) as ctx:
            self.chain.compact()
        compact_err = ctx.exception
        with self.assertRaises(AuditChainStateError) as ctx:
            self.chain.export_all()
        export_err = ctx.exception
        for attr in ("tenant", "seq", "reason", "line"):
            self.assertEqual(getattr(compact_err, attr),
                             getattr(export_err, attr), attr)
        self.assertEqual(self.snapshot(), raw)

    # --- concurrency: only complete pre/post snapshots are observable ---

    def test_concurrent_appends_and_compact_stay_consistent(self):
        self.seed()
        errors = []
        box = threading.Lock()
        stop = threading.Event()

        def reader():
            try:
                while not stop.is_set():
                    # Every observed snapshot must be a fully self-consistent
                    # pre/post-commit history, never a torn rewrite.
                    r = self.chain.verify_all()
                    if not r["ok"]:
                        with box:
                            errors.append(("verify_all", r))
                        return
                    h = self.chain.heads()
                    if not h["ok"]:
                        with box:
                            errors.append(("heads", h))
                        return
            except AuditChainStateError as e:  # noqa: BLE001
                with box:
                    errors.append(("state", e))

        def writer(i):
            try:
                for j in range(40):
                    self.chain.append("w", {"w": i, "j": j})
            except Exception as e:  # noqa: BLE001
                with box:
                    errors.append(("append", repr(e)))

        threads = [threading.Thread(target=reader) for _ in range(2)]
        threads += [threading.Thread(target=writer, args=(i,))
                    for i in range(4)]
        for t in threads:
            t.start()
        for _ in range(10):
            self.chain.compact()
        stop.set()
        for t in threads:
            t.join()

        self.assertEqual(errors, [])
        self.assertTrue(self.chain.verify_all()["ok"])
        self.assertEqual(self.chain.head("w")["count"], 160)
        # A final compact is a byte-identical no-op on the canonical log.
        before = self.snapshot()
        m = self.chain.compact()
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(m, self.chain.manifest())

    # --- no side effects beyond the data file ---

    def test_no_sidecar_files_are_created(self):
        self.seed()
        self.chain.compact()
        self.assertEqual(sorted(p.name for p in self.path.parent.iterdir()),
                         ["audit.jsonl"])

    def test_existing_entries_behave_unchanged_after_compact(self):
        self.seed()
        m = self.chain.compact()
        data = self.snapshot()
        other = AuditChain(self.path.with_name("other.jsonl"))
        self.assertEqual(other.verify_all_bytes(data),
                         {"ok": True, "tenants": [
                             {"tenant": "a", "count": 3},
                             {"tenant": "b", "count": 2}]})
        self.assertEqual(other.manifest_bytes(data), m)
        self.assertEqual(self.chain.read_tenant("a"),
                         [json.loads(line) for line in
                          data.decode().splitlines()
                          if json.loads(line)["tenant"] == "a"])
        self.assertEqual(
            self.chain.export_tenant("a"),
            b"".join(record_bytes(r) for r in chain_records(
                ("a", {"i": 1}), ("a", {"i": 2}), ("a", {"i": 3}))))


if __name__ == "__main__":
    unittest.main()
