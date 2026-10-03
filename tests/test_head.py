import json
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app import AuditChain, AuditChainStateError

ZERO = "0" * 64


class HeadTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.path = Path(self.tmp.name) / "audit.jsonl"
        self.chain = AuditChain(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def write_rows(self, rows):
        with self.path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(r if isinstance(r, str) else json.dumps(r, sort_keys=True))
                f.write("\n")

    # --- success shape ---

    def test_missing_file_is_zero_head_and_creates_nothing(self):
        self.assertFalse(self.path.exists())
        self.assertEqual(self.chain.head("t"),
                         {"tenant": "t", "count": 0, "hash": ZERO})
        self.assertFalse(self.path.exists())

    def test_empty_file_is_zero_head(self):
        self.path.write_bytes(b"")
        self.assertEqual(self.chain.head("t"),
                         {"tenant": "t", "count": 0, "hash": ZERO})

    def test_unknown_tenant_is_zero_head(self):
        self.chain.append("a", {})
        self.assertEqual(self.chain.head("zzz"),
                         {"tenant": "zzz", "count": 0, "hash": ZERO})

    def test_head_reports_verified_count_and_tail_hash(self):
        items = [self.chain.append("t", {"i": i}) for i in range(3)]
        h = self.chain.head("t")
        self.assertEqual(h, {"tenant": "t", "count": 3, "hash": items[-1]["hash"]})
        # the echoed tenant is the value that was passed in
        self.assertIsInstance(h["tenant"], str)

    def test_head_echoes_passed_tenant_verbatim(self):
        tenant = {"b": 2, "a": [1, {"x": True}]}
        self.chain.append(tenant, {})
        h = self.chain.head({"a": [1, {"x": True}], "b": 2})
        self.assertEqual(h["tenant"], {"a": [1, {"x": True}], "b": 2})
        self.assertEqual(h["count"], 1)

    def test_interleaved_tenants_do_not_count(self):
        a1 = self.chain.append("a", {})
        self.chain.append("b", {})
        a2 = self.chain.append("a", {})
        self.chain.append("b", {})
        self.assertEqual(self.chain.head("a"),
                         {"tenant": "a", "count": 2, "hash": a2["hash"]})
        self.assertEqual(self.chain.head("a")["hash"], a2["hash"])
        self.assertNotEqual(a1["hash"], a2["hash"])

    def test_distinct_json_identities_have_distinct_heads(self):
        for t in (1, 1.0, True, "1"):
            self.chain.append(t, {})
            self.chain.append(t, {})
        for t in (1, 1.0, True, "1"):
            h = self.chain.head(t)
            self.assertEqual(h["count"], 2)
            self.assertEqual(h["tenant"], t)
        hashes = {self.chain.head(t)["hash"] for t in (1, 1.0, True, "1")}
        self.assertEqual(len(hashes), 4)

    def test_head_feeds_append_if_head_and_batch(self):
        self.chain.append("t", {"i": 1})
        h = self.chain.head("t")
        item = self.chain.append_if_head("t", {"i": 2}, h["count"], h["hash"])
        self.assertEqual(item["seq"], 2)
        h = self.chain.head("t")
        batch = self.chain.append_batch_if_head(
            "t", [{"i": 3}, {"i": 4}], h["count"], h["hash"])
        self.assertEqual([r["seq"] for r in batch], [3, 4])
        self.assertEqual(self.chain.head("t"),
                         {"tenant": "t", "count": 4, "hash": batch[-1]["hash"]})

    def test_zero_head_feeds_conditional_append_on_missing_log(self):
        h = self.chain.head("t")
        item = self.chain.append_if_head("t", {}, h["count"], h["hash"])
        self.assertEqual((item["seq"], item["prev"]), (1, ZERO))

    # --- read-only ---

    def test_head_creates_or_modifies_nothing(self):
        self.chain.append("t", {})
        self.chain.append("u", {})
        before = self.path.read_bytes()
        entries = sorted(p.name for p in Path(self.tmp.name).iterdir())
        self.chain.head("t")
        self.chain.head("u")
        self.chain.head("zzz")
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(sorted(p.name for p in Path(self.tmp.name).iterdir()),
                         entries)

    # --- tenant input boundary: ValueError before any read ---

    def test_bad_tenant_raises_value_error_before_reading(self):
        bad_tenants = [
            float("nan"), float("inf"), float("-inf"),
            {"k": float("nan")}, [1, [float("-inf")]],
            {1: "x"}, {"a": {2: 3}}, object(), b"bytes", {1, 2}, ("a", 1),
        ]
        for bad in bad_tenants:
            with self.assertRaises(ValueError):
                self.chain.head(bad)
        self.assertFalse(self.path.exists())

    def test_cyclic_tenant_raises_value_error(self):
        cyc = []
        cyc.append(cyc)
        with self.assertRaises(ValueError):
            self.chain.head(cyc)
        d = {}
        d["self"] = d
        with self.assertRaises(ValueError):
            self.chain.head(d)

    def test_value_error_beats_corrupt_history_and_keeps_bytes(self):
        row = {"tenant": "t", "seq": 1, "event": {}, "prev": ZERO}
        row["hash"] = AuditChain._hash(row)
        row["event"] = {"tampered": True}
        self.write_rows([row])
        before = self.path.read_bytes()
        for bad in (float("nan"), {1: "x"}):
            with self.assertRaises(ValueError):
                self.chain.head(bad)
        self.assertEqual(self.path.read_bytes(), before)

    # --- state errors: identical location semantics to append/export ---

    def test_unparseable_line_raises_missing(self):
        other = {"tenant": "x", "seq": 1, "event": {}, "prev": ZERO}
        other["hash"] = AuditChain._hash(other)
        self.write_rows([other, "{not json"])
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.head("t")
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("t", 1, "missing", 2))

    def test_missing_field_raises_missing(self):
        self.write_rows([{"tenant": "t", "seq": 1, "event": {}, "prev": ZERO}])
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.head("t")
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("t", 1, "missing", 1))

    def test_sequence_gap_raises_sequence(self):
        good = {"tenant": "t", "seq": 1, "event": {}, "prev": ZERO}
        good["hash"] = AuditChain._hash(good)
        gap = {"tenant": "t", "seq": 3, "event": {}, "prev": good["hash"]}
        gap["hash"] = AuditChain._hash(gap)
        self.write_rows([good, gap])
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.head("t")
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("t", 2, "sequence", 2))

    def test_digest_mismatch_raises_digest(self):
        self.chain.append("t", {"i": 1})
        rows = [json.loads(l) for l in self.path.read_text().splitlines()]
        rows[0]["event"] = {"i": 99}
        self.write_rows(rows)
        before = self.path.read_bytes()
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.head("t")
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("t", 1, "digest", 1))
        self.assertEqual(self.path.read_bytes(), before)

    def test_corrupt_tail_not_ignored(self):
        # even though only the tail is reported, a broken later record fails
        items = [self.chain.append("t", {"i": i}) for i in range(2)]
        bad = {"tenant": "t", "seq": 3, "event": {}, "prev": items[-1]["hash"],
               "hash": "0" * 64}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(bad, sort_keys=True) + "\n")
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.head("t")
        self.assertEqual((cm.exception.seq, cm.exception.reason, cm.exception.line),
                         (3, "digest", 3))

    def test_illegal_utf8_raises_missing_with_physical_line(self):
        valid = {"tenant": "t", "seq": 1, "event": {}, "prev": ZERO}
        valid["hash"] = AuditChain._hash(valid)
        self.path.write_bytes(
            (json.dumps(valid, sort_keys=True) + "\n").encode("utf-8") + b"\xff")
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.head("t")
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("t", 2, "missing", 2))

    def test_other_tenant_corruption_does_not_block_head(self):
        self.chain.append("b", {})
        rows = [json.loads(l) for l in self.path.read_text().splitlines()]
        rows[0]["event"] = {"x": 1}  # b's digest broken; t unaffected
        self.write_rows(rows)
        item = self.chain.append("t", {"ok": True})
        self.assertEqual(self.chain.head("t"),
                         {"tenant": "t", "count": 1, "hash": item["hash"]})


class HeadsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.path = Path(self.tmp.name) / "audit.jsonl"
        self.chain = AuditChain(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def write_rows(self, rows):
        with self.path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(r if isinstance(r, str) else json.dumps(r, sort_keys=True))
                f.write("\n")

    # --- success shape ---

    def test_missing_and_empty_log_return_empty_tenant_list(self):
        self.assertEqual(self.chain.heads(), {"ok": True, "tenants": []})
        self.assertFalse(self.path.exists())
        self.path.write_bytes(b"")
        self.assertEqual(self.chain.heads(), {"ok": True, "tenants": []})

    def test_first_appearance_order_with_counts_and_hashes(self):
        b1 = self.chain.append("b", {})
        a1 = self.chain.append("a", {})
        b2 = self.chain.append("b", {})
        a2 = self.chain.append("a", {})
        c1 = self.chain.append("c", {})
        r = self.chain.heads()
        self.assertEqual(r, {"ok": True, "tenants": [
            {"tenant": "b", "count": 2, "hash": b2["hash"]},
            {"tenant": "a", "count": 2, "hash": a2["hash"]},
            {"tenant": "c", "count": 1, "hash": c1["hash"]},
        ]})
        self.assertNotEqual(b1["hash"], b2["hash"])
        self.assertEqual(a1["hash"], a1["hash"])  # sanity: unused but kept

    def test_tenant_types_kept_distinct_and_verbatim(self):
        for t in (1, 1.0, True, "1"):
            self.chain.append(t, {})
        r = self.chain.heads()
        self.assertTrue(r["ok"])
        self.assertEqual([e["tenant"] for e in r["tenants"]], [1, 1.0, True, "1"])
        self.assertTrue(all(e["count"] == 1 for e in r["tenants"]))
        self.assertTrue(all(len(e["hash"]) == 64 for e in r["tenants"]))

    def test_heads_agrees_with_head_per_tenant(self):
        for t in ("x", "y"):
            for i in range(3):
                self.chain.append(t, {"i": i})
        r = self.chain.heads()
        by_tenant = {e["tenant"]: e for e in r["tenants"]}
        for t in ("x", "y"):
            h = self.chain.head(t)
            self.assertEqual(by_tenant[t]["count"], h["count"])
            self.assertEqual(by_tenant[t]["hash"], h["hash"])

    def test_heads_feeds_conditional_appends(self):
        self.chain.append("a", {})
        self.chain.append("b", {})
        r = self.chain.heads()
        for e in r["tenants"]:
            item = self.chain.append_if_head(
                e["tenant"], {"more": True}, e["count"], e["hash"])
            self.assertEqual(item["seq"], 2)
        r = self.chain.heads()
        self.assertTrue(all(e["count"] == 2 for e in r["tenants"]))

    # --- read-only ---

    def test_heads_creates_or_modifies_nothing(self):
        self.chain.append("t", {})
        before = self.path.read_bytes()
        entries = sorted(p.name for p in Path(self.tmp.name).iterdir())
        self.chain.heads()
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(sorted(p.name for p in Path(self.tmp.name).iterdir()),
                         entries)

    # --- failures: verify_all's exact fields, no partial heads ---

    def test_unparseable_line_reports_missing_no_partial_heads(self):
        self.chain.append("t", {})
        with self.path.open("a", encoding="utf-8") as f:
            f.write("{oops\n")
        r = self.chain.heads()
        self.assertEqual(r, {"ok": False, "at": 2, "tenant": None,
                             "reason": "missing"})

    def test_missing_field_reports_tenant(self):
        self.write_rows([{"tenant": "t", "seq": 1, "event": {}, "prev": ZERO}])
        r = self.chain.heads()
        self.assertEqual(r, {"ok": False, "at": 1, "tenant": "t",
                             "reason": "missing"})

    def test_sequence_error_uses_physical_line(self):
        self.chain.append("a", {})
        row = {"tenant": "b", "seq": 2, "event": {}, "prev": ZERO}
        row["hash"] = AuditChain._hash(row)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        r = self.chain.heads()
        self.assertEqual(r, {"ok": False, "at": 2, "tenant": "b",
                             "reason": "sequence"})

    def test_digest_error_stops_at_first_physical_line(self):
        good = {"tenant": "a", "seq": 1, "event": {}, "prev": ZERO}
        good["hash"] = AuditChain._hash(good)
        bad = {"tenant": "b", "seq": 1, "event": {}, "prev": "f" * 64}
        bad["hash"] = AuditChain._hash(bad)
        later_bad = {"tenant": "a", "seq": 9, "event": {}, "prev": good["hash"]}
        later_bad["hash"] = AuditChain._hash(later_bad)
        self.write_rows([good, bad, later_bad])
        r = self.chain.heads()
        self.assertEqual(r, {"ok": False, "at": 2, "tenant": "b",
                             "reason": "digest"})

    def test_illegal_utf8_reports_missing_at_physical_line(self):
        valid = {"tenant": "t", "seq": 1, "event": {}, "prev": ZERO}
        valid["hash"] = AuditChain._hash(valid)
        self.path.write_bytes(
            (json.dumps(valid, sort_keys=True) + "\n").encode("utf-8") + b"\xff")
        r = self.chain.heads()
        self.assertEqual(r, {"ok": False, "at": 2, "tenant": None,
                             "reason": "missing"})

    def test_failure_matches_verify_all_and_keeps_bytes(self):
        self.chain.append("t", {})
        rows = [json.loads(l) for l in self.path.read_text().splitlines()]
        rows[0]["event"] = {"v": 2}
        self.write_rows(rows)
        before = self.path.read_bytes()
        self.assertEqual(self.chain.heads(), self.chain.verify_all())
        self.assertFalse(self.chain.heads()["ok"])
        self.assertEqual(self.path.read_bytes(), before)


class HeadConcurrencyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.path = Path(self.tmp.name) / "audit.jsonl"
        self.chain = AuditChain(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_heads_and_head_observe_complete_snapshots(self):
        for i in range(5):
            self.chain.append("t", {"i": i})
        problems = []
        box = threading.Lock()
        stop = threading.Event()

        def reader():
            while not stop.is_set():
                r = self.chain.heads()
                if not r["ok"]:
                    with box:
                        problems.append(("heads", r))
                    return
                if len(r["tenants"]) != 1:
                    with box:
                        problems.append(("tenants", r["tenants"]))
                    return
                e = r["tenants"][0]
                h = self.chain.head("t")
                for count, digest in ((e["count"], e["hash"]),
                                      (h["count"], h["hash"])):
                    if count < 5 or len(digest) != 64:
                        with box:
                            problems.append(("shape", count, digest))
                        return

        readers = [threading.Thread(target=reader) for _ in range(4)]
        for t in readers:
            t.start()

        def writer(i):
            for j in range(30):
                self.chain.append("t", {"w": i, "j": j})

        writers = [threading.Thread(target=writer, args=(i,)) for i in range(4)]
        for t in writers:
            t.start()
        for t in writers:
            t.join()
        stop.set()
        for t in readers:
            t.join(timeout=2)

        self.assertEqual(problems, [])
        total = 5 + 4 * 30
        h = self.chain.head("t")
        self.assertEqual(h["count"], total)
        r = self.chain.heads()
        self.assertEqual(r["tenants"],
                         [{"tenant": "t", "count": total, "hash": h["hash"]}])
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": total})


if __name__ == "__main__":
    unittest.main()
