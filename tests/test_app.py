import json
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import app
from app import AuditChain, AuditChainStateError


ZERO = "0" * 64


class AuditChainTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.path = Path(self.tmp.name) / "audit.jsonl"
        self.chain = AuditChain(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, rows):
        with self.path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(r if isinstance(r, str) else json.dumps(r, sort_keys=True))
                f.write("\n")

    def read(self):
        return self.path.read_text(encoding="utf-8")

    def test_empty_verify(self):
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": 0})

    def test_append_starts_at_seq_one_with_zero_prev(self):
        item = self.chain.append("t", {"a": 1})
        self.assertEqual(item["seq"], 1)
        self.assertEqual(item["prev"], ZERO)
        self.assertEqual(len(item["hash"]), 64)
        self.assertEqual(item["hash"], item["hash"].lower())
        row = json.loads(self.read())
        self.assertEqual(set(row), {"tenant", "seq", "event", "prev", "hash"})

    def test_append_chains(self):
        a = self.chain.append("t", {"i": 1})
        b = self.chain.append("t", {"i": 2})
        self.assertEqual(b["seq"], 2)
        self.assertEqual(b["prev"], a["hash"])
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": 2})

    def test_interleaved_tenants(self):
        a1 = self.chain.append("a", {})
        b1 = self.chain.append("b", {})
        a2 = self.chain.append("a", {})
        self.assertEqual(a2["seq"], 2)
        self.assertEqual(a2["prev"], a1["hash"])
        self.assertEqual(b1["seq"], 1)
        self.assertEqual(self.chain.verify("a"), {"ok": True, "count": 2})
        self.assertEqual(self.chain.verify("b"), {"ok": True, "count": 1})

    def test_expected_count_match_short_and_exact(self):
        self.chain.append("t", {})
        self.chain.append("t", {})
        self.assertEqual(self.chain.verify("t", 2), {"ok": True, "count": 2})
        r = self.chain.verify("t", 3)
        self.assertEqual((r["ok"], r["at"], r["reason"]), (False, 3, "missing"))
        r = self.chain.verify("t", 1)
        self.assertEqual((r["ok"], r["at"], r["reason"]), (False, 2, "sequence"))

    def test_expected_count_interleaved(self):
        self.chain.append("a", {})
        self.chain.append("b", {})
        self.assertEqual(self.chain.verify("a", 1), {"ok": True, "count": 1})

    # --- corruption: verify classification ---

    def test_unparseable_line_at_line_number(self):
        other = {"tenant": "x", "seq": 1, "event": {}, "prev": ZERO}
        other["hash"] = AuditChain._hash(other)
        with self.path.open("w", encoding="utf-8") as f:
            f.write(json.dumps(other, sort_keys=True) + "\n")
            f.write("{not json\n")
        r = self.chain.verify("t")
        self.assertFalse(r["ok"])
        self.assertEqual(r["at"], 2)  # file line number, not expected seq 1
        self.assertEqual(r["reason"], "missing")

    def test_unparseable_other_tenant_line_still_fatal(self):
        self.write(['{"tenant": "x", broken'])
        r = self.chain.verify("t")
        self.assertEqual((False, 1, "missing"), (r["ok"], r["at"], r["reason"]))

    def test_non_object_line(self):
        self.write(["[1,2,3]"])
        r = self.chain.verify("t")
        self.assertEqual((r["at"], r["reason"]), (1, "missing"))

    def test_missing_field(self):
        self.write([{"tenant": "t", "seq": 1, "event": {}, "prev": ZERO}])  # no hash
        r = self.chain.verify("t")
        self.assertEqual((r["at"], r["reason"]), (1, "missing"))

    def test_bad_first_prev_is_digest(self):
        row = {"tenant": "t", "seq": 1, "event": {}, "prev": "f" * 64}
        row["hash"] = AuditChain._hash(row)
        self.write([row])
        r = self.chain.verify("t")
        self.assertEqual((r["at"], r["reason"]), (1, "digest"))

    def test_bad_first_seq_is_sequence(self):
        row = {"tenant": "t", "seq": 2, "event": {}, "prev": ZERO}
        row["hash"] = AuditChain._hash(row)
        self.write([row])
        r = self.chain.verify("t")
        self.assertEqual((r["at"], r["reason"]), (1, "sequence"))

    def test_gap_is_sequence_at_expected(self):
        good = {"tenant": "t", "seq": 1, "event": {}, "prev": ZERO}
        good["hash"] = AuditChain._hash(good)
        row = {"tenant": "t", "seq": 3, "event": {}, "prev": good["hash"]}
        row["hash"] = AuditChain._hash(row)
        self.write([good, row])
        r = self.chain.verify("t")
        self.assertEqual((r["at"], r["reason"]), (2, "sequence"))

    def test_bad_prev_link_is_digest(self):
        self.chain.append("t", {})
        row = {"tenant": "t", "seq": 2, "event": {}, "prev": "9" * 64}
        row["hash"] = AuditChain._hash(row)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        r = self.chain.verify("t")
        self.assertEqual((r["at"], r["reason"]), (2, "digest"))

    def test_tampered_hash_is_digest(self):
        self.chain.append("t", {})
        row = {"tenant": "t", "seq": 2, "event": {}, "prev": ZERO}
        row["hash"] = "deadbeef" * 8
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        r = self.chain.verify("t")
        self.assertEqual((r["at"], r["reason"]), (2, "digest"))

    def test_tampered_event_is_digest(self):
        item = self.chain.append("t", {"v": 1})
        rows = [json.loads(l) for l in self.read().splitlines()]
        rows[0]["event"] = {"v": 2}
        self.write(rows)
        r = self.chain.verify("t")
        self.assertEqual((r["at"], r["reason"]), (1, "digest"))

    def test_first_error_reported_with_interleaving(self):
        # bad seq record for a at seq2 position, b in between is fine
        good = {"tenant": "a", "seq": 1, "event": {}, "prev": ZERO}
        good["hash"] = AuditChain._hash(good)
        b = {"tenant": "b", "seq": 1, "event": {}, "prev": ZERO}
        b["hash"] = AuditChain._hash(b)
        bad = {"tenant": "a", "seq": 3, "event": {}, "prev": good["hash"]}
        bad["hash"] = AuditChain._hash(bad)
        self.write([good, b, bad])
        r = self.chain.verify("a")
        self.assertEqual((r["at"], r["reason"]), (2, "sequence"))
        self.assertEqual(self.chain.verify("b"), {"ok": True, "count": 1})

    # --- corruption: illegal UTF-8 bytes in the file ---

    def _write_bytes(self, data):
        with self.path.open("wb") as f:
            f.write(data)

    def _valid_row(self, tenant="t", seq=1, prev=ZERO):
        row = {"tenant": tenant, "seq": seq, "event": {}, "prev": prev}
        row["hash"] = AuditChain._hash(row)
        return (json.dumps(row, sort_keys=True) + "\n").encode("utf-8")

    def test_illegal_utf8_first_byte_is_missing_line_one(self):
        self._write_bytes(b"\xff\xfe")
        self.assertEqual(self.chain.verify("t"),
                         {"ok": False, "at": 1, "reason": "missing"})
        self.assertEqual(self.chain.verify_all(),
                         {"ok": False, "at": 1, "tenant": None, "reason": "missing"})

    def test_illegal_utf8_on_last_line_without_newline(self):
        # bad byte is the second physical line even with no trailing newline
        self._write_bytes(self._valid_row() + b'{"tenant":\xff}')
        self.assertEqual(self.chain.verify("t"),
                         {"ok": False, "at": 2, "reason": "missing"})
        self.assertEqual(self.chain.verify_all(),
                         {"ok": False, "at": 2, "tenant": None, "reason": "missing"})

    def test_illegal_utf8_alone_after_newline_still_next_line(self):
        self._write_bytes(self._valid_row() + b"\xff")
        self.assertEqual(self.chain.verify("t"),
                         {"ok": False, "at": 2, "reason": "missing"})
        self.assertEqual(self.chain.verify_all(),
                         {"ok": False, "at": 2, "tenant": None, "reason": "missing"})

    def test_truncated_multibyte_counts_lfs_before_it(self):
        self._write_bytes(self._valid_row("x") + self._valid_row("y") + b"abc\xc2\n")
        self.assertEqual(self.chain.verify("t"),
                         {"ok": False, "at": 3, "reason": "missing"})
        self.assertEqual(self.chain.verify_all(),
                         {"ok": False, "at": 3, "tenant": None, "reason": "missing"})

    def test_illegal_utf8_verify_result_keeps_only_existing_fields(self):
        self._write_bytes(b"\xff")
        self.assertEqual(set(self.chain.verify("t")), {"ok", "at", "reason"})

    def test_earlier_digest_error_beats_later_bad_bytes(self):
        good = {"tenant": "t", "seq": 1, "event": {}, "prev": ZERO}
        good["hash"] = AuditChain._hash(good)
        tampered = dict(good)
        tampered["event"] = {"z": 9}
        self._write_bytes(
            (json.dumps(tampered, sort_keys=True) + "\n").encode("utf-8") + b"\xff\n"
        )
        self.assertEqual(self.chain.verify("t"),
                         {"ok": False, "at": 1, "reason": "digest"})
        self.assertEqual(self.chain.verify_all(),
                         {"ok": False, "at": 1, "tenant": "t", "reason": "digest"})

    def test_earlier_sequence_error_beats_later_bad_bytes(self):
        row = {"tenant": "t", "seq": 2, "event": {}, "prev": ZERO}
        row["hash"] = AuditChain._hash(row)
        self._write_bytes((json.dumps(row, sort_keys=True) + "\n").encode("utf-8") + b"\xff")
        self.assertEqual(self.chain.verify("t"),
                         {"ok": False, "at": 1, "reason": "sequence"})

    def test_earlier_unparseable_json_beats_later_bad_bytes(self):
        self._write_bytes(b"{oops\n\xff")
        self.assertEqual(self.chain.verify("t"),
                         {"ok": False, "at": 1, "reason": "missing"})
        self.assertEqual(self.chain.verify_all(),
                         {"ok": False, "at": 1, "tenant": None, "reason": "missing"})

    def test_valid_utf8_garbage_still_missing_not_digest_or_sequence(self):
        for raw in (b"\n", b"   \n", b"[1,2]\n"):
            self._write_bytes(raw)
            self.assertEqual(self.chain.verify("t"),
                             {"ok": False, "at": 1, "reason": "missing"}, raw)
        # complete object minus hash -> missing at its own seq, never digest
        self.write([{"tenant": "t", "seq": 1, "event": {}, "prev": ZERO}])
        self.assertEqual(self.chain.verify("t"),
                         {"ok": False, "at": 1, "reason": "missing"})

    def test_append_rejects_illegal_utf8_with_expected_seq_and_line(self):
        before = self._valid_row() + b"\xffgarbage"
        self._write_bytes(before)
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.append("t", {})
        # one valid t record -> next expected seq is 2; bad byte is on line 2
        self.assertEqual(cm.exception.tenant, "t")
        self.assertEqual(cm.exception.seq, 2)
        self.assertEqual(cm.exception.reason, "missing")
        self.assertEqual(cm.exception.line, 2)
        self.assertEqual(self.path.read_bytes(), before)

    def test_append_rejects_illegal_utf8_seq_one_for_unseen_tenant(self):
        before = self._valid_row("x") + b"\xff"
        self._write_bytes(before)
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.append("t", {})
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("t", 1, "missing", 2))
        self.assertEqual(self.path.read_bytes(), before)

    def test_append_missing_or_empty_file_starts_at_seq_one(self):
        # nonexistent file starts at seq 1 / ZERO
        item = self.chain.append("t", {})
        self.assertEqual((item["seq"], item["prev"]), (1, ZERO))
        self.path.unlink()
        # completely empty file behaves identically
        self._write_bytes(b"")
        item = self.chain.append("t", {})
        self.assertEqual((item["seq"], item["prev"]), (1, ZERO))

    # --- append must reject corrupt history without modifying file ---

    def test_append_rejects_and_keeps_file(self):
        self.chain.append("t", {})
        rows = [json.loads(l) for l in self.read().splitlines()]
        rows[0]["event"] = {"x": 9}
        self.write(rows)
        before = self.read()
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.append("t", {})
        self.assertEqual(cm.exception.tenant, "t")
        self.assertEqual(cm.exception.seq, 1)
        self.assertEqual(self.read(), before)

    def test_append_error_seq_falls_back_to_expected(self):
        # unparseable line: seq unknown -> first affected seq is expected (1)
        self.write(["{oops"])
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.append("t", {})
        self.assertEqual(cm.exception.seq, 1)

    def test_append_after_other_tenant_corruption_irrelevant(self):
        # corruption belongs to another tenant only: t appends fine
        self.chain.append("b", {})
        rows = [json.loads(l) for l in self.read().splitlines()]
        rows[0]["event"] = {"x": 1}
        self.write(rows)
        item = self.chain.append("t", {})
        self.assertEqual(item["seq"], 1)
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": 1})

    def test_unicode_event_hash_roundtrip(self):
        item = self.chain.append("t", {"name": "审计"})
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": 1})
        # independently recompute from the file
        row = json.loads(self.read())
        payload = json.dumps(
            {k: row[k] for k in ("tenant", "seq", "event", "prev")},
            sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
        import hashlib
        self.assertEqual(hashlib.sha256(payload).hexdigest(), row["hash"])

    # --- verify_all ---

    def test_verify_all_empty_and_missing_file(self):
        self.assertEqual(self.chain.verify_all(), {"ok": True, "tenants": []})
        self.path.write_text("", encoding="utf-8")
        self.assertEqual(self.chain.verify_all(), {"ok": True, "tenants": []})

    def test_verify_all_interleaved_first_appearance_order(self):
        self.chain.append("b", {})
        self.chain.append("a", {})
        self.chain.append("b", {})
        self.chain.append("a", {})
        self.chain.append("c", {})
        r = self.chain.verify_all()
        self.assertEqual(r, {"ok": True, "tenants": [
            {"tenant": "b", "count": 2},
            {"tenant": "a", "count": 2},
            {"tenant": "c", "count": 1},
        ]})
        # per-tenant verify still agrees
        self.assertEqual(self.chain.verify("a"), {"ok": True, "count": 2})

    def test_verify_all_tenant_types_kept_distinct_and_verbatim(self):
        for t in (1, "1"):
            self.chain.append(t, {})
            self.chain.append(t, {})
        r = self.chain.verify_all()
        self.assertEqual(r, {"ok": True, "tenants": [
            {"tenant": 1, "count": 2},
            {"tenant": "1", "count": 2},
        ]})

    def test_verify_all_unparseable_line(self):
        self.chain.append("t", {})
        with self.path.open("a", encoding="utf-8") as f:
            f.write("{oops\n")
        r = self.chain.verify_all()
        self.assertEqual(r, {"ok": False, "at": 2, "tenant": None, "reason": "missing"})

    def test_verify_all_non_object_and_missing_tenant(self):
        self.write(["[1,2,3]"])
        r = self.chain.verify_all()
        self.assertEqual(r, {"ok": False, "at": 1, "tenant": None, "reason": "missing"})
        self.write([{"seq": 1, "event": {}, "prev": ZERO, "hash": "x"}])
        r = self.chain.verify_all()
        self.assertEqual(r, {"ok": False, "at": 1, "tenant": None, "reason": "missing"})

    def test_verify_all_missing_field_reports_tenant(self):
        self.write([{"tenant": "t", "seq": 1, "event": {}, "prev": ZERO}])  # no hash
        r = self.chain.verify_all()
        self.assertEqual(r, {"ok": False, "at": 1, "tenant": "t", "reason": "missing"})

    def test_verify_all_sequence_error_uses_line_number(self):
        self.chain.append("a", {})
        row = {"tenant": "b", "seq": 2, "event": {}, "prev": ZERO}
        row["hash"] = AuditChain._hash(row)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        r = self.chain.verify_all()
        self.assertEqual(r, {"ok": False, "at": 2, "tenant": "b", "reason": "sequence"})

    def test_verify_all_digest_error_stops_at_first(self):
        good = {"tenant": "a", "seq": 1, "event": {}, "prev": ZERO}
        good["hash"] = AuditChain._hash(good)
        bad = {"tenant": "b", "seq": 1, "event": {}, "prev": "f" * 64}
        bad["hash"] = AuditChain._hash(bad)
        later_bad = {"tenant": "a", "seq": 9, "event": {}, "prev": good["hash"]}
        later_bad["hash"] = AuditChain._hash(later_bad)
        self.write([good, bad, later_bad])
        r = self.chain.verify_all()
        # first physical corruption wins; the later seq error must not override
        self.assertEqual(r, {"ok": False, "at": 2, "tenant": "b", "reason": "digest"})

    def test_verify_all_tampered_hash_is_digest(self):
        self.chain.append("t", {})
        rows = [json.loads(l) for l in self.read().splitlines()]
        rows[0]["event"] = {"v": 2}
        self.write(rows)
        before = self.read()
        r = self.chain.verify_all()
        self.assertEqual(r, {"ok": False, "at": 1, "tenant": "t", "reason": "digest"})
        self.assertEqual(self.read(), before)  # read-only, no side effects

    # --- tenant identity: canonical JSON data identity, not loose equality ---

    def test_numeric_bool_string_tenants_are_independent_chains(self):
        # 1, 1.0, true and "1" must not share a chain under loose equality.
        for t in (1, 1.0, True, "1"):
            item = self.chain.append(t, {})
            self.assertEqual(item["seq"], 1)
            self.assertEqual(item["prev"], ZERO)
        for t in (1, 1.0, True, "1"):
            self.assertEqual(self.chain.verify(t), {"ok": True, "count": 1})
            item = self.chain.append(t, {})
            self.assertEqual(item["seq"], 2)
        r = self.chain.verify_all()
        self.assertEqual(r, {"ok": True, "tenants": [
            {"tenant": 1, "count": 2},
            {"tenant": 1.0, "count": 2},
            {"tenant": True, "count": 2},
            {"tenant": "1", "count": 2},
        ]})

    def test_object_tenant_key_order_is_same_identity(self):
        a = self.chain.append({"a": 1, "b": 2}, {})
        b = self.chain.append({"b": 2, "a": 1}, {})
        self.assertEqual(b["seq"], 2)
        self.assertEqual(b["prev"], a["hash"])
        self.assertEqual(self.chain.verify({"b": 2, "a": 1}), {"ok": True, "count": 2})
        # nested objects normalize too, but different values stay distinct
        c = self.chain.append({"a": 1, "b": 3}, {})
        self.assertEqual(c["seq"], 1)

    def test_legacy_confused_chain_reports_first_sequence_error(self):
        # File written under loose-equality semantics: tenant 1 at seq 1,
        # then 1.0 continuing at seq 2. Under JSON identity, 1.0's first
        # record is a sequence error; nothing is reordered or repaired.
        one = {"tenant": 1, "seq": 1, "event": {}, "prev": ZERO}
        one["hash"] = AuditChain._hash(one)
        confused = {"tenant": 1.0, "seq": 2, "event": {}, "prev": one["hash"]}
        confused["hash"] = AuditChain._hash(confused)
        self.write([one, confused])
        before = self.read()
        r = self.chain.verify(1.0)
        self.assertEqual((r["ok"], r["at"], r["reason"]), (False, 1, "sequence"))
        self.assertEqual(self.chain.verify(1), {"ok": True, "count": 1})
        r = self.chain.verify_all()
        self.assertEqual(r, {"ok": False, "at": 2, "tenant": 1.0, "reason": "sequence"})
        self.assertEqual(self.read(), before)
        # append to the confused identity must refuse and keep bytes intact
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.append(1.0, {})
        self.assertEqual((cm.exception.tenant, cm.exception.seq, cm.exception.reason),
                         (1.0, 1, "sequence"))
        self.assertEqual(self.read(), before)
        # the unaffected identity still appends on its own chain
        item = self.chain.append(1, {})
        self.assertEqual(item["seq"], 2)

    def test_true_and_one_do_not_share_history(self):
        self.chain.append(1, {})
        item = self.chain.append(True, {})
        self.assertEqual(item["seq"], 1)
        self.assertEqual(item["prev"], ZERO)
        self.assertEqual(self.chain.verify(True), {"ok": True, "count": 1})
        self.assertEqual(self.chain.verify(1), {"ok": True, "count": 1})


class AuditChainConcurrencyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.path = Path(self.tmp.name) / "audit.jsonl"
        self.chain = AuditChain(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def run_threads(self, workers):
        errors = []

        def guard(fn):
            try:
                fn()
            except Exception as exc:  # noqa: BLE001 - collected and re-raised
                errors.append(exc)

        threads = [threading.Thread(target=guard, args=(w,)) for w in workers]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if errors:
            raise errors[0]

    def test_concurrent_appends_same_tenant_form_one_contiguous_chain(self):
        threads, per_thread = 8, 25
        results = [[] for _ in range(threads)]

        def make(i):
            def work():
                for j in range(per_thread):
                    results[i].append(self.chain.append("t", {"i": i, "j": j}))
            return work

        self.run_threads([make(i) for i in range(threads)])
        total = threads * per_thread
        # every successful append produced a distinct, contiguous seq
        seqs = sorted(item["seq"] for slot in results for item in slot)
        self.assertEqual(seqs, list(range(1, total + 1)))
        # returned items correspond 1:1 to records in the file
        rows = [json.loads(l) for l in
                self.path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), total)
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": total})

    def test_concurrent_appends_interleaved_tenants_each_count_from_one(self):
        tenants = ["a", "b", "c"]
        per_tenant = 20

        def make(tenant):
            def work():
                for i in range(per_tenant):
                    self.chain.append(tenant, {"i": i})
            return work

        self.run_threads([make(t) for t in tenants for _ in range(3)])
        for t in tenants:
            self.assertEqual(self.chain.verify(t),
                             {"ok": True, "count": 3 * per_tenant})
        r = self.chain.verify_all()
        self.assertTrue(r["ok"])
        self.assertEqual(sorted(x["count"] for x in r["tenants"]),
                         [3 * per_tenant] * 3)

    def test_verify_and_verify_all_during_appends_always_see_legal_history(self):
        stop = threading.Event()
        observations = [[], []]

        def writer():
            for i in range(60):
                self.chain.append("t", {"i": i})
            stop.set()

        def make_reader(slot):
            def reader():
                while not stop.is_set():
                    r = self.chain.verify("t")
                    self.assertTrue(r["ok"], r)
                    observations[slot].append(r["count"])
                    r = self.chain.verify_all()
                    self.assertTrue(r["ok"], r)
            return reader

        self.run_threads([writer, make_reader(0), make_reader(1)])
        # each reader observes a non-decreasing sequence of legal histories
        for seen in observations:
            self.assertTrue(seen)
            self.assertEqual(seen, sorted(seen))
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": 60})

    def test_concurrent_append_to_corrupt_history_all_refused_and_file_kept(self):
        row = {"tenant": "t", "seq": 1, "event": {}, "prev": ZERO}
        row["hash"] = AuditChain._hash(row)
        row["event"] = {"tampered": True}
        with self.path.open("w", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        before = self.path.read_bytes()
        outcomes = []

        def work():
            try:
                self.chain.append("t", {})
                outcomes.append("appended")
            except AuditChainStateError as e:
                outcomes.append((e.tenant, e.seq, e.reason, e.line))

        self.run_threads([work] * 6)
        self.assertEqual(len(outcomes), 6)
        for o in outcomes:
            self.assertEqual(o, ("t", 1, "digest", 1))
        self.assertEqual(self.path.read_bytes(), before)

    def test_concurrent_value_error_still_priority_over_state(self):
        # corrupt history + illegal event: ValueError must win in every thread
        self.write_corrupt()

        def work():
            with self.assertRaises(ValueError):
                self.chain.append("t", float("nan"))

        self.run_threads([work] * 4)

    def write_corrupt(self):
        row = {"tenant": "t", "seq": 5, "event": {}, "prev": ZERO}
        row["hash"] = AuditChain._hash(row)
        with self.path.open("w", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    def test_concurrent_appends_across_processes(self):
        import subprocess, sys
        workers, per_worker = 4, 10
        script = (
            "import sys; sys.path.insert(0, %r);"
            "from app import AuditChain;"
            "c = AuditChain(%r);"
            "[c.append('p', {'i': i}) for i in range(%d)]"
        ) % (str(Path(app.__file__).parent), str(self.path), per_worker)
        procs = [subprocess.Popen([sys.executable, "-c", script])
                 for _ in range(workers)]
        for p in procs:
            self.assertEqual(p.wait(), 0)
        self.assertEqual(self.chain.verify("p"),
                         {"ok": True, "count": workers * per_worker})


if __name__ == "__main__":
    unittest.main()
