import json
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app import AuditChain, AuditChainStateError

ZERO = "0" * 64


class AppendManyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.path = Path(self.tmp.name) / "audit.jsonl"
        self.chain = AuditChain(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def read_rows(self):
        return [json.loads(l) for l in self.path.read_text().splitlines()]

    def test_empty_list_is_a_noop_and_creates_nothing(self):
        self.assertEqual(self.chain.append_many([]), [])
        self.assertFalse(self.path.exists())
        # also a no-op against an existing file: bytes untouched
        self.chain.append("u", {"v": 1})
        before = self.path.read_bytes()
        self.assertEqual(self.chain.append_many([]), [])
        self.assertEqual(self.path.read_bytes(), before)

    def test_entries_must_be_a_list(self):
        for bad in (None, True, 1, 1.5, "x",
                    {"tenant": "t", "event": {}}, ("x",), {"a", "b"}):
            with self.assertRaises(ValueError):
                self.chain.append_many(bad)
        self.assertFalse(self.path.exists())

    def test_member_shape_and_key_set_are_checked(self):
        bad_members = [
            None, True, 1, "x", [], ["tenant", "event"],
            {},                                  # missing both keys
            {"tenant": "t"},                     # missing event
            {"event": {}},                       # missing tenant
            {"tenant": "t", "event": {}, "seq": 1},   # extra key
            {"tenant": "t", "event": {}, "x": 1},
            {"tenant": "t", 1: {}},              # non-string key
        ]
        for member in bad_members:
            with self.assertRaises(ValueError, msg=repr(member)):
                self.chain.append_many([member])
        # one bad member anywhere rejects the whole call
        with self.assertRaises(ValueError):
            self.chain.append_many(
                [{"tenant": "t", "event": {}}, {"tenant": "u"}])
        self.assertFalse(self.path.exists())

    def test_tenant_and_event_cross_the_json_boundary(self):
        with self.assertRaises(ValueError):
            self.chain.append_many([{"tenant": float("nan"), "event": {}}])
        with self.assertRaises(ValueError):
            self.chain.append_many(
                [{"tenant": "t", "event": {"bad": float("inf")}}])
        cyc = []
        cyc.append(cyc)
        with self.assertRaises(ValueError):
            self.chain.append_many([{"tenant": "t", "event": cyc}])
        with self.assertRaises(ValueError):
            self.chain.append_many([{"tenant": "t", "event": {1: "x"}}])
        # a boundary violation in any member rejects the whole call
        with self.assertRaises(ValueError):
            self.chain.append_many([
                {"tenant": "t", "event": {}},
                {"tenant": "u", "event": float("nan")},
            ])
        self.assertFalse(self.path.exists())

    def test_validation_happens_before_history_is_read(self):
        # a corrupt log must not turn a malformed call into a state error
        self.path.write_bytes(b"\xff")
        before = self.path.read_bytes()
        with self.assertRaises(ValueError):
            self.chain.append_many([{"tenant": "t"}])
        with self.assertRaises(ValueError):
            self.chain.append_many("not-a-list")
        self.assertEqual(self.path.read_bytes(), before)

    def test_single_tenant_entries_number_consecutively(self):
        items = self.chain.append_many([
            {"tenant": "t", "event": {"i": 1}},
            {"tenant": "t", "event": {"i": 2}},
            {"tenant": "t", "event": {"i": 3}},
        ])
        self.assertEqual([it["seq"] for it in items], [1, 2, 3])
        self.assertEqual(items[0]["prev"], ZERO)
        self.assertEqual(items[1]["prev"], items[0]["hash"])
        self.assertEqual(items[2]["prev"], items[1]["hash"])
        self.assertEqual(items, self.read_rows())
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": 3})

    def test_tenants_interleave_and_each_continues_own_tail(self):
        a0 = self.chain.append("a", {"i": 0})
        b0 = self.chain.append("b", {"i": 0})
        items = self.chain.append_many([
            {"tenant": "a", "event": {"i": 1}},
            {"tenant": "b", "event": {"i": 1}},
            {"tenant": "a", "event": {"i": 2}},
            {"tenant": "c", "event": {"i": 1}},
            {"tenant": "b", "event": {"i": 2}},
        ])
        self.assertEqual([it["tenant"] for it in items],
                         ["a", "b", "a", "c", "b"])
        self.assertEqual([it["seq"] for it in items], [2, 2, 3, 1, 3])
        self.assertEqual(items[0]["prev"], a0["hash"])
        self.assertEqual(items[1]["prev"], b0["hash"])
        self.assertEqual(items[2]["prev"], items[0]["hash"])
        self.assertEqual(items[3]["prev"], ZERO)
        self.assertEqual(items[4]["prev"], items[1]["hash"])
        # physical JSONL order matches input order exactly
        self.assertEqual(self.read_rows()[2:], items)
        self.assertEqual(self.chain.verify("a"), {"ok": True, "count": 3})
        self.assertEqual(self.chain.verify("b"), {"ok": True, "count": 3})
        self.assertEqual(self.chain.verify("c"), {"ok": True, "count": 1})
        self.assertTrue(self.chain.verify_all()["ok"])

    def test_returned_records_carry_exactly_the_five_fields(self):
        items = self.chain.append_many([
            {"tenant": "t", "event": {}},
            {"tenant": "u", "event": {}},
        ])
        for item in items:
            self.assertEqual(set(item), {"tenant", "seq", "event",
                                         "prev", "hash"})

    def test_canonical_tenant_identity_shared_within_one_call(self):
        # distinct spellings of one JSON identity share a chain; distinct
        # identities (1 vs "1") stay separate chains
        items = self.chain.append_many([
            {"tenant": {"a": 1, "b": 2}, "event": {}},
            {"tenant": {"b": 2, "a": 1}, "event": {}},
            {"tenant": 1, "event": {}},
            {"tenant": "1", "event": {}},
        ])
        self.assertEqual([it["seq"] for it in items], [1, 2, 1, 1])
        self.assertEqual(items[1]["prev"], items[0]["hash"])
        self.assertEqual(items[2]["prev"], ZERO)
        self.assertEqual(items[3]["prev"], ZERO)
        self.assertTrue(self.chain.verify_all()["ok"])

    def test_byte_equivalent_to_sequential_appends(self):
        entries = [
            {"tenant": "a", "event": {"i": 1, "s": "审计"}},
            {"tenant": "b", "event": [1, 2]},
            {"tenant": "a", "event": {"i": 2}},
            {"tenant": "a", "event": {"i": 3}},
            {"tenant": "b", "event": None},
        ]
        self.chain.append("z", {})
        many_items = self.chain.append_many(entries)
        self.chain.append("z", {})
        grouped = self.path.read_bytes()

        other = AuditChain(self.path.with_name("other.jsonl"))
        other.append("z", {})
        sequential = [other.append(e["tenant"], e["event"]) for e in entries]
        other.append("z", {})
        self.assertEqual(other.path.read_bytes(), grouped)
        self.assertEqual(sequential, many_items)
        self.assertEqual(other.verify_all(), self.chain.verify_all())

    def test_corrupt_history_raises_without_partial_records(self):
        self.chain.append("t", {})
        rows = self.read_rows()
        rows[0]["event"] = {"x": 9}
        with self.path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, sort_keys=True) + "\n")
        before = self.path.read_bytes()
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.append_many([
                {"tenant": "t", "event": {"i": 1}},
                {"tenant": "u", "event": {"i": 1}},
            ])
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("t", 1, "digest", 1))
        self.assertEqual(self.path.read_bytes(), before)

    def test_corrupt_history_uninvolved_tenant_does_not_block(self):
        self.chain.append("b", {})
        rows = self.read_rows()
        rows[0]["event"] = {"x": 1}
        with self.path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, sort_keys=True) + "\n")
        items = self.chain.append_many([
            {"tenant": "t", "event": {}},
            {"tenant": "u", "event": {}},
        ])
        self.assertEqual([it["seq"] for it in items], [1, 1])
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": 1})
        self.assertEqual(self.chain.verify("u"), {"ok": True, "count": 1})

    def test_earliest_physical_line_wins_across_chains(self):
        # two broken chains: "a" breaks at line 3, "b" breaks at line 2;
        # the error of "b" is reported even though "a" comes first in input
        def record(tenant, seq, prev):
            item = {"tenant": tenant, "seq": seq, "event": {}, "prev": prev}
            item["hash"] = AuditChain._hash(item)
            return item

        a1 = record("a", 1, ZERO)
        broken_b1 = record("b", 1, ZERO)
        broken_b1["event"] = {"tampered": True}  # hash no longer matches
        broken_a2 = {"tenant": "a", "seq": 5, "event": {},
                     "prev": a1["hash"], "hash": ZERO}
        # physical layout: line1 a1(ok), line2 b1(digest), line3 a2(sequence)
        rows = [a1, broken_b1, broken_a2]
        with self.path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, sort_keys=True) + "\n")
        before = self.path.read_bytes()
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.append_many([
                {"tenant": "a", "event": {}},
                {"tenant": "b", "event": {}},
            ])
        self.assertEqual((cm.exception.tenant, cm.exception.reason,
                          cm.exception.line),
                         ("b", "digest", 2))
        self.assertEqual(self.path.read_bytes(), before)

    def test_same_line_tie_breaks_by_input_order(self):
        # one unparseable line breaks every involved chain at the same
        # physical line; the tenant first in input order is reported
        self.chain.append("t", {})
        with self.path.open("a", encoding="utf-8") as f:
            f.write("not json\n")
        before = self.path.read_bytes()
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.append_many([
                {"tenant": "u", "event": {}},
                {"tenant": "t", "event": {}},
            ])
        self.assertEqual((cm.exception.tenant, cm.exception.reason,
                          cm.exception.line),
                         ("u", "missing", 2))
        self.assertEqual(self.path.read_bytes(), before)

    def test_illegal_utf8_reports_same_first_broken_point_as_append(self):
        valid = {"tenant": "t", "seq": 1, "event": {}, "prev": ZERO}
        valid["hash"] = AuditChain._hash(valid)
        before = (json.dumps(valid, sort_keys=True) + "\n").encode("utf-8") + b"\xff"
        self.path.write_bytes(before)
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.append_many([
                {"tenant": "t", "event": {}},
                {"tenant": "t", "event": {}},
            ])
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("t", 2, "missing", 2))
        self.assertEqual(self.path.read_bytes(), before)

    def test_legacy_file_without_trailing_newline_still_one_line_each(self):
        self.chain.append_many([{"tenant": "t", "event": {"v": 1}}])
        self.path.write_bytes(self.path.read_bytes().rstrip(b"\n"))
        items = self.chain.append_many([
            {"tenant": "t", "event": {"v": 2}},
            {"tenant": "u", "event": {"v": 1}},
        ])
        lines = self.path.read_text().splitlines()
        self.assertEqual(len(lines), 3)
        for raw in lines:
            json.loads(raw)  # records must never be glued together
        self.assertEqual([it["seq"] for it in items], [2, 1])
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": 2})
        self.assertEqual(self.chain.verify("u"), {"ok": True, "count": 1})

    def test_result_usable_by_existing_readers_and_offline_verification(self):
        self.chain.append_many([
            {"tenant": "a", "event": {"i": 1}},
            {"tenant": "b", "event": {"i": 1}},
            {"tenant": "a", "event": {"i": 2}},
        ])
        self.assertEqual([r["seq"] for r in self.chain.read_tenant("a")],
                         [1, 2])
        exported = self.chain.export_tenant("a")
        other = AuditChain(self.path.with_name("other.jsonl"))
        imported = other.import_tenant("a", exported)
        self.assertEqual([r["seq"] for r in imported], [1, 2])
        self.assertEqual(other.verify("a"), {"ok": True, "count": 2})
        self.assertEqual(
            self.chain.verify_bytes(self.path.read_bytes(), "a"),
            {"ok": True, "count": 2})
        self.assertTrue(
            self.chain.verify_all_bytes(self.path.read_bytes())["ok"])


class AppendManyConcurrencyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.path = Path(self.tmp.name) / "audit.jsonl"
        self.chain = AuditChain(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_groups_are_indivisible_intervals(self):
        n_thread, size = 8, 6
        results, errors = [], []
        box = threading.Lock()

        def worker(i):
            local = []
            try:
                local = self.chain.append_many([
                    {"tenant": "t", "event": {"w": i, "j": j}}
                    for j in range(size)
                ])
            except Exception as e:  # noqa: BLE001
                with box:
                    errors.append(repr(e))
            with box:
                results.extend((i, it) for it in local)

        threads = [threading.Thread(target=worker, args=(i,))
                   for i in range(n_thread)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [])
        total = n_thread * size
        self.assertEqual(sorted(it["seq"] for _, it in results),
                         list(range(1, total + 1)))
        rows = [json.loads(l) for l in self.path.read_text().splitlines()]
        self.assertEqual(len(rows), total)
        by_worker = {}
        for row in rows:
            by_worker.setdefault(row["event"]["w"], []).append(row)
        for i, group in by_worker.items():
            seqs = [r["seq"] for r in group]
            self.assertEqual(seqs, list(range(seqs[0], seqs[0] + size)), i)
            for prev_row, row in zip(group, group[1:]):
                self.assertEqual(row["prev"], prev_row["hash"])
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": total})
        self.assertTrue(self.chain.verify_all()["ok"])

    def test_groups_interleave_with_single_appends_of_other_tenants(self):
        done = []
        box = threading.Lock()

        def many_worker():
            items = self.chain.append_many([
                {"tenant": "a", "event": {"i": j}} for j in range(20)])
            with box:
                done.extend(("a", it) for it in items)

        def single_worker():
            local = [self.chain.append("b", {"i": j}) for j in range(20)]
            with box:
                done.extend(("b", it) for it in local)

        threads = [threading.Thread(target=many_worker) for _ in range(4)]
        threads += [threading.Thread(target=single_worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        counts = {"a": 0, "b": 0}
        for tenant, it in done:
            counts[tenant] += 1
        self.assertEqual(counts, {"a": 80, "b": 80})
        self.assertEqual(self.chain.verify("a"), {"ok": True, "count": 80})
        self.assertEqual(self.chain.verify("b"), {"ok": True, "count": 80})
        self.assertTrue(self.chain.verify_all()["ok"])

    def test_value_error_under_contention_leaves_no_trace(self):
        outcomes = []
        box = threading.Lock()

        def good(i):
            self.chain.append_many([
                {"tenant": "t", "event": {"i": i}},
                {"tenant": "u", "event": {"i": i}},
            ])
            with box:
                outcomes.append("ok")

        def bad(i):
            try:
                self.chain.append_many([
                    {"tenant": "t", "event": {"i": i}},
                    {"tenant": "u"},  # missing key
                ])
            except ValueError:
                with box:
                    outcomes.append("value")

        threads = []
        for i in range(20):
            threads.append(threading.Thread(target=good, args=(i,)))
            threads.append(threading.Thread(target=bad, args=(i,)))
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(outcomes.count("ok"), 20)
        self.assertEqual(outcomes.count("value"), 20)
        self.assertEqual(self.chain.verify("t")["count"], 20)
        self.assertEqual(self.chain.verify("u")["count"], 20)


if __name__ == "__main__":
    unittest.main()
