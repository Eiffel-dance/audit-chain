import json
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app import (
    AuditChain,
    AuditChainConflictError,
    AuditChainStateError,
    ZERO,
)


def record_bytes(item):
    return (json.dumps(item, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


class ImportTenantTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.path = Path(self.tmp.name) / "audit.jsonl"
        self.chain = AuditChain(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def write_rows(self, rows, path=None):
        path = path or self.path
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(r if isinstance(r, str) else json.dumps(r, sort_keys=True))
                f.write("\n")

    def make_export(self, tenant="t", n=5, chain=None):
        chain = chain or AuditChain(self.path.with_name("source.jsonl"))
        for i in range(n):
            chain.append(tenant, {"i": i, "s": "审计"})
        return chain, chain.export_tenant(tenant)

    def valid_row(self, tenant="t", seq=1, prev=ZERO, event=None):
        row = {"tenant": tenant, "seq": seq, "event": event or {}, "prev": prev}
        row["hash"] = AuditChain._hash(row)
        return row

    # --- boundary: validation order, empty no-op, no file/no byte touched ---

    def test_data_must_be_bytes(self):
        for bad in ("", "not jsonl", bytearray(b""), 1, None, [], object()):
            with self.assertRaises(ValueError):
                self.chain.import_tenant("t", bad)
        self.assertFalse(self.path.exists())

    def test_illegal_tenant_is_value_error_even_for_empty_data(self):
        bad_tenants = [
            float("nan"), float("inf"), float("-inf"),
            {"k": float("nan")}, [1, [float("-inf")]],
            {1: "x"}, {"a": {2: 3}}, object(), b"bytes", {1, 2}, ("a", 1),
        ]
        for bad in bad_tenants:
            with self.assertRaises(ValueError):
                self.chain.import_tenant(bad, b"")
        self.assertFalse(self.path.exists())

    def test_cyclic_tenant_is_value_error(self):
        cyc = []
        cyc.append(cyc)
        with self.assertRaises(ValueError):
            self.chain.import_tenant(cyc, b"")
        d = {}
        d["self"] = d
        with self.assertRaises(ValueError):
            self.chain.import_tenant(d, b"")

    def test_empty_data_returns_empty_creates_and_reads_nothing(self):
        # missing path: no creation
        self.assertFalse(self.path.exists())
        self.assertEqual(self.chain.import_tenant("t", b""), [])
        self.assertFalse(self.path.exists())
        # corrupt target: the no-op never reads history and leaves bytes
        self.path.write_bytes(b"\xff")
        corrupt = self.path.read_bytes()
        self.assertEqual(self.chain.import_tenant("t", b""), [])
        self.assertEqual(self.path.read_bytes(), corrupt)
        # existing healthy target with other tenants: nothing appended
        self.path.unlink()
        self.chain.append("other", {})
        healthy = self.path.read_bytes()
        self.assertEqual(self.chain.import_tenant("t", b""), [])
        self.assertEqual(self.path.read_bytes(), healthy)

    def test_value_error_beats_target_state_and_conflict_and_keeps_bytes(self):
        # a target that is both corrupt (for t) and would conflict
        row = self.valid_row("t", 1)
        row["event"] = {"tampered": True}  # digest broken
        self.write_rows([row])
        before = self.path.read_bytes()
        for bad in (float("nan"), {"k": float("inf")}, {1: "x"}):
            with self.assertRaises(ValueError):
                self.chain.import_tenant(bad, b'{"tenant":"t"}')
        with self.assertRaises(ValueError):
            self.chain.import_tenant("t", "not bytes")
        self.assertEqual(self.path.read_bytes(), before)

    # --- success: grafting, verbatim semantics and offline verification ---

    def test_import_into_missing_path_creates_verifiable_log(self):
        _, data = self.make_export("t", 5)
        records = self.chain.import_tenant("t", data)
        self.assertEqual([r["seq"] for r in records], [1, 2, 3, 4, 5])
        self.assertEqual(self.path.read_bytes(), data)
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": 5})
        self.assertEqual(self.chain.verify_all(),
                         {"ok": True, "tenants": [{"tenant": "t", "count": 5}]})
        self.assertEqual(self.chain.export_tenant("t"), data)
        self.assertEqual(self.chain.verify_bytes(self.path.read_bytes(), "t"),
                         {"ok": True, "count": 5})

    def test_import_preserves_tenant_seq_event_prev_hash_verbatim(self):
        src, data = self.make_export("t", 4)
        records = self.chain.import_tenant("t", data)
        src_rows = [json.loads(l) for l in data.decode("utf-8").splitlines()]
        self.assertEqual(records, src_rows)
        self.assertEqual(records[0]["prev"], ZERO)
        for prev, r in zip([ZERO] + [x["hash"] for x in records], records):
            self.assertEqual(r["prev"], prev)
            self.assertEqual(r["hash"], AuditChain._hash(r))
            self.assertEqual(set(r), {"tenant", "seq", "event", "prev", "hash"})
        # appending on the target continues the imported chain without gaps
        nxt = self.chain.append("t", {"i": 99})
        self.assertEqual((nxt["seq"], nxt["prev"]),
                         (5, records[-1]["hash"]))
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": 5})

    def test_returned_records_are_ordered_by_seq(self):
        _, data = self.make_export("t", 3)
        records = self.chain.import_tenant("t", data)
        self.assertEqual([r["seq"] for r in records], [1, 2, 3])
        self.assertEqual([r["event"] for r in records],
                         [{"i": i, "s": "审计"} for i in range(3)])
        # paged read works against the grafted history
        self.assertEqual([r["seq"] for r in self.chain.read_tenant("t")],
                         [1, 2, 3])
        self.assertEqual([r["seq"] for r in self.chain.read_tenant("t", 2, 1)],
                         [2])

    def test_import_single_record(self):
        row = self.valid_row("t", 1, event={"only": True})
        records = self.chain.import_tenant("t", record_bytes(row))
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0], row)
        self.assertEqual(self.path.read_bytes(), record_bytes(row))

    def test_import_coexists_with_other_tenants_records(self):
        # target already carries a valid interleaved history of other tenants
        for i in range(3):
            self.chain.append("a", {"i": i})
            self.chain.append("b", {"i": i})
        target_before = self.path.read_bytes()
        _, data = self.make_export("t", 4)
        records = self.chain.import_tenant("t", data)
        self.assertEqual(len(records), 4)
        # the import is one indivisible block appended after existing bytes
        self.assertEqual(self.path.read_bytes(), target_before + data)
        self.assertEqual(self.chain.verify("a"), {"ok": True, "count": 3})
        self.assertEqual(self.chain.verify("b"), {"ok": True, "count": 3})
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": 4})
        self.assertTrue(self.chain.verify_all()["ok"])
        # t's first record still anchors at ZERO despite the interleaving
        self.assertEqual(self.chain.export_tenant("t"), data)
        self.assertEqual(
            self.chain.verify_all_bytes(self.path.read_bytes())["ok"], True)

    def test_import_after_target_without_trailing_newline_still_valid(self):
        # a hand-me-down target whose last physical line lacks a newline is
        # healed exactly the way a plain append heals it
        other = self.valid_row("a", 1)
        raw = json.dumps(other, sort_keys=True).encode("utf-8")  # no \n
        self.path.write_bytes(raw)
        _, data = self.make_export("t", 2)
        self.chain.import_tenant("t", data)
        merged = self.path.read_bytes()
        self.assertTrue(self.chain.verify_all()["ok"])
        self.assertEqual(merged, raw + b"\n" + data)

    def test_repeated_imports_of_distinct_tenants_share_one_log(self):
        _, data_a = self.make_export("a", 3)
        _, data_b = self.make_export("b", 2)
        self.chain.import_tenant("a", data_a)
        self.chain.import_tenant("b", data_b)
        self.assertEqual(self.chain.verify("a"), {"ok": True, "count": 3})
        self.assertEqual(self.chain.verify("b"), {"ok": True, "count": 2})
        self.assertEqual(self.chain.export_tenant("a"), data_a)
        self.assertEqual(self.chain.export_tenant("b"), data_b)
        self.assertEqual(self.chain.verify_all(), {
            "ok": True,
            "tenants": [{"tenant": "a", "count": 3},
                        {"tenant": "b", "count": 2}],
        })

    def test_distinct_json_identities_never_conflict_on_import(self):
        # target already has tenant 1; importing the unrelated tenant "1"
        # must see an empty chain for its own identity
        self.chain.append(1, {"k": "int"})
        rows = [self.valid_row("1", 1, event={"k": "str"})]
        data = record_bytes(rows[0])
        self.chain.import_tenant("1", data)
        for tenant, count in ((1, 1), (1.0, 0), (True, 0), ("1", 1)):
            self.assertEqual(self.chain.verify(tenant),
                             {"ok": True, "count": count})
        self.assertTrue(self.chain.verify_all()["ok"])

    def test_object_tenant_identity_key_order_is_normalized(self):
        t1 = {"a": [1, {"x": 1}], "b": None}
        src = AuditChain(self.path.with_name("src.jsonl"))
        src.append(t1, {"v": 1})
        data = src.export_tenant(t1)
        # import under a textually different but canonically equal tenant
        returned = self.chain.import_tenant({"b": None, "a": [1, {"x": 1}]}, data)
        self.assertEqual(len(returned), 1)
        self.assertEqual(self.chain.verify(t1), {"ok": True, "count": 1})

    def test_non_canonical_input_spelling_is_accepted_and_normalized(self):
        # whitespace and textual key order differ, JSON identity does not
        row = self.valid_row("t", 1, event={"a": 1, "b": 2})
        raw = (b'{"hash": "' + row["hash"].encode() + b'",'
               b' "tenant": "t", "seq": 1, '
               b'"event": {"b": 2, "a": 1}, "prev": "'
               + ZERO.encode() + b'"}\n')
        self.assertEqual(len(raw.splitlines()), 1)
        records = self.chain.import_tenant("t", raw)
        self.assertEqual(records[0], row)
        self.assertEqual(self.path.read_bytes(), record_bytes(row))
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": 1})

    def test_imported_export_reexports_byte_identical(self):
        _, data = self.make_export("t", 6)
        self.chain.import_tenant("t", data)
        self.assertEqual(self.chain.export_tenant("t"), data)

    # --- input contract: foreign tenant is ValueError ---

    def test_mixed_tenant_bytes_raise_value_error(self):
        # full interleaved source log (not a single-tenant export)
        src = AuditChain(self.path.with_name("src.jsonl"))
        src.append("t", {})
        src.append("other", {})
        src.append("t", {})
        mixed = self.path.with_name("src.jsonl").read_bytes()
        target = self.path.with_name("dst.jsonl")
        chain = AuditChain(target)
        with self.assertRaises(ValueError):
            chain.import_tenant("t", mixed)
        self.assertFalse(target.exists())

    def test_foreign_record_after_valid_prefix_raises_and_writes_nothing(self):
        good1 = record_bytes(self.valid_row("t", 1))
        foreign = record_bytes(self.valid_row("other", 1))
        good3 = record_bytes(self.valid_row(
            "t", 3, prev=AuditChain._hash(json.loads(good1))))
        # foreign at physical line 2 even though line 3 is also broken
        with self.assertRaises(ValueError):
            self.chain.import_tenant("t", good1 + foreign + good3)
        self.assertFalse(self.path.exists())

    def test_foreign_distinct_json_identity_is_value_error(self):
        int_row = record_bytes(self.valid_row(1, 1))
        str_row = record_bytes(self.valid_row("1", 1))
        with self.assertRaises(ValueError):
            self.chain.import_tenant(1, int_row + str_row)
        self.assertFalse(self.path.exists())

    def test_first_physical_problem_wins_between_foreign_and_chain_defect(self):
        # chain defect on line 1 precedes the foreign record on line 2
        bad_first = self.valid_row("t", 2)  # seq 2 where seq 1 expected
        foreign = self.valid_row("other", 1)
        data = record_bytes(bad_first) + record_bytes(foreign)
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.import_tenant("t", data)
        self.assertEqual((cm.exception.seq, cm.exception.reason,
                          cm.exception.line), (1, "sequence", 1))
        # foreign on line 1 precedes the chain defect on line 2
        good = self.valid_row("t", 1)
        gap = self.valid_row("t", 3, prev=good["hash"])
        data = record_bytes(foreign) + record_bytes(gap)
        with self.assertRaises(ValueError):
            self.chain.import_tenant("t", data)

    # --- input chain state errors: same locations as append/verify/export ---

    def test_input_not_starting_at_seq_one_is_sequence(self):
        row = self.valid_row("t", 2)
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.import_tenant("t", record_bytes(row))
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("t", 1, "sequence", 1))
        self.assertFalse(self.path.exists())

    def test_input_first_prev_not_zero_is_digest(self):
        row = self.valid_row("t", 1, prev="f" * 64)
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.import_tenant("t", record_bytes(row))
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("t", 1, "digest", 1))

    def test_input_hash_mismatch_is_digest(self):
        row = self.valid_row("t", 1)
        row["event"] = {"tampered": True}
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.import_tenant("t", record_bytes(row))
        self.assertEqual((cm.exception.seq, cm.exception.reason,
                          cm.exception.line), (1, "digest", 1))

    def test_input_missing_field_is_missing_at_seq_and_line(self):
        raw = json.dumps(
            {"tenant": "t", "seq": 1, "event": {}, "prev": ZERO}).encode()
        raw += b"\n"
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.import_tenant("t", raw)
        self.assertEqual((cm.exception.seq, cm.exception.reason,
                          cm.exception.line), (1, "missing", 1))

    def test_input_unparseable_blank_or_non_object_lines_are_missing(self):
        cases = [
            (b"\n", 1),
            (b"   \n", 1),
            (b"[1,2,3]\n", 1),
            (b"{not json\n", 1),
            (record_bytes(self.valid_row("t", 1)) + b"{bad\n", 2),
            (record_bytes(self.valid_row("t", 1)) + b"\n", 2),
        ]
        for raw, line in cases:
            with self.assertRaises(AuditChainStateError) as cm:
                self.chain.import_tenant("t", raw)
            self.assertEqual((cm.exception.seq, cm.exception.reason,
                              cm.exception.line), (line, "missing", line), raw)

    def test_input_duplicate_keys_and_non_standard_numbers_are_missing(self):
        for raw in (
            b'{"tenant":"t","tenant":"t","seq":1,"event":{},'
            b'"prev":"' + ZERO.encode() + b'","hash":"x"}\n',
            b'{"tenant":"t","seq":1,"event":{"x":NaN},"prev":"'
            + ZERO.encode() + b'","hash":"x"}\n',
            b'{"tenant":"t","seq":1e999,"event":{},"prev":"'
            + ZERO.encode() + b'","hash":"x"}\n',
        ):
            with self.assertRaises(AuditChainStateError) as cm:
                self.chain.import_tenant("t", raw)
            self.assertEqual((cm.exception.reason, cm.exception.line),
                             ("missing", 1), raw)

    def test_input_illegal_utf8_is_missing_at_physical_line(self):
        raw = record_bytes(self.valid_row("t", 1)) + b'{"tenant":\xff}\n'
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.import_tenant("t", raw)
        self.assertEqual((cm.exception.seq, cm.exception.reason,
                          cm.exception.line), (2, "missing", 2))

    def test_input_mid_chain_gap_and_broken_prev(self):
        first = self.valid_row("t", 1)
        gap = self.valid_row("t", 3, prev=first["hash"])
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.import_tenant("t",
                                     record_bytes(first) + record_bytes(gap))
        self.assertEqual((cm.exception.seq, cm.exception.reason,
                          cm.exception.line), (2, "sequence", 2))
        bad_prev = self.valid_row("t", 2, prev="9" * 64)
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.import_tenant(
                "t", record_bytes(first) + record_bytes(bad_prev))
        self.assertEqual((cm.exception.seq, cm.exception.reason,
                          cm.exception.line), (2, "digest", 2))

    def test_input_error_priority_matches_existing_scan(self):
        # digest on line 1 beats bad bytes on line 2: store a hash computed
        # over the original event, then tamper the event
        tampered = self.valid_row("t", 1)
        tampered["event"] = {"z": 9}
        raw = record_bytes(tampered) + b"\xff\n"
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.import_tenant("t", raw)
        self.assertEqual((cm.exception.seq, cm.exception.reason,
                          cm.exception.line), (1, "digest", 1))
        # sequence on line 1 beats bad bytes on line 2
        wrong_seq = self.valid_row("t", 2)
        raw = record_bytes(wrong_seq) + b"\xff"
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.import_tenant("t", raw)
        self.assertEqual((cm.exception.seq, cm.exception.reason,
                          cm.exception.line), (1, "sequence", 1))
        # missing on line 1 beats a sequence defect later
        raw = b"{oops\n" + record_bytes(self.valid_row("t", 5))
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.import_tenant("t", raw)
        self.assertEqual((cm.exception.seq, cm.exception.reason,
                          cm.exception.line), (1, "missing", 1))

    def test_input_state_error_writes_no_partial_records(self):
        self.chain.append("other", {"keep": True})
        before = self.path.read_bytes()
        good1 = record_bytes(self.valid_row("t", 1))
        good2 = record_bytes(self.valid_row(
            "t", 2, prev=json.loads(good1)["hash"]))
        bad3 = self.valid_row("t", 4, prev=json.loads(good2)["hash"])
        with self.assertRaises(AuditChainStateError):
            self.chain.import_tenant("t",
                                     good1 + good2 + record_bytes(bad3))
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": 0})

    def test_input_state_error_takes_priority_over_target_conflict(self):
        # target already has t (would conflict) but the input chain itself is
        # broken: the input is validated first, so it is a state error
        self.chain.append("t", {})
        broken = record_bytes(self.valid_row("t", 2))
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.import_tenant("t", broken)
        self.assertEqual(cm.exception.reason, "sequence")

    # --- target conflict: empty-chain assertion only ---

    def test_target_with_existing_tenant_records_conflicts(self):
        first = self.chain.append("t", {"i": 0})
        second = self.chain.append("t", {"i": 1})
        before = self.path.read_bytes()
        _, data = self.make_export("t", 3,
                                   AuditChain(self.path.with_name("s.jsonl")))
        with self.assertRaises(AuditChainConflictError) as cm:
            self.chain.import_tenant("t", data)
        e = cm.exception
        self.assertEqual(e.tenant, "t")
        self.assertEqual(e.reason, "conflict")
        self.assertEqual(e.expected_count, 0)
        self.assertEqual(e.expected_hash, ZERO)
        self.assertEqual((e.actual_count, e.actual_hash),
                         (2, second["hash"]))
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": 2})

    def test_target_conflict_after_interleaving_reports_target_tail(self):
        a1 = self.chain.append("a", {})
        t1 = self.chain.append("t", {})
        self.chain.append("a", {})
        before = self.path.read_bytes()
        _, data = self.make_export("t", 1,
                                   AuditChain(self.path.with_name("s.jsonl")))
        with self.assertRaises(AuditChainConflictError) as cm:
            self.chain.import_tenant("t", data)
        e = cm.exception
        self.assertEqual((e.expected_count, e.expected_hash), (0, ZERO))
        self.assertEqual((e.actual_count, e.actual_hash), (1, t1["hash"]))
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(self.chain.verify("a"), {"ok": True, "count": 2})

    def test_foreign_tenant_digest_corruption_does_not_block_import(self):
        # mirrors export semantics: an unparseable line anywhere is fatal, but
        # another tenant's own digest break is invisible to t's scan
        bad_other = self.valid_row("b", 1)
        bad_other["event"] = {"x": 1}
        self.write_rows([bad_other])
        _, data = self.make_export("t", 2)
        records = self.chain.import_tenant("t", data)
        self.assertEqual(len(records), 2)
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": 2})

    # --- target corruption: same AuditChainStateError as append ---

    def test_corrupt_target_history_raises_state_error_and_writes_nothing(self):
        tampered = self.valid_row("t", 1)
        tampered["event"] = {"tampered": True}
        self.write_rows([tampered])
        before = self.path.read_bytes()
        _, data = self.make_export("t", 2,
                                   AuditChain(self.path.with_name("s.jsonl")))
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.import_tenant("t", data)
        # state error takes priority over what would otherwise be a conflict
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("t", 1, "digest", 1))
        self.assertEqual(self.path.read_bytes(), before)

    def test_unparseable_target_line_raises_missing_at_physical_line(self):
        other = self.valid_row("x", 1)
        self.write_rows([other, "{not json"])
        before = self.path.read_bytes()
        _, data = self.make_export("t", 1,
                                   AuditChain(self.path.with_name("s.jsonl")))
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.import_tenant("t", data)
        self.assertEqual((cm.exception.seq, cm.exception.reason,
                          cm.exception.line), (1, "missing", 2))
        self.assertEqual(self.path.read_bytes(), before)


class ImportConcurrencyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.path = Path(self.tmp.name) / "audit.jsonl"
        self.chain = AuditChain(self.path)
        src = AuditChain(self.path.with_name("source.jsonl"))
        for i in range(20):
            src.append("t", {"i": i})
        self.data = src.export_tenant("t")
        self.tail_hash = json.loads(self.data.decode().splitlines()[-1])["hash"]

    def tearDown(self):
        self.tmp.cleanup()

    def test_parallel_imports_one_wins_rest_conflict(self):
        successes, conflicts, others = [], [], []
        box = threading.Lock()

        def importer():
            try:
                records = self.chain.import_tenant("t", self.data)
                with box:
                    successes.append(len(records))
            except AuditChainConflictError as e:
                with box:
                    conflicts.append((e.expected_count, e.expected_hash,
                                      e.actual_count, e.actual_hash))
            except Exception as e:  # noqa: BLE001
                with box:
                    others.append(repr(e))

        threads = [threading.Thread(target=importer) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(others, [])
        self.assertEqual(successes, [20])
        self.assertEqual(len(conflicts), 7)
        for exp_count, exp_hash, act_count, act_hash in conflicts:
            self.assertEqual((exp_count, exp_hash), (0, ZERO))
            self.assertEqual((act_count, act_hash), (20, self.tail_hash))
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": 20})
        self.assertEqual(self.chain.export_tenant("t"), self.data)

    def test_import_indivisible_against_appends_and_readers(self):
        stop = threading.Event()
        problems = []
        box = threading.Lock()

        def reader():
            while not stop.is_set():
                r = self.chain.verify("t")
                if not r["ok"]:
                    with box:
                        problems.append(("verify", r))
                    return
                r2 = self.chain.verify_all()
                if not r2["ok"]:
                    with box:
                        problems.append(("verify_all", r2))
                    return
                # t must always be either absent or present as the full
                # 20-record graft; a partial prefix would mean a torn block
                if r["count"] not in (0, 20):
                    with box:
                        problems.append(("partial", r["count"]))
                    return

        readers = [threading.Thread(target=reader) for _ in range(4)]
        for t in readers:
            t.start()

        def other_writer(i):
            c = AuditChain(self.path)
            for j in range(50):
                c.append("other", {"w": i, "j": j})

        writers = [threading.Thread(target=other_writer, args=(i,))
                   for i in range(4)]
        outcomes = {"ok": 0, "conflict": 0}

        def importer():
            try:
                self.chain.import_tenant("t", self.data)
                with box:
                    outcomes["ok"] += 1
            except AuditChainConflictError:
                with box:
                    outcomes["conflict"] += 1

        importers = [threading.Thread(target=importer) for _ in range(6)]
        for t in writers + importers:
            t.start()
        for t in writers:
            t.join()
        for t in importers:
            t.join()
        stop.set()
        for t in readers:
            t.join(timeout=2)

        self.assertEqual(problems, [])
        self.assertEqual(outcomes["ok"], 1)
        self.assertEqual(outcomes["conflict"], 5)
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": 20})
        self.assertEqual(self.chain.verify("other"),
                         {"ok": True, "count": 200})
        self.assertTrue(self.chain.verify_all()["ok"])
        # the grafted bytes sit intact as one contiguous, re-exportable block
        self.assertEqual(self.chain.export_tenant("t"), self.data)
        self.assertTrue(self.chain.verify_bytes(
            self.path.read_bytes(), "t")["ok"])


if __name__ == "__main__":
    unittest.main()
