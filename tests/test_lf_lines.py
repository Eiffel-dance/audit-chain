"""Acceptance tests for the LF-only physical line boundary.

Only the byte 0x0A terminates a JSONL record; a CR immediately before it
(CRLF) is still one line. U+0085, U+2028, U+2029 and a bare CR are not line
boundaries: the Unicode separators stay on the current line as ordinary
strict-JSON string content (tenant values, events, object keys, arrays and
nested values), while a bare CR in a spot strict JSON forbids must surface as
a parse failure. Every entry point shares that one boundary, and physical
line numbers count LFs.
"""
import json
import os
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

from app import AuditChain, AuditChainStateError, ZERO

NEL = chr(0x85)
LS = chr(0x2028)
PS = chr(0x2029)
SEPARATORS = (NEL, LS, PS)


def dumps(obj):
    # keep the separator characters as raw UTF-8, the way an external writer
    # that did not ASCII-escape them would put them on the wire
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, allow_nan=False)


class LFLineBoundaryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.path = Path(self.tmp.name) / "audit.jsonl"
        self.chain = AuditChain(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    # ---- construction helpers ----------------------------------------

    def record(self, tenant, seq, event, prev):
        item = {"tenant": tenant, "seq": seq, "event": event, "prev": prev}
        item["hash"] = AuditChain._hash(item)
        return item

    def build_bytes(self, spec, nl=b"\n", trailing=True):
        """spec: list of (tenant, event). Weaves independent per-tenant
        chains and returns raw JSONL bytes with the chosen line ending."""
        state = {}
        chunks = []
        for tenant, event in spec:
            key = dumps(tenant)
            count, prev = state.get(key, (0, ZERO))
            item = self.record(tenant, count + 1, event, prev)
            chunks.append(dumps(item).encode("utf-8"))
            state[key] = (count + 1, item["hash"])
        ending = nl if trailing else b""
        return nl.join(chunks) + ending

    def every_split(self, data):
        # 2-way cuts at every byte offset (so every cut through the middle
        # of a multi-byte UTF-8 sequence is exercised), plus a bounded set
        # of 3-way cuts sampled on a stride to keep the suite fast.
        for cut in range(0, len(data) + 1):
            yield [data[:cut], data[cut:]]
        if len(data) >= 3:
            for cut1 in range(1, len(data) - 1, 5):
                for cut2 in range(cut1 + 1, len(data), 7):
                    yield [data[:cut1], data[cut1:cut2], data[cut2:]]

    # ---- raw separators are content, in every JSON position ----------

    def positions(self, ch):
        return {
            "tenant": ("t" + ch + "x", "e"),
            "event": ("t", "a" + ch + "b"),
            "object_key": ("t", {"k" + ch: 1}),
            "array": ("t", ["a", ch, 3, [ch]]),
            "nested": ("t", {"n": {"deep": [{"x": ch}]}}),
        }

    def test_raw_separator_in_every_position_is_one_record(self):
        for ch in SEPARATORS:
            for label, (tenant, event) in self.positions(ch).items():
                spec = [(tenant, event), (tenant, {"again": ch})]
                for nl in (b"\n", b"\r\n"):
                    data = self.build_bytes(spec, nl=nl)
                    # one physical line per record regardless of splitlines()
                    self.assertEqual(data.count(b"\n"), 2, (ch, label, nl))
                    self.assertGreaterEqual(
                        len(data.decode("utf-8").splitlines()), 2,
                        (ch, label, nl))
                    self.assertEqual(
                        self.chain.verify_bytes(data, tenant),
                        {"ok": True, "count": 2}, (ch, label, nl))
                    result = self.chain.verify_all_bytes(data)
                    self.assertTrue(result["ok"], (ch, label, nl, result))
                    self.assertEqual(result["tenants"][0]["count"], 2,
                                     (ch, label))

    def test_separator_value_round_trips_verbatim(self):
        for ch in SEPARATORS:
            data = self.build_bytes([("t" + ch, ["x" + ch])])
            self.path.write_bytes(data)
            records = self.chain.read_tenant("t" + ch)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["event"], ["x" + ch])
            self.assertEqual(records[0]["tenant"], "t" + ch)
            os.unlink(self.path)

    # ---- interleaving exotic tenants ---------------------------------

    def test_separator_tenants_interleave(self):
        a, b = "A" + LS, "B" + NEL
        spec = [(a, 1), (b, "x"), (a, 2), (b, "y"), (a, 3)]
        for nl in (b"\n", b"\r\n"):
            data = self.build_bytes(spec, nl=nl)
            self.assertEqual(
                self.chain.verify_bytes(data, a), {"ok": True, "count": 3})
            self.assertEqual(
                self.chain.verify_bytes(data, b), {"ok": True, "count": 2})
            result = self.chain.verify_all_bytes(data)
            self.assertTrue(result["ok"])
            self.assertEqual([t["tenant"] for t in result["tenants"]], [a, b])
            self.assertEqual([t["count"] for t in result["tenants"]], [3, 2])
            self.assertEqual(
                self.chain.verify_bytes(data, "C" + PS),
                {"ok": True, "count": 0})

    # ---- no final newline ---------------------------------------------

    def test_no_trailing_newline_still_one_physical_line(self):
        data = self.build_bytes([("t", "a"), ("t", "b")], trailing=False)
        self.assertFalse(data.endswith(b"\n"))
        self.assertEqual(self.chain.verify_bytes(data, "t"),
                         {"ok": True, "count": 2})
        self.assertTrue(self.chain.verify_all_bytes(data)["ok"])
        # file snapshot path: head/read/append all cope, and append inserts
        # exactly one separating LF without touching existing bytes
        self.path.write_bytes(data)
        self.assertEqual(self.chain.head("t")["count"], 2)
        before = self.path.read_bytes()
        item = self.chain.append("t", "c")
        after = self.path.read_bytes()
        # exactly one separating LF is inserted; appends always finish with
        # a trailing LF, which is appended after the new record too
        self.assertEqual(after, before + b"\n" + dumps(item).encode() + b"\n")
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": 3})

    # ---- CRLF ----------------------------------------------------------

    def test_crlf_and_mixed_endings(self):
        spec = [("t", 1), ("u", 1), ("t", 2)]
        items_state = {}
        lines = []
        for tenant, event in spec:
            key = dumps(tenant)
            count, prev = items_state.get(key, (0, ZERO))
            item = self.record(tenant, count + 1, event, prev)
            lines.append(dumps(item).encode("utf-8"))
            items_state[key] = (count + 1, item["hash"])
        data = lines[0] + b"\r\n" + lines[1] + b"\n" + lines[2] + b"\r\n"
        self.assertEqual(data.count(b"\n"), 3)
        self.assertEqual(self.chain.verify_bytes(data, "t"),
                         {"ok": True, "count": 2})
        self.assertEqual(self.chain.verify_bytes(data, "u"),
                         {"ok": True, "count": 1})
        self.assertTrue(self.chain.verify_all_bytes(data)["ok"])
        self.path.write_bytes(data)
        self.assertEqual(len(self.chain.read_tenant("t")), 2)
        # appending keeps CRLF bytes intact and verifies afterwards
        self.chain.append("u", 2)
        self.assertTrue(self.path.read_bytes().startswith(lines[0] + b"\r\n"))
        self.assertEqual(self.chain.verify_all()["ok"], True)

    # ---- arbitrary byte chunking --------------------------------------

    def test_chunks_split_through_multibyte_separators(self):
        data = self.build_bytes(
            [("t" + LS, [NEL]), ("u", {"k" + PS: 1}),
             ("t" + LS, "z" + NEL + PS)])
        # sanity: the bytes really carry raw multi-byte sequences
        self.assertIn(LS.encode(), data)
        for chunks in self.every_split(data):
            self.assertEqual(
                self.chain.verify_all_chunks(iter(chunks))["ok"], True,
                chunks)
            self.assertEqual(
                self.chain.verify_chunks(iter(chunks), "t" + LS),
                {"ok": True, "count": 2}, chunks)
            self.assertEqual(
                self.chain.verify_chunks(iter(chunks), "u"),
                {"ok": True, "count": 1}, chunks)
            # empty chunks may appear anywhere
            padded = [b""] + sum(([c, b""] for c in chunks), [])
            self.assertTrue(
                self.chain.verify_all_chunks(iter(padded))["ok"])

    # ---- bare CR is illegal wherever strict JSON forbids it -----------

    def test_bare_cr_inside_string_is_missing(self):
        good = self.build_bytes([("t", 1)])
        # build the offending record explicitly with a valid-looking hash
        first = json.loads(good)
        bad = {"tenant": "t", "seq": 2, "event": "a\rb",
               "prev": first["hash"]}
        bad["hash"] = AuditChain._hash(bad)
        line = (json.dumps(bad, ensure_ascii=False)
                .replace("\\r", "\r")).encode("utf-8") + b"\n"
        self.assertIn(b"\r", line)
        data = good + line
        r = self.chain.verify_bytes(data, "t")
        self.assertEqual((r["ok"], r["at"], r["reason"]),
                         (False, 2, "missing"))
        self.assertEqual(self.chain.verify_all_bytes(data),
                         {"ok": False, "at": 2, "tenant": None,
                          "reason": "missing"})

    def test_bare_cr_between_records_without_lf_is_missing(self):
        # CR is whitespace to JSON, so two objects separated only by a CR on
        # one LF physical line are extra-data on that single line -> missing
        one = dumps(self.record("t", 1, {}, ZERO)).encode()
        two = dumps(self.record("t", 2, {}, "x" * 64)).encode()
        data = one + b"\r" + two + b"\n"
        self.assertEqual(data.count(b"\n"), 1)
        r = self.chain.verify_bytes(data, "t")
        self.assertEqual((r["ok"], r["at"], r["reason"]),
                         (False, 1, "missing"))

    # ---- separators outside a structure are structural garbage -------

    def test_separator_outside_structure_is_missing(self):
        one = dumps(self.record("t", 1, {}, ZERO)).encode()
        other = dumps(self.record("t", 2, {}, "x" * 64)).encode()
        for ch in SEPARATORS:
            sep = ch.encode("utf-8")
            # two valid objects joined only by a separator: one LF line now,
            # strict JSON must reject it (previously splitlines() wrongly made
            # it two parseable lines)
            data = one + sep + other + b"\n"
            r = self.chain.verify_bytes(data, "t")
            self.assertEqual((r["ok"], r["at"], r["reason"]),
                             (False, 1, "missing"), ch)
            # trailing separator after one object: extra data
            data = one + sep + b"\n"
            r = self.chain.verify_bytes(data, "t")
            self.assertEqual((r["at"], r["reason"]), (1, "missing"), ch)
            # a separator-only physical line of its own
            data = one + b"\n" + sep + b"\n"
            r = self.chain.verify_bytes(data, "t")
            self.assertEqual((r["at"], r["reason"]), (2, "missing"), ch)

    # ---- first-error priority, lines counted by LF --------------------

    def test_first_error_priority_with_separator_records(self):
        good1 = self.build_bytes([("t" + LS, LS), ("o", 1), ("t" + LS, 2)])
        exotic_line = good1.count(b"\n")
        self.assertEqual(exotic_line, 3)
        good_records = [json.loads(l) for l in
                        good1.decode("utf-8").split("\n") if l]
        tail = good_records[-1]["hash"]
        # For a tenant-specific verify_bytes, "at" is the expected seq (3);
        # for verify_all_bytes it is the LF physical line (4). A wrong seq
        # masks a wrong prev, so the digest case must carry seq == expected.
        cases = [
            ("sequence", self.record("t" + LS, 9, {}, tail), 3),
            ("digest", self.record("t" + LS, 3, {}, "9" * 64), 3),
            ("missing", None, 4),
        ]
        for reason, item, tenant_at in cases:
            if item is None:
                rest = b"{not json\n"
            else:
                rest = dumps(item).encode() + b"\n"
            r = self.chain.verify_bytes(good1 + rest, "t" + LS)
            self.assertEqual((r["ok"], r["at"], r["reason"]),
                             (False, tenant_at, reason), reason)
            # every failure is located on LF physical line 4, even though
            # str.splitlines() counts more lines because of the raw chars
            self.assertEqual(
                self.chain.verify_all_bytes(good1 + rest),
                {"ok": False, "at": 4,
                 "tenant": None if item is None else "t" + LS,
                 "reason": reason}, reason)

    def test_earlier_corruption_beats_later_separator_record(self):
        # line 1 has a bad digest; a perfectly valid raw-LS record sits on
        # LF line 2 -- the line-1 defect still wins
        tampered = {"tenant": "t", "seq": 1, "event": {"z": 9}, "prev": ZERO}
        tampered["hash"] = AuditChain._hash(
            {"tenant": "t", "seq": 1, "event": {}, "prev": ZERO})
        line1 = dumps(tampered).encode() + b"\n"
        line2 = self.build_bytes([("t", "a" + LS + "b")])
        r = self.chain.verify_bytes(line1 + line2, "t")
        self.assertEqual((r["at"], r["reason"]), (1, "digest"))

    def test_illegal_utf8_line_number_counts_lf(self):
        # raw NEL makes splitlines() see more lines than there are; the bad
        # UTF-8 byte's location must still be the LF physical line
        good = self.build_bytes([("t", "a" + NEL), ("t", "b" + LS)])
        self.assertEqual(good.count(b"\n"), 2)
        data = good + b"{bad:\xff}\n"
        r = self.chain.verify_bytes(data, "t")
        self.assertEqual((r["ok"], r["at"], r["reason"]),
                         (False, 3, "missing"))
        self.assertEqual(self.chain.verify_all_bytes(data),
                         {"ok": False, "at": 3, "tenant": None,
                          "reason": "missing"})

    # ---- same chain rules as equivalent ASCII content -----------------

    def test_exotic_content_follows_same_chain_as_ascii_equivalent(self):
        ascii_spec = [("t", ["a"]), ("t", {"k": 2}), ("o", 1)]
        exotic_spec = [("t", ["a" + LS]), ("t", {"k" + NEL: 2}),
                       ("o" + PS, 1)]
        ascii_data = self.build_bytes(ascii_spec)
        exotic_data = self.build_bytes(exotic_spec)
        for data, tenant, count in ((ascii_data, "t", 2),
                                    (exotic_data, "t", 2),
                                    (exotic_data, "o" + PS, 1)):
            self.assertEqual(self.chain.verify_bytes(data, tenant),
                             {"ok": True, "count": count})
        # the reported head drives a successful conditional append
        import tempfile
        p = Path(tempfile.mkdtemp()) / "x.jsonl"
        try:
            target = AuditChain(p)
            p.write_bytes(exotic_data)
            h = target.head("t")
            self.assertEqual(h["count"], 2)
            item = target.append_if_head(
                "t", "next" + PS, h["count"], h["hash"])
            self.assertEqual(item["seq"], 3)
            self.assertEqual(target.verify("t"), {"ok": True, "count": 3})
            # stale head now conflicts
            with self.assertRaises(Exception):
                target.append_if_head("t", "x", h["count"], h["hash"])
        finally:
            import shutil
            shutil.rmtree(p.parent, ignore_errors=True)

    def test_tampering_exotic_event_is_digest(self):
        data = bytearray(self.build_bytes([("t", "a" + LS + "b")]))
        # flip a byte inside the raw LS sequence's content region: simpler
        # and deterministic -- replace the event letter while keeping length
        text = bytes(data).decode("utf-8").replace("a" + LS + "b",
                                                   "a" + LS + "c")
        r = self.chain.verify_bytes(text.encode("utf-8"), "t")
        self.assertEqual((r["at"], r["reason"]), (1, "digest"))

    # ---- read / head / export over externally written files -----------

    def test_file_reads_and_exports_handle_raw_separators(self):
        spec = [("t" + NEL, "e1" + LS), ("u", ["x" + PS]),
                ("t" + NEL, {"k": LS})]
        data = self.build_bytes(spec, nl=b"\r\n")
        self.path.write_bytes(data)
        records = self.chain.read_tenant("t" + NEL)
        self.assertEqual([r["seq"] for r in records], [1, 2])
        self.assertEqual(records[0]["event"], "e1" + LS)
        page = self.chain.read_tenant("t" + NEL, start_seq=2, page_size=1)
        self.assertEqual(page[0]["event"], {"k": LS})
        self.assertEqual(self.chain.head("t" + NEL)["count"], 2)
        self.assertEqual(self.chain.heads()["tenants"][0]["tenant"], "t" + NEL)
        self.assertTrue(self.chain.verify_all()["ok"])
        # export_tenant canonicalizes (ASCII-escapes) but stays equivalent
        exported = self.chain.export_tenant("t" + NEL)
        self.assertEqual(
            self.chain.verify_bytes(exported, "t" + NEL),
            {"ok": True, "count": 2})
        for line in exported.decode("utf-8").split("\n"):
            if line:
                json.loads(line)  # strict standard JSON
        # export_all returns the original snapshot bytes unchanged, including
        # CRLF endings and raw separator characters
        self.assertEqual(self.chain.export_all(), data)

    # ---- imports: acceptance, first broken LF line, no partial write --

    def test_import_paths_accept_raw_separators_and_crlf(self):
        spec = [("t", "a" + LS), ("t", ["b" + NEL, {"k" + PS: 1}])]
        data = self.build_bytes(spec, nl=b"\r\n")
        self.chain.import_tenant("t", data)
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": 2})
        self.assertEqual(self.chain.read_tenant("t")[0]["event"], "a" + LS)

        target = AuditChain(self.path.with_name("all.jsonl"))
        interleaved = self.build_bytes(
            [("a" + LS, 1), ("b" + NEL, 2), ("a" + LS, 3)], nl=b"\r\n")
        target.import_all(interleaved)
        self.assertEqual(target.verify_all()["ok"], True)

    def test_import_chunks_arbitrary_splits_roundtrip(self):
        data = self.build_bytes(
            [("t", NEL), ("t", LS), ("t", PS)])
        for chunks in self.every_split(data):
            p = self.path.with_name(f"c{len(chunks)}_{chunks[0][:1]!r}.jsonl")
            target = AuditChain(p)
            target.import_tenant_chunks("t", iter(chunks))
            self.assertEqual(target.verify("t"), {"ok": True, "count": 3})
            os.unlink(p)

    def test_cross_file_import_locates_first_broken_lf_line(self):
        # raw separators make str.splitlines() over-count lines; the broken
        # record sits on LF physical line 3 even though splitlines sees 5
        good = self.build_bytes([("t", "a" + LS), ("t", NEL),
                                 ("t", PS)])
        broken = self.record("t", 5, {}, "dead")
        payload = good + dumps(broken).encode() + b"\n"
        self.assertGreater(len(payload.decode().splitlines()), 4)
        self.assertEqual(payload.count(b"\n"), 4)
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.import_tenant("t", payload)
        self.assertEqual(cm.exception.reason, "sequence")
        self.assertEqual(cm.exception.line, 4)
        self.assertFalse(self.path.exists())  # nothing created on failure

        # same for import_all, and no partial content on a pre-existing file
        self.chain.append("other", 1)
        before = self.path.read_bytes()
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.import_all(payload)
        self.assertEqual(cm.exception.line, 4)
        self.assertEqual(self.path.read_bytes(), before)

        # an unparseable exotic-content line is missing at its LF line
        bad_utf8 = good + b"{\xff}\n"
        with self.assertRaises(AuditChainStateError) as cm:
            AuditChain(self.path.with_name("m.jsonl")).import_all(bad_utf8)
        self.assertEqual((cm.exception.reason, cm.exception.line),
                         ("missing", 4))

    def test_bad_input_import_never_writes_partial_bytes(self):
        good = self.build_bytes([("t", LS)])
        cases = [
            good + b"{not json\n",                       # missing at 2
            good + b'{"tenant":"t","seq":3}\n',         # missing fields at 2
            b'{"tenant":"x"' + LS.encode() + b':1}\n',  # garbage line
        ]
        for i, payload in enumerate(cases):
            p = self.path.with_name(f"fail{i}.jsonl")
            target = AuditChain(p)
            with self.assertRaises(AuditChainStateError):
                target.import_all(payload)
            self.assertFalse(p.exists(), i)

    # ---- append-side normalization stays ASCII/LF ----------------------

    def test_append_normalizes_separators_to_escaped_lf_jsonl(self):
        item = self.chain.append("t" + LS, {"e": NEL})
        raw = self.path.read_bytes()
        self.assertTrue(raw.endswith(b"\n"))
        self.assertNotIn(LS.encode(), raw)
        self.assertNotIn(NEL.encode(), raw)
        self.assertIn(b"\\u2028", raw)
        # the on-disk escaped record parses back to the exact value
        parsed = json.loads(raw)
        self.assertEqual(parsed["tenant"], item["tenant"])
        self.assertEqual(parsed["event"], item["event"])
        self.assertEqual(self.chain.verify_all()["ok"], True)

    def test_crlf_snapshot_import_all_then_export_all_verbatim(self):
        data = self.build_bytes([("a", LS), ("b", NEL), ("a", PS)],
                                nl=b"\r\n")
        target = AuditChain(self.path.with_name("raw.jsonl"))
        target.import_all(data)
        # import_all re-serializes canonically (LF endings); that verifies
        self.assertTrue(target.verify_all()["ok"])
        # but export_all over an untouched raw snapshot stays byte-for-byte
        raw_path = self.path.with_name("snapshot.jsonl")
        raw_path.write_bytes(data)
        self.assertEqual(AuditChain(raw_path).export_all(), data)

    # ---- blank CRLF lines and stray CR-only lines ---------------------

    def test_blank_or_cr_only_lines_are_missing(self):
        good = self.build_bytes([("t", 1)])
        for garbage in (b"\r\n", b"\n", b"   \r\n", b"\t\r\n"):
            data = good + garbage
            r = self.chain.verify_bytes(data, "t")
            self.assertEqual((r["at"], r["reason"]), (2, "missing"), garbage)
            self.assertEqual(
                self.chain.verify_all_bytes(data)["at"], 2, garbage)

    # ---- appends against raw CRLF/exotic snapshots --------------------

    def test_batch_and_many_append_continue_raw_snapshots(self):
        data = self.build_bytes(
            [("t" + LS, "a" + NEL), ("u", 1), ("t" + LS, "b" + PS)],
            nl=b"\r\n")
        self.path.write_bytes(data)
        items = self.chain.append_batch("t" + LS, ["c", ["d" + LS]])
        self.assertEqual([i["seq"] for i in items], [3, 4])
        items = self.chain.append_many([
            {"tenant": "u", "event": "2" + PS},
            {"tenant": "v" + NEL, "event": {}},
        ])
        self.assertEqual([i["seq"] for i in items], [2, 1])
        self.assertTrue(self.chain.verify_all()["ok"])
        # the pre-existing raw CRLF bytes are untouched at the front
        self.assertTrue(self.path.read_bytes().startswith(data))
        # the new block is normalized JSON + LF, no raw separators
        tail = self.path.read_bytes()[len(data):]
        self.assertNotIn(LS.encode(), tail)
        self.assertNotIn(PS.encode(), tail)
        self.assertTrue(tail.endswith(b"\n"))

    # ---- segmented migration carries the boundary ---------------------

    def test_range_export_import_with_raw_separators(self):
        spec = [("t", "a" + LS), ("t", "b" + NEL),
                ("t", ["c" + PS]), ("t", {"k": LS})]
        data = self.build_bytes(spec, nl=b"\r\n")
        src = AuditChain(self.path.with_name("src.jsonl"))
        src.path.write_bytes(data)
        segment = src.export_tenant_range("t", 3)
        # source bytes are CRLF raw; the export re-serializes canonically but
        # keeps seq/prev/hash values. A mid-chain segment is not a standalone
        # chain (seqs start at 3), so its contents are inspected directly and
        # validated through the graft below.
        rows = [json.loads(l) for l in segment.decode().split("\n") if l]
        self.assertEqual([r["seq"] for r in rows], [3, 4])
        self.assertEqual(rows[0]["event"], ["c" + PS])

        # graft the segment onto a target that holds exactly records 1..2,
        # asserted by head; CRLF input into import_tenant_range works too
        head2 = AuditChain(self.path.with_name("h2.jsonl"))
        head2.import_tenant("t", self.build_bytes(spec[:2]))
        h = head2.head("t")
        crlf_segment = segment.replace(b"\n", b"\r\n")
        records = head2.import_tenant_range(
            "t", crlf_segment, h["count"], h["hash"])
        self.assertEqual([r["seq"] for r in records], [3, 4])
        self.assertEqual(head2.verify("t"), {"ok": True, "count": 4})

    def test_import_all_range_with_separator_tenants(self):
        a, b = "a" + LS, "b" + NEL
        base = self.build_bytes([(a, 1), (b, 1), (a, 2), (b, 2)])
        target = AuditChain(self.path.with_name("base.jsonl"))
        target.path.write_bytes(base)
        ha, hb = target.head(a), target.head(b)
        # build the continuing segments with the right seqs/prevs by exporting
        # them from an equivalent raw source
        src = AuditChain(self.path.with_name("full.jsonl"))
        full = self.build_bytes(
            [(a, 1), (b, 1), (a, 2), (b, 2), (b, "x" + PS),
             (a, ["y" + NEL])], nl=b"\r\n")
        src.path.write_bytes(full)
        seg_a = src.export_tenant_range(a, 3)
        seg_b = src.export_tenant_range(b, 3)
        stream = seg_b.replace(b"\n", b"\r\n") + seg_a.replace(b"\n", b"\r\n")
        records = target.import_all_range(stream, [
            {"tenant": b, "expected_count": hb["count"],
             "expected_hash": hb["hash"]},
            {"tenant": a, "expected_count": ha["count"],
             "expected_hash": ha["hash"]},
        ])
        self.assertEqual([(r["tenant"], r["seq"]) for r in records],
                         [(b, 3), (a, 3)])
        self.assertTrue(target.verify_all()["ok"])
        self.assertEqual(target.read_tenant(a)[-1]["event"], ["y" + NEL])

    # ---- chunked export over raw snapshots -----------------------------

    def test_chunked_exports_reassemble_to_raw_snapshot(self):
        data = self.build_bytes(
            [("t" + NEL, LS), ("u", PS), ("t" + NEL, "x")], nl=b"\r\n")
        raw_path = self.path.with_name("raw.jsonl")
        raw_path.write_bytes(data)
        src = AuditChain(raw_path)
        for size in (1, 2, 3, 7, 1000):
            chunks = list(src.export_all_chunks(size))
            self.assertEqual(b"".join(chunks), data, size)
        chunks = list(src.export_tenant_chunks("t" + NEL, 5))
        exported = src.export_tenant("t" + NEL)
        self.assertEqual(b"".join(chunks), exported)
        self.assertEqual(
            self.chain.verify_bytes(b"".join(chunks), "t" + NEL),
            {"ok": True, "count": 2})

    # ---- read paging and heads over exotic interleaved CRLF ------------

    def test_paging_and_heads_over_crlf_exotic_snapshot(self):
        spec = [("a" + LS, 1), ("b" + NEL, 1), ("a" + LS, 2),
                ("b" + NEL, 2), ("a" + LS, 3)]
        self.path.write_bytes(self.build_bytes(spec, nl=b"\r\n"))
        self.assertEqual(
            [r["seq"] for r in self.chain.read_tenant("a" + LS)], [1, 2, 3])
        self.assertEqual(
            self.chain.read_tenant("a" + LS, start_seq=2, page_size=1)[0]["seq"],
            2)
        heads = self.chain.heads()
        self.assertEqual(
            [(t["tenant"], t["count"]) for t in heads["tenants"]],
            [("a" + LS, 3), ("b" + NEL, 2)])
        # conditional batch append from the reported exotic-tenant head
        entry = heads["tenants"][0]
        self.chain.append_batch_if_head(
            entry["tenant"], ["z"], entry["count"], entry["hash"])
        self.assertEqual(self.chain.verify(entry["tenant"]),
                         {"ok": True, "count": 4})


if __name__ == "__main__":
    unittest.main()
