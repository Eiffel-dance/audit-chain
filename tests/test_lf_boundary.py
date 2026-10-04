"""LF-only physical record boundary.

Only the byte 0x0A ends a record: a CR immediately before it (CRLF) is still
one line, and U+0085, U+2028, U+2029 and a bare CR are never boundaries --
raw inside a JSON string they are UTF-8 content of the current line handed to
strict JSON, while outside any string they make strict JSON fail (missing).
"""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app import AuditChain, AuditChainStateError, ZERO

SPECIAL = ("\x85", " ", " ")


def row(tenant="t", seq=1, prev=ZERO, event=None):
    item = {"tenant": tenant, "seq": seq, "event": event if event is not None else {},
            "prev": prev}
    item["hash"] = AuditChain._hash(item)
    return item


def lf_line(item, newline=b"\n"):
    body = json.dumps(item, sort_keys=True, allow_nan=False,
                      ensure_ascii=False).encode("utf-8")
    return body + newline


def ascii_line(item):
    # The bytes append() itself writes (ensure_ascii escapes U+2028 etc.).
    return (json.dumps(item, sort_keys=True, allow_nan=False) + "\n").encode()


class LFBoundaryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.path = Path(self.tmp.name) / "audit.jsonl"
        self.chain = AuditChain(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def write_raw(self, data):
        self.path.write_bytes(data)

    # --- raw special characters are content, not line boundaries ----------

    def test_special_chars_in_tenant_event_key_array_nested(self):
        tenants = ["a" + ch + "b" for ch in SPECIAL] + [
            ["x", ch] for ch in SPECIAL
        ] + [{"k" + ch: 1} for ch in SPECIAL]
        events = []
        for ch in SPECIAL:
            events.extend([
                "e" + ch + "f",                      # event string
                {"key" + ch: "v"},                  # object key
                ["arr", ch, 1, None],               # array
                {"nested": {"k": ["deep", ch]}},    # nested value
                [ch, {"x": [ch]}],
                {"mix": ["a" + ch, {"k" + ch: ch}]},
            ])
        records = []
        data = b""
        seq_for = {}
        prev_for = {}
        for i, tenant in enumerate(tenants):
            for j, event in enumerate(events):
                key = AuditChain._tenant_key(tenant)
                seq = seq_for.get(key, 0) + 1
                prev = prev_for.get(key, ZERO)
                item = row(tenant, seq, prev, event)
                seq_for[key] = seq
                prev_for[key] = item["hash"]
                records.append(item)
                data += lf_line(item)
        # the whole stream is one object per LF line despite the raw chars
        self.assertEqual(data.count(b"\n"), len(records))
        self.write_raw(data)
        self.assertTrue(self.chain.verify_all()["ok"])
        for tenant in tenants:
            key = AuditChain._tenant_key(tenant)
            self.assertEqual(self.chain.verify(tenant),
                             {"ok": True, "count": len(events)})
            got = self.chain.read_tenant(tenant)
            self.assertEqual([r["event"] for r in got], events)
            self.assertEqual(self.chain.head(tenant)["hash"], prev_for[key])
        # verify_all_bytes and export_all agree on the raw snapshot
        self.assertTrue(self.chain.verify_all_bytes(data)["ok"])
        self.assertEqual(self.chain.export_all(), data)

    def test_raw_and_ascii_escaped_spellings_share_one_chain(self):
        # Same logical value, two byte spellings: raw UTF-8 vs the \uXXXX
        # escaping append itself writes. Digests/counts must be identical.
        for ch in SPECIAL:
            tenant, event = "t" + ch, {"s": "v" + ch}
            r1 = row(tenant, 1, ZERO, event)
            raw = lf_line(r1)                       # raw U+0085/U+2028/U+2029
            asc = ascii_line(r1)                    # \uXXXX escaped
            self.assertNotEqual(raw, asc)           # bytes differ ...
            self.assertEqual(
                AuditChain(self.path.with_name("r.jsonl")).verify_bytes(raw, tenant),
                AuditChain(self.path.with_name("a.jsonl")).verify_bytes(asc, tenant))
            # append() of the same value produces the escaped spelling and the
            # exact same hash as the raw-spelled record
            self.write_raw(b"")
            produced = self.chain.append(tenant, event)
            self.assertEqual(produced["hash"], r1["hash"])
            self.assertEqual(self.path.read_bytes(), asc)
            # a raw-spelled second record links straight onto it
            r2 = row(tenant, 2, r1["hash"], event)
            with self.path.open("ab") as f:
                f.write(lf_line(r2))
            self.assertEqual(self.chain.verify(tenant), {"ok": True, "count": 2})

    def test_tenants_carrying_special_chars_stay_partitioned_and_interleave(self):
        ta, tb = "x" + SPECIAL[1], "y" + SPECIAL[2]
        data = lf_line(row(ta, 1)) + lf_line(row(tb, 1)) \
            + lf_line(row(ta, 2, AuditChain._hash(row(ta, 1))))
        self.write_raw(data)
        self.assertEqual(self.chain.verify(ta), {"ok": True, "count": 2})
        self.assertEqual(self.chain.verify(tb), {"ok": True, "count": 1})
        self.assertEqual(
            [t["tenant"] for t in self.chain.verify_all()["tenants"]], [ta, tb])
        # exports never mix the two identities
        exported_a = self.chain.export_tenant(ta)
        self.assertEqual(exported_a.count(b"\n"), 2)
        self.assertTrue(
            AuditChain(self.path.with_name("o.jsonl")).verify_bytes(exported_a, ta)["ok"])

    def test_line_with_several_special_chars_is_one_record(self):
        # One LF line packed with all three characters; universal-newline
        # splitting would have fragmented it into several lines.
        event = SPECIAL[0] + SPECIAL[1] + SPECIAL[2] + SPECIAL[1]
        item = row("t", 1, ZERO, {"k": [event, {"j": event}]})
        self.write_raw(lf_line(item) + lf_line(row("t", 2, item["hash"])))
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": 2})
        self.assertEqual(
            self.chain.read_tenant("t")[0]["event"], {"k": [event, {"j": event}]})

    # --- CRLF is one line ---------------------------------------------------

    def test_crlf_terminated_history_is_valid(self):
        a1, a2 = row("a", 1), None
        data = lf_line(a1, b"\r\n")
        a2 = row("a", 2, a1["hash"], {"v": 1})
        data += lf_line(a2, b"\r\n")
        data += lf_line(row("b", 1), b"\r\n")  # interleaved other tenant
        self.write_raw(data)
        self.assertEqual(self.chain.verify("a"), {"ok": True, "count": 2})
        self.assertEqual(self.chain.verify("b"), {"ok": True, "count": 1})
        self.assertTrue(self.chain.verify_all()["ok"])
        self.assertEqual(
            [r["seq"] for r in self.chain.read_tenant("a")], [1, 2])
        # export_all preserves the original bytes of an existing snapshot
        self.assertEqual(self.chain.export_all(), data)
        # in-memory entries agree
        self.assertTrue(self.chain.verify_all_bytes(data)["ok"])
        self.assertEqual(self.chain.verify_bytes(data, "a"),
                         {"ok": True, "count": 2})
        # export_tenant re-emits canonical LF records (CR normalized away by
        # parse/reserialize), still verifiable as the same chain
        exported = self.chain.export_tenant("a")
        self.assertNotIn(b"\r", exported)
        other = AuditChain(self.path.with_name("o.jsonl"))
        self.assertEqual(other.import_tenant("a", exported)[0]["hash"], a1["hash"])

    def test_crlf_without_final_newline(self):
        item = row("t", 1)
        self.write_raw(json.dumps(item, ensure_ascii=False).encode() + b"\r")
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": 1})
        self.assertEqual(self.chain.read_tenant("t")[0]["hash"], item["hash"])
        # appending after an LF-less tail inserts exactly one healing LF
        nxt = self.chain.append("t", {})
        self.assertEqual((nxt["seq"], nxt["prev"]), (2, item["hash"]))
        self.assertTrue(self.chain.verify("t")["ok"])

    def test_lf_without_final_newline(self):
        item = row("t", 1, ZERO, {"s": "x" + SPECIAL[1]})
        self.write_raw(lf_line(item)[:-1])  # strip the terminating LF
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": 1})
        self.assertTrue(self.chain.verify_all_bytes(lf_line(item)[:-1])["ok"])
        nxt = self.chain.append("t", {})
        self.assertEqual(nxt["prev"], item["hash"])
        self.assertTrue(self.chain.verify("t")["ok"])

    # --- bare CR and other structural separators are never boundaries ------

    def test_bare_carriage_return_is_missing_not_a_separator(self):
        good_body = json.dumps(row("t", 1), ensure_ascii=False).encode()
        # CR between two JSON objects on a single physical LF line: under
        # splitlines this would look like two records; it is one broken line
        r = self.chain.verify_bytes(good_body + b"\r" + good_body + b"\n", "t")
        self.assertEqual((r["ok"], r["at"], r["reason"]),
                         (False, 1, "missing"))
        # raw CR inside a JSON string is a strict-JSON parse failure
        raw_cr = good_body.replace(b'"event": {}', b'"event": "a\rb"', 1)
        r = self.chain.verify_bytes(raw_cr + b"\n", "t")
        self.assertEqual((r["at"], r["reason"]), (1, "missing"))
        # CR junk after an otherwise valid object, CRLF-less
        r = self.chain.verify_bytes(good_body + b"\rjunk\n", "t")
        self.assertEqual((r["at"], r["reason"]), (1, "missing"))
        # a lone CR-only file has zero LF terminators but is one unterminated
        # line that strict JSON rejects
        r = self.chain.verify_bytes(b"\r", "t")
        self.assertEqual((r["at"], r["reason"]), (1, "missing"))

    def test_other_unicode_separators_outside_json_are_missing(self):
        # U+0085 / U+2028 / U+2029 are not JSON whitespace outside a string
        for ch in SPECIAL:
            body = json.dumps(row("t", 1), ensure_ascii=False).encode("utf-8")
            bad = body + ch.encode("utf-8") + b"\n"  # trailing junk, one LF line
            r = self.chain.verify_bytes(bad, "t")
            self.assertEqual((r["at"], r["reason"]), (1, "missing"), ch)
        for sep in ("\x0b", "\x0c"):
            bad = (json.dumps(row("t", 1)) + sep + "\n").encode()
            r = self.chain.verify_bytes(bad, "t")
            self.assertEqual((r["at"], r["reason"]), (1, "missing"), sep)

    # --- first-error priority with special chars present --------------------

    def test_first_error_priority_counts_lf_lines(self):
        ch = SPECIAL[1]
        # line 1: a valid LF line that itself contains U+2028/U+0085 (would be
        # three lines under splitlines); line 2: digest-broken -> line 2
        good = row("t", 1, ZERO, {"s": ch + SPECIAL[0]})
        broken = row("t", 2, good["hash"])
        broken["event"] = {"tampered": True}  # hash was over event {}
        data = lf_line(good) + lf_line(broken)
        self.write_raw(data)
        # verify_all reports the failed result (it never raises); line 2 is
        # the digest defect even though line 1's raw chars are several lines
        # under splitlines
        self.assertEqual(self.chain.verify_all(),
                         {"ok": False, "at": 2, "tenant": "t", "reason": "digest"})
        r = self.chain.verify("t")
        self.assertEqual((r["at"], r["reason"]), (2, "digest"))
        # the raising read entries locate the same LF line
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.read_tenant("t")
        self.assertEqual((cm.exception.tenant, cm.exception.seq,
                          cm.exception.reason, cm.exception.line),
                         ("t", 2, "digest", 2))
        # damage on the *earlier* line wins even though a special-char line
        # follows it
        tampered = row("t", 1, ZERO, {"x": 1})
        tampered["hash"] = AuditChain._hash(row("t", 1))  # hash of event {}
        later_ok = row("t", 2, tampered["hash"], {"s": ch})
        data = lf_line(tampered) + lf_line(later_ok)
        r = AuditChain(self.path.with_name("m.jsonl")).verify_bytes(data, "t")
        self.assertEqual((r["at"], r["reason"]), (1, "digest"))
        # unparseable special-char-free line beats a later digest defect
        r = self.chain.verify_bytes(b"{oops\n" + lf_line(broken), "t")
        self.assertEqual((r["at"], r["reason"]), (1, "missing"))

    def test_illegal_utf8_line_number_is_lf_physical_line(self):
        # earlier lines carry raw U+2028 (would inflate splitlines counts);
        # the illegal byte sits on LF line 3
        item1 = row("t", 1, ZERO, {"a": SPECIAL[1] + SPECIAL[2]})
        item2 = row("t", 2, item1["hash"], {"b": SPECIAL[0]})
        data = lf_line(item1) + lf_line(item2) + b'{"broken":\xff}'
        r = self.chain.verify_bytes(data, "t")
        self.assertEqual((r["at"], r["reason"]), (3, "missing"))
        self.write_raw(data)
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.head("t")
        self.assertEqual((cm.exception.reason, cm.exception.line),
                         ("missing", 3))
        # earlier digest error still takes priority over the bad UTF-8 line
        tampered = row("t", 1, ZERO, {"x": 1})
        tampered["hash"] = AuditChain._hash(row("t", 1))
        data = lf_line(tampered) + lf_line(item2) + b"\xff\n"
        r = self.chain.verify_bytes(data, "t")
        self.assertEqual((r["at"], r["reason"]), (1, "digest"))

    # --- arbitrary byte chunking -------------------------------------------

    def test_verify_chunks_at_every_byte_split(self):
        items = []
        for i in range(4):
            items.append(row("t", i + 1,
                             items[-1]["hash"] if items else ZERO,
                             {"s": SPECIAL[i % len(SPECIAL)] + "crlf"}))
        # mix CRLF and LF terminators and drop the final newline
        data = b"".join(lf_line(it, b"\r\n" if i % 2 else b"\n")
                        for i, it in enumerate(items))
        self.assertIn(b"\r\n", data)
        for cut in range(len(data) + 1):
            chunks = [data[:cut], data[cut:]] if cut else [data]
            r = self.chain.verify_chunks(chunks, "t")
            self.assertEqual(r, {"ok": True, "count": 4}, cut)
            self.assertTrue(self.chain.verify_all_chunks(
                [data[:cut], data[cut:]])["ok"], cut)
            # finer 3-way splits at a sample of positions
        for c1 in range(0, len(data), 7):
            for c2 in range(c1, len(data), 13):
                chunks = [data[:c1], data[c1:c2], data[c2:]]
                self.assertTrue(
                    self.chain.verify_chunks(chunks, "t")["ok"], (c1, c2))

    def test_import_chunks_at_arbitrary_splits(self):
        src = AuditChain(self.path.with_name("src.jsonl"))
        for i, ch in enumerate(SPECIAL):
            src.append("t" + ch, {"i": i, "s": "v" + ch})
            src.append("other", {"i": i})
        single = AuditChain(self.path.with_name("single.jsonl"))
        data = src.export_tenant("t" + SPECIAL[0])
        for cut in range(len(data) + 1):
            p = self.path.with_name(f"imp{cut}.jsonl")
            got = AuditChain(p).import_tenant_chunks(
                "t" + SPECIAL[0], [data[:cut], data[cut:]])
            self.assertEqual(len(got), 1)
            self.assertEqual(AuditChain(p).verify("t" + SPECIAL[0]),
                             {"ok": True, "count": 1})
        full = src.export_all()
        dest = AuditChain(self.path.with_name("allchunks.jsonl"))
        parts = [full[i:i + 3] for i in range(0, len(full), 3)]
        records = dest.import_all_chunks(parts)
        self.assertEqual(len(records), 6)
        self.assertTrue(dest.verify_all()["ok"])

    # --- cross-file migration and damage localization -----------------------

    def test_cross_file_import_then_verify_finds_first_broken_lf_line(self):
        # raw-UTF-8 snapshot with special chars, CRLF and no final newline,
        # built by hand as a legacy file
        a1 = row("t", 1, ZERO, {"s": "a" + SPECIAL[1]})
        a2 = row("t", 2, a1["hash"], {"s": ["x", SPECIAL[2]]})
        b1 = row("u", 1, ZERO, {"k": SPECIAL[0]})
        snapshot = lf_line(a1, b"\r\n") + lf_line(b1) + lf_line(a2)
        # migrate it verbatim through export_all/import_all on another chain
        middle = AuditChain(self.path.with_name("m.jsonl"))
        self.assertTrue(middle.verify_all_bytes(snapshot)["ok"])
        dest = AuditChain(self.path.with_name("d.jsonl"))
        records = dest.import_all(snapshot)
        self.assertEqual(len(records), 3)
        # import_all re-serializes (escaping U+2028) but every value, seq and
        # hash is preserved: compare parsed lines rather than raw bytes
        self.assertTrue(dest.verify_all()["ok"])
        exported = [json.loads(l) for l in dest.export_all().decode().split("\n")[:-1]]
        self.assertEqual(exported, [a1, b1, a2])
        # now damage LF line 3 (a2) on a legacy-style file; lines 1-2 carry
        # CRLF and raw special chars that must not move the numbering
        broken = dict(a2)
        broken["event"] = {"tampered": True}
        damaged = lf_line(a1, b"\r\n") + lf_line(b1) + lf_line(broken)
        target_path = self.path.with_name("bad.jsonl")
        target_path.write_bytes(damaged)
        target = AuditChain(target_path)
        self.assertEqual(target.verify_all(),
                         {"ok": False, "at": 3, "tenant": "t", "reason": "digest"})
        # single-tenant verification locates the same LF line
        with self.assertRaises(AuditChainStateError) as cm:
            target.read_tenant("t")
        self.assertEqual((cm.exception.reason, cm.exception.line),
                         ("digest", 3))
        # and import over a corrupt target refuses, line 3, no bytes written
        good_path = self.path.with_name("good.jsonl")
        good_path.write_bytes(damaged)
        goodc = AuditChain(good_path)
        before = good_path.read_bytes()
        with self.assertRaises(AuditChainStateError) as cm:
            goodc.import_tenant("t", lf_line(row("t", 1, ZERO, {"fresh": 1})))
        self.assertEqual((cm.exception.reason, cm.exception.line),
                         ("digest", 3))
        self.assertEqual(good_path.read_bytes(), before)

    def test_special_chars_in_imported_single_tenant_bytes(self):
        # import_tenant accepts raw special chars and preserves the chain;
        # failure leaves no file and no partial content
        item = row("t", 1, ZERO, {"s": SPECIAL[0] + SPECIAL[1]})
        dest = AuditChain(self.path.with_name("imp1.jsonl"))
        records = dest.import_tenant("t", lf_line(item))
        self.assertEqual(records[0]["event"], {"s": SPECIAL[0] + SPECIAL[1]})
        self.assertEqual(dest.verify("t"), {"ok": True, "count": 1})
        # raw CR inside the imported stream is missing at line 1
        fail = AuditChain(self.path.with_name("imp2.jsonl"))
        body = json.dumps(item, ensure_ascii=False).encode()
        with self.assertRaises(AuditChainStateError) as cm:
            fail.import_tenant("t", body + b"\rjunk\n")
        self.assertEqual((cm.exception.reason, cm.exception.line),
                         ("missing", 1))
        self.assertFalse(self.path.with_name("imp2.jsonl").exists())

    def test_failure_appends_nothing_and_keeps_bytes(self):
        # A trailing unparseable physical line (no LF, carrying a raw U+2028
        # that splitlines would wrongly treat as a boundary): every writer
        # refuses and leaves the bytes untouched.
        self.chain.append("keep", {"ok": True})
        healthy = self.path.read_bytes()
        corrupt = healthy + b"garbage-without-lf" + SPECIAL[1].encode("utf-8")
        self.path.write_bytes(corrupt)
        with self.assertRaises(AuditChainStateError) as cm:
            self.chain.append("keep", {"y": 2})
        self.assertEqual(cm.exception.reason, "missing")
        with self.assertRaises(AuditChainStateError):
            self.chain.append_batch("keep", [{"z": 1}])
        with self.assertRaises(AuditChainStateError):
            self.chain.append_many([{"tenant": "keep", "event": {"z": 1}}])
        self.assertEqual(self.path.read_bytes(), corrupt)
        # the healthy prefix is restored for the next test phase: the special
        # chars in real values still append and verify normally
        self.path.write_bytes(healthy)
        self.assertEqual(
            self.chain.append("keep", {"s": "v" + SPECIAL[2]})["seq"], 2)
        self.assertTrue(self.chain.verify("keep")["ok"])

    # --- every appending/conditional/migration entry shares the boundary ----

    def test_all_append_entries_accept_special_values(self):
        ch = SPECIAL[1]
        tenant = "z" + ch
        events = [{"s": ch}, ["a", ch], {"k" + ch: [ch, 1]}]
        r = self.chain.append(tenant, events[0])
        self.assertEqual(r["seq"], 1)
        batch = self.chain.append_batch(tenant, events[1:])
        self.assertEqual([x["seq"] for x in batch], [2, 3])
        # conditional append continues from the observed head
        h = self.chain.head(tenant)
        nxt = self.chain.append_if_head(tenant, {"next": ch},
                                        h["count"], h["hash"])
        self.assertEqual(nxt["seq"], 4)
        h = self.chain.head(tenant)
        cond_batch = self.chain.append_batch_if_head(
            tenant, [{"q": ch}, {}], h["count"], h["hash"])
        self.assertEqual([x["seq"] for x in cond_batch], [5, 6])
        # cross-tenant atomic append
        many = self.chain.append_many([
            {"tenant": tenant, "event": {"m": ch}},
            {"tenant": "other" + ch, "event": ch},
        ])
        self.assertEqual([x["seq"] for x in many], [7, 1])
        heads = {t["tenant"]: (t["count"], t["hash"])
                 for t in self.chain.heads()["tenants"]}
        ok = self.chain.append_many_if_heads([
            {"tenant": tenant, "events": [ch], "expected_count": heads[tenant][0],
             "expected_hash": heads[tenant][1]},
            {"tenant": "other" + ch, "events": [ch, ch],
             "expected_count": heads["other" + ch][0],
             "expected_hash": heads["other" + ch][1]},
        ])
        self.assertEqual([x["seq"] for x in ok], [8, 2, 3])
        self.assertTrue(self.chain.verify_all()["ok"])
        # values survive paged reading
        self.assertEqual(
            self.chain.read_tenant(tenant, 1, 1)[0]["event"], events[0])
        self.assertEqual(self.chain.head(tenant)["count"], 8)

    def test_expected_count_and_chunks_share_lf_line_numbers(self):
        item1 = row("t", 1, ZERO, {"a": SPECIAL[2]})
        item2 = row("t", 2, item1["hash"], {"b": SPECIAL[0]})
        valid = lf_line(item1) + lf_line(item2)
        # expected_count short/long verdicts unaffected by raw special chars
        self.assertEqual(self.chain.verify_bytes(valid, "t", 3),
                         {"ok": False, "at": 3, "reason": "missing"})
        self.assertEqual(self.chain.verify_bytes(valid, "t", 1),
                         {"ok": False, "at": 2, "reason": "sequence"})
        # corrupt LF line 2 and feed it in arbitrarily split chunks
        broken = dict(item2)
        broken["event"] = {"x": 1}  # digest wrong
        bad = lf_line(item1) + lf_line(broken)
        for cut in range(len(bad) + 1):
            r = self.chain.verify_chunks([bad[:cut], bad[cut:]], "t")
            self.assertEqual((r["at"], r["reason"]), (2, "digest"), cut)
        self.assertEqual(
            self.chain.verify_all_chunks([bad[:4], bad[4:40], bad[40:]]),
            {"ok": False, "at": 2, "tenant": "t", "reason": "digest"})

    def test_range_migration_roundtrip_with_special_values(self):
        ch = SPECIAL[0]
        for i in range(5):
            self.chain.append("t", {"i": i, "s": ch})
            self.chain.append("other", {"i": i})
        seg = self.chain.export_tenant_range("t", 3)
        self.assertEqual(seg.count(b"\n"), 3)
        dest = AuditChain(self.path.with_name("r1.jsonl"))
        # seed the target with the first two records by exporting/importing
        head_part = self.chain.export_tenant_range("t", 1, 2)
        dest.import_tenant("t", head_part)
        h = dest.head("t")
        records = dest.import_tenant_range("t", seg, h["count"], h["hash"])
        self.assertEqual([r["seq"] for r in records], [3, 4, 5])
        self.assertEqual(dest.verify("t"), {"ok": True, "count": 5})
        # multi-tenant range import interleaving, special-char tenant names
        src = AuditChain(self.path.with_name("src_r.jsonl"))
        src.append("p" + SPECIAL[1], {"v": 1})
        src.append("q" + SPECIAL[2], {"v": 1})
        src.append("p" + SPECIAL[1], {"v": 2})
        src.append("q" + SPECIAL[2], {"v": 2})
        pa = src.export_tenant_range("p" + SPECIAL[1], 2)
        qa = src.export_tenant_range("q" + SPECIAL[2], 2)
        # interleave the two segments into one byte stream physically
        p_line = pa
        q_line = qa
        interleaved = q_line + p_line  # q first physically
        d2 = AuditChain(self.path.with_name("r2.jsonl"))
        d2.append("q" + SPECIAL[2], {"v": 1})
        d2.append("p" + SPECIAL[1], {"v": 1})
        got = d2.import_all_range(interleaved, [
            {"tenant": "q" + SPECIAL[2], "expected_count": 1,
             "expected_hash": d2.head("q" + SPECIAL[2])["hash"]},
            {"tenant": "p" + SPECIAL[1], "expected_count": 1,
             "expected_hash": d2.head("p" + SPECIAL[1])["hash"]},
        ])
        self.assertEqual([r["seq"] for r in got], [2, 2])
        self.assertTrue(d2.verify_all()["ok"])

    def test_chunked_exports_concatenate_and_carry_special_chars(self):
        for i, ch in enumerate(SPECIAL * 3):
            self.chain.append("t", {"i": i, "s": "v" + ch})
        ref = self.chain.export_tenant("t")
        for size in (1, 2, 3, 7, 64, 10_000):
            chunks = list(self.chain.export_tenant_chunks("t", size))
            self.assertEqual(b"".join(chunks), ref)
        full = self.chain.export_all()
        for size in (1, 5, 13):
            chunks = list(self.chain.export_all_chunks(size))
            self.assertEqual(b"".join(chunks), full)
        # chunked export refuses a corrupt source and yields nothing first
        self.path.write_bytes(full + b"{bad")
        with self.assertRaises(AuditChainStateError) as cm:
            list(self.chain.export_tenant_chunks("t", 10))
        self.assertEqual(cm.exception.reason, "missing")

    def test_crlf_snapshot_byte_behavior_preserved_by_export_all(self):
        # export_all returns the raw bytes of an existing snapshot unchanged,
        # CRLF and raw special chars included
        a1 = row("t", 1, ZERO, {"s": "x" + SPECIAL[1]})
        legacy = lf_line(a1, b"\r\n")
        self.write_raw(legacy)
        self.assertEqual(self.chain.export_all(), legacy)
        # append on a CRLF history still works and keeps old bytes verbatim
        nxt = self.chain.append("t", {})
        merged = self.path.read_bytes()
        self.assertTrue(merged.startswith(legacy))
        self.assertTrue(merged.endswith(ascii_line(nxt)))
        self.assertEqual(nxt["prev"], a1["hash"])

    def test_verify_all_bytes_structure_with_special_tenants(self):
        ta, tb = {"k": SPECIAL[1]}, ["a", SPECIAL[2]]
        data = lf_line(row(ta, 1)) + lf_line(row(tb, 1)) \
            + lf_line(row(ta, 2, AuditChain._hash(row(ta, 1))))
        r = self.chain.verify_all_bytes(data)
        self.assertTrue(r["ok"])
        self.assertEqual(
            [(x["tenant"], x["count"]) for x in r["tenants"]],
            [(ta, 2), (tb, 1)])


if __name__ == "__main__":
    unittest.main()
