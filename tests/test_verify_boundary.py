import json
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app import AuditChain, ZERO


class VerifyTenantBoundaryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.path = Path(self.tmp.name) / "audit.jsonl"
        self.chain = AuditChain(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def write_bytes(self, data):
        self.path.write_bytes(data)

    def valid_row(self, tenant="t", seq=1, prev=ZERO):
        row = {"tenant": tenant, "seq": seq, "event": {}, "prev": prev}
        row["hash"] = AuditChain._hash(row)
        return (json.dumps(row, sort_keys=True) + "\n").encode("utf-8")

    # --- illegal tenants: ValueError, exactly, before any read/path effect ---

    def assert_value_error(self, tenant):
        with self.assertRaises(ValueError) as cm:
            self.chain.verify(tenant)
        # must not leak TypeError / RecursionError / json encoding errors
        self.assertIs(type(cm.exception), ValueError)
        with self.assertRaises(ValueError):
            self.chain.verify(tenant, 3)  # expected_count must not change it

    def test_non_finite_floats_rejected(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            self.assertFalse(self.path.exists())
            self.assert_value_error(value)
            self.assertFalse(self.path.exists())  # never created a file

    def test_non_finite_nested_in_containers_rejected(self):
        self.assert_value_error({"a": [float("nan")]})
        self.assert_value_error([{"x": float("inf")}])
        self.assert_value_error({"a": {"b": float("-inf")}})

    def test_non_string_object_keys_rejected(self):
        self.assert_value_error({1: "x"})
        self.assert_value_error({1.0: "x"})
        self.assert_value_error({True: "x"})
        self.assert_value_error({None: "x"})
        self.assert_value_error({("a",): "x"})
        self.assert_value_error({"ok": {2: "deep"}})

    def test_cyclic_containers_rejected(self):
        cyc_list = []
        cyc_list.append(cyc_list)
        self.assert_value_error(cyc_list)
        cyc_dict = {}
        cyc_dict["self"] = cyc_dict
        self.assert_value_error(cyc_dict)
        cyc_nested = {"a": []}
        cyc_nested["a"].append(cyc_nested)
        self.assert_value_error(cyc_nested)

    def test_other_unencodable_values_rejected(self):
        for value in ({1, 2}, ("x",), object(), b"bytes", frozenset({"a"})):
            self.assert_value_error(value)

    # --- corrupt history must not override the input boundary ---

    def test_corrupt_file_plus_illegal_tenant_value_error_first_bytes_kept(self):
        for payload in (b"\xff", b"{oops\n", self.valid_row() + b"\xffgarbage"):
            self.write_bytes(payload)
            for bad in (float("nan"), float("inf"), {1: "x"}):
                with self.assertRaises(ValueError):
                    self.chain.verify(bad)
                self.assertEqual(self.path.read_bytes(), payload)

    def test_cyclic_tenant_against_corrupt_file_value_error_first(self):
        self.write_bytes(b'{"tenant": "x", broken\n')
        cyc = []
        cyc.append(cyc)
        with self.assertRaises(ValueError):
            self.chain.verify(cyc)
        self.assertEqual(self.path.read_bytes(), b'{"tenant": "x", broken\n')

    # --- legal tenants: existing verify semantics are preserved ---

    def test_legal_complex_object_tenant_key_order_normalized(self):
        tenant_a = {"a": 1, "b": [1, 2, {"c": True, "d": None}], "e": "s"}
        tenant_b = {"e": "s", "b": [1, 2, {"d": None, "c": True}], "a": 1}
        self.assertEqual(json.dumps(tenant_a, sort_keys=True),
                         json.dumps(tenant_b, sort_keys=True))
        item = self.chain.append(tenant_a, {})
        self.assertEqual(item["seq"], 1)
        self.assertEqual(self.chain.verify(tenant_b), {"ok": True, "count": 1})
        self.assertEqual(self.chain.verify(tenant_b, 1), {"ok": True, "count": 1})
        self.assertEqual(self.chain.verify(tenant_b, 2)["reason"], "missing")
        item2 = self.chain.append(tenant_b, {})
        self.assertEqual((item2["seq"], item2["prev"]), (2, item["hash"]))
        # a different value, same key-ordering shape, stays its own partition
        other = {"a": 1, "b": [1, 2, {"c": False, "d": None}], "e": "s"}
        self.assertEqual(self.chain.verify(other), {"ok": True, "count": 0})

    def test_int_float_bool_string_are_separate_partitions_in_verify(self):
        for t in (1, 1.0, True, "1"):
            self.chain.append(t, {})
        for t in (1, 1.0, True, "1"):
            self.assertEqual(self.chain.verify(t), {"ok": True, "count": 1})
        # cross-queries must never match another type's record
        self.assertEqual(self.chain.verify(1, 2)["reason"], "missing")
        self.assertEqual(self.chain.verify(True, 1), {"ok": True, "count": 1})
        # and a second record per tenant stays independent
        for t in (1, 1.0, True, "1"):
            self.chain.append(t, {})
        for t in (1, 1.0, True, "1"):
            self.assertEqual(self.chain.verify(t), {"ok": True, "count": 2})

    def test_legal_scalar_and_container_tenants(self):
        for t in (None, 0, -3, 1.25, "s", [], {}, [None, True, 1, 1.0, "x"]):
            self.assertEqual(self.chain.verify(t), {"ok": True, "count": 0})
            self.chain.append(t, {})
            self.assertEqual(self.chain.verify(t), {"ok": True, "count": 1})

    def test_interleaved_tenants_verify_independently(self):
        self.chain.append("a", {})
        self.chain.append(1, {})
        self.chain.append("a", {})
        self.chain.append(1, {})
        self.chain.append({"k": "v"}, {})
        self.assertEqual(self.chain.verify("a"), {"ok": True, "count": 2})
        self.assertEqual(self.chain.verify(1), {"ok": True, "count": 2})
        self.assertEqual(self.chain.verify({"k": "v"}), {"ok": True, "count": 1})
        self.assertEqual(self.chain.verify("absent"), {"ok": True, "count": 0})

    def test_expected_count_results_unchanged(self):
        self.chain.append("t", {})
        self.chain.append("t", {})
        self.assertEqual(self.chain.verify("t", 2), {"ok": True, "count": 2})
        self.assertEqual(self.chain.verify("t", 3),
                         {"ok": False, "at": 3, "reason": "missing"})
        self.assertEqual(self.chain.verify("t", 1),
                         {"ok": False, "at": 2, "reason": "sequence"})

    def test_physical_line_damage_still_reports_first_error(self):
        # missing record for an unseen tenant on a damaged later line
        self.write_bytes(self.valid_row("x") + b"\xff")
        self.assertEqual(self.chain.verify("t"),
                         {"ok": False, "at": 2, "reason": "missing"})
        # digest error on the tenant's own record beats later damage
        good = {"tenant": "t", "seq": 1, "event": {}, "prev": ZERO}
        good["hash"] = AuditChain._hash(good)
        tampered = dict(good)
        tampered["event"] = {"z": 9}
        self.write_bytes(
            (json.dumps(tampered, sort_keys=True) + "\n").encode("utf-8") + b"\xff"
        )
        self.assertEqual(self.chain.verify("t"),
                         {"ok": False, "at": 1, "reason": "digest"})

    def test_nan_is_never_a_usable_tenant(self):
        self.chain.append("nan", {})  # string tenant with similar spelling
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValueError):
                self.chain.verify(bad)
        self.assertEqual(self.chain.verify("nan"), {"ok": True, "count": 1})

    # --- verify_all shape and priority unchanged ---

    def test_verify_all_takes_no_tenant_and_never_creates_file(self):
        self.assertFalse(self.path.exists())
        self.assertEqual(self.chain.verify_all(), {"ok": True, "tenants": []})
        self.assertFalse(self.path.exists())

    def test_verify_all_still_reports_corruption(self):
        self.write_bytes(b"\xff")
        self.assertEqual(self.chain.verify_all(),
                         {"ok": False, "at": 1, "tenant": None, "reason": "missing"})


class VerifyDuringConcurrentAppendTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.path = Path(self.tmp.name) / "audit.jsonl"
        self.chain = AuditChain(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_repeated_verify_while_appending(self):
        n_writer, per, n_reader = 8, 25, 4
        total = n_writer * per
        stop = threading.Event()
        failures, leaks = [], []
        box = threading.Lock()

        def reader():
            cyc = []
            cyc.append(cyc)
            while not stop.is_set():
                r = self.chain.verify("t")
                if not (r.get("ok") and r["count"] in range(total + 1)):
                    with box:
                        failures.append(("verify", r))
                    return
                # complex object tenant: key-order normalized every time
                r2 = self.chain.verify({"b": 2, "a": 1})
                if r2 != {"ok": True, "count": 0}:
                    with box:
                        failures.append(("object", r2))
                    return
                # illegal tenants must keep raising ValueError, never leak
                for bad in (float("nan"), float("inf"), {1: "x"}, cyc):
                    try:
                        self.chain.verify(bad)
                        with box:
                            failures.append(("no-raise", bad))
                    except ValueError:
                        pass
                    except Exception as e:  # noqa: BLE001
                        with box:
                            leaks.append(type(e).__name__)

        readers = [threading.Thread(target=reader) for _ in range(n_reader)]
        for t in readers:
            t.start()

        def writer(i):
            for j in range(per):
                self.chain.append("t", {"w": i, "j": j})

        threads = [threading.Thread(target=writer, args=(i,))
                   for i in range(n_writer)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        stop.set()
        for t in readers:
            t.join(timeout=5)

        self.assertEqual(failures, [])
        self.assertEqual(leaks, [])
        self.assertEqual(self.chain.verify("t"), {"ok": True, "count": total})
        self.assertTrue(self.chain.verify_all()["ok"])


if __name__ == "__main__":
    unittest.main()
