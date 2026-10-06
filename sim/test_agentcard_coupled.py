#!/usr/bin/env python3
"""Independent, standard-library oracles for the coupled storage experiment.

Tiny byte counts make every schedule hand-checkable. Expected times below are
literal consequences of command, array, channel and fill durations; they are
not computed by a second invocation of the implementation under test.

Run: python3 -m unittest discover -s sim -p 'test_agentcard_coupled.py' -v
No model data, external packages, network, RTL, or hardware are involved.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import fields, replace
import unittest

from agentcard_coupled import (Cache, DeviceConfig, FastBus, PageService, Read,
                               kv_reads, validate_trace)


def tiny_config(**changes):
    """An eight-byte page takes 2 command + 5 array + 4 IO + 2 fill ns."""
    base = DeviceConfig(page_bytes=8, channels=1, dies_per_channel=1,
                        planes_per_die=1, channel_bytes_per_ns=2,
                        command_ns=2, read_ns=5, bridge_pages=2,
                        fast_bytes_per_ns=4)
    return replace(base, **changes)


def operation_rows(service, operation):
    return [r for r in service.records if r["operation"] == operation]


def times(service, operation):
    return [(r["start_ns"], r["end_ns"]) for r in operation_rows(service, operation)]


class PageTimingTests(unittest.TestCase):
    def test_one_page_exact_timing_with_arrival_offset(self):
        service = PageService(tiny_config(), audit=True)
        self.assertEqual(service.run([Read("e", "expert", 8, 0)], 10), {"e": 23})
        self.assertEqual(times(service, "command"), [(10, 12)])
        self.assertEqual(times(service, "array"), [(12, 17)])
        self.assertEqual(times(service, "data"), [(17, 21)])
        self.assertEqual(times(service, "bridge"), [(21, 23)])
        self.assertEqual(service.stats["physical_expert_bytes"], 8)
        self.assertEqual(service.fast.bytes["fill_expert"], 8)
        service.check()

    def test_partial_page_and_boundary_round_up_storage_only(self):
        # Data always transfers eight bytes. The fast bus fills useful bytes.
        # Two same-plane reads issue at 0 and 11; second data ends at 22.
        for size, pages, end in [(1, 1, 12), (7, 1, 13), (8, 1, 13),
                                 (9, 2, 23), (16, 2, 24)]:
            with self.subTest(size=size):
                service = PageService(tiny_config(), audit=True)
                self.assertEqual(service.run([Read("x", "kv", size, 0)], 0), {"x": end})
                self.assertEqual(service.stats["submitted_pages"], pages)
                self.assertEqual(service.stats["physical_kv_bytes"], 8 * pages)
                self.assertEqual(service.stats["useful_kv_bytes"], size)
                self.assertEqual(service.fast.bytes["fill_kv"], size)
                self.assertEqual(len(operation_rows(service, "data")), pages)
                service.check()

    def test_same_die_distinct_planes_cannot_overlap_array_service(self):
        c = tiny_config(planes_per_die=2, shared_command_data_bus=False)
        service = PageService(c, audit=True)
        ready = service.run([Read("a", "expert", 8, 0), Read("b", "expert", 8, 1)], 0)
        # Plane 1 may issue while plane 0 drains data, but only after its array
        # is released at t=7. Its data finishes at 18, and fill at 20.
        self.assertEqual(ready, {"a": 13, "b": 20})
        self.assertEqual(times(service, "command"), [(0, 2), (7, 9)])
        self.assertEqual(times(service, "array"), [(2, 7), (9, 14)])
        self.assertEqual(times(service, "data"), [(7, 11), (14, 18)])
        service.check()

    def test_same_plane_stays_occupied_through_data(self):
        c = tiny_config(planes_per_die=2, shared_command_data_bus=False)
        service = PageService(c, audit=True)
        # Addresses 0 and 2 both map to plane zero. The second command must
        # wait until 11, not merely until the first array completes at 7.
        ready = service.run([Read("a", "expert", 8, 0), Read("b", "expert", 8, 2)], 0)
        self.assertEqual(ready, {"a": 13, "b": 24})
        self.assertEqual(times(service, "command"), [(0, 2), (11, 13)])
        self.assertEqual(service.stats["peak_plane_buffers"], 1)
        service.check()

    def test_distinct_dies_overlap_arrays_and_serialize_channel_data(self):
        service = PageService(tiny_config(dies_per_channel=2), audit=True)
        ready = service.run([Read("a", "expert", 8, 0), Read("b", "expert", 8, 1)], 0)
        self.assertEqual(ready, {"a": 13, "b": 17})
        self.assertEqual(times(service, "command"), [(0, 2), (2, 4)])
        self.assertEqual(times(service, "array"), [(2, 7), (4, 9)])
        self.assertEqual(times(service, "data"), [(7, 11), (11, 15)])
        self.assertEqual([r["resource"] for r in operation_rows(service, "array")], [0, 1])
        service.check()

    def test_distinct_channels_still_share_one_fast_bus(self):
        service = PageService(tiny_config(channels=2), audit=True)
        ready = service.run([Read("a", "expert", 8, 0), Read("b", "kv", 8, 1)], 0)
        # Both data transfers finish at 11; one common fast bus fills 11..13
        # and 13..15. A second fast bus would incorrectly finish both at 13.
        self.assertEqual(ready, {"a": 13, "b": 15})
        self.assertEqual(times(service, "data"), [(7, 11), (7, 11)])
        self.assertEqual(service.fast.busy_ns, 4)
        self.assertEqual(service.fast.until, 15)
        service.check()

    def test_command_data_shared_bus_cost_is_observable(self):
        requests = [Read("a", "expert", 8, 0), Read("b", "expert", 8, 1)]
        shared = PageService(tiny_config(planes_per_die=2), audit=True)
        separate = PageService(tiny_config(planes_per_die=2, shared_command_data_bus=False), audit=True)
        self.assertEqual(shared.run(requests, 0), {"a": 13, "b": 24})
        self.assertEqual(separate.run(requests, 0), {"a": 13, "b": 20})
        self.assertEqual(times(shared, "command"), [(0, 2), (11, 13)])
        self.assertEqual(times(separate, "command"), [(0, 2), (7, 9)])
        shared.check()
        separate.check()

    def test_finite_bridge_propagates_fast_bus_backpressure(self):
        # An existing compute transfer occupies the fast bus until 50. The
        # first page holds a bridge slot until 58. With one slot the second
        # page cannot start its four-ns data transfer until 58, finishing fill
        # at 70. Two slots permit data at 6..10 and fast fills ending 58,66.
        for slots, expected, data in [(1, {"a": 58, "b": 70}, [(2, 6), (58, 62)]),
                                      (2, {"a": 58, "b": 66}, [(2, 6), (6, 10)])]:
            with self.subTest(bridge_pages=slots):
                fast = FastBus(1)
                fast.transfer(0, 50, "compute")
                c = tiny_config(dies_per_channel=2, command_ns=1, read_ns=1,
                                bridge_pages=slots, fast_bytes_per_ns=1)
                service = PageService(c, fast=fast, audit=True)
                ready = service.run([Read("a", "expert", 8, 0), Read("b", "kv", 8, 1)], 0)
                self.assertEqual(ready, expected)
                self.assertEqual(times(service, "data"), data)
                self.assertEqual(service.stats["peak_bridge_pages"], slots)
                self.assertEqual(fast.bytes["compute"], 50)
                self.assertEqual(fast.bytes["fill_expert"] + fast.bytes["fill_kv"], 16)
                service.check()

    def test_retry_repeats_command_array_and_full_page_io(self):
        c = tiny_config(retry_every=1, retry_ns=7)
        service = PageService(c, audit=True)
        # First attempt 0..11 fails. Retry command 11..13, array 13..25,
        # data 25..29, then one ns for the three useful bytes.
        self.assertEqual(service.run([Read("x", "expert", 3, 0)], 0), {"x": 30})
        self.assertEqual(times(service, "command"), [(0, 2), (11, 13)])
        self.assertEqual(times(service, "array"), [(2, 7), (13, 25)])
        self.assertEqual(times(service, "data"), [(7, 11), (25, 29)])
        self.assertEqual(times(service, "bridge"), [(29, 30)])
        self.assertEqual(service.stats["retries"], 1)
        self.assertEqual(service.stats["physical_expert_bytes"], 16)
        self.assertEqual(service.stats["useful_expert_bytes"], 3)
        self.assertEqual(service.fast.bytes["fill_expert"], 3)
        self.assertEqual(service.stats["issued_attempts"], 2)
        service.check()

    def test_retry_selection_uses_physical_page_address(self):
        c = tiny_config(retry_every=2, retry_ns=7)
        service = PageService(c, audit=True)
        service.run([Read("x", "kv", 16, 0)], 0)
        self.assertEqual([r["page"] for r in operation_rows(service, "data")], [0, 1, 1])
        self.assertEqual(service.stats["retries"], 1)
        self.assertEqual(service.stats["physical_kv_bytes"], 24)
        self.assertEqual(service.fast.bytes["fill_kv"], 16)
        service.check()

    def test_mixed_classes_alternate_when_both_are_eligible(self):
        service = PageService(tiny_config(), audit=True)
        ready = service.run([Read("bulk", "expert", 24, 0), Read("context", "kv", 16, 3)], 0)
        commands = operation_rows(service, "command")
        self.assertEqual([r["kind"] for r in commands], ["expert", "kv", "expert", "kv", "expert"])
        self.assertEqual([r["start_ns"] for r in commands], [0, 11, 22, 33, 44])
        self.assertEqual(ready, {"context": 46, "bulk": 57})
        self.assertEqual(service.stats["physical_expert_bytes"], 24)
        self.assertEqual(service.stats["physical_kv_bytes"], 16)
        service.check()

    def test_empty_run_and_sequential_calls_remain_causal(self):
        service = PageService(tiny_config(), audit=True)
        self.assertEqual(service.run([], 4), {})
        self.assertEqual(service.time, 4)
        self.assertEqual(service.run([Read("a", "expert", 8, 0)], 4), {"a": 17})
        self.assertEqual(service.run([Read("b", "kv", 8, 1)], 17), {"b": 30})
        self.assertEqual(service.pending, 0)
        self.assertEqual(service.stats["completed_pages"], 2)
        service.check()

    def test_independent_aggregate_service_lower_bounds(self):
        c = tiny_config(channels=2, dies_per_channel=2, planes_per_die=2,
                        bridge_pages=3, fast_bytes_per_ns=1, retry_every=3, retry_ns=2)
        service = PageService(c, audit=True)
        service.run([Read("e", "expert", 65, 0), Read("k", "kv", 31, 9)], 0)
        # Nine expert pages and four KV pages. Addresses 2,5,8,11 retry.
        self.assertEqual(service.stats["completed_pages"], 13)
        self.assertEqual(service.stats["completed_attempts"], 17)
        self.assertEqual(service.stats["physical_expert_bytes"], 96)
        self.assertEqual(service.stats["physical_kv_bytes"], 40)
        self.assertEqual(service.fast.bytes["fill_expert"] + service.fast.bytes["fill_kv"], 96)
        self.assertGreaterEqual(service.time * 2, 17 * (2 + 4))  # shared channel command+data
        self.assertGreaterEqual(service.time * 4, 13 * 5 + 4 * 7)  # four die arrays
        self.assertGreaterEqual(service.time, 96)  # one-byte/ns fast bus
        self.assertLessEqual(service.stats["peak_bridge_pages"], 3)
        self.assertLessEqual(service.stats["peak_plane_buffers"], 8)
        service.check()


class CacheTests(unittest.TestCase):
    def test_pending_coalescing_is_not_a_hit_or_second_allocation(self):
        cache = Cache(8)
        r = Read("x", "expert", 8, 0)
        self.assertEqual(cache.acquire(r, 0), "miss")
        self.assertEqual(cache.acquire(r, 0), "pending")
        self.assertEqual(cache.used, 8)
        self.assertEqual(cache.entries["x"]["pins"], 2)
        cache.mark_ready("x", 10)
        self.assertEqual(cache.acquire(r, 9), "pending")
        self.assertEqual(cache.acquire(r, 10), "hit")
        self.assertEqual(cache.stats["expert_miss_objects"], 1)
        self.assertEqual(cache.stats["expert_pending_objects"], 2)
        self.assertEqual(cache.stats["expert_hit_objects"], 1)
        self.assertEqual(cache.stats["expert_pending_bytes"], 16)
        for _ in range(4):
            cache.release("x", 10)
        cache.check()

    def test_full_cache_pins_block_eviction_then_release_allows_it(self):
        cache = Cache(8)
        a, b = Read("a", "expert", 8, 0), Read("b", "kv", 8, 1)
        cache.acquire(a, 0)
        cache.mark_ready("a", 5)
        with self.assertRaisesRegex(ValueError, "pinned"):
            cache.acquire(b, 5)
        self.assertEqual(list(cache.entries), ["a"])
        self.assertEqual(cache.used, 8)
        cache.release("a", 5)
        self.assertEqual(cache.acquire(b, 5), "miss")
        self.assertEqual(list(cache.entries), ["b"])
        self.assertEqual(cache.stats["evicted_bytes"], 8)
        self.assertEqual(cache.stats["evicted_objects"], 1)
        cache.mark_ready("b", 9)
        cache.release("b", 9)
        cache.check()

    def test_lru_eviction_skips_pinned_eldest(self):
        cache = Cache(16)
        for index, key in enumerate(("a", "b")):
            cache.acquire(Read(key, "expert", 8, index), 0)
            cache.mark_ready(key, 5)
        cache.release("b", 5)
        cache.acquire(Read("c", "kv", 8, 2), 5)
        self.assertEqual(list(cache.entries), ["a", "c"])
        self.assertEqual(cache.stats["evicted_objects"], 1)
        cache.release("a", 5)
        cache.mark_ready("c", 10)
        cache.release("c", 10)
        cache.check()

    def test_cache_hit_updates_lru_recency(self):
        cache = Cache(16)
        a, b, c = (Read(k, "expert", 8, i) for i, k in enumerate("abc"))
        for r in (a, b):
            cache.acquire(r, 0)
            cache.mark_ready(r.key, 1)
            cache.release(r.key, 1)
        self.assertEqual(cache.acquire(a, 1), "hit")
        cache.release("a", 1)
        cache.acquire(c, 1)
        self.assertEqual(list(cache.entries), ["a", "c"])
        cache.mark_ready("c", 2)
        cache.release("c", 2)
        cache.check()

    def test_cache_rejects_identity_changes_and_bad_lifecycle(self):
        cache = Cache(8)
        cache.acquire(Read("x", "expert", 8, 0), 0)
        for changed in (Read("x", "expert", 7, 0), Read("x", "expert", 8, 1), Read("x", "kv", 8, 0)):
            with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, "identity"):
                cache.acquire(changed, 0)
        with self.assertRaises(ValueError):
            cache.release("x", 0)
        cache.mark_ready("x", 5)
        with self.assertRaises(ValueError):
            cache.mark_ready("x", 6)
        with self.assertRaises(ValueError):
            cache.release("x", 4)
        cache.release("x", 5)
        with self.assertRaises(ValueError):
            cache.release("x", 5)
        cache.check()


class FastBusTests(unittest.TestCase):
    def test_rounding_serialization_and_idle_gap(self):
        fast = FastBus(4)
        fast.audit = True
        self.assertEqual(fast.transfer(3, 5, "fill_expert"), 5)
        self.assertEqual(fast.transfer(3, 4, "compute"), 6)
        self.assertEqual(fast.transfer(10, 1, "fill_kv"), 11)
        self.assertEqual(fast.busy_ns, 4)
        self.assertEqual(fast.intervals, [(3, 5, 5, "fill_expert"), (5, 6, 4, "compute"), (10, 11, 1, "fill_kv")])
        self.assertEqual(fast.transfer(1, 0, "compute"), 1)
        self.assertEqual(fast.until, 11)
        self.assertEqual(fast.bytes["compute"], 4)


class InputValidationTests(unittest.TestCase):
    def test_device_parameters_reject_nonintegers_and_nonpositive_values(self):
        for field in fields(DeviceConfig):
            if field.name == "shared_command_data_bus":
                bad_values = [0, 1, "true", None]
            else:
                bad_values = [-1, 1.5, True, "8", None]
                if field.name not in ("retry_every", "retry_ns"):
                    bad_values.append(0)
            for value in bad_values:
                with self.subTest(field=field.name, value=value), self.assertRaises(ValueError):
                    replace(tiny_config(), **{field.name: value})

    def test_geometry_metadata_bounds(self):
        with self.assertRaises(ValueError):
            tiny_config(page_bytes=1024 * 1024 + 1)
        with self.assertRaises(ValueError):
            tiny_config(channels=4097)
        self.assertEqual(tiny_config(page_bytes=1024 * 1024).page_bytes, 1024 * 1024)
        self.assertEqual(tiny_config(channels=16, dies_per_channel=16, planes_per_die=16).channels, 16)

    def test_reads_reject_invalid_keys_kinds_sizes_and_addresses(self):
        good = dict(key="x", kind="expert", size=8, first_page=0)
        invalid = {"key": ["", None, 1], "kind": ["compute", "KV", None],
                   "size": [0, -1, 1.5, True, "8"],
                   "first_page": [-1, 1.5, True, "0"]}
        for key, values in invalid.items():
            for value in values:
                with self.subTest(field=key, value=value), self.assertRaises(ValueError):
                    Read(**(good | {key: value}))

    def test_duplicate_storage_objects_and_noncausal_arrivals_rejected(self):
        service = PageService(tiny_config())
        r = Read("x", "expert", 8, 0)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            service.run([r, r], 0)
        self.assertEqual(service.stats["submitted_pages"], 0)
        service.run([r], 0)
        for arrival in (-1, 1.5, True, 12):
            with self.subTest(arrival=arrival), self.assertRaises(ValueError):
                service.run([], arrival)

    def test_fast_bus_negative_and_noninteger_inputs_rejected(self):
        for rate in (0, -1, 1.5, True):
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                FastBus(rate)
        for value in (-1, 1.5, True):
            with self.subTest(ready=value), self.assertRaises(ValueError):
                FastBus(4).transfer(value, 8, "compute")
            with self.subTest(size=value), self.assertRaises(ValueError):
                FastBus(4).transfer(0, value, "compute")

    def test_cache_capacity_and_oversized_objects_rejected(self):
        for capacity in (0, -1, 1.5, True):
            with self.subTest(capacity=capacity), self.assertRaises(ValueError):
                Cache(capacity)
        cache = Cache(8)
        with self.assertRaisesRegex(ValueError, "capacity"):
            cache.acquire(Read("large", "expert", 9, 0), 0)
        self.assertEqual(cache.used, 0)
        self.assertFalse(cache.entries)

    def test_cache_times_reject_negative_and_noninteger_values(self):
        for now in (-1, 1.5, True):
            with self.subTest(acquire_time=now), self.assertRaises(ValueError):
                Cache(8).acquire(Read("x", "expert", 8, 0), now)
            cache = Cache(8)
            cache.acquire(Read("x", "expert", 8, 0), 0)
            with self.subTest(completion_time=now), self.assertRaises(ValueError):
                cache.mark_ready("x", now)
            cache.mark_ready("x", 0)
            with self.subTest(release_time=now), self.assertRaises(ValueError):
                cache.release("x", now)


class CheckerMutationTests(unittest.TestCase):
    def make_valid_service(self):
        service = PageService(tiny_config(retry_every=2, retry_ns=3), audit=True)
        service.run([Read("e", "expert", 8, 0), Read("k", "kv", 9, 1)], 0)
        service.check()  # Positive control before each intentionally bad copy.
        return service

    def test_conservation_and_capacity_counter_mutations_are_detected(self):
        control = self.make_valid_service()
        mutations = [("submitted_pages", 1), ("completed_pages", 1),
                     ("issued_attempts", 1), ("completed_attempts", 1),
                     ("retries", 1), ("physical_expert_bytes", 1),
                     ("physical_kv_bytes", 1), ("useful_expert_bytes", 1),
                     ("useful_kv_bytes", 1), ("peak_bridge_pages", 3),
                     ("peak_plane_buffers", 2),
                     ("channel_0_data_ns", control.time + 1),
                     ("channel_0_command_ns", control.time + 1),
                     ("die_0_array_ns", control.time + 1)]
        for key, delta in mutations:
            with self.subTest(counter=key):
                damaged = deepcopy(control)
                damaged.stats[key] += delta
                with self.assertRaises(AssertionError):
                    damaged.check()
                control.check()  # Unchanged run still passes the same checker.

    def test_fast_bus_and_terminal_state_mutations_are_detected(self):
        control = self.make_valid_service()
        for target in ("pending", "fast_bytes", "fast_busy"):
            with self.subTest(target=target):
                damaged = deepcopy(control)
                if target == "pending":
                    damaged.pending = 1
                elif target == "fast_bytes":
                    damaged.fast.bytes["fill_expert"] += 1
                else:
                    damaged.fast.busy_ns = damaged.fast.until + 1
                with self.assertRaises(AssertionError):
                    damaged.check()
                control.check()

    def test_audit_resource_overlap_mutation_is_detected(self):
        control = self.make_valid_service()
        damaged = deepcopy(control)
        commands = operation_rows(damaged, "command")
        commands[1]["start_ns"] = commands[0]["start_ns"]
        with self.assertRaisesRegex(AssertionError, "overlap"):
            damaged.check()
        control.check()

    def test_cache_accounting_and_unreleased_pin_mutations_are_detected(self):
        control = Cache(8)
        control.acquire(Read("x", "expert", 8, 0), 0)
        control.mark_ready("x", 1)
        control.release("x", 1)
        control.check()
        for target in ("used", "peak", "pins", "ready"):
            with self.subTest(target=target):
                damaged = deepcopy(control)
                if target == "used":
                    damaged.used -= 1
                elif target == "peak":
                    damaged.peak = 9
                elif target == "pins":
                    damaged.entries["x"]["pins"] = 1
                else:
                    damaged.entries["x"]["ready"] = None
                with self.assertRaises(AssertionError):
                    damaged.check()
                control.check()


def native_trace(attention="none_control", context=257, token_count=1):
    """Literal native schema fixture, independent of trace generation code."""
    return dict(schema_version=1, provenance="independent unittest fixture",
                context_tokens=context, attention=attention,
                tokens=[dict(token=ti, not_before_ns=ti, layers=[
                    dict(layer=li, experts=list(range(8)),
                         kv_blocks_by_query_head=([[0] for _ in range(32)]
                             if attention == "selected_hypothesis" and li % 4 == 3 else []),
                         selection_delay_ns=0, attention_compute_ns=0,
                         route_delay_ns=0, expert_compute_ns=0)
                    for li in range(48)]) for ti in range(token_count)])


class TraceValidationTests(unittest.TestCase):
    def test_valid_minimum_native_schema_in_all_attention_modes(self):
        for attention in ("none_control", "selected_hypothesis", "full"):
            with self.subTest(attention=attention):
                trace = native_trace(attention)
                self.assertIs(validate_trace(trace), trace)

    def test_root_schema_is_strict(self):
        valid = native_trace()
        cases = [None, [], {}, valid | {"extra": 0}]
        for field in valid:
            cases.append({key: value for key, value in valid.items() if key != field})
        for trace in cases:
            with self.subTest(trace=trace), self.assertRaises(ValueError):
                validate_trace(trace)

    def test_root_values_and_metadata_bounds(self):
        invalid = {"schema_version": [0, 2, True, "1"],
                   "provenance": ["", "  ", None, 1],
                   "context_tokens": [0, -1, 1.5, True, 1010001],
                   "attention": ["selected", "sparse", None],
                   "tokens": [[], None, {}, [None]]}
        for field, values in invalid.items():
            for value in values:
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    validate_trace(native_trace() | {field: value})
        validate_trace(native_trace(context=1010000))
        too_many = native_trace()
        too_many["tokens"] = [deepcopy(too_many["tokens"][0]) for _ in range(257)]
        with self.assertRaisesRegex(ValueError, "256"):
            validate_trace(too_many)

    def test_token_layer_order_and_noncausal_timestamps_rejected(self):
        cases = []
        for field, value in [("token", 1), ("token", True), ("not_before_ns", -1),
                             ("not_before_ns", True), ("not_before_ns", 0.5),
                             ("layers", []), ("layers", None)]:
            trace = native_trace()
            trace["tokens"][0][field] = value
            cases.append(trace)
        trace = native_trace(token_count=2)
        trace["tokens"][0]["not_before_ns"] = 10
        cases.append(trace)
        trace = native_trace()
        trace["tokens"][0]["layers"][0]["layer"] = 1
        cases.append(trace)
        trace = native_trace()
        trace["tokens"][0]["layers"][0]["future_route_time"] = 0
        cases.append(trace)
        for index, trace in enumerate(cases):
            with self.subTest(case=index), self.assertRaises(ValueError):
                validate_trace(trace)

    def test_native_routes_require_eight_distinct_bounded_integer_ids(self):
        invalid_routes = [[], list(range(7)), list(range(9)), [0] * 8,
                          [-1] + list(range(1, 8)), [256] + list(range(1, 8)),
                          [True] + list(range(1, 8)), [0.0] + list(range(1, 8)),
                          [None] + list(range(1, 8)), None]
        for route in invalid_routes:
            trace = native_trace()
            trace["tokens"][0]["layers"][0]["experts"] = route
            with self.subTest(route=route), self.assertRaises(ValueError):
                validate_trace(trace)

    def test_stage_offsets_require_nonnegative_integer_durations(self):
        for field in ("selection_delay_ns", "attention_compute_ns", "route_delay_ns", "expert_compute_ns"):
            for value in (-1, 0.5, True, None):
                trace = native_trace()
                trace["tokens"][0]["layers"][0][field] = value
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    validate_trace(trace)

    def test_selected_heads_require_32_valid_nonempty_block_lists(self):
        malformed = [[], [[0]] * 31, [[0]] * 33, None]
        for heads in malformed:
            trace = native_trace("selected_hypothesis")
            trace["tokens"][0]["layers"][3]["kv_blocks_by_query_head"] = heads
            with self.subTest(heads=heads), self.assertRaises(ValueError):
                validate_trace(trace)
        for blocks in ([], [0, 0], [-1], [3], [True], [1.0], [None], [[0]], [{}], None):
            trace = native_trace("selected_hypothesis")
            trace["tokens"][0]["layers"][3]["kv_blocks_by_query_head"][0] = blocks
            with self.subTest(blocks=blocks), self.assertRaises(ValueError):
                validate_trace(trace)

    def test_selections_for_recurrent_or_control_layers_rejected(self):
        for attention, layer in [("none_control", 3), ("full", 3), ("selected_hypothesis", 0)]:
            trace = native_trace(attention)
            trace["tokens"][0]["layers"][layer]["kv_blocks_by_query_head"] = [[0] for _ in range(32)]
            with self.subTest(attention=attention, layer=layer), self.assertRaises(ValueError):
                validate_trace(trace)


class KVGeometryTests(unittest.TestCase):
    def test_grouped_query_union_is_per_shared_kv_head(self):
        trace = native_trace("selected_hypothesis", context=257)
        layer = trace["tokens"][0]["layers"][3]
        layer["kv_blocks_by_query_head"] = [[0] for _ in range(16)] + [[1] for _ in range(16)]
        layer["kv_blocks_by_query_head"][0] = [2, 0]
        layer["kv_blocks_by_query_head"][17] = [2, 1]
        validate_trace(trace)
        reads = kv_reads(layer, trace, 16384)
        # 32 query heads share two KV heads, 16 each. Their unions are {0,2}
        # and {1,2}; the tail is one token, 1 KiB per KV head. No cross-head
        # deduplication is legal, and repeated query demand is not 32 objects.
        self.assertEqual([(r.key, r.size) for r in reads],
                         [("k:3:0:0", 131072), ("k:3:0:2", 1024),
                          ("k:3:1:1", 131072), ("k:3:1:2", 1024)])
        # Padded-expert region = ceil(5,541,888/16,384)*48*256 pages.
        # Per-head stride = ceil(1,010,000/128)*8 pages = 63,128.
        self.assertEqual([r.first_page for r in reads], [4544400, 4544416, 4607536, 4607544])

    def test_common_query_selection_still_has_two_kv_objects(self):
        trace = native_trace("selected_hypothesis")
        reads = kv_reads(trace["tokens"][0]["layers"][3], trace, 16384)
        self.assertEqual([r.key for r in reads], ["k:3:0:0", "k:3:1:0"])
        self.assertEqual(sum(r.size for r in reads), 262144)
        self.assertNotEqual(reads[0].first_page, reads[1].first_page)

    def test_full_attention_tail_and_exact_block_boundary(self):
        for context, sizes in [(1, [1024, 1024]), (128, [131072, 131072]),
                               (129, [131072, 1024, 131072, 1024])]:
            trace = native_trace("full", context=context)
            reads = kv_reads(trace["tokens"][0]["layers"][3], trace, 16384)
            with self.subTest(context=context):
                self.assertEqual([r.size for r in reads], sizes)
                self.assertEqual(sum(r.size for r in reads), context * 2048)

    def test_stable_addresses_do_not_depend_on_selection_order(self):
        trace = native_trace("selected_hypothesis")
        layer = trace["tokens"][0]["layers"][3]
        layer["kv_blocks_by_query_head"][0] = [2, 0, 1]
        before = kv_reads(layer, trace, 16384)
        layer["kv_blocks_by_query_head"][0] = [1, 2, 0]
        self.assertEqual(kv_reads(layer, trace, 16384), before)
        full = native_trace("full")
        full_by_key = {r.key: r for r in kv_reads(full["tokens"][0]["layers"][3], full, 16384)}
        self.assertTrue(all(r == full_by_key[r.key] for r in before))

    def test_none_control_and_recurrent_layers_have_no_kv_requests(self):
        trace = native_trace("none_control")
        self.assertEqual(kv_reads(trace["tokens"][0]["layers"][3], trace, 16384), [])
        trace = native_trace("full")
        self.assertEqual(kv_reads(trace["tokens"][0]["layers"][0], trace, 16384), [])

    def test_page_size_must_divide_selection_block(self):
        trace = native_trace("full", context=129)
        layer = trace["tokens"][0]["layers"][3]
        for page_bytes in (3, 16385, 262144):
            with self.subTest(page_bytes=page_bytes), self.assertRaisesRegex(ValueError, "divide"):
                kv_reads(layer, trace, page_bytes)
        self.assertEqual(len(kv_reads(layer, trace, 131072)), 4)


if __name__ == "__main__":
    if not __debug__:
        raise RuntimeError("Run without python -O: validation assertions are required")
    unittest.main()
