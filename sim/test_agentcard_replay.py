#!/usr/bin/env python3
"""Causality, audited geometry, cache-stage, and accounting regression tests."""
import copy
import unittest

from agentcard_coupled import Cache, DeviceConfig, GEOMETRY, Read, Replay
from run_agentcard_coupled import synthetic_trace


class ReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = DeviceConfig(page_bytes=131072, channels=1, dies_per_channel=1,
                                  planes_per_die=1, channel_bytes_per_ns=131072,
                                  command_ns=1, read_ns=1, bridge_pages=2,
                                  fast_bytes_per_ns=131072)
        cls.trace = synthetic_trace(tokens=2, context_tokens=128,
                                    attention="none_control", blocks_per_query=1)
        cls.base = Replay(cls.device).run(cls.trace, 1)

    def test_audited_geometry_and_dense_matrix_traffic(self):
        g = GEOMETRY
        self.assertEqual(g["expert_weights"], 3 * 3072 * 1024)
        self.assertEqual(g["compact_expert_bytes"], g["expert_weights"] * 9 // 16)
        self.assertEqual(g["resident_core_bytes"], 6145327104 * 9 // 16 + 2082816 * 2)
        self.assertEqual(g["recurrent_state_bytes"], 36 * 64 * 128 * 128 * 4)
        self.assertEqual(g["conv_state_bytes"], 36 * 12288 * 4 * 2)
        f = self.base["fast_byte_counters"]
        weight = sum(f[k] for k in ("resident_attention", "resident_shared_router", "input_embedding", "output_head", "consume_expert"))
        self.assertEqual(weight, 2 * (5066081280 + 2082816 * 2 + 1728))
        self.assertEqual(f["recurrent_conv_rw"], 2 * 2 * (150994944 + 3538944))
        self.assertEqual(f.get("consume_kv", 0), 0)

    def test_route_delay_propagates_without_future_route_oracle(self):
        changed = copy.deepcopy(self.trace)
        changed["tokens"][0]["layers"][5]["route_delay_ns"] = 1234
        result = Replay(self.device).run(changed, 1)
        before = [s for s in self.base["stages"] if s["token"] == 0 and s["layer"] < 5]
        after = [s for s in result["stages"] if s["token"] == 0 and s["layer"] < 5]
        self.assertEqual(before, after)
        b = next(s for s in self.base["stages"] if s["token"] == 0 and s["layer"] == 5 and s["kind"] == "expert")
        a = next(s for s in result["stages"] if s["token"] == 0 and s["layer"] == 5 and s["kind"] == "expert")
        self.assertEqual(a["request_ns"] - b["request_ns"], 1234)
        self.assertEqual(result["tokens"][0]["done_ns"] - self.base["tokens"][0]["done_ns"], 1234)
        self.assertEqual(result["tokens"][1]["done_ns"] - self.base["tokens"][1]["done_ns"], 1234)

    def test_future_routes_cannot_change_prefix(self):
        changed = copy.deepcopy(self.trace)
        changed["tokens"][1]["layers"][47]["experts"] = list(range(128, 136))
        result = Replay(self.device).run(changed, 1)
        self.assertEqual(result["tokens"][0], self.base["tokens"][0])
        self.assertEqual(result["stages"][:-1], self.base["stages"][:-1])

    def test_warmup_measurement_and_terminal_drain(self):
        r = self.base
        self.assertEqual([t["phase"] for t in r["tokens"]], ["warmup", "measurement"])
        self.assertEqual(r["summary"]["mean_service_ms"], r["tokens"][1]["service_ns"] / 1e6)
        self.assertEqual(r["summary"]["terminal_pending_pages"], 0)
        self.assertEqual(sum(t["fast_bytes"] for t in r["tokens"]), sum(r["fast_byte_counters"].values()))
        for t in r["tokens"]:
            self.assertLessEqual(t["necessary_resource_bound_ns"], t["service_ns"])
        self.assertLessEqual(r["summary"]["card_allocated_bytes"], r["summary"]["assumed_card_bytes"])

    def test_prepin_stage_does_not_evict_later_demanded_hit(self):
        r = Replay(DeviceConfig(page_bytes=16, channels=1), cache_bytes=20)
        a, old, new = Read("a", "expert", 10, 0), Read("old", "expert", 10, 1), Read("new", "expert", 10, 2)
        for item in (a, old):
            r.cache.acquire(item, 0); r.cache.mark_ready(item.key, 0); r.cache.release(item.key, 0)
        # a is oldest; naive acquire(new), acquire(a) would evict that useful hit.
        r.stage([new, a], 0, 0, 0, "expert")
        self.assertEqual(r.stages[0]["hit_bytes"], 10)
        self.assertEqual(set(r.cache.entries), {"a", "new"})
        self.assertEqual(r.fast.bytes["fill_expert"], 10)
        self.assertEqual(r.fast.bytes["consume_expert"], 20)
        r.cache.check(); r.service.check()

    def test_hits_still_charge_fast_consumption(self):
        r = Replay(DeviceConfig(page_bytes=16, channels=1), cache_bytes=32)
        a = Read("a", "expert", 10, 0)
        now = r.stage([a], 0, 0, 0, "expert")
        r.stage([a], now, 1, 0, "expert")
        self.assertEqual(r.fast.bytes["fill_expert"], 10)
        self.assertEqual(r.fast.bytes["consume_expert"], 20)
        self.assertEqual(r.service.stats["completed_pages"], 1)
        self.assertEqual(r.stages[1]["dependency_wait_ns"], 0)

    def test_capacity_rejections_and_no_partial_stage_mutation(self):
        r = Replay(cache_bytes=20)
        with self.assertRaisesRegex(ValueError, "pinned stage"):
            r.stage([Read("a", "expert", 11, 0), Read("b", "expert", 10, 1)], 0, 0, 0, "expert")
        self.assertEqual(r.cache.used, 0)
        with self.assertRaisesRegex(ValueError, "card capacity"):
            Replay(self.device, card_bytes=4 * 2**30).run(self.trace, 1)
        with self.assertRaises(ValueError):
            Replay(self.device).run(self.trace, 2)
        with self.assertRaises(ValueError):
            Replay(format="imaginary")

    def test_stage_duplicate_identity_rejected(self):
        a = Read("a", "expert", 10, 0)
        with self.assertRaises(ValueError):
            Replay().stage([a, a], 0, 0, 0, "expert")

    def test_sparse_summary_and_gqa_bytes(self):
        trace = synthetic_trace(tokens=2, context_tokens=128, blocks_per_query=1)
        r = Replay(self.device).run(trace, 1)
        # All16 grouped queries select same128-token block: two heads, not32.
        self.assertEqual(r["summary"]["kv_demand_bytes_per_token"], 12 * 2 * 128 * 1024)
        self.assertEqual(r["fast_byte_counters"]["selection_summary"], 2 * 12 * 2048)
        self.assertEqual(r["summary"]["kv_byte_hit_fraction"], 1)
        self.assertEqual(r["summary"]["kv_physical_bytes_per_token"], 0)


if __name__ == "__main__":
    unittest.main()
