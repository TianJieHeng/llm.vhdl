#!/usr/bin/env python3
"""Independent small oracles for Experiment 003 selection and cache policies.

Fixtures spell out two physical KV groups and 32 query heads. Expected byte
counts, retained block identities and eviction victims are hand-derived, not
computed by another policy invocation. These tests use metadata only: no model
weights, RTL, hardware, downloads or external packages.

Run: python3 -m unittest discover -s sim -p 'test_agentcard_policies.py' -v
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import unittest

from agentcard_coupled import DeviceConfig, Read, Replay
from agentcard_policies import (Candidate, CausalCache, PolicyReplay, SelectionPolicy,
                               audit_selection, select_pages)


BLOCK_BYTES = 131072
PAGE_BYTES = 16384


def shared_heads(*candidates):
    """Thirty-two distinct lists referring to the same logical block IDs."""
    return [list(candidates) for _ in range(32)]


def capped(blocks, minimum=1, **kwargs):
    return SelectionPolicy(mode="capped", cap_bytes=blocks * BLOCK_BYTES,
                           min_blocks_per_query=minimum, **kwargs)


def fill_and_release(cache, read, now=0):
    state = cache.acquire(read, now)
    if state == "miss":
        cache.mark_ready(read.key, now)
    cache.release(read.key, now)
    return state


class PhysicalSelectionTests(unittest.TestCase):
    def test_lossless_union_has_two_physical_groups_not_32_copies(self):
        heads = shared_heads(Candidate(0, 9.0), Candidate(1, 2.0), Candidate(2, 1.0))
        result = select_pages(heads, 384, SelectionPolicy(mode="union"))
        self.assertEqual(result["selected_keys"],
                         [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)])
        self.assertEqual(result["selected_by_query"], [[0, 1, 2] for _ in range(32)])
        self.assertEqual(result["requested_union_blocks"], 6)
        self.assertEqual(result["served_union_blocks"], 6)
        self.assertEqual(result["omitted_union_blocks"], 0)
        self.assertEqual(result["requested_union_bytes"], 786432)
        self.assertEqual(result["served_union_bytes"], 786432)
        self.assertEqual(result["omitted_union_bytes"], 0)
        self.assertEqual(result["requested_physical_pages"], 48)
        self.assertEqual(result["served_physical_pages"], 48)
        self.assertFalse(result["fallback"])
        self.assertEqual(result["min_query_byte_coverage"], 1.0)
        self.assertEqual(result["zero_coverage_queries"], 0)

    def test_cap_is_one_total_for_both_groups(self):
        heads = shared_heads(Candidate(0, 9.0), Candidate(1, 2.0), Candidate(2, 1.0))
        result = select_pages(heads, 384, capped(3))
        # Each group must first serve block zero. Only ONE additional block
        # fits globally. A cap independently applied to each group serves six.
        self.assertEqual(result["selected_keys"], [(0, 0), (0, 1), (1, 0)])
        self.assertEqual(result["served_union_blocks"], 3)
        self.assertEqual(result["served_union_bytes"], 393216)
        self.assertEqual(result["served_physical_pages"], 24)
        self.assertEqual(result["selected_by_query"],
                         [[0, 1] for _ in range(16)] + [[0] for _ in range(16)])
        self.assertFalse(result["fallback"])

    def test_explicit_omissions_and_query_score_mass_are_exact(self):
        heads = shared_heads(Candidate(0, 9.0), Candidate(1, 2.0), Candidate(2, 1.0))
        result = select_pages(heads, 384, capped(2))
        self.assertEqual(result["selected_keys"], [(0, 0), (1, 0)])
        self.assertEqual(result["requested_union_blocks"], 6)
        self.assertEqual(result["served_union_blocks"], 2)
        self.assertEqual(result["omitted_union_blocks"], 4)
        self.assertEqual(result["requested_union_bytes"], 786432)
        self.assertEqual(result["served_union_bytes"], 262144)
        self.assertEqual(result["omitted_union_bytes"], 524288)
        self.assertEqual(result["requested_physical_pages"], 48)
        self.assertEqual(result["served_physical_pages"], 16)
        self.assertEqual(result["requested_physical_bytes"], 786432)
        self.assertEqual(result["served_physical_bytes"], 262144)
        self.assertEqual(result["omitted_physical_bytes"], 524288)
        self.assertEqual(result["requested_score_mass"], 384.0)
        self.assertEqual(result["omitted_score_mass"], 96.0)
        self.assertEqual(result["zero_coverage_queries"], 0)
        self.assertAlmostEqual(result["min_query_byte_coverage"], 1 / 3)
        self.assertEqual(len(result["per_query"]), 32)
        for row in result["per_query"]:
            self.assertEqual(row["requested_blocks"], 3)
            self.assertEqual(row["served_blocks"], 1)
            self.assertEqual(row["omitted_blocks"], 2)
            self.assertAlmostEqual(row["byte_coverage"], 1 / 3)
            self.assertEqual(row["score_mass_coverage"], 0.75)
            self.assertEqual(row["mandatory_requested"], 0)
            self.assertEqual(row["mandatory_served"], 0)

    def test_tail_uses_actual_bytes_but_whole_physical_pages(self):
        heads = shared_heads(Candidate(2, 1.0))
        result = select_pages(heads, 257, SelectionPolicy(mode="union"))
        self.assertEqual(result["selected_keys"], [(0, 2), (1, 2)])
        self.assertEqual(result["requested_union_bytes"], 2048)
        self.assertEqual(result["served_union_bytes"], 2048)
        self.assertEqual(result["requested_physical_pages"], 2)
        self.assertEqual(result["served_physical_pages"], 2)
        self.assertEqual(result["requested_physical_bytes"], 32768)
        self.assertEqual(result["served_physical_bytes"], 32768)
        # Two 1 KiB tails require two separate 16 KiB physical pages. Charging
        # useful bytes alone would falsely declare a 16 KiB cap feasible.
        physical_cap = SelectionPolicy(mode="capped", cap_bytes=16384,
                                       min_blocks_per_query=1)
        overflow = select_pages(heads, 257, physical_cap)
        self.assertTrue(overflow["fallback"])
        self.assertEqual(overflow["selected_keys"], [(0, 2), (1, 2)])
        self.assertEqual(overflow["omitted_union_blocks"], 0)

    def test_partial_tail_query_byte_coverage_is_not_block_fraction(self):
        heads = shared_heads(Candidate(0, 9.0), Candidate(2, 1.0))
        result = select_pages(heads, 257, capped(2))
        self.assertEqual(result["selected_keys"], [(0, 0), (1, 0)])
        self.assertEqual(result["requested_union_bytes"], 264192)
        self.assertEqual(result["served_union_bytes"], 262144)
        self.assertEqual(result["omitted_union_bytes"], 2048)
        self.assertEqual(result["requested_physical_pages"], 18)
        self.assertEqual(result["served_physical_pages"], 16)
        self.assertAlmostEqual(result["per_query"][0]["byte_coverage"], 128 / 129)
        self.assertEqual(result["per_query"][0]["score_mass_coverage"], 0.9)

    def test_cache_hits_still_count_against_the_union_cap(self):
        heads = shared_heads(Candidate(0, 9.0), Candidate(1, 2.0), Candidate(2, 1.0))
        result = select_pages(heads, 384, capped(2, cache_tiebreak=True),
                              cached=frozenset((g, b) for g in range(2) for b in range(3)))
        self.assertEqual(result["selected_keys"], [(0, 0), (1, 0)])
        self.assertEqual(result["served_physical_pages"], 16)
        self.assertEqual(result["omitted_union_blocks"], 4)

    def test_page_size_changes_physical_tail_accounting(self):
        heads = shared_heads(Candidate(2, 1.0))
        result = select_pages(heads, 257, SelectionPolicy(mode="capped", cap_bytes=2048,
                                                        min_blocks_per_query=1),
                              page_bytes=1024)
        self.assertFalse(result["fallback"])
        self.assertEqual(result["served_physical_pages"], 2)
        self.assertEqual(result["served_union_bytes"], 2048)


class ProtectionAndFairnessTests(unittest.TestCase):
    def test_low_score_unique_query_gets_its_minimum(self):
        heads = shared_heads(Candidate(0, 100.0), Candidate(1, 90.0))
        heads[15] = [Candidate(7, 0.000001)]
        result = select_pages(heads, 1024, capped(3))
        self.assertEqual(result["selected_keys"], [(0, 0), (0, 7), (1, 0)])
        self.assertEqual(result["selected_by_query"][15], [7])
        self.assertTrue(all(row["served_blocks"] >= 1 for row in result["per_query"]))
        self.assertEqual(result["zero_coverage_queries"], 0)
        self.assertFalse(result["fallback"])

    def test_minimum_two_is_enforced_per_query_not_per_group(self):
        heads = shared_heads(Candidate(0, 100.0), Candidate(1, 90.0), Candidate(4, 80.0))
        heads[15] = [Candidate(2, 0.000002), Candidate(3, 0.000001)]
        result = select_pages(heads, 640, capped(6, minimum=2))
        self.assertEqual(result["selected_keys"],
                         [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1)])
        self.assertTrue(all(row["served_blocks"] >= 2 for row in result["per_query"]))
        self.assertEqual(result["selected_by_query"][15], [2, 3])
        self.assertFalse(result["fallback"])

    def test_minimum_larger_than_candidate_count_needs_only_available_blocks(self):
        result = select_pages(shared_heads(Candidate(0, 1.0)), 128,
                              capped(2, minimum=8))
        self.assertEqual(result["selected_keys"], [(0, 0), (1, 0)])
        self.assertFalse(result["fallback"])
        self.assertEqual(result["min_query_byte_coverage"], 1.0)

    def test_rare_critical_page_survives_only_with_explicit_protection(self):
        heads = shared_heads(Candidate(0, 100.0), Candidate(1, 90.0))
        heads[15].append(Candidate(7, 0.000000001, mandatory=True))
        protected = select_pages(heads, 1024, capped(3))
        self.assertEqual(protected["selected_keys"], [(0, 0), (0, 7), (1, 0)])
        self.assertEqual(protected["per_query"][15]["mandatory_requested"], 1)
        self.assertEqual(protected["per_query"][15]["mandatory_served"], 1)
        self.assertIn(7, protected["selected_by_query"][15])
        self.assertNotIn(7, protected["selected_by_query"][0])
        self.assertFalse(protected["fallback"])
        # Attribution control: remove only the mandatory flag. An attractive
        # aggregate score must now beat this rare, tiny-score page.
        heads[15][-1] = Candidate(7, 0.000000001)
        unprotected = select_pages(heads, 1024, capped(3))
        self.assertNotIn((0, 7), unprotected["selected_keys"])
        self.assertEqual(unprotected["per_query"][15]["mandatory_requested"], 0)
        self.assertEqual(unprotected["omitted_union_blocks"], 2)

    def test_mandatory_block_is_not_dropped_to_make_budget_look_feasible(self):
        heads = shared_heads(Candidate(0, 100.0), Candidate(1, 0.0, mandatory=True))
        result = select_pages(heads, 256, capped(1))
        self.assertTrue(result["fallback"])
        self.assertEqual(result["selected_keys"], [(0, 0), (0, 1), (1, 0), (1, 1)])
        self.assertEqual(result["served_union_bytes"], 524288)
        self.assertEqual(result["omitted_union_bytes"], 0)
        self.assertEqual(result["omitted_union_blocks"], 0)
        self.assertEqual(result["omitted_score_mass"], 0.0)
        self.assertTrue(all(row["mandatory_served"] == 1 for row in result["per_query"]))

    def test_disjoint_query_fairness_overflow_falls_back_to_full_union(self):
        heads = [[Candidate(q % 16, float(32 - q))] for q in range(32)]
        result = select_pages(heads, 2048, capped(31))
        self.assertTrue(result["fallback"])
        self.assertEqual(result["selected_keys"], [(g, b) for g in range(2) for b in range(16)])
        self.assertEqual(result["requested_union_blocks"], 32)
        self.assertEqual(result["served_union_blocks"], 32)
        self.assertEqual(result["served_union_bytes"], 4194304)
        self.assertEqual(result["served_physical_pages"], 256)
        self.assertEqual(result["omitted_union_blocks"], 0)
        self.assertEqual(result["min_query_byte_coverage"], 1.0)
        self.assertEqual(result["zero_coverage_queries"], 0)
        self.assertEqual(result["selected_by_query"], [[q % 16] for q in range(32)])

    def test_zero_score_mass_is_explicitly_undefined_and_does_not_hide_omissions(self):
        result = select_pages(shared_heads(Candidate(0, 0.0), Candidate(1, 0.0)),
                              256, capped(2))
        self.assertEqual(result["requested_score_mass"], 0.0)
        self.assertEqual(result["omitted_score_mass"], 0.0)
        self.assertEqual(result["omitted_union_blocks"], 2)
        self.assertEqual(result["min_query_byte_coverage"], 0.5)
        for row in result["per_query"]:
            self.assertIsNone(row["score_mass_coverage"])
            self.assertEqual(row["omitted_blocks"], 1)


class DeterminismAndValidationTests(unittest.TestCase):
    def test_candidate_reordering_cannot_change_ties_or_mutate_inputs(self):
        heads = shared_heads(Candidate(2, 1.0), Candidate(0, 3.0), Candidate(1, 1.0))
        before = deepcopy(heads)
        result = select_pages(heads, 384, capped(3))
        reordered = [list(reversed(items)) for items in heads]
        self.assertEqual(result, select_pages(reordered, 384, capped(3)))
        self.assertEqual(heads, before)
        self.assertEqual(result["selected_keys"], [(0, 0), (0, 1), (1, 0)])

    def test_cache_only_breaks_exact_priority_ties(self):
        tied = shared_heads(Candidate(0, 3.0), Candidate(1, 1.0), Candidate(2, 1.0))
        cached = frozenset({(1, 2)})
        plain = select_pages(tied, 384, capped(3), cached=cached)
        biased = select_pages(tied, 384, capped(3, cache_tiebreak=True), cached=cached)
        self.assertEqual(plain["selected_keys"], [(0, 0), (0, 1), (1, 0)])
        self.assertEqual(biased["selected_keys"], [(0, 0), (1, 0), (1, 2)])
        unequal = shared_heads(Candidate(0, 3.0), Candidate(1, 2.0), Candidate(2, 1.0))
        higher = select_pages(unequal, 384, capped(3, cache_tiebreak=True), cached=cached)
        self.assertEqual(higher["selected_keys"], [(0, 0), (0, 1), (1, 0)])

    def test_identical_queries_coalesce_but_duplicate_ids_within_query_reject(self):
        heads = shared_heads(Candidate(0, 1.0))
        result = select_pages(heads, 128, capped(2))
        self.assertEqual(result["served_union_blocks"], 2)
        self.assertEqual(result["served_physical_pages"], 16)
        heads[0].append(Candidate(0, 2.0))
        with self.assertRaises(ValueError):
            select_pages(heads, 128, capped(2))

    def test_invalid_scores_are_rejected_before_any_selection(self):
        for score in (-1.0, float("nan"), float("inf"), -float("inf"), True, "1", None):
            with self.subTest(score=score), self.assertRaises(ValueError):
                heads = shared_heads(Candidate(0, score))
                select_pages(heads, 128, capped(2))

    def test_invalid_or_unknown_block_ids_are_rejected(self):
        for block in (-1, 1, True, 0.0, "0", None):
            with self.subTest(block=block), self.assertRaises(ValueError):
                select_pages(shared_heads(Candidate(block, 1.0)), 128, capped(2))

    def test_mandatory_is_boolean_not_a_truthy_coercion(self):
        for mandatory in (0, 1, "yes", None):
            with self.subTest(mandatory=mandatory), self.assertRaises(ValueError):
                select_pages(shared_heads(Candidate(0, 1.0, mandatory=mandatory)),
                             128, capped(2))

    def test_malformed_head_counts_and_empty_queries_are_rejected(self):
        cases = [None, [], shared_heads(Candidate(0, 1.0))[:-1],
                 shared_heads(Candidate(0, 1.0)) + [[Candidate(0, 1.0)]],
                 [[] for _ in range(32)]]
        for index, heads in enumerate(cases):
            with self.subTest(case=index), self.assertRaises(ValueError):
                select_pages(heads, 128, capped(2))

    def test_policy_rejects_unknown_modes_and_invalid_numbers(self):
        base = capped(2)
        cases = [("mode", "guess"), ("cap_bytes", -1), ("cap_bytes", 1.5),
                 ("cap_bytes", True), ("min_blocks_per_query", -1),
                 ("min_blocks_per_query", 0.5), ("min_blocks_per_query", True),
                 ("cache_tiebreak", 1)]
        for field, value in cases:
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                select_pages(shared_heads(Candidate(0, 1.0)), 128,
                             replace(base, **{field: value}))


class CausalCacheTests(unittest.TestCase):
    def test_reuse_policy_protects_observed_reuse_while_lru_does_not(self):
        a, b, c = (Read(key, kind, 8, index)
                   for index, (key, kind) in enumerate((("a", "expert"), ("b", "kv"), ("c", "kv"))))
        for policy, survivors in (("lru", {"b", "c"}), ("windowed_reuse", {"a", "c"})):
            with self.subTest(policy=policy):
                cache = CausalCache(16, policy=policy)
                cache.begin_token(0)
                self.assertEqual(fill_and_release(cache, a), "miss")
                self.assertEqual(fill_and_release(cache, a), "hit")
                self.assertEqual(fill_and_release(cache, b), "miss")
                # a is older but has two observed demands; b has one.
                self.assertEqual(fill_and_release(cache, c), "miss")
                self.assertEqual(set(cache.entries), survivors)
                self.assertEqual(cache.stats["evicted_bytes"], 8)
                cache.check()

    def test_reuse_counts_saturate_so_historical_popularity_cannot_pin_forever(self):
        cache = CausalCache(16, policy="windowed_reuse")
        cache.begin_token(0)
        a, b, c = (Read(k, "expert", 8, i) for i, k in enumerate("abc"))
        for _ in range(10):
            fill_and_release(cache, a)
        for _ in range(2):
            fill_and_release(cache, b)
        fill_and_release(cache, c)
        # Both historical counts are capped at two, so LRU breaks the tie.
        self.assertEqual(set(cache.entries), {"b", "c"})
        cache.check()

    def test_eight_token_epoch_forgets_stale_reuse(self):
        cache = CausalCache(16, policy="windowed_reuse")
        cache.begin_token(0)
        a, b, c = (Read(k, "expert", 8, i) for i, k in enumerate("abc"))
        fill_and_release(cache, a)
        fill_and_release(cache, a)
        fill_and_release(cache, b)
        for token in range(1, 9):
            cache.begin_token(token)
        fill_and_release(cache, c)
        self.assertEqual(set(cache.entries), {"b", "c"})
        cache.check()

    def test_pinned_and_pending_objects_never_become_eviction_candidates(self):
        for policy in ("lru", "windowed_reuse"):
            with self.subTest(policy=policy):
                cache = CausalCache(16, policy=policy)
                cache.begin_token(0)
                a, b, c = (Read(k, "kv", 8, i) for i, k in enumerate("abc"))
                cache.acquire(a, 0)
                for _ in range(3):
                    fill_and_release(cache, b)
                fill_and_release(cache, c)
                self.assertEqual(set(cache.entries), {"a", "c"})
                self.assertIsNone(cache.entries["a"]["ready"])
                cache.mark_ready("a", 0)
                cache.release("a", 0)
                cache.check()

    def test_policy_does_not_use_unobserved_future_object_identity(self):
        snapshots = []
        for future_key in ("future_popular_a", "future_popular_b"):
            cache = CausalCache(16, policy="windowed_reuse")
            cache.begin_token(0)
            for index, key in enumerate("abc"):
                fill_and_release(cache, Read(key, "expert", 8, index))
            snapshots.append((list(cache.entries), dict(cache.stats)))
            cache.begin_token(1)
            for _ in range(10):
                fill_and_release(cache, Read(future_key, "expert", 8, 10))
        self.assertEqual(snapshots[0], snapshots[1])
        self.assertEqual(snapshots[0][0], ["b", "c"])


def replay_fixture(tokens=2):
    """Native 48-layer trace with literal routes, separate scored annotations."""
    trace = dict(schema_version=1, provenance="independent policy test fixture",
                 context_tokens=256, attention="selected_hypothesis", tokens=[])
    candidates = {}
    for token in range(tokens):
        layers = []
        for layer in range(48):
            heads = [[0, 1] for _ in range(32)] if layer % 4 == 3 else []
            layers.append(dict(layer=layer, experts=list(range(8)),
                               kv_blocks_by_query_head=heads, selection_delay_ns=0,
                               attention_compute_ns=0, route_delay_ns=0, expert_compute_ns=0))
            if heads:
                candidates[token, layer] = shared_heads(Candidate(0, 2.0), Candidate(1, 1.0))
        trace["tokens"].append(dict(token=token, not_before_ns=0, layers=layers))
    return trace, candidates


def metadata_device():
    return DeviceConfig(page_bytes=131072, channels=1, dies_per_channel=1,
                        planes_per_die=1, channel_bytes_per_ns=131072,
                        command_ns=1, read_ns=1, bridge_pages=2,
                        fast_bytes_per_ns=131072)


class ReplayPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trace, cls.candidates = replay_fixture()
        cls.reference = Replay(metadata_device()).run(cls.trace, 1)
        cls.union = PolicyReplay(metadata_device(), candidates=cls.candidates,
                                selection=SelectionPolicy(mode="union"),
                                cache_policy="lru").run(cls.trace, 1)

    def test_lossless_union_lru_is_exactly_experiment_002_for_all_legacy_results(self):
        for key, value in self.reference.items():
            with self.subTest(key=key):
                if key == "summary":
                    self.assertEqual({field: self.union[key][field] for field in value}, value)
                else:
                    self.assertEqual(self.union[key], value)
        self.assertEqual(self.union["summary"]["omitted_union_bytes_per_token"], 0)
        self.assertEqual(self.union["summary"]["lossy_stages"], 0)
        self.assertEqual(self.union["summary"]["union_byte_coverage"], 1.0)

    def test_capped_selections_change_actual_storage_and_consumer_traffic(self):
        result = PolicyReplay(metadata_device(), candidates=self.candidates,
                              selection=capped(2)).run(self.trace, 1)
        # Twelve full-attention layers, two KV groups, one 128 KiB block/group.
        self.assertEqual(result["tokens"][0]["physical_kv_bytes"], 3145728)
        self.assertEqual(self.union["tokens"][0]["physical_kv_bytes"], 6291456)
        self.assertEqual(result["fast_byte_counters"]["consume_kv"], 6291456)
        self.assertEqual(result["fast_byte_counters"]["fill_kv"], 3145728)
        self.assertEqual(result["summary"]["kv_demand_bytes_per_token"], 3145728)
        self.assertEqual(result["summary"]["requested_union_bytes_per_token"], 6291456)
        self.assertEqual(result["summary"]["served_union_bytes_per_token"], 3145728)
        self.assertEqual(result["summary"]["omitted_union_bytes_per_token"], 3145728)
        self.assertEqual(result["summary"]["lossy_stages"], 12)
        self.assertEqual(result["summary"]["union_byte_coverage"], 0.5)
        self.assertEqual(result["summary"]["terminal_pending_pages"], 0)
        self.assertEqual(len(result["coverage"]), 24)
        self.assertTrue(all(s["objects"] == 2 for s in result["stages"]
                            if s["kind"] == "kv" and s["objects"]))

    def test_future_scores_and_routes_cannot_change_completed_prefix(self):
        base = PolicyReplay(metadata_device(), candidates=self.candidates,
                            selection=capped(2, cache_tiebreak=True),
                            cache_policy="windowed_reuse").run(self.trace, 1)
        trace, candidates = deepcopy(self.trace), deepcopy(self.candidates)
        trace["tokens"][1]["layers"][47]["experts"] = list(range(128, 136))
        candidates[1, 47] = shared_heads(Candidate(0, 1.0), Candidate(1, 100.0, mandatory=True))
        changed = PolicyReplay(metadata_device(), candidates=candidates,
                               selection=capped(2, cache_tiebreak=True),
                               cache_policy="windowed_reuse").run(trace, 1)
        self.assertEqual(changed["tokens"][0], base["tokens"][0])
        self.assertEqual(changed["stages"][:-2], base["stages"][:-2])
        self.assertEqual(changed["coverage"][:-1], base["coverage"][:-1])
        self.assertNotEqual(changed["coverage"][-1]["selected_keys"],
                            base["coverage"][-1]["selected_keys"])
        self.assertGreater(changed["tokens"][1]["physical_expert_bytes"],
                           base["tokens"][1]["physical_expert_bytes"])

    def test_expert_hits_are_prepinned_under_both_cache_policies(self):
        for policy in ("lru", "windowed_reuse"):
            with self.subTest(policy=policy):
                replay = PolicyReplay(DeviceConfig(page_bytes=16, channels=1),
                                      cache_bytes=16, cache_policy=policy)
                replay.cache.begin_token(0)
                a, old, new = (Read(k, "expert", 8, i) for i, k in enumerate(("a", "old", "new")))
                fill_and_release(replay.cache, a)
                fill_and_release(replay.cache, old)
                replay.stage([new, a], 0, 0, 0, "expert")
                self.assertEqual(set(replay.cache.entries), {"a", "new"})
                self.assertEqual(replay.stages[0]["hit_bytes"], 8)
                self.assertEqual(replay.fast.bytes["fill_expert"], 8)
                self.assertEqual(replay.fast.bytes["consume_expert"], 16)
                replay.cache.check()
                replay.service.check()

    def test_fallback_too_large_for_cache_is_rejected_without_partial_allocation(self):
        candidates = {(0, 3): shared_heads(Candidate(0, 2.0), Candidate(1, 1.0))}
        replay = PolicyReplay(metadata_device(), cache_bytes=3 * BLOCK_BYTES,
                              candidates=candidates, selection=capped(1))
        replay.context_tokens = 256
        reads = [Read(f"k:3:{g}:{b}", "kv", BLOCK_BYTES, g * 2 + b)
                 for g in range(2) for b in range(2)]
        with self.assertRaisesRegex(ValueError, "pinned stage"):
            replay.stage(reads, 0, 0, 3, "kv")
        self.assertEqual(replay.cache.used, 0)
        self.assertFalse(replay.cache.entries)
        self.assertEqual(replay.service.stats["submitted_pages"], 0)

    def test_candidate_annotations_must_match_each_query_not_just_union(self):
        trace, candidates = replay_fixture()
        # Other heads keep both IDs in the grouped union, so a union-only
        # comparison cannot notice that query zero requests a different set.
        trace["tokens"][0]["layers"][3]["kv_blocks_by_query_head"][0] = [0]
        with self.assertRaises(ValueError):
            PolicyReplay(metadata_device(), candidates=candidates,
                         selection=capped(2)).run(trace, 1)


class SelectionAuditMutationTests(unittest.TestCase):
    def setUp(self):
        self.heads = shared_heads(Candidate(0, 9.0), Candidate(1, 2.0), Candidate(2, 1.0))
        self.policy = capped(2)
        self.control = select_pages(self.heads, 384, self.policy)

    def assert_rejected(self, damaged, heads=None, policy=None, context=384, message=None):
        heads = self.heads if heads is None else heads
        policy = self.policy if policy is None else policy
        # Positive control and explicit unchecked attribution control: merely
        # carrying/reading corrupted result data must not raise by itself.
        audit_selection(self.heads, 384, self.policy, self.control)
        self.assertIsInstance(dict(damaged), dict)
        if message is None:
            with self.assertRaises(AssertionError):
                audit_selection(heads, context, policy, damaged)
        else:
            with self.assertRaisesRegex(AssertionError, message):
                audit_selection(heads, context, policy, damaged)

    def test_aggregate_accounting_mutants_are_detected(self):
        fields = ("requested_union_bytes", "served_union_bytes", "omitted_union_bytes",
                  "requested_union_blocks", "served_union_blocks", "omitted_union_blocks",
                  "requested_physical_bytes", "served_physical_bytes", "omitted_physical_bytes",
                  "requested_physical_pages", "served_physical_pages", "omitted_physical_pages",
                  "requested_score_mass", "omitted_score_mass", "min_query_byte_coverage",
                  "zero_coverage_queries")
        for field in fields:
            with self.subTest(field=field):
                damaged = deepcopy(self.control)
                damaged[field] += 1
                self.assert_rejected(damaged)
        damaged = deepcopy(self.control)
        damaged["lossy"] = False
        self.assert_rejected(damaged)

    def test_per_query_accounting_mutants_are_detected(self):
        fields = ("requested_blocks", "served_blocks", "omitted_blocks", "requested_bytes",
                  "served_bytes", "byte_coverage", "requested_score_mass", "served_score_mass",
                  "score_mass_coverage", "mandatory_requested", "mandatory_served")
        for field in fields:
            with self.subTest(field=field):
                damaged = deepcopy(self.control)
                damaged["per_query"][15][field] += 1
                self.assert_rejected(damaged)

    def test_membership_and_duplicate_key_mutants_are_detected(self):
        damaged = deepcopy(self.control)
        damaged["selected_keys"].append((0, 0))
        self.assert_rejected(damaged, message="duplicate physical key")
        damaged = deepcopy(self.control)
        damaged["selected_keys"].append((0, 3))
        self.assert_rejected(damaged, message="unrequested page")
        damaged = deepcopy(self.control)
        damaged["selected_by_query"][0] = [1]
        self.assert_rejected(damaged)

    def test_physical_cap_guard_catches_an_otherwise_consistent_overfull_decision(self):
        # A valid full union has internally consistent accounting. Re-labeling
        # it as a capped, non-fallback choice must hit the cap guard itself.
        damaged = select_pages(self.heads, 384, SelectionPolicy(mode="union"))
        self.assert_rejected(damaged, message="physical union cap")

    def test_false_fallback_cannot_hide_an_omitted_request(self):
        damaged = deepcopy(self.control)
        damaged["fallback"] = True
        self.assert_rejected(damaged, message="lossless/fallback omitted request")

    def test_required_page_guard_catches_a_consistent_unprotected_decision(self):
        # The candidate request and scores are identical in both arms. Only
        # the fixture's required flag changes. Every byte counter stays valid.
        heads = shared_heads(Candidate(0, 100.0), Candidate(1, 90.0))
        heads[15].append(Candidate(7, 0.000000001))
        policy = capped(3)
        damaged = select_pages(heads, 1024, policy)
        self.assertNotIn((0, 7), damaged["selected_keys"])
        heads[15][-1] = Candidate(7, 0.000000001, mandatory=True)
        self.assert_rejected(damaged, heads=heads, policy=policy, context=1024,
                             message="required page omitted")

    def test_fairness_guard_catches_consistent_but_insufficient_query_minimum(self):
        # All accounting is valid under minimum one. Under minimum two this
        # same physically feasible result starves every query of its contract.
        policy = capped(3, minimum=2)
        self.assert_rejected(self.control, policy=policy, message="query fairness")


if __name__ == "__main__":
    unittest.main()
