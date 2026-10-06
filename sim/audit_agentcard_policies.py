#!/usr/bin/env python3
"""Independent, standard-library audit of experiment003's fixed 32-token matrix.

Usage: python3 sim/audit_agentcard_policies.py path/to/results.json

No simulator module is imported. A separate heap-based, metadata-only cache
replay reconstructs hits, misses, evictions, physical page placement, occupied
resource times and query coverage from the published fixture definition.
It does not independently reconstruct the kernel's scheduled service time.
Native/reference and fallback schedule identities are checked separately.
"""
import argparse
from collections import Counter
import hashlib
from heapq import heapify, heappop, heappush
import json
import math
from random import Random

TOKENS = 32
WARMUP = 8
N = TOKENS - WARMUP
CACHE = 8 * 2**30
BLOCK = 131072
EXPERT = 5308416
PAGE = 16384
EXPERT_STRIDE = (5541888 + PAGE - 1) // PAGE
KV_BASE = EXPERT_STRIDE * 48 * 256
KV_STRIDE = 7891 * 8
FIXED_FAST = 5379316416
DEVICE = dict(page_bytes=16384, channels=32, dies_per_channel=8,
              planes_per_die=2, channel_bytes_per_ns=2, command_ns=100,
              read_ns=40000, shared_command_data_bus=True, bridge_pages=32,
              fast_bytes_per_ns=256, retry_every=0, retry_ns=40000)
CASES = {
    "native_full_lru": ("native", "lru", "union"),
    "disjoint_union_lru": ("disjoint", "lru", "union"),
    "disjoint_union_reuse": ("disjoint", "reuse", "union"),
    "disjoint_cap_lru": ("disjoint", "lru", "cap"),
    "phase_churn_union_lru": ("phase", "lru", "union"),
    "phase_churn_union_reuse": ("phase", "reuse", "union"),
    "phase_churn_cap_lru": ("phase", "lru", "cap"),
    "phase_churn_cap_cached": ("phase", "lru", "cached"),
    "disjoint_overflow_fallback": ("disjoint", "lru", "fallback"),
}


def equal(actual, expected, label):
    if isinstance(expected, float):
        assert isinstance(actual, (int, float)) and math.isclose(
            actual, expected, rel_tol=1e-12, abs_tol=1e-9), (label, actual, expected)
    else:
        assert actual == expected, (label, actual, expected)


def reconstruct(scenario, policy, selection):
    rng = Random(20261006)
    routes = [[sorted(rng.sample(range(16), 8)) for _ in range(48)]
              for _ in range(TOKENS)]
    entries, history, heap, seen = {}, {}, [], set()
    used = stamp = peak = 0
    counters = Counter()
    demand, hits, misses = Counter(), Counter(), Counter()
    tokens, coverage = [], []
    capped = selection in ("cap", "cached")
    for t in range(TOKENS):
        if t % 8 == 0:
            history = {}
            heap = [(0, value[1], key) for key, value in entries.items()]
            heapify(heap)
        token_misses = Counter()
        kv_bytes = 0
        pages_by_resource = [[0] * 32, [0] * 256]
        min_coverage = [1.0] * 32
        min_score = [1.0] * 32
        for layer in range(48):
            kv = []
            if layer % 4 == 3:
                offset = t * 128 if scenario == "phase" else 0
                heads = [list(range(offset + q * 128, offset + (q + 1) * 128))
                         for q in range(16)]
                chosen = {(g, b) for g in range(2) for row in heads for b in row}
                if capped:
                    resident = {(g, b) for g, b in chosen
                                if ("k", layer, g, b) in entries}
                    # Independent closed form for disjoint requests: each query
                    # gets its anchor plus seven optional blocks, then the global
                    # residual capacity is filled by the equal-score tie order.
                    chosen = {(g, row[-1]) for g in range(2) for row in heads}
                    for g in range(2):
                        for row in heads:
                            optional = sorted(row[:-1], key=lambda b: (
                                -(selection == "cached" and (g, b) in resident), b))
                            chosen.update((g, b) for b in optional[:7])
                    extras = sorted(((g, b) for g in range(2) for row in heads
                                     for b in row[:-1] if (g, b) not in chosen),
                                    key=lambda key: (
                                        -(selection == "cached" and key in resident), key))
                    chosen.update(extras[:512 - len(chosen)])
                    for q in range(32):
                        served = sum((q // 16, b) in chosen for b in heads[q % 16])
                        min_coverage[q] = min(min_coverage[q], served / 128)
                        min_score[q] = min(min_score[q], (served - 1) / 127)
                kv = [(("k", layer, g, b), BLOCK) for g, b in sorted(chosen)]
                kv_bytes += len(kv) * BLOCK
            route_base = 128 if scenario == "phase" and t >= 19 else 0
            experts = [(("e", layer, e + route_base), EXPERT) for e in routes[t][layer]]
            for stage in (kv, experts):
                wanted = {key for key, _ in stage}
                existing = [r for r in stage if r[0] in entries]
                new = [r for r in stage if r[0] not in entries]
                for key, size in existing + new:
                    stamp += 1
                    hit = key in entries
                    kind = "expert" if key[0] == "e" else "kv"
                    state = "hit" if hit else "miss"
                    counters[kind + "_" + state + "_bytes"] += size
                    counters[kind + "_" + state + "_objects"] += 1
                    if t >= WARMUP:
                        demand[kind] += size
                        hits[kind] += size * hit
                        misses[kind] += size * (not hit)
                    if not hit:
                        token_misses[kind] += size
                        first_page = ((key[1] * 256 + key[2]) * EXPERT_STRIDE
                                      if key[0] == "e" else
                                      KV_BASE + (key[1] * 2 + key[2]) * KV_STRIDE + key[3] * 8)
                        count = size // PAGE
                        for bins in pages_by_resource:
                            quotient, remainder = divmod(count, len(bins))
                            if quotient:
                                for i in range(len(bins)):
                                    bins[i] += quotient
                            for i in range(remainder):
                                bins[(first_page + i) % len(bins)] += 1
                        while used + size > CACHE:
                            _, old_stamp, victim = heappop(heap)
                            if victim not in entries or entries[victim][1] != old_stamp or victim in wanted:
                                continue
                            evicted_size = entries.pop(victim)[0]
                            used -= evicted_size
                            counters["evicted_bytes"] += evicted_size
                            counters["evicted_objects"] += 1
                        if key in seen:
                            counters[kind + "_refetch_bytes"] += size
                            counters[kind + "_refetch_objects"] += 1
                        seen.add(key)
                        used += size
                        peak = max(peak, used)
                    history[key] = min(2, history.get(key, 0) + 1)
                    entries[key] = (size, stamp)
                for key, _ in stage:
                    heappush(heap, (history[key] if policy == "reuse" else 0,
                                    entries[key][1], key))
                if len(heap) > max(1, 3 * len(entries)):
                    heap = [(history.get(key, 0) if policy == "reuse" else 0,
                             value[1], key) for key, value in entries.items()]
                    heapify(heap)
        summaries = 0 if scenario == "native" else (7891 if scenario == "phase" else 2048) * 2048 * 12
        fast_bytes = FIXED_FAST + summaries + kv_bytes + sum(token_misses.values())
        # Each of 48 shared/router transfers and the embedding transfer rounds
        # up by 0.25 ns. The other transfers divide exactly at 256 bytes/ns.
        assert (fast_bytes + 3136) % 256 == 0
        fast = (fast_bytes + 3136) // 256
        channel = max(pages_by_resource[0]) * (8192 + 100)
        array = max(pages_by_resource[1]) * 40000
        tokens.append(dict(token=t, physical_expert_bytes=token_misses["expert"],
                           physical_kv_bytes=token_misses["kv"], fast_bytes=fast_bytes,
                           channel_busy_bound_ns=channel, die_array_busy_bound_ns=array,
                           fast_busy_bound_ns=fast,
                           necessary_resource_bound_ns=max(fast, channel, array)))
        coverage.append((min_coverage, min_score))
    measured = tokens[WARMUP:]
    summary = dict(measured_tokens=N, warmup_tokens=WARMUP, cache_peak_bytes=peak,
                   cache_capacity_bytes=CACHE, terminal_pending_pages=0,
                   mean_necessary_resource_ms=sum(t["necessary_resource_bound_ns"] for t in measured) / N / 1e6,
                   count_resource_bound_over_20ms=sum(t["necessary_resource_bound_ns"] > 20000000 for t in measured),
                   count_resource_bound_over_50ms=sum(t["necessary_resource_bound_ns"] > 50000000 for t in measured),
                   fast_bytes_per_token=sum(t["fast_bytes"] for t in measured) / N,
                   requested_union_blocks_per_token=49152,
                   served_union_blocks_per_token=6144 if capped else 49152,
                   omitted_union_blocks_per_token=43008 if capped else 0,
                   query_byte_coverage_min=0.0625 if capped else 1.0,
                   zero_coverage_queries=0, fallback_stages=288 if selection == "fallback" else 0,
                   lossy_stages=288 if capped else 0, mandatory_requested=9216, mandatory_served=9216,
                   union_byte_coverage=0.125 if capped else 1.0,
                   synthetic_score_mass_coverage=15 / 127 if capped else 1.0,
                   requested_score_mass_per_token=786048 if scenario == "native" else 48768,
                   omitted_score_mass_per_token=43008 if capped else 0)
    for suffix in ("union_bytes", "physical_bytes"):
        summary["requested_" + suffix + "_per_token"] = 49152 * BLOCK
        summary["served_" + suffix + "_per_token"] = (6144 if capped else 49152) * BLOCK
        summary["omitted_" + suffix + "_per_token"] = (43008 if capped else 0) * BLOCK
    for kind in ("expert", "kv"):
        summary[kind + "_byte_hit_fraction"] = hits[kind] / demand[kind]
        summary[kind + "_demand_bytes_per_token"] = demand[kind] / N
        summary[kind + "_miss_file_bytes_per_token"] = misses[kind] / N
        summary[kind + "_physical_bytes_per_token"] = misses[kind] / N
    return summary, tokens, coverage, dict(counters)


def main():
    if not __debug__:
        raise RuntimeError("Run without -O: audit assertions are required")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results")
    args = parser.parse_args()
    with open(args.results, "rb") as handle:
        raw = handle.read()
    data = json.loads(raw)
    assert set(data["results"]) == set(CASES), "audit supports the fixed nine-case matrix"
    for name, parameters in CASES.items():
        manifest = data["metadata"]["cases"][name]
        for field, expected in (("tokens", TOKENS), ("warmup_tokens", WARMUP),
                                ("seed", 20261006), ("cache_bytes", CACHE), ("device", DEVICE)):
            equal(manifest[field], expected, name + ":manifest:" + field)
        scenario, cache_policy, selection = parameters
        expected_policy = dict(mode="capped" if selection != "union" else "union",
                               cap_bytes=2**20 if selection == "fallback" else 64 * 2**20,
                               min_blocks_per_query=8, cache_tiebreak=selection == "cached")
        equal(manifest["selection_policy"], expected_policy, name + ":manifest:selection")
        equal(manifest["cache_policy"], "windowed_reuse" if cache_policy == "reuse" else "lru",
              name + ":manifest:cache policy")
        equal(manifest["fixture"], {"native": "native_full", "disjoint": "disjoint", "phase": "phase_churn"}[scenario],
              name + ":manifest:fixture")
        summary, tokens, coverage, counters = reconstruct(*parameters)
        actual = data["results"][name]
        for field, expected in summary.items():
            equal(actual["summary"][field], expected, name + ":summary:" + field)
        equal(actual["cache_counters"], counters, name + ":cache counters")
        equal(len(actual["tokens"]), TOKENS, name + ":token count")
        equal(len(actual["coverage"]), TOKENS, name + ":coverage token count")
        for expected, row, cov, expected_cov in zip(tokens, actual["tokens"], actual["coverage"], coverage):
            for field, value in expected.items():
                equal(row[field], value, name + ":token:" + str(row["token"]) + ":" + field)
            assert row["service_ns"] >= expected["necessary_resource_bound_ns"]
            equal(cov["minimum_byte_coverage_by_query"], expected_cov[0], name + ":query bytes")
            equal(cov["minimum_score_mass_coverage_by_query"], expected_cov[1], name + ":query scores")
        print("PASS", name, "independent cache/coverage/physical resources", flush=True)
    baseline = data["results"]["disjoint_union_lru"]
    fallback = data["results"]["disjoint_overflow_fallback"]
    native = data["results"]["native_full_lru"]
    for field in ("tokens", "cache_counters", "storage_counters", "fast_byte_counters"):
        equal(fallback[field], baseline[field], "fallback identity:" + field)
    for a, b in zip(baseline["tokens"], native["tokens"]):
        equal(a["service_ns"] - b["service_ns"], 196608, "native summary-only service difference")
    print("PASS all 9 cases; exact fallback schedule; native summary-only schedule delta")
    print("results_sha256=" + hashlib.sha256(raw).hexdigest())


if __name__ == "__main__":
    main()
