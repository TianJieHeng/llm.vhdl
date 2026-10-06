#!/usr/bin/env python3
"""Experiment 003: causal lossless cache policies and EXPLICITLY LOSSY KV caps.

Synthetic candidate scores and required-page flags are fixture metadata, never
learned attention or semantic ground truth. The experiment-002 kernel is reused
without changing its schedule or accounting.
"""
from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import dataclass
import math

from agentcard_coupled import Cache, GEOMETRY, Replay, ceildiv, integer, validate_trace


@dataclass(frozen=True)
class Candidate:
    block: int
    score: float
    mandatory: bool = False

    def __post_init__(self):
        integer(self.block, "candidate block")
        if type(self.score) not in (int, float) or not math.isfinite(self.score) or not 0 <= self.score <= 1e12:
            raise ValueError("score must be finite, nonnegative and <= 1e12")
        if type(self.mandatory) is not bool:
            raise ValueError("mandatory must be bool")


@dataclass(frozen=True)
class SelectionPolicy:
    mode: str = "union"
    cap_bytes: int = 64 * 2**20
    min_blocks_per_query: int = 8
    cache_tiebreak: bool = False

    def __post_init__(self):
        if self.mode not in ("union", "capped"):
            raise ValueError("unknown selection mode")
        integer(self.cap_bytes, "cap_bytes")
        integer(self.min_blocks_per_query, "min_blocks_per_query", 1)
        if type(self.cache_tiebreak) is not bool:
            raise ValueError("cache_tiebreak must be bool")


def select_pages(heads, context_tokens, policy=SelectionPolicy(), cached=frozenset(), page_bytes=16384):
    """Cap physical union bytes across BOTH KV groups; cache hits still count.

    Mandatory union and per-query round-robin minima come first. If that greedy
    feasible set exceeds the cap, serve the full request union. This conservative
    fallback does not claim to solve minimum-union set cover optimally. Extras
    rank by summed fixture score per physical byte; cached status breaks only
    exact score-density ties. There is no inferred permission to omit anything.
    """
    integer(context_tokens, "context_tokens", 1)
    integer(page_bytes, "page_bytes", 1)
    if context_tokens > 1010000 or 131072 % page_bytes:
        raise ValueError("unsupported context or physical page size")
    if not isinstance(policy, SelectionPolicy):
        raise ValueError("expected SelectionPolicy")
    if not isinstance(heads, list) or len(heads) != 32:
        raise ValueError("expected 32 query heads")
    blocks = ceildiv(context_tokens, 128)
    queries, sizes, scores, mandatory = [], {}, Counter(), set()
    for q, candidates in enumerate(heads):
        if not isinstance(candidates, list) or not candidates:
            raise ValueError("query candidates must be a nonempty list")
        if any(not isinstance(c, Candidate) or c.block >= blocks for c in candidates):
            raise ValueError("unknown candidate block")
        if len({c.block for c in candidates}) != len(candidates):
            raise ValueError("duplicate candidate within query")
        row = {(q // 16, c.block): c for c in candidates}
        queries.append(row)
        for key, candidate in row.items():
            sizes[key] = min(128, context_tokens - key[1] * 128) * 1024
            scores[key] += candidate.score
            if candidate.mandatory:
                mandatory.add(key)
    try:
        cached = frozenset(cached)
    except TypeError as e:
        raise ValueError("invalid cached keys") from e
    if any(not isinstance(k, tuple) or len(k) != 2 or type(k[0]) is not int or k[0] not in (0, 1)
           or type(k[1]) is not int or not 0 <= k[1] < blocks for k in cached):
        raise ValueError("invalid cached keys")
    physical = {k: ceildiv(v, page_bytes) * page_bytes for k, v in sizes.items()}
    selected = set(sizes) if policy.mode == "union" else set(mandatory)
    fallback = False
    if policy.mode == "capped":
        # Round-robin fairness is measured per query, even when another query
        # selected its page first. High global coverage cannot hide starvation.
        order = [sorted(row, key=lambda k: (-row[k].score,
                 -(policy.cache_tiebreak and k in cached), k)) for row in queries]
        for level in range(policy.min_blocks_per_query):
            for row, ranked in zip(queries, order):
                if len(set(row) & selected) < min(level + 1, len(row)):
                    selected.add(next(k for k in ranked if k not in selected))
        if sum(physical[k] for k in selected) > policy.cap_bytes:
            fallback = True
            selected = set(sizes)
        else:
            used = sum(physical[k] for k in selected)
            ranked = sorted(set(sizes) - selected,
                            key=lambda k: (-scores[k] / physical[k],
                                           -(policy.cache_tiebreak and k in cached), k))
            for key in ranked:
                if used + physical[key] <= policy.cap_bytes:
                    selected.add(key)
                    used += physical[key]
    per_query = []
    selected_by_query = []
    for row in queries:
        served = set(row) & selected
        requested_bytes = sum(sizes[k] for k in row)
        served_bytes = sum(sizes[k] for k in served)
        requested_score = sum(c.score for c in row.values())
        served_score = sum(row[k].score for k in served)
        pins = {k for k, c in row.items() if c.mandatory}
        selected_by_query.append(sorted(k[1] for k in served))
        per_query.append(dict(requested_blocks=len(row), served_blocks=len(served),
                              omitted_blocks=len(row)-len(served), requested_bytes=requested_bytes,
                              served_bytes=served_bytes, byte_coverage=served_bytes/requested_bytes,
                              requested_score_mass=requested_score, served_score_mass=served_score,
                              score_mass_coverage=served_score/requested_score if requested_score else None,
                              mandatory_requested=len(pins), mandatory_served=len(pins & served)))
    result = dict(selected_keys=sorted(selected), selected_by_query=selected_by_query,
                  fallback=fallback, lossy=len(selected) < len(sizes), per_query=per_query,
                  min_query_byte_coverage=min(r["byte_coverage"] for r in per_query),
                  zero_coverage_queries=sum(not r["served_blocks"] for r in per_query),
                  requested_score_mass=sum(scores.values()),
                  omitted_score_mass=sum(scores[k] for k in sizes.keys()-selected))
    for suffix, amount in (("union_bytes", sizes), ("physical_bytes", physical),
                           ("union_blocks", {k: 1 for k in sizes}),
                           ("physical_pages", {k: v//page_bytes for k, v in physical.items()})):
        result["requested_"+suffix] = sum(amount.values())
        result["served_"+suffix] = sum(amount[k] for k in selected)
        result["omitted_"+suffix] = result["requested_"+suffix]-result["served_"+suffix]
    audit_selection(heads, context_tokens, policy, result, page_bytes)
    return result


def audit_selection(heads, context_tokens, policy, result, page_bytes=16384):
    """Refuse omitted fixture-required pages, starvation, cap/accounting failures.

    These are policy safety checks, NOT a reasoning-quality oracle.
    """
    selected = set(map(tuple, result["selected_keys"]))
    requested = {(q//16, c.block) for q, row in enumerate(heads) for c in row}
    assert selected <= requested, "unrequested page"
    assert len(selected) == len(result["selected_keys"]), "duplicate physical key"
    logical = {k: min(128, context_tokens-k[1]*128)*1024 for k in requested}
    physical = {k: ceildiv(v, page_bytes)*page_bytes for k, v in logical.items()}
    for suffix, values in (("union_bytes", logical), ("physical_bytes", physical),
                           ("union_blocks", {k: 1 for k in requested}),
                           ("physical_pages", {k: v//page_bytes for k, v in physical.items()})):
        requested_amount, served_amount = sum(values.values()), sum(values[k] for k in selected)
        assert result["requested_"+suffix] == requested_amount, "requested "+suffix+" accounting"
        assert result["served_"+suffix] == served_amount, "served "+suffix+" accounting"
        assert result["omitted_"+suffix] == requested_amount-served_amount, "omitted "+suffix+" accounting"
    if policy.mode == "union" or result["fallback"]:
        assert selected == requested, "lossless/fallback omitted request"
    else:
        assert sum(physical[k] for k in selected) <= policy.cap_bytes, "physical union cap"
    coverages, zero, total_score, omitted_score = [], 0, 0.0, 0.0
    for q, row in enumerate(heads):
        wanted = {c.block for c in row}
        served = {b for group, b in selected if group == q//16} & wanted
        pins = {c.block for c in row if c.mandatory}
        assert pins <= served, "required page omitted"
        assert len(served) >= min(policy.min_blocks_per_query, len(wanted)), "query fairness"
        assert sorted(served) == result["selected_by_query"][q], "query served accounting"
        req_bytes = sum(logical[(q//16,b)] for b in wanted)
        got_bytes = sum(logical[(q//16,b)] for b in served)
        req_score = sum(c.score for c in row)
        got_score = sum(c.score for c in row if c.block in served)
        total_score += req_score
        omitted_score += sum(c.score for c in row if c.block not in served)
        coverages.append(got_bytes/req_bytes)
        zero += not served
        expected = dict(requested_blocks=len(wanted), served_blocks=len(served),
                        omitted_blocks=len(wanted)-len(served), requested_bytes=req_bytes,
                        served_bytes=got_bytes, byte_coverage=got_bytes/req_bytes,
                        requested_score_mass=req_score, served_score_mass=got_score,
                        score_mass_coverage=got_score/req_score if req_score else None,
                        mandatory_requested=len(pins), mandatory_served=len(pins & served))
        for key, value in expected.items():
            got = result["per_query"][q][key]
            assert (got == value if value is None or type(value) is int else
                    isinstance(got, (int, float)) and math.isclose(got, value, rel_tol=1e-12, abs_tol=1e-9)), "query "+key+" accounting"
    assert result["min_query_byte_coverage"] == min(coverages), "minimum query coverage"
    assert result["zero_coverage_queries"] == zero, "zero query coverage"
    assert result["lossy"] == (selected != requested), "lossy classification"
    assert math.isclose(result["requested_score_mass"], total_score, rel_tol=1e-12, abs_tol=1e-9), "requested score accounting"
    assert math.isclose(result["omitted_score_mass"], omitted_score, rel_tol=1e-12, abs_tol=1e-9), "omitted score accounting"


class CausalCache(Cache):
    """Byte-capacity LRU or eight-token windowed, frequency-prioritized LRU.

    Frequencies are capped at two and count demanded objects only, after their
    causal request becomes available. They reset on fixed token-epoch boundaries.
    Both classes share one capacity; pending fills and demanded hits stay pinned.
    No future route/page trace is visible to this class.
    """
    def __init__(self, capacity, policy="lru"):
        super().__init__(capacity)
        if policy not in ("lru", "windowed_reuse"):
            raise ValueError("unknown cache policy")
        self.policy = policy
        self.observations = Counter()
        self.token = -1
        self.tiers = [OrderedDict() for _ in range(3)]

    def begin_token(self, token):
        integer(token, "token")
        if token < self.token:
            raise ValueError("noncausal cache token")
        if token//8 != self.token//8:
            self.observations.clear()
            self.tiers = [OrderedDict.fromkeys(self.entries), OrderedDict(), OrderedDict()]
        self.token = token

    def acquire(self, read, now):
        integer(now, "cache acquire time")
        if self.policy == "lru":
            state = super().acquire(read, now)
            self.observations[read.key] = min(2, self.observations[read.key]+1)
            return state
        # Same allocation/pinning semantics as002; only the victim rank changes.
        if read.key in self.entries:
            if self.entries[read.key]["read"] != read:
                raise ValueError("object identity changed")
        elif read.size > self.capacity:
            raise ValueError("object exceeds cache capacity; tiling is not modeled")
        while read.key not in self.entries and self.used + read.size > self.capacity:
            victim = None
            for tier in self.tiers:
                victim = next((key for key in tier if not self.entries[key]["pins"]
                               and self.entries[key]["ready"] is not None
                               and self.entries[key]["ready"] <= now), None)
                if victim is not None:
                    tier.pop(victim)
                    break
            if victim is None:
                raise ValueError("pinned stage exceeds cache capacity; tiling is not modeled")
            old = self.entries.pop(victim)
            self.used -= old["read"].size
            self.stats["evicted_bytes"] += old["read"].size
            self.stats["evicted_objects"] += 1
        state = super().acquire(read, now)
        old_count = self.observations[read.key]
        self.tiers[old_count].pop(read.key, None)
        self.observations[read.key] = min(2, old_count+1)
        self.tiers[self.observations[read.key]][read.key] = None
        return state


class PolicyReplay(Replay):
    def __init__(self, *args, selection=SelectionPolicy(), cache_policy="lru", candidates=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache = CausalCache(self.cache.capacity, cache_policy)
        self.selection = selection
        self.candidates = candidates or {}
        self.coverage = []

    def stage(self, reads, now, token, layer, kind):
        self.cache.begin_token(token)
        if kind == "kv" and reads:
            heads = self.candidates[(token, layer)]
            resident = frozenset((int(r.key.split(":")[2]), int(r.key.split(":")[3]))
                                 for r in reads if r.key in self.cache.entries
                                 and self.cache.entries[r.key]["ready"] is not None
                                 and self.cache.entries[r.key]["ready"] <= now)
            decision = select_pages(heads, self.context_tokens, self.selection, resident, self.c.page_bytes)
            chosen = set(decision["selected_keys"])
            all_keys = {(int(r.key.split(":")[2]), int(r.key.split(":")[3])) for r in reads}
            expected = {(q//16, c.block) for q, row in enumerate(heads) for c in row}
            if all_keys != expected:
                raise ValueError("candidate fixture differs from trace request union")
            reads = [r for r in reads if (int(r.key.split(":")[2]), int(r.key.split(":")[3])) in chosen]
            self.coverage.append(dict(token=token, layer=layer, **decision))
        return super().stage(reads, now, token, layer, kind)

    def run(self, trace, warmup_tokens=8):
        validate_trace(trace)
        self.context_tokens = trace["context_tokens"]
        expected_keys = set()
        for token in trace["tokens"]:
            for layer in token["layers"]:
                if layer["layer"] not in GEOMETRY["full_attention_layers"] or trace["attention"] == "none_control":
                    continue
                key = (token["token"], layer["layer"])
                expected_keys.add(key)
                heads = self.candidates.get(key)
                expected = layer["kv_blocks_by_query_head"] if trace["attention"] == "selected_hypothesis" else [list(range(ceildiv(self.context_tokens,128)))]*32
                if not isinstance(heads, list) or len(heads) != 32 or any(
                    not isinstance(row, list) or any(not isinstance(c, Candidate) for c in row)
                    or len(row) != len({c.block for c in row})
                    or {c.block for c in row} != set(wanted) for row, wanted in zip(heads, expected)):
                    raise ValueError("candidate fixture differs from per-query trace requests")
        if set(self.candidates) != expected_keys:
            raise ValueError("candidate fixture has missing or extra stages")
        result = super().run(trace, warmup_tokens)
        measured = [r for r in self.coverage if r["token"] >= warmup_tokens]
        summary = result["summary"]
        for name in ("requested_union_bytes", "served_union_bytes", "omitted_union_bytes",
                     "requested_union_blocks", "served_union_blocks", "omitted_union_blocks",
                     "requested_physical_bytes", "served_physical_bytes", "omitted_physical_bytes",
                     "requested_score_mass", "omitted_score_mass"):
            summary[name+"_per_token"] = sum(r[name] for r in measured)/summary["measured_tokens"]
        summary["query_byte_coverage_min"] = min((r["min_query_byte_coverage"] for r in measured), default=1)
        summary["zero_coverage_queries"] = sum(r["zero_coverage_queries"] for r in measured)
        summary["fallback_stages"] = sum(r["fallback"] for r in measured)
        summary["lossy_stages"] = sum(r["lossy"] for r in measured)
        summary["mandatory_requested"] = sum(q["mandatory_requested"] for r in measured for q in r["per_query"])
        summary["mandatory_served"] = sum(q["mandatory_served"] for r in measured for q in r["per_query"])
        summary["union_byte_coverage"] = (summary["served_union_bytes_per_token"]/summary["requested_union_bytes_per_token"]
                                         if measured else 1)
        summary["synthetic_score_mass_coverage"] = (1-summary["omitted_score_mass_per_token"]/summary["requested_score_mass_per_token"]
                                                   if summary["requested_score_mass_per_token"] else None)
        result["coverage"] = self.coverage
        result["cache_policy"] = self.cache.policy
        return result
