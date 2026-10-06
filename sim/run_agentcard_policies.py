#!/usr/bin/env python3
"""Bounded experiment003 controlled policy matrix; no weights or real quality."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import csv
import hashlib
import json
from pathlib import Path
import sys

from agentcard_coupled import DeviceConfig, GEOMETRY, ceildiv
from agentcard_policies import Candidate, PolicyReplay, SelectionPolicy, select_pages
from run_agentcard_coupled import digest, synthetic_trace

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ["sim/agentcard_coupled.py", "sim/run_agentcard_coupled.py",
           "sim/agentcard_policies.py", "sim/run_agentcard_policies.py",
           "sim/test_agentcard_coupled.py", "sim/test_agentcard_replay.py",
           "sim/test_agentcard_policies.py"]


def cases():
    # Every case uses the SAME8GiB cache,256GB/s bus and32-channel envelope.
    return [
        ("native_full_lru", "native_full", SelectionPolicy(), "lru"),
        ("disjoint_union_lru", "disjoint", SelectionPolicy(), "lru"),
        ("disjoint_union_reuse", "disjoint", SelectionPolicy(), "windowed_reuse"),
        ("disjoint_cap_lru", "disjoint", SelectionPolicy("capped"), "lru"),
        ("phase_churn_union_lru", "phase_churn", SelectionPolicy(), "lru"),
        ("phase_churn_union_reuse", "phase_churn", SelectionPolicy(), "windowed_reuse"),
        ("phase_churn_cap_lru", "phase_churn", SelectionPolicy("capped"), "lru"),
        ("phase_churn_cap_cached", "phase_churn", SelectionPolicy("capped", cache_tiebreak=True), "lru"),
        ("disjoint_overflow_fallback", "disjoint", SelectionPolicy("capped", cap_bytes=2**20), "lru"),
    ]


def fixture(name, tokens):
    phase = name == "phase_churn"
    trace = synthetic_trace(tokens=tokens, context_tokens=1010000 if phase else 262144,
                            routes="hot", query_overlap="disjoint",
                            kv_locality="churn" if phase else "hot",
                            attention="full" if name == "native_full" else "selected_hypothesis")
    if phase:
        # Off-epoch shift: do not hand the cache a reset aligned to the change.
        for token in trace["tokens"][min(tokens-1, tokens//2+3):]:
            for layer in token["layers"]:
                layer["experts"] = [e+128 for e in layer["experts"]]
    candidates, intern = {}, {}
    for token in trace["tokens"]:
        for layer in token["layers"]:
            if layer["layer"] not in GEOMETRY["full_attention_layers"]:
                continue
            raw = layer["kv_blocks_by_query_head"] if name != "native_full" else [list(range(ceildiv(trace["context_tokens"], 128)))]*32
            signature = tuple(map(tuple, raw))
            if signature not in intern:
                # All optional candidates have equal toy scores. A rare low-score
                # required page is deliberately inconsistent with popularity.
                # Last page in each supplied query list acts as an explicit anchor.
                intern[signature] = [[Candidate(b, 0 if i == len(row)-1 else 1,
                                                mandatory=i == len(row)-1)
                                      for i, b in enumerate(row)] for row in raw]
            candidates[(token["token"], layer["layer"])] = intern[signature]
    trace["provenance"] = f"experiment003 synthetic {name}; fixture scores/required flags; no observed attention or semantic labels"
    h = hashlib.sha256()
    for key, heads in candidates.items():
        h.update(json.dumps([key, [[asdict(c) for c in row] for row in heads]],
                            sort_keys=True, separators=(",", ":")).encode())
    return trace, candidates, h.hexdigest()


def compact_coverage(rows, tokens):
    """Keep per-token minimum coverage for EACH query; do not hide starvation.

    Full per-stage requested/served identities are available with--include-ledger.
    Every decision has already been audited against its actual selected keys.
    """
    out = []
    for ti in range(tokens):
        stages = [r for r in rows if r["token"] == ti]
        out.append(dict(token=ti, full_attention_stages=len(stages),
                        fallback_stages=sum(r["fallback"] for r in stages),
                        lossy_stages=sum(r["lossy"] for r in stages),
                        requested_union_bytes=sum(r["requested_union_bytes"] for r in stages),
                        served_union_bytes=sum(r["served_union_bytes"] for r in stages),
                        omitted_union_blocks=sum(r["omitted_union_blocks"] for r in stages),
                        minimum_byte_coverage_by_query=[min(r["per_query"][q]["byte_coverage"] for r in stages) for q in range(32)],
                        minimum_score_mass_coverage_by_query=[min((r["per_query"][q]["score_mass_coverage"] for r in stages if r["per_query"][q]["score_mass_coverage"] is not None), default=None) for q in range(32)]))
    return out


def negative_fixtures():
    # Tiny page-rounding/overflow fixtures separate from throughput cases.
    heads = [[Candidate(q%16, 1), Candidate(16+q%16, 0, True)] for q in range(32)]
    tiny = SelectionPolicy("capped", cap_bytes=16384, min_blocks_per_query=1)
    overflow = select_pages(heads, 4096, tiny)
    return dict(mandatory_overflow=dict(requested_blocks=overflow["requested_union_blocks"],
                served_blocks=overflow["served_union_blocks"], fallback=overflow["fallback"],
                requested_physical_bytes=overflow["requested_physical_bytes"], cap_bytes=tiny.cap_bytes),
                interpretation="Full union fallback preserves supplied requests, may exceed the deadline, and never claims reasoning preservation from fixture coverage")


def main():
    if not __debug__:
        raise RuntimeError("Run without python -O: checkers require assertions")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--warmup-tokens", type=int, default=8)
    ap.add_argument("--case", action="append")
    ap.add_argument("--include-ledger", action="store_true")
    args = ap.parse_args()
    if not 2 <= args.tokens <= 64 or not 0 <= args.warmup_tokens < args.tokens:
        ap.error("need2..64tokens and warmup smaller than token count")
    choices = cases()
    if args.case:
        unknown = set(args.case)-{r[0] for r in choices}
        if unknown:
            ap.error(f"unknown cases: {sorted(unknown)}")
        choices = [r for r in choices if r[0] in args.case]
    hashes = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCES}
    device = DeviceConfig(channels=32)
    results, manifest, summaries = {}, {}, []
    for name, kind, selection, cache in choices:
        trace, candidates, candidate_hash = fixture(kind, args.tokens)
        result = PolicyReplay(device, selection=selection, cache_policy=cache, candidates=candidates).run(trace, args.warmup_tokens)
        if not args.include_ledger:
            result["coverage"] = compact_coverage(result["coverage"], args.tokens)
            result.pop("stages")
        result["selection_policy"] = asdict(selection)
        results[name] = result
        manifest[name] = dict(fixture=kind, trace_sha256=digest(trace), candidate_sha256=candidate_hash,
                              selection_policy=asdict(selection), cache_policy=cache,
                              seed=20261006, tokens=args.tokens, warmup_tokens=args.warmup_tokens,
                              cache_bytes=8*2**30, assumed_card_bytes=16*2**30, device=asdict(device))
        row = dict(case=name, **result["summary"])
        summaries.append(row)
        print(f"{name}: service {row['mean_service_ms']:.3f}ms; bound {row['mean_necessary_resource_ms']:.3f}ms; expert hits {row['expert_byte_hit_fraction']:.2%}; union coverage {row['union_byte_coverage']:.2%}; minimum query {row['query_byte_coverage_min']:.2%}; fallbacks {row['fallback_stages']}", flush=True)
    metadata = dict(schema_version=3, base_git_sha="29ae898f86e5e03b5cd88937c92930008bbb1e7e",
                    scope="synthetic policy/storage replay; omission and score coverage are not quality; timings are not achieved tokens/s",
                    python=sys.version.split()[0], source_sha256=hashes, cases=manifest,
                    unchanged_fast_placement_floor_bytes=5379316416,
                    placement_sensitivity=dict(weight_only_bytes=5070248640, offchip_state_rw_bytes=309067776,
                      weight_only_ms_at256=5070248640/256/1e6, with_state_ms_at256=5379316416/256/1e6,
                      note="Analytical zero-KV/no-fill bounds only. Moving state on chip changes placement; no case does so. Weight-only floor leaves about0.195ms of20ms for everything else."),
                    negative_fixtures=negative_fixtures())
    if hashes != {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCES}:
        raise RuntimeError("source changed during run; discard results")
    args.out.mkdir(parents=True, exist_ok=True)
    for name, value in (("results.json", dict(metadata=metadata, results=results)), ("manifest.json", metadata)):
        (args.out/name).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n")
    with (args.out/"summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0])); writer.writeheader(); writer.writerows(summaries)
    print(f"PASS:{len(results)}cases; all requested native top-8 experts served; zero pending pages")


if __name__ == "__main__":
    main()
