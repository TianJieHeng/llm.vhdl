#!/usr/bin/env python3
"""Run bounded synthetic Experiment 002, or import a schema-v1 native-route trace."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import csv
import hashlib
import json
from pathlib import Path
import random
import sys

from agentcard_coupled import DeviceConfig, GEOMETRY, Replay, ceildiv, validate_trace

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ["sim/agentcard_coupled.py", "sim/run_agentcard_coupled.py",
           "sim/test_agentcard_coupled.py", "sim/test_agentcard_replay.py"]


def synthetic_trace(*, tokens=6, seed=20261006, context_tokens=262144,
                    routes="hot", attention="selected_hypothesis",
                    kv_locality="hot", query_overlap="shared", blocks_per_query=128):
    """Explicit toy locality. It contains no measurements of model routing/quality."""
    if routes not in ("hot", "uniform", "shift") or kv_locality not in ("hot", "churn") or query_overlap not in ("shared", "disjoint"):
        raise ValueError("unknown synthetic locality")
    rng = random.Random(seed)
    available = ceildiv(context_tokens, 128)
    if not 1 <= blocks_per_query <= available:
        raise ValueError("invalid per-query selection budget")
    out = dict(schema_version=1, provenance=f"synthetic seed={seed}; never model-observed",
               context_tokens=context_tokens, attention=attention, tokens=[])
    for ti in range(tokens):
        layers = []
        for li in range(48):
            # Hot routes draw from16/layer (6.25% of256). A shift changes every
            # layer's hot set halfway through, adversarially invalidating reuse.
            base = 128 if routes == "shift" and ti >= tokens // 2 else 0
            pool = range(256) if routes == "uniform" else range(base, base + 16)
            experts = sorted(rng.sample(pool, 8))
            heads = []
            if li % 4 == 3 and attention == "selected_hypothesis":
                for q in range(32):
                    offset = (ti * blocks_per_query if kv_locality == "churn" else 0)
                    if query_overlap == "disjoint":
                        offset += (q % 16) * blocks_per_query
                    heads.append(sorted((offset + b) % available for b in range(blocks_per_query)))
            layers.append(dict(layer=li, experts=experts, kv_blocks_by_query_head=heads,
                               selection_delay_ns=0, attention_compute_ns=0,
                               route_delay_ns=0, expert_compute_ns=0))
        out["tokens"].append(dict(token=ti, not_before_ns=0, layers=layers))
    return validate_trace(out)


def cases(tokens=6):
    # 32 physical channels is an assumed sensitivity envelope, NOT a board BOM.
    baseline = DeviceConfig(channels=32)
    common = dict(tokens=tokens)
    return [
        ("expert_only_hot_8g", common | dict(attention="none_control"), baseline, 8, "compact"),
        ("native_full_attention_8g", common | dict(attention="full"), baseline, 8, "compact"),
        ("selected_hot_8g", common, baseline, 8, "compact"),
        ("selected_hot_1m_context", common | dict(context_tokens=1010000), baseline, 8, "compact"),
        ("selected_churn_8g", common | dict(kv_locality="churn"), baseline, 8, "compact"),
        ("selected_disjoint_heads_8g", common | dict(query_overlap="disjoint"), baseline, 8, "compact"),
        ("selected_uniform_routes_8g", common | dict(routes="uniform"), baseline, 8, "compact"),
        ("selected_shift_routes_8g", common | dict(routes="shift"), baseline, 8, "compact"),
        ("selected_hot_4g", common, baseline, 4, "compact"),
        ("selected_churn_8channels", common | dict(kv_locality="churn"), replace(baseline, channels=8), 8, "compact"),
        ("selected_hot_fast128", common, replace(baseline, fast_bytes_per_ns=128), 8, "compact"),
        ("selected_churn_repo_padded", common | dict(kv_locality="churn"), baseline, 8, "repo_padded"),
    ]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main():
    if not __debug__:
        raise RuntimeError("Run without python -O: checkers require assertions")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--trace", type=Path, help="import schema-v1 JSON instead of the synthetic matrix")
    ap.add_argument("--device", type=Path, help="JSON DeviceConfig fields for imported trace")
    ap.add_argument("--cache-gib", type=int, default=8)
    ap.add_argument("--card-gib", type=int, default=16)
    ap.add_argument("--warmup-tokens", type=int, default=2)
    ap.add_argument("--tokens", type=int, default=6)
    ap.add_argument("--case", action="append", help="run only named matrix case; repeatable")
    ap.add_argument("--include-stages", action="store_true", help="include every stage timestamp in results.json")
    ap.add_argument("--format", choices=("compact", "repo_padded"), default="compact")
    ap.add_argument("--emit-example", type=Path, help="also write a small strict-schema example")
    args = ap.parse_args()
    source_hashes = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in SOURCES}
    args.out.mkdir(parents=True, exist_ok=True)
    if args.emit_example:
        args.emit_example.write_text(json.dumps(synthetic_trace(tokens=2, context_tokens=256, blocks_per_query=1), indent=2) + "\n")
    if args.trace:
        if args.trace.stat().st_size > 64 * 2**20:
            raise ValueError("trace exceeds 64MiB importer limit")
        trace = validate_trace(json.loads(args.trace.read_text()))
        device = DeviceConfig(**json.loads(args.device.read_text())) if args.device else DeviceConfig(channels=32)
        work = [("imported_trace", trace, device, args.cache_gib, args.format, dict(import_path=args.trace.name, provenance=trace["provenance"]))]
    else:
        if args.device:
            ap.error("--device requires --trace")
        choices = cases(args.tokens)
        if args.case:
            unknown = set(args.case) - {c[0] for c in choices}
            if unknown:
                ap.error(f"unknown cases: {sorted(unknown)}")
            choices = [c for c in choices if c[0] in args.case]
        work = [(name, synthetic_trace(**params), device, cache, fmt, params)
                for name, params, device, cache, fmt in choices]
    results, rows, manifest = {}, [], {}
    for name, trace, device, cache, fmt, params in work:
        result = Replay(device, cache * 2**30, args.card_gib * 2**30, fmt).run(trace, args.warmup_tokens)
        if not args.include_stages:
            result.pop("stages")
        results[name] = result
        row = dict(case=name, **result["summary"])
        rows.append(row)
        manifest[name] = dict(trace_sha256=digest(trace), trace_parameters=params, device=asdict(device),
                              cache_gib=cache, card_gib=args.card_gib, format=fmt)
        print(f"{name}: service {row['mean_service_ms']:.3f}ms, necessary-resource {row['mean_necessary_resource_ms']:.3f}ms, expert byte hit {row['expert_byte_hit_fraction']:.3%}, KV byte hit {row['kv_byte_hit_fraction']}", flush=True)
    metadata = dict(schema_version=1, base_git_sha="52cbfc7b2a6c01302d806da8337117c17a2440f9",
                    units=dict(time="ns", bytes="bytes", bandwidth="bytes/ns (1 = 1 decimal GB/s)", capacity="GiB = 2^30 bytes"),
                    scope="synthetic storage-only stage-barrier replay; not physical throughput, real routing locality, quality, or achieved tokens/s",
                    python=sys.version.split()[0], source_sha256=source_hashes,
                    cases=manifest)
    if source_hashes != {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in SOURCES}:
        raise RuntimeError("source changed during run; discard this result")
    (args.out / "results.json").write_text(json.dumps(dict(metadata=metadata, results=results), indent=2, sort_keys=True) + "\n")
    (args.out / "manifest.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    with (args.out / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(f"PASS: {len(rows)} cases; all service/cache/fast-bus checks; terminal pending=0")


if __name__ == "__main__":
    main()
