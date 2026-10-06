#!/usr/bin/env python3
"""Local-only synthetic AgentCard experiment; standard library, cc and GHDL.

Writes bounded logs/CSV/JSON to --out. No CI, models, hardware, network or RTL edits.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
RTL = ["rtl/util_pkg.vhd", "rtl/async_fifo.vhd", "rtl/axi_rd_fsm.vhd",
       "rtl/stream_fifo.vhd", "rtl/axi_rd_port.vhd", "rtl/weight_streamer.vhd"]
SOURCES = RTL + ["sim/tb_weight_streamer.vhd", "sim/tb_agentcard_storage.vhd",
                 "sim/run_agentcard_storage.py", "ref/matvec_int4.c", "ref/mv4i_arith.h"]
DEFAULTS = dict(BANKS=27, SERVICE_CYCLES=1, LATENCY=8, SKEW=0,
                PAGE_BEATS=128, PAGE_DELAY=0, FIFO_DEPTH=64, BURST_BEATS=16,
                OUTSTANDING=16, CONSUMER_PERIOD=1, PAUSE_EVERY=0,
                PAUSE_CYCLES=0, WARM_GROUPS=512, MEASURE_GROUPS=2048,
                DRAIN_GROUPS=393)
CASES = {
    "independent_fast": {},
    "shared_9_banks": dict(BANKS=9),
    "shared_3_banks": dict(BANKS=3),
    "shared_1_bank": dict(BANKS=1),
    "bandwidth_quarter": dict(SERVICE_CYCLES=4),
    "latency_small_buffer": dict(LATENCY=128, FIFO_DEPTH=32),
    "latency_large_buffer": dict(LATENCY=128, FIFO_DEPTH=256),
    "latency_large_buffer_one_outstanding": dict(LATENCY=128, FIFO_DEPTH=256, OUTSTANDING=1),
    "page_skew_small_buffer": dict(LATENCY=32, SKEW=7, PAGE_DELAY=96, FIFO_DEPTH=32),
    "page_skew_large_buffer": dict(LATENCY=32, SKEW=7, PAGE_DELAY=96, FIFO_DEPTH=256),
    "consumer_pauses": dict(CONSUMER_PERIOD=2, PAUSE_EVERY=256, PAUSE_CYCLES=80),
    "burst_8": dict(BURST_BEATS=8, LATENCY=32, FIFO_DEPTH=64),
}


def main() -> None:
    if not __debug__:
        raise RuntimeError("Run without python -O: validation assertions are required")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ghdl", default=os.environ.get("GHDL", "ghdl"))
    ap.add_argument("--cc", default=os.environ.get("CC", "cc"))
    args = ap.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    work = out / "ghdl"
    work.mkdir(exist_ok=True)
    commands = []

    def run(label, argv, expected_failure=None):
        commands.append(dict(label=label, argv=[str(x) for x in argv]))
        proc = subprocess.run(argv, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, timeout=120)
        (out / f"{label}.log").write_text(proc.stdout)
        if expected_failure:
            if proc.returncode == 0 or expected_failure not in proc.stdout:
                raise RuntimeError(f"{label}: failed to detect intended defect; see log")
        elif proc.returncode:
            raise RuntimeError(f"{label}: return {proc.returncode}; see log")
        return proc.stdout

    def unique_line(log, marker):
        lines = [line for line in log.splitlines() if marker in line]
        if len(lines) != 1:
            raise RuntimeError(f"Expected one {marker!r} completion record, got {len(lines)}")
        return lines[0]

    common = ["--std=08", "-frelaxed", f"--workdir={work}"]
    versions = {"python": sys.version.splitlines()[0],
                "ghdl": run("ghdl_version", [args.ghdl, "--version"]).splitlines()[0],
                "cc": run("cc_version", [args.cc, "--version"]).splitlines()[0]}
    base_sha = run("git_head", ["git", "rev-parse", "HEAD"]).strip()
    source_hashes = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in SOURCES}
    run("compile_c", [args.cc, "-O2", "-Wall", "-Wextra", "-o", str(out / "mv4i"), "ref/matvec_int4.c"])
    c_log = run("c_baseline", [str(out / "mv4i")])
    unique_line(c_log, "OK (0 failures)")
    unique_line(c_log, "PASS (4178 checked)")
    run("emit_synthetic", [str(out / "mv4i"), "--emit", str(out / "synthetic-fk33.mv4i"), "100", "200", "48", "256"])
    assert (out / "synthetic-fk33.mv4i").stat().st_size == 114688
    run("analyze", [args.ghdl, "-a", *common, *RTL,
                    "sim/tb_weight_streamer.vhd", "sim/tb_agentcard_storage.vhd"])
    original = run("streamer_baseline", [args.ghdl, "-r", *common, "tb_weight_streamer",
                                         "--assert-level=error", "--stop-time=100us", "--stop-delta=2000000"])
    unique_line(original, "weight_streamer: 0 reassembly errors across both geometries")
    unique_line(original, "weight_streamer CADENCE (cycles spanned by 21 accepts")
    rows = []
    raw = {}
    for name, overrides in CASES.items():
        params = DEFAULTS | overrides
        log = run(name, [args.ghdl, "-r", *common, "tb_agentcard_storage",
                         *[f"-g{k}={v}" for k, v in params.items()], "--assert-level=error"])
        measure = unique_line(log, "MEASURE cycles=")
        final = unique_line(log, "PASS agentcard_storage ")
        m = {k: int(v) for k, v in re.findall(r"(\w+)=(-?\d+)", measure)}
        f = {k: int(v) for k, v in re.findall(r"(\w+)=(-?\d+)", final)}
        assert f["total_groups"] == sum(params[x] for x in ("WARM_GROUPS", "MEASURE_GROUPS", "DRAIN_GROUPS"))
        assert f["total_beats"] == f["total_groups"] * 27
        assert m["groups"] == params["MEASURE_GROUPS"]
        assert m["start_inventory"] + m["input_beats"] == 27*m["groups"] + m["end_inventory"]
        assert m["demand_cycles"] == m["groups"] + m["starved_cycles"]
        # At most ceil(C/T) accepted beats per bank in ANY C consecutive cycles.
        # This holds even with a response pending at the first edge because
        # the bank is reserved until acceptance, then spaced by T cycles.
        service_bound = params["BANKS"] * ((m["cycles"]-1)//params["SERVICE_CYCLES"]+1)
        assert m["input_beats"] <= service_bound
        # Output may additionally use accepted inventory present before the window.
        assert 27*m["groups"] <= m["start_inventory"]+service_bound
        assert m["peak_lane"] <= params["FIFO_DEPTH"]+4
        assert m["input_beats"] > 4*m["start_inventory"]
        row = dict(case=name, groups_per_cycle=m["groups"]/m["cycles"],
                   input_bytes_per_cycle=32*m["input_beats"]/m["cycles"],
                   useful_bytes_per_cycle=864*m["groups"]/m["cycles"],
                   weight_elements_per_cycle=1536*m["groups"]/m["cycles"],
                   starved_demand_percent=100*m["starved_cycles"]/m["demand_cycles"],
                   average_inventory_bytes=32*m["sum_inventory"]/m["cycles"],
                   nominal_fifo_bytes=27*32*params["FIFO_DEPTH"],
                   long_run_service_ceiling_groups_per_cycle=min(1/params["CONSUMER_PERIOD"], params["BANKS"]/(27*params["SERVICE_CYCLES"])),
                   **m)
        rows.append(row)
        raw[name] = dict(parameters=params, measurement=m, completion=f,
                         finite_window_accepted_beats_ceiling=service_bound)
        print(f"{name}: {row['groups_per_cycle']:.6f} group/cycle, "
              f"{row['starved_demand_percent']:.3f}% demand starvation", flush=True)
    by_name = {row["case"]: row for row in rows}
    assert by_name["independent_fast"]["groups_per_cycle"] == 1
    for name, ideal in (("shared_9_banks", 1/3), ("shared_3_banks", 1/9),
                        ("shared_1_bank", 1/27), ("bandwidth_quarter", 1/4)):
        assert abs(by_name[name]["groups_per_cycle"] / ideal - 1) < .01
    assert by_name["latency_large_buffer"]["groups_per_cycle"] > 2*by_name["latency_small_buffer"]["groups_per_cycle"]
    assert by_name["latency_large_buffer"]["groups_per_cycle"] > 2*by_name["latency_large_buffer_one_outstanding"]["groups_per_cycle"]
    # Mutation attribution: identical corrupt source both with and without
    # the new data oracle, so another structural check cannot claim the kill.
    mutation_results = []
    for mutation, reason in [(1, "weight mismatch"), (2, "weight mismatch"), (3, "scale mismatch"),
                             (4, "weight mismatch"), (5, "weight mismatch")]:
        prefix = [args.ghdl, "-r", *common, "tb_agentcard_storage", f"-gMUTATION={mutation}"]
        run(f"mutation_{mutation}_checked", prefix + ["--assert-level=error"], reason)
        control = run(f"mutation_{mutation}_unchecked", prefix + ["-gCHECK_DATA=false", "--assert-level=error"])
        unique_line(control, "PASS agentcard_storage ")
        mutation_results.append(dict(mutation=mutation, checked="detected", unchecked="completes", reason=reason))
    assert {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCES} == source_hashes
    result = dict(schema_version=1, scope="synthetic sustained streamer feed, not LLM tokens/s or device performance",
                  simulated_clock_ns=10, seed="none; deterministic address and cycle functions",
                  base_git_sha=base_sha, tool_versions=versions, source_sha256=source_hashes,
                  baseline=dict(c="OK (0 failures)", golden_vectors=4178, streamer="0 reassembly errors; cadence completed"),
                  cases=raw, summary=rows, mutation_controls=mutation_results, commands=commands)
    (out / "results.json").write_text(json.dumps(result, indent=2)+"\n")
    with (out / "summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(f"PASS: {len(rows)} sustained cases, 5 attributed data-oracle mutations, both upstream baselines")


if __name__ == "__main__":
    main()
