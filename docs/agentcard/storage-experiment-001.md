# AgentCard experiment 001: sustained synthetic storage feed

Status: simulation only. Base: `d07fe5550aff5e7ef15c31fc63b49d55aa53cf9a` on
this fork's `fpga` branch. Date: 2026-10-06 UTC.

## Question and boundary

How do shared read bandwidth, request delay, outstanding requests, and small
buffers limit the stream of weight/scale groups accepted by a compute consumer?

This experiment drives the **unchanged** `rtl/weight_streamer.vhd` and its read
ports/FIFOs with sustained synthetic AXI reads. It uses the FK33 packing geometry:
24 weight lanes plus 3 scale lanes, each 256 bits. One joint consumption is
48 rows x 32 4-bit weight indices plus 48 16-bit scales: **864 bytes for 1,536
weight elements**, or 4.5 bits/weight before file padding. It does not run the
matvec arithmetic, decode an LLM, or execute an expert-routing policy.

This measures a mechanism needed by the proposed storage-backed AgentCard. It
**does not establish** 120B+ MoE feasibility, 20 or 50 tokens/s, useful 250K-1M
context, accuracy, PCIe/host compatibility, a 16 GB host memory budget, physical
flash bandwidth/latency, power, FPGA timing/resource use, or a final on-card
memory budget. There are no model downloads, device accesses, hardware changes,
CI jobs, or FPGA tools in this experiment.

## Model and precise assumptions

- Single clock, 10 ns simulation period. Reported rates are per simulated cycle;
  this is not a measured or timing-closed 100 MHz device. `DUAL_CLK=false`,
  `FAST_POP=true`. The upstream baseline separately covers both FIFO clock paths
  and both FAST_POP settings; that does not make this storage model a CDC test.
- 27 independent AXI address lanes, 32 bytes/beat, sequential reads, bounded
  per-lane request queues, in-order responses per lane, no response reordering.
  All base addresses are 1 MiB apart and 4 KiB aligned. AR addresses, sizes,
  lengths, INCR mode, 4 KiB boundaries, and exact requested/returned totals are
  checked. Burst length defaults to 16; the final burst is deliberately partial.
- `BANKS` must divide 27. Lane p belongs to bank p mod BANKS. A bank rotates fairly
  among eligible lane heads. A selected response reserves the bank until its
  handshake; RVALID/data/RLAST remain held under backpressure. Consecutive accepted
  beats in that bank are at least SERVICE_CYCLES apart, asserted in simulation.
  Default 27 banks means 27 separate beat-service resources, not a single shared
  864-byte/cycle link. No PCIe, NAND, HBM, or NVMe implementation is implied.
- A burst's earliest first response is AR acceptance +1+LATENCY+(lane mod 5)*SKEW
  cycles, plus PAGE_DELAY when its start beat is divisible by PAGE_BEATS.
  Subsequent responses in the burst obey bank service limits. Requests can
  overlap these delays up to OUTSTANDING and the RTL's FIFO-space reservations.
  A later request never overtakes its lane's head. PAGE_DELAY is a synthetic
  per-page-start request penalty, not serialized NAND die service, a cache, or
  a measured device property. Cross-lane address issue is unconstrained apart
  from queue bounds; no shared command bus is modeled.
- The consumer accepts weight and scale together. It requests work every
  CONSUMER_PERIOD cycles except for PAUSE_CYCLES in each PAUSE_EVERY cycle block.
  These pauses exercise consumer backpressure and upstream request throttling.
  The returned R-stall and AR-stall counts describe only those physical
  handshakes, not all time the RTL intentionally refrains from issuing reads.
- Source values are deterministic functions of group, row and element,
  including higher group bits and row/element cross-terms. No random seed is
  used. The source writes lane-bit slices; the checker independently walks
  rows/elements and checks every bit of both output streams. This proves the
  tested data/order cases, not every possible AXI schedule or corruption.

## Windows and accounting

Every case starts empty, consumes 512 warmup groups, measures 2048 groups, and then
consumes 393 tail groups to full drain, for 2953 groups / 79,731 beats total.
Measurement begins on the edge after the 512th consumption and ends on the edge
accepting the last measured group, inclusive. Initial inventory is sampled before
that first edge; ending inventory is sampled after the final edge. All incoming
R handshakes and output consumptions on those edges are included.

Inventory means **accepted input beats minus joint consumed groups, per lane**.
It includes FIFO output stages and the scale assembly hold; it is not a probe of
raw BRAM occupancy. Average/minimum/maximum inventory use post-handshake samples
on every measured edge. Byte counts multiply beats by 32. Nominal FIFO memory is
27 x 32 x FIFO_DEPTH bytes, excluding small output stages, request metadata and
any other proposed card memory.

Per-lane and aggregate conservation are asserted at the measurement boundaries:

`start_inventory + accepted_input_beats = 27 * measured_groups + end_inventory`

The window is longer than 4 x (FIFO_DEPTH+4) groups, incoming beats exceed 4 times
initial inventory, the tail is longer than FIFO_DEPTH+BURST_BEATS+4, and every
lane's last input handshake must occur **after** the measured window. The result
therefore cannot be a prefilled-FIFO drain result. All 2,953 groups are checked,
and final per-lane input/output totals, outstanding request queues and inventory
must drain exactly. A 2,000,000-cycle watchdog and 120-second subprocess timeouts
bound execution; a timeout or stop-time exit cannot be mistaken for completion.

Starvation is the fraction of requested consumer cycles with either input stream
unavailable. Intentional consumer pauses are excluded from that denominator.
Feed is measured groups / all measured cycles. "Weight elements/cycle" is feed
x 1,536, a **potential arithmetic-input rate**, not an executed MAC or token rate.

Analytical cross-check: in C consecutive cycles, a bank can accept at most
ceil(C/SERVICE_CYCLES) beats. The runner checks the aggregate finite-window bound
and allows initial inventory when bounding output. The loose long-run ceiling is
min(1/CONSUMER_PERIOD, BANKS/(27*SERVICE_CYCLES)); periodic consumer pauses, request
delays, head-of-line blocking and limited buffers can lower it further.

## Results

See [summary.csv](results/experiment-001/summary.csv) for the numeric matrix and
[results.json](results/experiment-001/results.json) for parameters, measured
counts, source SHA-256 hashes, commands, tool versions and mutation outcomes.
A clean rerun reproduced every numeric row and mutation outcome exactly.
Only those bounded text results and [completion excerpts](results/experiment-001/verification.txt) are committed. Generated
executables, simulator work libraries, synthetic images and scratch logs stay
outside the repository.

The upstream C reference self-test completed with 0 failures and 4,178 arithmetic
golden-vector checks; its synthetic 100 x 200 FK33 image was 114,688 bytes. The
original streamer test completed with 0 reassembly errors and both fast/slow,
single/dual-clock cadence verdicts. Those original cadence probes prefill FIFOs;
only the new bench supplies evidence about sustained fresh-read feed.

The independent-bank fast case attains 1 group/cycle without starvation. Sharing
lanes over 9, 3, 1 banks gives exactly 1/3, 1/9, 1/27 group/cycle; setting service spacing
to 4 gives 1/4. These equal analytical bandwidth ceilings. A 128-cycle request delay
with 32-beat per-lane buffers gives approximately 0.215 group/cycle; 256-beat buffers
recover 1 group/cycle **under these independent-bank and outstanding-request
assumptions**. More buffering hides bounded delay only if enough requests can be
in flight; it cannot manufacture shared bandwidth.

Nominal FIFO storage at depths 32, 64, 256 is 27,648 / 55,296 / 221,184 bytes. These are
small, bounded streamer buffers; they are not the total memory requirement of an
LLM accelerator. The next experiment would need an explicit target-device read
model (including realistic page/die concurrency), expert access/reuse traces and
all shared-link bottlenecks before selecting a physical memory budget.

## Verification that the checker can fail

Five source-side mutations are each run twice, once with the data oracle and
once with it disabled: one weight bit flipped, one-group payload replay from a burst earlier, one scale
bit flipped, weight lanes swapped by half the lane array, and a weight-only
176-group replay. The checked run must fail with the intended data mismatch;
the unchecked control must reach normal completion. This attributes the failure
to the new oracle rather than another structural assertion. Earlier draft
patterns aliased half-array lane swaps; the independent review identified this,
and cross-coordinate terms plus explicit lane/replay mutations closed the gap.

## Reproduce locally

Requirements: Python 3 standard library, C compiler, GHDL with VHDL-2008 support.
No package installation or network access is performed by the runner.

```sh
python3 sim/run_agentcard_storage.py --out /path/outside/repo/agentcard-001
# Optional executable override (GHDL environment variable also works):
python3 sim/run_agentcard_storage.py --ghdl /path/to/ghdl-mcode \
  --out /path/outside/repo/agentcard-001
```

The run must end with a PASS summary for all 12 cases and all 5 attributed
mutations, with both upstream baseline completion markers. Each case needs a
unique MEASURE record and PASS completion, not merely process exit 0.

The recorded cloud run used GHDL 5.0.1 (`5.0.1+dfsg-1+b1`) from Debian's signed
trixie snapshot package index dated 20260925T181338Z. Because system installation
was unnecessary, `ghdl-mcode` and `ghdl-common` were downloaded with APT signature/
package-hash verification and extracted to a workspace-local tools directory.
`GHDL_PREFIX` pointed to that installation's `usr/lib/ghdl/mcode/vhdl` directory.
No global security settings were changed. See the result metadata for actual
compiler/Python version strings and source hashes.

Existing license and attribution remain intact; this is a fork-only experiment.
No upstream RTL, arithmetic tables, hardware scripts, or historical worklog
entries were changed.
