# AgentCard experiment 003: coordinated history and causal reuse policies

Date: 2026-10-06 UTC. Base: `29ae898f86e5e03b5cd88937c92930008bbb1e7e`,
this fork's `fpga` branch. A bounded, standard-library **software experiment**.
No model inference, trained selector, model download, RTL, remote compute,
physical hardware, paid run, dependency installation or CI run is involved.
[Experiment 002](storage-experiment-002.md) remains unchanged.

## Question and answer

Can coordinated attention-page selection and causal cache policies reduce
traffic without concealing what was omitted?

This experiment separates two different interventions:

- **Lossless relative to supplied requests:** physical KV union coalescing,
  native top-8 expert retention, whole-stage pinning, and a different causal
  eviction order. Every supplied requirement completes before layer progress.
- **LOSSY:** a cap on the combined physical KV demand of all 32 query heads,
  across both KV groups. Omitted requested pages are counted explicitly.
  Preserving fixture-required pages is not evidence of preserved reasoning.

**MEASURED in this simulator:** the tested lossless reuse policy regresses
both workloads. Capping history lowers scheduled service from 66.1 to 24.4 ms
in the disjoint fixture and from 74.3 to 32.2 ms in the phase/churn fixture,
but omits **87.5% of requested KV bytes**. Every supplied required anchor
survives; some queries retain only 6.25% of their requested bytes. The
cache-aware tie-break adds no traffic or timing gain on this fixture.

The phase/churn cap still has a 65.5 ms scheduled token at the route shift.
A cap too small for protected/fair service falls back to the full union,
recovering its slower schedule exactly. Every measured token in every case
exceeds the 20 ms necessary resource budget under the unchanged placement.
**This is a useful accounting result, not preserved reasoning or 50 tokens/s.**

The target is still a fixed 120B+ MoE agent, 50 tokens/s goal and 20 tokens/s
floor, useful 250K–1M history, and compatibility with a 16 GB **host**. The
8 GiB shared cache and 16 GiB **card** ceiling are unchanged sensitivity
assumptions inherited from 002, not an approved card-memory budget or BOM.
This metadata-only simulator's host cost does not establish an implementation's
host-memory requirement.

## What stays fixed

All nine cases use 32 physical channels, 8 dies/channel, 2 plane buffers/die,
16 KiB physical pages, 2 GB/s/channel, 40 microsecond array-read latency,
100 ns command service, 32 bridge pages, one 256 GB/s shared fast-memory bus,
8 GiB combined expert/KV cache, compact 4.5-bpw experts and the same off-chip
state placement. Units and service assumptions are inherited verbatim from
002, including zero unspecified compute, no prefetch and the stage barriers.

The frozen reference geometry is Qwen3.5-122B-A10B: 48 layers, 36 Gated
DeltaNet layers, 12 full-attention layers, 32 query heads, 2 KV heads, and
native 8-of-256 routed experts per layer. **No selected expert is dropped.**
The kernel still consumes 384 expert instances/token, including cache hits.
Routes become available only after that layer's attention dependency.

Cache hits remove cold service and fill writes; they do not remove consumer
reads. Pending fills consume capacity and do not count as hits. All existing
members of a demanded stage are pinned before allocating any misses. Both
classes use the same byte capacity, and whole-stage cache capacity checks
remain active. No future route or page choice is available to the cache.

The immutable KV snapshot excludes prefill, append writes, mutable-tail
coherence, cross-request batching and attention arithmetic. NVMe/PCIe links,
FTL behavior, endurance, writes/garbage collection, power and thermal limits
are not modeled. These are counterexample and accounting controls, not a
prototype card or hardware performance measurements.

## Policies

### Request-union baseline and native reference

The independent per-query fixture requests 128 logical blocks per query,
16,384 tokens, with disjoint lists among the 16 queries sharing a KV head.
The storage kernel deduplicates each KV group's physical union. At 262,144
history tokens this reaches the entire 6 GiB history across all 12 attention
layers, despite the per-query limit.

`native_full_lru` is a separate full-attention reference: every query requests
all history. Its physical union is identical to `disjoint_union_lru`, but it
does not read the sparse selection summaries. A request-union policy is
lossless **relative to its input candidate lists**; those independent lists
already change native attention semantics. They do not establish native model
accuracy, even when their physical union happens to cover the whole history.

### Causal windowed reuse

`windowed_reuse` changes only the eviction victim order. It counts past and
current demanded object accesses, saturating each count at two. Counts reset
at fixed token-epoch boundaries 0, 8, 16, 24. Eligible objects with fewer
observed accesses are evicted first; equal-count objects use LRU order.
Pending/pinned objects are never eligible. Both experts and KV objects share
one byte-accounted capacity; it is not a second cache or additional memory.
The policy makes no prefetch requests and has no route oracle.

This is a simple workload-sensitive control, not an optimal cache algorithm.
Counts describe demand events, not predicted semantic importance, saved work
per byte, or probability of future reuse. Its fixed epochs can discard useful
history. Variable object sizes and mixed-class eviction are checked against
an independent victim-selection oracle.

### Explicitly lossy union cap

`capped` allows 64 MiB of page-rounded KV demand **per full-attention layer**,
combined across both KV groups. Cached and cold blocks both count. Logical
128-token blocks use 128 KiB except a partial final block; the physical cap
rounds each object to 16 KiB pages. It is neither a cold-miss-only budget nor
a per-query cap.

Selection first takes every fixture-required page, then round-robins over
queries until each has at least eight requested blocks (or all available
blocks when it requests fewer). Optional extras are greedily ranked by
summed fixture score per physical byte. This is not an optimal knapsack or
minimum-union solver. Stable physical identities break ties; the cache-aware
variant prefers resident pages only on exact score-density ties. The fairness
pass likewise uses residency only after equal per-query scores. Residency never outranks score density for extras or per-query score in the
fairness pass.

The toy optional scores are all one. The final page in each query's supplied
list is an explicit required anchor with score zero. These scores and flags
are **synthetic fixture metadata**. They are not learned attention, actual
critical evidence, calibrated relevance probabilities or semantic truth.
Equal-score residency choices can still discard the real answer. The tests
include a rare low-score required-page control that is lost when its explicit
protection is removed, exposing precisely that unresolved dependence on
correctly identifying necessary information.

The capped cases serve 512 of the 4,096 requested blocks per layer: 12.5% of
the physical union. The minimum per-query byte coverage is 6.25%. Fairness
prevents zero coverage but does not promise equal coverage beyond the minimum;
stable tie ordering can favor some queries. A global average must not conceal
that weaker per-query guarantee. JSON retains the minimum across layers for
**each** query on **each** token, with score-mass coverage separately labeled.

### Safe overflow and capacity rejection

If the required-page plus greedy fairness set exceeds the cap, the selector
serves the entire original request union and flags fallback. It does not
quietly drop a required request or pretend to meet the cap. This conservative
rule can fall back even when a different combinatorial choice could fit.
`disjoint_overflow_fallback` deliberately uses a 1 MiB cap, too small even for
the required union. Its exact requests and schedule must equal the uncapped
disjoint baseline, including missed deadlines.

If this full union itself exceeds cache capacity, the existing whole-stage
check rejects the stage before allocation or I/O. It does not simulate a
streaming fallback or partially execute the stage. Tests exercise that
boundary; it is a simulator capability limit, not proof that streaming is
impossible on a different architecture.

## Controlled traces and measured window

Every main case has seed 20261006 and 32 synthetic tokens: eight startup/
warmup tokens plus 24 measured tokens. Cold-start rows are retained. The
phase/churn fixture uses a 1,010,000-token context, advancing each query's
128-block window by 128 blocks/token. It shifts all hot expert sets from
0–15 to 128–143 at token 19, off the cache's eight-token epoch boundary.
There are 11 measured tokens before this shift and 13 after it.

The 1,010,000-token context is an extension hypothesis, not a native-context
support guarantee. The phase/churn rows are compared only with matching
phase/churn rows; their larger selection-summary traffic is not silently
attributed to a policy. Likewise native full versus independent selection
has a known summary-read difference. Within each comparison, the exact trace
and candidate hashes match across policy variants.

A 24-token deterministic window exposes adaptation and churn but is not a
steady-state or production-tail study. With nearest-rank quantiles, p99 is
still the sample maximum; no population percentile claim is made. All
numbers below are **MEASURED executions of an assumed simulator**, not
measurements of model behavior or hardware throughput.

## Results

| Case | Expert / KV byte hits | Mean / max service (ms) | Mean necessary bound (ms) | Bound >50 ms (of 24) |
|---|---:|---:|---:|---:|
| `native_full_lru` | 49.89% / 100.00% | 65.925 / 67.234 | 50.169 | 17 |
| `disjoint_union_lru` | 49.89% / 100.00% | 66.122 / 67.431 | 50.365 | 24 |
| `disjoint_union_reuse` | 44.68% / 100.00% | 67.908 / 73.379 | 50.780 | 24 |
| `disjoint_cap_lru` | 99.96% / 100.00% | 24.377 / 24.488 | 24.359 | 0 |
| `phase_churn_union_lru` | 47.71% / 93.75% | 74.270 / 90.406 | 52.673 | 24 |
| `phase_churn_union_reuse` | 39.17% / 88.42% | 82.625 / 96.818 | 54.693 | 24 |
| `phase_churn_cap_lru` | 91.62% / 73.44% | 32.209 / 65.498 | 26.514 | 0 |
| `phase_churn_cap_cached` | 91.62% / 73.44% | 32.209 / 65.498 | 26.514 | 0 |
| `disjoint_overflow_fallback` | 49.89% / 100.00% | 66.122 / 67.431 | 50.365 | 24 |

What the controlled comparisons establish:

- **Lossless reuse is not an automatic win.** Disjoint expert hits fall from
  49.89% to 44.68% under windowed reuse, while KV hits stay at 100%. Phase/churn
  expert hits fall from 47.71% to 39.17%, and KV hits fall from 93.75% to 88.42%.
  This particular observed-frequency eviction rule is rejected for these
  fixtures. It does not reject every causal cache policy.
- **The cap buys room by omitting history.** Requested KV demand is 6 GiB/token;
  served demand is 0.75 GiB, omitting 5.25 GiB and 43,008 logical blocks/token.
  Disjoint expert byte hits reach 99.96%; phase/churn hits reach 91.62%.
  All 9,216 measured per-query required-anchor requests survive. Optional
  synthetic score-mass coverage is 11.81%, a fixture proxy, not semantic recall.
- **Cache-aware ties are neutral here.** The capped phase/churn variants have
  identical cache counters, physical traffic, scheduled times and resource
  bounds. No gain is claimed from introducing that knob; equal proxy scores
  do not imply equivalent real answers.
- **The longer window exposes the shift.** Phase-capped scheduled service peaks
  at 65.498 ms on token 19, so one of 24 measured tokens misses the serialized
  50 ms budget. Its 35.987 ms necessary bound on that token is channel-dominated,
  above the 33.714 ms fast-bus component. Passing the necessary 50 ms test does
  not prove a different overlap schedule or real compute can meet it.
- **Full fallback is genuinely full.** All 288 measured attention stages in
  `disjoint_overflow_fallback` flag fallback. Token timings, storage/fast/cache
  counters and expert hits match `disjoint_union_lru` exactly. All 24 measured
  tokens exceed both the scheduled and necessary 50 ms budgets.
- **Native full is separately accounted.** It differs from disjoint-union LRU
  by exactly 0.196608 ms/token of sparse-summary reads. Its necessary bound
  exceeds 50 ms on 17 of 24 measured tokens; its scheduled service exceeds
  50 ms on all 24. A mean does not replace the per-token budget checks.

Cold startup is retained rather than relabeled as steady state. The first
native-full token takes 184.798 ms in this schedule; disjoint capped starts
at 79.631 ms and phase-capped at 80.191 ms. These are excluded from the measured
window but remain in JSON alongside all other warmup tokens.

Machine-readable outputs are in `results/experiment-003/`: source/trace/candidate
provenance, all token timing/resource rows, per-token per-query minimum coverage,
full counters, summary CSV and verification evidence.

Scheduled service is the existing causal stage-barrier schedule, with all
unspecified selection and model computation set to zero. It is not achieved
tokens/s, a hardware prediction, or a universal latency lower bound. Necessary
resource-time bounds use actual policy misses and count occupied channel,
die-array and shared fast-bus time. Passing a bound does not prove feasibility;
failing a bound rejects that particular traffic/placement assumption.

## The unchanged fast-memory floor

**DERIVED:** 5,070,248,640 weight/small-parameter/input-row bytes plus
309,067,776 recurrent/convolution-state read/write bytes equals
5,379,316,416 bytes/token before KV, selection summaries and miss fills.
At 256 GB/s that alone occupies 21.013 ms, exceeding the 20 ms goal.
**No policy case changes this placement or increases the bandwidth.**

A separate analytical sensitivity removes the assumed off-chip state R/W.
Weight traffic alone then occupies 19.806 ms, leaving about 0.194 ms of a
20 ms budget for KV, fills, selection, computation and all other work.
That is a changed placement assumption, not a demonstrated policy gain,
working on-chip state design or a solved 50 tokens/s path. No 512 GB/s case
is substituted into this experiment to conceal the unchanged floor.

## What would establish useful reasoning

This work supplies omission accounting and falsifiable resource conditions.
It cannot establish attention-weight accuracy, answer retention, reasoning
quality, tool-use correctness or useful 250K–1M history.

Before a deployment or hardware commitment, the next evidence would need:

1. A real causal route/attention trace from the chosen fixed model, including
   candidate generation and selection costs, without future information
2. Paired native-full versus selected-history evaluation on long-range
   retrieval, rare decisive evidence, conflicting instructions/evidence,
   multi-step reasoning and tool-use tasks, with omissions logged by query
3. A selector that can identify necessary evidence, and a broader fallback
   whose deadline misses are reported rather than hidden
4. A device-specific placement/compute/bandwidth model covering actual state,
   cache fills, attention arithmetic, PCIe/NVMe paths, writes and power

These are requirements for later evidence, not experiments launched here.
Experiment 003 ends after publication and verification.

## Reproduction and verification

Python 3.10+ standard library, from repository root:

```sh
python3 -m unittest discover -s sim -p 'test_agentcard_*.py' -v
python3 sim/run_agentcard_policies.py --out ../agentcard003-a
python3 sim/run_agentcard_policies.py --out ../agentcard003-b
cmp ../agentcard003-a/results.json ../agentcard003-b/results.json
cmp ../agentcard003-a/summary.csv ../agentcard003-b/summary.csv
cmp ../agentcard003-a/manifest.json ../agentcard003-b/manifest.json
python3 sim/audit_agentcard_policies.py ../agentcard003-a/results.json
# Optional full per-stage identities and coverage, larger output:
python3 sim/run_agentcard_policies.py --case disjoint_cap_lru \
  --include-ledger --out ../agentcard003-ledger
```

The runner bounds token count to 2–64, defaults to 32, and accepts named-case
filters. It rejects optimized Python (`-O`) because conservation assertions
are required. It hashes all seven source/test files before and after the run
and records exact trace and candidate hashes. Only metadata is allocated.
The candidate interface is a Python fixture extension, not a claimed captured
real-attention trace format. Per-query candidate membership and exact stage
keys must match the validated native-route trace before replay.

- **93 local tests passed:** 53 existing checks and 40 new policy tests
- **Two complete nine-case runs reproduced exactly:** `results.json`,
  `summary.csv` and `manifest.json` are byte-for-byte identical
- **Independent audit passed all nine cases** without importing simulator code;
  every report result was also reviewed against the emitted data
- All seven execution-source/test hashes were unchanged during both runs;
  the independent auditor has its own recorded SHA-256
- Every case drained to zero pending pages; all 216 measured tokens/run
  exceeded the unchanged 20 ms necessary resource budget
- Optimized Python was refused, and the final full local test gate was rerun
- Observed simulator wall times were 775.639 and 760.840 seconds, with peak
  RSS 474,044 and 474,004 KiB (about 463 MiB per process). The runs partially
  overlapped; these are local simulator costs, not device/model throughput

Detailed commands, hashes and checks are in `results/experiment-003/verification.json`;
independent audit output is retained alongside it.

The separately hashed `sim/audit_agentcard_policies.py` imports no simulator
code. Its heap-based cache replay and closed-form selection reconstruct cache
counters, physical expert/KV traffic, per-query coverage, and each token's
channel/die/fast-memory resource bounds for the fixed nine-case matrix. It
checks full-fallback schedule identity and the native selection-summary delta;
it does not independently reconstruct every scheduled event.

The independent tests use literal set/page/byte arithmetic and hand-built
cache traces, not a second invocation of the implementation for expected
values. Coverage includes two physical KV groups, duplicate query coalescing,
partial tail pages, cached demand counting against the cap, rare protected
pages, conflicting/disjoint requests, per-query fairness, full-union overflow,
cache-capacity rejection, strict/nonfinite inputs, deterministic ties,
cache-hit pinning, pending fills, epoch reset, future-score/route prefix
invariance and exact preservation of all experiment-002 legacy result fields
under union+LRU. Deliberately corrupted ledgers and otherwise-consistent
protection/cap/fairness violations are caught with unchecked attribution
controls. These demonstrate policy/checker failures, not simulated accuracy.

No unrelated RTL regression was rerun: this change does not modify RTL,
experiment-001/002 sources, their outputs, regression discovery or CI workflows.
