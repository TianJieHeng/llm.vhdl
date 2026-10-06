# AgentCard experiment 002: coupled expert and attention storage

Date: 2026-10-06 UTC. Base: `52cbfc7b2a6c01302d806da8337117c17a2440f9`,
this fork's `fpga` branch. **A software experiment, not a card implementation.**
No RTL, application behavior, CI, license, model weights or physical hardware
is changed. This extends [experiment 001](storage-experiment-001.md).

## What the experiment answers

Can a proposed storage and fast-memory budget serve native top-8 experts and
attention history together, after counting shared resources, cache eviction,
page rounding, fill writes and layer dependencies?

There are two different outputs:

1. **Scheduled service time:** an executable, deliberately simple, causal
   single-stream schedule with stage barriers and zero unspecified compute.
   It is not a hardware prediction, achieved tokens/s, or a universal latency
   lower bound. Better pipelining could reduce its barriers; real compute,
   selection, synchronization and unmodeled work add costs.
2. **Necessary resource-time bounds:** the maximum of the most-loaded physical
   channel's occupied bus time, the most-loaded die's array-read service and
   the shared fast-memory bus's occupied time. These can reject a target for
   the stated actual LRU misses, packing, precision and traffic/placement assumptions, even with ideal overlap. Passing
   them cannot establish feasibility or model quality.

The targets remain 50 tokens/s (20 ms/token), a 20 tokens/s floor
(50 ms/token), useful 250K–1M history, and compatibility with a 16 GB **host**.
The simulated 16 GiB **card** memory ceiling is a sensitivity assumption, not
an approved memory budget, component choice or bill of materials. No host
memory requirement or useful-history quality is established here.

## Audited geometry versus assumed implementation

The architecture constants are derived from the
[Qwen3.5-122B-A10B configuration](https://huggingface.co/Qwen/Qwen3.5-122B-A10B/blob/main/config.json)
and its
[reference implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py),
not a downloaded checkpoint, fitted route trace or measured hardware result.
The frozen arithmetic lives in `GEOMETRY` in `sim/agentcard_coupled.py`.

| Quantity | Architecture-derived value / explicit format assumption |
|---|---:|
| Layers | 48: 36 Gated DeltaNet, full attention at zero-based 3, 7, …, 47 |
| Routed experts | 256/layer, native top-8; one resident shared expert/layer |
| One routed expert | 3 × 3,072 × 1,024 = 9,437,184 weights |
| Selected instances/token | 48 × 8 = 384 |
| Compact expert | 5,308,416 bytes at Q4 + one FP16 scale/32 weights (4.5 bpw) |
| Compact routed pool | 65,229,815,808 bytes |
| Compact selected expert bytes/token | 2,038,431,744 |
| Resident language core | 3,460,912,128 bytes with explicit FP16 small parameters |
| Recurrent state | 150,994,944 bytes, assumed FP32 |
| Convolution history | 3,538,944 bytes, assumed four BF16 slots/channel |
| Matrix work/token, before attention | 9,006,366,720 MACs |
| Full-attention heads | 32 query, 2 KV, dimension 256; 16 queries share each KV head |
| KV bytes/history token | 1,024/head/layer; 24,576 across all 12 full-attention layers |
| Full KV at 262,144 history | 6 GiB |
| Full KV at 1,010,000 history | 24,821,760,000 bytes (about 23.12 GiB) |

The resident core starts with 6,147,409,920 non-routed language parameters.
Its 6,145,327,104 matrix parameters use the assumed 4.5-bpw representation;
2,082,816 convolution/small/norm parameters use assumed FP16. Both untied
embedding and output matrices are resident, but decode reads only one input
embedding row and the full output head. Vision, mixed-precision alternatives,
additional runtime/OS/metadata allocations and actual packing of resident
matrices are outside this budget. Four-slot convolution allocation is a stated
implementation choice; three past samples suffice for the causal history.

The `repo_padded` sensitivity follows the existing
[`tools/pack_int4.py` geometry](https://github.com/TianJieHeng/llm.vhdl/blob/52cbfc7b2a6c01302d806da8337117c17a2440f9/tools/pack_int4.py):
three independent expert tensor files occupy 5,541,888 bytes/expert, while their
MAC consumer reads 5,419,008 bytes/expert. File headers, row padding and
per-lane 4 KiB alignment differ from physical NAND-page rounding, which is
charged separately. No full model was packed. Across 384 instances, these are 2,128,084,992
file bytes and 2,080,899,072 consumer bytes before flash-page rounding,
versus 2,038,431,744 compact logical weight/scale bytes. In the service
counters, `useful_*_bytes` means requested **file** bytes (including packing
overhead in the padded case); physical bytes include full read pages and
failed retry transfers. Logical compact weight payload remains separate in
the geometry table.

### Fast-memory accounting matters even on a cache hit

Compact matrix weights read per token: 5,066,081,280 bytes. Adding explicit
FP16 small parameters and the input row gives 5,070,248,640 bytes. The current
placement reads and writes the complete off-chip recurrent and convolution
state, adding 309,067,776 bytes/token. Thus the no-KV/no-fill floor is
**5,379,316,416 bytes/token**, under this placement.

At an assumed 256 GB/s this alone occupies about 21.01 ms, already beyond the
50 tokens/s budget. This is a **conditional rejection of this reference
placement**, not a universal impossibility at 256 GB/s. On-chip state, direct
storage-to-compute streaming, a different format or a different schedule
change the accounting and require a new model. Weight traffic alone is about
253.5 GB/s at 50 tokens/s. Every miss here adds a cache-fill write as well as
its subsequent consumer read. Hitting in cache removes cold traffic only.

## Service model

Time is integer nanoseconds; bandwidth is integer bytes/ns (1 byte/ns =
1 decimal GB/s). Capacity suffix GiB means 2^30 bytes. Transfer durations use
ceiling division. There is no arbitrary conversion from simulation cycles to
tokens; 1 ns is the simulation time quantum.

Baseline **assumed**, uncalibrated envelope:

- 32 physical flash channels, 8 dies/channel, 2 plane buffers/die
- 16 KiB physical read pages, 2 bytes/ns per channel, 100 ns command service
- 40,000 ns array-read service; one array operation at a time per die
- Command and data share each channel's IO bus by default
- One page buffer/plane, held through its channel transfer
- 32 page-sized controller bridge buffers, held until their fast-memory fills
  complete; fast-memory backpressure can block flash transfers
- One shared 256 bytes/ns fast-memory bus; one case uses 128 bytes/ns
- One shared whole-object LRU cache: whole experts and 128-token, per-KV-head
  blocks (128 KiB except the final partial block); baseline capacity 8 GiB
- No prefetch, no future-route oracle, no model-derived locality assumption

The 27 logical AXI lanes in experiment 001 are **not** interpreted as 27 NAND
channels. These physical-channel counts are parameters for falsification, not
an available product or measured throughput. NVMe, PCIe host/upstream links,
FTL behavior, writes, garbage collection, wear, refresh, thermal throttling,
energy and power are not silently folded into this NAND service model.

Commands are class-round-robin among eligible expert/KV queues and
round-robin over planes. Ready data transfers take priority over new commands;
finite plane buffers bound the ready-data set. An optional separate
command/data-bus mode exists as an explicit optimistic sensitivity.
`retry_every=N` deterministically fails the first attempt for physical addresses
whose `(page_id + 1) % N == 0`: it charges another command, array read,
`retry_ns` of extra retry service, and another full-page transfer. Only the
successful attempt fills fast memory. This is a testable retry abstraction,
not an empirical error distribution or a calibrated ECC controller.

### Causal schedule and cache policy

Each token and all 48 layers execute in order. A layer reads its resident
attention weights and state/selection summaries, waits its selection delay,
fetches/consumes the required KV union, waits attention compute, reads shared
expert/router weights, waits the route-ready delay, then requests the selected
native top-8 experts. Expert consumption and its compute delay finish before
the next layer. Output-head traffic is charged once per token.

Consequently native replay's expert and KV cold service is **additive through
causal dependencies and coupled through the shared cache**. It does not claim
simultaneous expert/KV flash queue contention in native single-stream decode.
Concurrent mixed queues, arbitration and bridge contention are independently
exercised by kernel tests. There is no free all-layers-at-token-start route.

All already-cached members of a demanded stage are pinned before any miss
allocation, so arbitrary expert-ID order cannot evict a demanded hit. Pending
allocations count against capacity and cannot become hits until fills complete.
Pinned/pending entries cannot be evicted. A stage whose complete working set
exceeds cache capacity is rejected before allocation. This is **whole-stage
pinning, not tiled or streaming attention**; its capacity rejection does not
prove a tiled design impossible. The cache starts empty. Warmup, measurement
and final drain are reported separately; the short measured window is not
claimed to be steady state. All queues and pending fills drain to zero.

The simulated history is an immutable KV snapshot. Prefill, new-token KV
append writes, mutable tail-block coherence and cross-request batching are
outside scope. Physical KV block addresses are stable and disjoint from expert
addresses. Within each KV head the selections of all 16 grouped query heads
are unioned before I/O. Multiple queries selecting the same block do not cause
multiple reads. Whole KV blocks, rather than individual physical flash pages,
are the independently evictable cache entries.

## Scenarios and results

All scenarios use seed 20261006, six synthetic tokens, two startup/warmup
tokens and four measured tokens. Every token has all 48 layers and native
8-of-256 routing. The hot synthetic route set is 16 experts/layer; uniform
routes sample all 256; the shift case changes every hot set halfway through.
These locality patterns are controls, **not observations of Qwen routing**.

Selected-attention cases select 128 logical blocks (16,384 tokens) per query
head. The shared-head case unions to 16,384 tokens/KV head. The disjoint-head
control expands the union 16× to the complete 262,144-token history, exposing
why per-query selection caps are not physical KV traffic caps. Churn advances
block selections each token. All selected-attention cases change native
attention semantics; selection quality and recall have not been tested.
The 1,010,000 history case is a context-extension hypothesis, not a native
context-length guarantee.

| Case | Expert byte hits | KV byte hits | Mean / max service (ms) | Mean necessary bound (ms) | Bound >50 ms (of 4) |
|---|---:|---:|---:|---:|---:|
| `expert_only_hot_8g` | 88.61% | n/a | 26.144 / 31.826 | 21.920 | 0 |
| `native_full_attention_8g` | 49.80% | 100.00% | 66.045 / 66.459 | 50.176 | 4 |
| `selected_hot_8g` | 88.61% | 100.00% | 27.913 / 33.595 | 23.690 | 0 |
| `selected_hot_1m_context` | 88.61% | 100.00% | 28.474 / 34.156 | 24.251 | 0 |
| `selected_churn_8g` | 88.61% | 0.00% | 34.783 / 40.465 | 25.263 | 0 |
| `selected_disjoint_heads_8g` | 49.80% | 100.00% | 66.242 / 66.656 | 50.372 | 4 |
| `selected_uniform_routes_8g` | 9.77% | 100.00% | 56.999 / 58.191 | 30.018 | 0 |
| `selected_shift_routes_8g` | 50.20% | 100.00% | 42.231 / 59.615 | 27.215 | 0 |
| `selected_hot_4g` | 88.15% | 100.00% | 28.135 / 33.595 | 23.726 | 0 |
| `selected_churn_8channels` | 88.61% | 0.00% | 87.919 / 116.713 | 40.210 | 1 |
| `selected_hot_fast128` | 88.61% | 100.00% | 50.655 / 56.149 | 47.379 | 0 |
| `selected_churn_repo_padded` | 88.61% | 0.00% | 35.056 / 40.900 | 25.468 | 0 |

**What changed when resources were coupled:**

- Native full attention warms all 6 GiB of KV, but that leaves little space in
  the 8 GiB cache for experts. Its 49.80% expert byte-hit rate still causes
  refills; fast-memory traffic averages 12.845 GB/token. The fast-memory
  resource bound exceeds 50 ms in all four measured tokens. This rejects
  that particular 8 GiB-cache/256 GB/s placement at the 20 tokens/s floor,
  not every use of a 16 GiB card or a larger cache.
- Shared selected blocks keep expert byte hits at 88.61% in this short window.
  Making the grouped-query selections disjoint expands KV to full history and
  drops expert hits to 49.80%. The resulting 66.242 ms mean scheduled service
  is not rescued by having a per-query selection cap.
- Changing only selected-page locality from hot to churn makes KV cold reads
  recur every token: mean scheduled service rises from 27.913 to 34.783 ms.
  Reducing physical channels from 32 to 8 raises the churn case to 87.919 ms;
  one measured token exceeds even the 50 ms necessary resource bound.
- Uniform native routes achieve only 9.77% expert byte hits in this bounded
  trace. Its 56.999 ms scheduled service fails the serialized 50 ms budget,
  while its 30.018 ms necessary resource bound does **not** establish a
  universal bandwidth impossibility. Phase shifts likewise expose a
  59.615 ms worst scheduled token hidden by the 42.231 ms average.
- Every measured token in every case exceeds the 20 ms necessary resource
  bound under the stated off-chip-state/fast-memory placement. None proves
  the 50 tokens/s target. Conversely, low sparse-case bounds do not prove
  preserved reasoning quality or sufficient compute.

Startup is materially different: the native-full first token takes 184.798 ms
in this schedule; selected-hot starts at 65.878 ms. Both startup tokens are
retained in JSON and excluded from the four-token measurement window.

The machine-readable files are in `results/experiment-002/`. Per-token rows
retain startup and measurement separately, physical expert/KV traffic, all
fast-memory traffic, scheduled service time and necessary resource bounds.
The CSV includes p50/p95/p99/max scheduled latency, dependency-wait quantiles,
byte-weighted hits, cache peak/allocation, physical/file amplification and
counts above both 20 ms and 50 ms budgets. Four measured tokens cannot estimate
production tail probabilities; token p95 and p99 are simply their nearest-rank
sample maxima. Stage quantiles summarize the observed dependency waits.

### How to use a rejection

- A necessary resource bound above 20 ms rejects 50 tokens/s for that
  traffic/placement. Above 50 ms rejects the 20 tokens/s floor for that
  traffic/placement. Examine every token, not only averages.
- A scheduled service time above a budget rejects this particular serialized
  schedule. It does not by itself reject all possible pipelined implementations.
- A bound below budget leaves compute and omitted work unresolved. Zero-delay
  compute and selection in these synthetic traces do not grant free real MACs.
- Native full attention, selected attention and hypothetical factor/codebook
  architectures are different quality/compute problems. Only the first two
  are represented here; no factor/codebook claim is made.

The next useful evidence is a real causal expert-route/QKV selection trace
and a placement/timing model for a specific device. Establish head-union
locality and phase-change behavior before buying storage. If selection cannot
preserve reasoning/tool accuracy, the sparse cases are invalid regardless of
storage speed. If a chosen device cannot meet the byte/resource bound or its
buffers cannot support the required schedule, reject that design before
hardware spending. This experiment manufactures neither locality nor quality.

## Reproduction and importer

Python 3.10+ standard library only. No dependency install, model download,
network access, CI, paid compute, remote machine or hardware operation is used.
Run from the repository root:

```sh
python3 -m unittest discover -s sim -p 'test_agentcard_*.py' -v
python3 sim/run_agentcard_coupled.py --out ../agentcard-002-a
python3 sim/run_agentcard_coupled.py --out ../agentcard-002-b
cmp ../agentcard-002-a/results.json ../agentcard-002-b/results.json
cmp ../agentcard-002-a/summary.csv ../agentcard-002-b/summary.csv
# Full per-stage timing ledger (larger JSON), or a focused matrix case:
python3 sim/run_agentcard_coupled.py --case selected_hot_8g \
  --include-stages --out ../agentcard-002-stages
# Schema example contains two synthetic tokens, so warmup must be < 2:
python3 sim/run_agentcard_coupled.py \
  --trace sim/fixtures/agentcard/trace-v1-example.json --warmup-tokens 1 \
  --out ../agentcard-002-import
```

The default matrix has 12 cases. Source SHA-256 hashes, exact geometry,
configuration, seed parameters and canonical generated-trace SHA-256 hashes
are recorded; the runner aborts if its source files change during execution.
Results are deterministic for the recorded Python version. The runner refuses
`python -O`, because it disables the conservation assertions. Each invocation
has a strict 64 MiB imported-trace limit, 1–256 native tokens, a maximum
1,010,000-token context and one million queued physical pages/stage. Only
metadata is allocated; no multi-gigabyte weight/KV arrays are created.

`--device file.json` accepts exact `DeviceConfig` fields for an imported trace;
unknown fields and invalid types/ranges are rejected. `--format repo_padded` and `--cache-gib` also apply only to imported traces;
the matrix defines those settings per named case. `--card-gib` and
`--warmup-tokens` apply to either mode. The
example is a fixture, not a captured model trace. A real trace exporter must
write this schema and supply honest provenance:

- Root: `schema_version: 1`, nonempty `provenance`, `context_tokens`,
  `attention` (`full`, `selected_hypothesis` or `none_control`), `tokens`
- Each token: contiguous zero-based `token`, nondecreasing `not_before_ns`
  and exactly 48 ordered `layers`
- Each layer: exact zero-based `layer`, eight distinct native `experts` IDs
  in 0–255, `kv_blocks_by_query_head`, and nonnegative integer
  `selection_delay_ns`, `attention_compute_ns`, `route_delay_ns`,
  `expert_compute_ns`
- For selected full-attention layers, exactly 32 query-head lists of unique
  valid 128-token block IDs. Each ID maps deterministically to physical page
  IDs using layer, KV head, block and configured physical page size. Other
  layers/modes use an empty list; `full` enumerates the complete valid history
- Delays are relative to the preceding dependency, not future absolute-route
  availability promises. Measured delays must be converted consistently and
  must not include costs already charged as storage/fast-memory service

Unknown fields, reordered layers/tokens, malformed or duplicate selections,
unknown expert/block IDs and noncausal release order are rejected. This
interface does not make an imported trace a quality evaluation.

## Verification

- 53 local standard-library tests passed, including independent kernel tests
  and replay/geometry/causality tests
- Independent review found no blocking model, accounting, causal or claim
  defects; all 12 fast-memory budgets were reconstructed independently
- Two complete 12-case executions reproduced `results.json`, `summary.csv`
  and `manifest.json` byte-for-byte
- Observed local execution times: 117.83 s and 116.07 s; peak simulator RSS
  103,380 KiB and 103,280 KiB (about 101 MiB). These are simulator host costs,
  not device throughput measurements
- The committed example passed the importer; `python -O` was rejected as
  required. Every run's input source hashes remained unchanged
- All service/cache/resource checks passed, with zero terminal pending pages
  in every case. Detailed verification and hashes are in `verification.json`

Independent tests use literal hand-derived schedules, rather than deriving
expected values with the implementation under test: page boundaries, exact
shared/separate bus timing, die/plane serialization, bounded bridge stalls,
retry transfers, class fairness, pending coalescence without a hit, pinned
cache eviction, demand pre-pinning, head union, partial blocks, stable physical
addresses and causal route-delay/prefix invariance. Deliberately corrupted
counters and overlapping resource records demonstrate checker failures; these
are **checker-corruption controls**, not source-side mutation coverage.

No unrelated RTL suite was rerun because no RTL, streamer runner or regression
discovery rule changed. Experiment 001's baseline results remain unchanged;
this experiment's Python tests are a separate local-only gate.
