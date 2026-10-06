#!/usr/bin/env python3
"""Experiment 002: causal metadata-only coupled expert/KV storage replay.

All bandwidths are integer bytes/ns, all times integer ns. No model or hardware
is loaded. Device parameters and synthetic locality are assumptions, not facts.
"""
from __future__ import annotations

from collections import Counter, OrderedDict, deque
from dataclasses import asdict, dataclass
import heapq
import math


def ceildiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True)
class DeviceConfig:
    page_bytes: int = 16384
    channels: int = 8
    dies_per_channel: int = 8
    planes_per_die: int = 2
    channel_bytes_per_ns: int = 2
    command_ns: int = 100
    read_ns: int = 40000
    shared_command_data_bus: bool = True
    bridge_pages: int = 32
    fast_bytes_per_ns: int = 256
    retry_every: int = 0
    retry_ns: int = 40000

    def __post_init__(self):
        for key, value in asdict(self).items():
            if key == "shared_command_data_bus":
                if type(value) is not bool:
                    raise ValueError("shared_command_data_bus must be bool")
            else:
                integer(value, key, 0 if key in ("retry_every", "retry_ns") else 1)
        if self.page_bytes > 1024 * 1024 or self.channels * self.dies_per_channel * self.planes_per_die > 4096:
            raise ValueError("device exceeds bounded experiment limits")


@dataclass(frozen=True)
class Read:
    key: str
    kind: str
    size: int
    first_page: int

    def __post_init__(self):
        if not isinstance(self.key, str) or not self.key or self.kind not in ("expert", "kv"):
            raise ValueError("read needs a nonempty key and expert/kv kind")
        integer(self.size, "read size", 1)
        integer(self.first_page, "first_page")


class FastBus:
    def __init__(self, bytes_per_ns):
        self.rate = integer(bytes_per_ns, "fast bandwidth", 1)
        self.until = 0
        self.bytes = Counter()
        self.busy_ns = 0
        self.expected_busy_ns = 0
        self.expected_bytes = Counter()
        self.intervals = []
        self.audit = False

    def transfer(self, ready, size, kind):
        integer(ready, "transfer ready")
        integer(size, "transfer size")
        if not size:
            return ready
        start = max(ready, self.until)
        end = start + ceildiv(size, self.rate)
        self.until = end
        self.bytes[kind] += size
        self.busy_ns += end - start
        self.expected_busy_ns += ceildiv(size, self.rate)
        self.expected_bytes[kind] += size
        if self.audit:
            self.intervals.append((start, end, size, kind))
        return end


class PageService:
    """Nonpreemptive page reads, FIFO per kind/plane, fair class alternation.

    A die performs at most one array read at once; planes each have one page
    buffer and retain it through channel transfer. Commands share channel IO
    with data by default. A bounded controller bridge retains transferred pages
    until the single fast-memory bus accepts them. Retry is extra array service
    after a failed full-page transfer; the repeated command, array read and
    full-page data transfer are charged. Failed data is not filled into cache.
    """
    def __init__(self, config=DeviceConfig(), fast=None, audit=False):
        self.c = config
        self.fast = fast or FastBus(config.fast_bytes_per_ns)
        self.audit = audit
        self.records = []
        self.stats = Counter()
        self.time = 0
        self.pending = 0

    def run(self, reads, ready_ns):
        c = self.c
        integer(ready_ns, "ready_ns")
        if ready_ns < self.time:
            raise ValueError("noncausal storage arrival")
        reads = list(reads)
        if len({r.key for r in reads}) != len(reads):
            raise ValueError("duplicate storage object: coalesce before service")
        if not reads:
            self.time = ready_ns
            return {}
        if sum(ceildiv(r.size, c.page_bytes) for r in reads) > 1000000:
            raise ValueError("stage exceeds one-million queued-page metadata limit")
        nplanes = c.channels * c.dies_per_channel * c.planes_per_die
        queues = [[deque(), deque()] for _ in range(nplanes)]
        occupied = [False] * nplanes
        die_busy = [False] * (c.channels * c.dies_per_channel)
        cmd_busy = [False] * c.channels
        data_busy = [False] * c.channels
        ready_data = [deque() for _ in range(c.channels)]
        cursor = [0] * c.channels
        preferred = [0] * c.channels
        channel_planes = [list(range(ch, nplanes, c.channels)) for ch in range(c.channels)]
        events = []
        sequence = 0
        bridge = 0
        plane_count = 0
        todo = Counter()
        complete = {}
        now = ready_ns
        pages = 0
        for r in reads:
            count = ceildiv(r.size, c.page_bytes)
            todo[r.key] = count
            self.stats[f"useful_{r.kind}_bytes"] += r.size
            for offset in range(count):
                address = r.first_page + offset
                p = address % nplanes
                amount = min(c.page_bytes, r.size - offset * c.page_bytes)
                queues[p][r.kind == "kv"].append((r.key, r.kind, address, amount, p, 0))
                pages += 1
        self.pending = pages
        self.stats["submitted_pages"] += pages
        self.stats["peak_queued_pages"] = max(self.stats["peak_queued_pages"], pages)

        def event(at, kind, payload):
            nonlocal sequence
            sequence += 1
            heapq.heappush(events, (at, sequence, kind, payload))

        def record(kind, start, end, page, resource):
            if self.audit:
                self.records.append(dict(operation=kind, start_ns=start, end_ns=end,
                                         key=page[0], kind=page[1], page=page[2], resource=resource))

        def pump(ch):
            nonlocal bridge, plane_count
            shared_free = not (cmd_busy[ch] or data_busy[ch])
            data_free = shared_free if c.shared_command_data_bus else not data_busy[ch]
            if data_free and ready_data[ch] and bridge < c.bridge_pages:
                page = ready_data[ch].popleft()
                data_busy[ch] = True
                bridge += 1
                self.stats["peak_bridge_pages"] = max(self.stats["peak_bridge_pages"], bridge)
                duration = ceildiv(c.page_bytes, c.channel_bytes_per_ns)
                self.stats[f"physical_{page[1]}_bytes"] += c.page_bytes
                self.stats[f"channel_{ch}_data_ns"] += duration
                record("data", now, now + duration, page, ch)
                event(now + duration, "data_end", page)
            command_free = not cmd_busy[ch] and (not data_busy[ch] or not c.shared_command_data_bus)
            if not command_free:
                return
            ps = channel_planes[ch]
            for k in (preferred[ch], 1 - preferred[ch]):
                for step in range(len(ps)):
                    idx = (cursor[ch] + step) % len(ps)
                    p = ps[idx]
                    # Address striping: channel, then die, then plane.
                    die = p % (c.channels * c.dies_per_channel)
                    if queues[p][k] and not occupied[p] and not die_busy[die]:
                        page = queues[p][k].popleft()
                        occupied[p] = True
                        die_busy[die] = True
                        cmd_busy[ch] = True
                        plane_count += 1
                        self.stats["peak_plane_buffers"] = max(self.stats["peak_plane_buffers"], plane_count)
                        cursor[ch] = (idx + 1) % len(ps)
                        preferred[ch] = 1 - k
                        array_time = c.read_ns + page[5] * c.retry_ns
                        self.stats["issued_attempts"] += 1
                        self.stats[f"channel_{ch}_command_ns"] += c.command_ns
                        self.stats[f"die_{die}_array_ns"] += array_time
                        record("command", now, now + c.command_ns, page, ch)
                        record("array", now + c.command_ns, now + c.command_ns + array_time, page, die)
                        event(now + c.command_ns, "command_end", ch)
                        event(now + c.command_ns + array_time, "array_end", page)
                        return

        for ch in range(c.channels):
            pump(ch)
        while events:
            now, _, operation, page = heapq.heappop(events)
            affected = range(c.channels) if operation == "fill_end" else [page if operation == "command_end" else page[4] % c.channels]
            if operation == "command_end":
                cmd_busy[page] = False
            elif operation == "array_end":
                p = page[4]
                die_busy[p % (c.channels * c.dies_per_channel)] = False
                ready_data[p % c.channels].append(page)
            elif operation == "data_end":
                p = page[4]
                data_busy[p % c.channels] = False
                occupied[p] = False
                plane_count -= 1
                self.stats["completed_attempts"] += 1
                failed = c.retry_every and (page[2] + 1) % c.retry_every == 0 and page[5] == 0
                if failed:
                    self.stats["retries"] += 1
                    affected = range(c.channels)
                    bridge -= 1
                    queues[p][page[1] == "kv"].appendleft((*page[:5], 1))
                else:
                    end = self.fast.transfer(now, page[3], "fill_" + page[1])
                    record("bridge", now, end, page, 0)
                    event(end, "fill_end", page)
            elif operation == "fill_end":
                bridge -= 1
                todo[page[0]] -= 1
                self.pending -= 1
                self.stats["completed_pages"] += 1
                if not todo[page[0]]:
                    complete[page[0]] = now
            # Multiple same-time events may unlock resources; pumping after each
            # is deterministic, conservative, and never starts a busy resource.
            for ch in affected:
                pump(ch)
        if self.pending or bridge or plane_count or any(todo.values()) or any(die_busy):
            raise AssertionError("terminal drain failed")
        self.time = now
        self.stats["calls"] += 1
        return complete

    def check(self):
        c, s = self.c, self.stats
        assert not self.pending
        assert s["submitted_pages"] == s["completed_pages"]
        assert s["issued_attempts"] == s["completed_attempts"]
        assert s["completed_attempts"] == s["completed_pages"] + s["retries"]
        assert s["completed_attempts"] * c.page_bytes == s["physical_expert_bytes"] + s["physical_kv_bytes"]
        assert s["useful_expert_bytes"] == self.fast.bytes["fill_expert"]
        assert s["useful_kv_bytes"] == self.fast.bytes["fill_kv"]
        assert s["peak_bridge_pages"] <= c.bridge_pages
        assert s["peak_plane_buffers"] <= c.channels * c.dies_per_channel * c.planes_per_die
        for ch in range(c.channels):
            assert s[f"channel_{ch}_data_ns"] <= self.time
            if c.shared_command_data_bus:
                assert s[f"channel_{ch}_data_ns"] + s[f"channel_{ch}_command_ns"] <= self.time
        assert self.fast.busy_ns <= max(self.time, self.fast.until)
        assert self.fast.busy_ns == self.fast.expected_busy_ns
        assert self.fast.bytes == self.fast.expected_bytes
        for die in range(c.channels * c.dies_per_channel):
            assert s[f"die_{die}_array_ns"] <= self.time
        if self.audit:
            groups = {}
            for r in self.records:
                op = r["operation"]
                if op not in ("array", "command", "data"):
                    continue
                resource = ("io" if op != "array" and c.shared_command_data_bus else op, r["resource"])
                groups.setdefault(resource, []).append((r["start_ns"], r["end_ns"]))
            for intervals in groups.values():
                intervals.sort()
                assert all(a[1] <= b[0] for a, b in zip(intervals, intervals[1:])), "resource overlap"


class Cache:
    """Shared whole-object LRU. Allocations include pending fills; pins cannot evict."""
    def __init__(self, capacity):
        self.capacity = integer(capacity, "cache capacity", 1)
        self.entries = OrderedDict()
        self.seen = set()
        self.used = self.peak = 0
        self.stats = Counter()

    def acquire(self, read, now):
        integer(now, "cache acquire time")
        if read.key in self.entries:
            e = self.entries[read.key]
            if e["read"] != read:
                raise ValueError("object identity changed")
            e["pins"] += 1
            self.entries.move_to_end(read.key)
            state = "hit" if e["ready"] is not None and e["ready"] <= now else "pending"
        else:
            if read.size > self.capacity:
                raise ValueError("object exceeds cache capacity; tiling is not modeled")
            while self.used + read.size > self.capacity:
                victim = next((key for key, e in self.entries.items() if not e["pins"] and e["ready"] is not None and e["ready"] <= now), None)
                if victim is None:
                    raise ValueError("pinned stage exceeds cache capacity; tiling is not modeled")
                old = self.entries.pop(victim)
                self.used -= old["read"].size
                self.stats["evicted_bytes"] += old["read"].size
                self.stats["evicted_objects"] += 1
            if read.key in self.seen:
                self.stats[f"{read.kind}_refetch_bytes"] += read.size
                self.stats[f"{read.kind}_refetch_objects"] += 1
            self.entries[read.key] = dict(read=read, ready=None, pins=1)
            self.used += read.size
            self.peak = max(self.peak, self.used)
            state = "miss"
        self.stats[f"{read.kind}_{state}_bytes"] += read.size
        self.stats[f"{read.kind}_{state}_objects"] += 1
        return state

    def mark_ready(self, key, at):
        e = self.entries[key]
        if e["ready"] is not None:
            raise ValueError("duplicate completion")
        e["ready"] = integer(at, "fill completion")
        self.seen.add(key)

    def release(self, key, now):
        integer(now, "cache release time")
        e = self.entries[key]
        if e["pins"] <= 0 or e["ready"] is None or now < e["ready"]:
            raise ValueError("release before ready / unpinned")
        e["pins"] -= 1

    def check(self):
        assert self.used == sum(e["read"].size for e in self.entries.values())
        assert self.used <= self.peak <= self.capacity
        assert all(e["pins"] == 0 and e["ready"] is not None for e in self.entries.values())


# Architecture-derived, explicit hypothetical precision (no tensors downloaded).
GEOMETRY = dict(layers=48, experts_per_layer=256, top_k=8,
                hidden=3072, expert_intermediate=1024, query_heads=32,
                kv_heads=2, head_dim=256, full_attention_layers=list(range(3, 48, 4)),
                kv_bytes_per_token_per_head=1024, selection_block_tokens=128,
                expert_weights=9437184, compact_expert_bytes=5308416,
                padded_expert_file_bytes=5541888, padded_expert_consumed_bytes=5419008,
                resident_core_bytes=3460912128, recurrent_state_bytes=150994944,
                conv_state_bytes=3538944, matrix_mac_per_token=9006366720,
                compact_matrix_bytes_per_token=5066081280)


def strict_keys(value, required, optional=()):
    if not isinstance(value, dict) or not set(required) <= value.keys() or set(value) - set(required) - set(optional):
        raise ValueError(f"expected keys {required}, optional {optional}")


def validate_trace(trace):
    """Strict importer contract. Delays are causal offsets, never future-route timestamps."""
    strict_keys(trace, ("schema_version", "provenance", "context_tokens", "attention", "tokens"))
    if trace["schema_version"] != 1 or type(trace["schema_version"]) is not int:
        raise ValueError("unknown trace schema")
    if not isinstance(trace["provenance"], str) or not trace["provenance"].strip():
        raise ValueError("trace provenance is required")
    context = integer(trace["context_tokens"], "context_tokens", 1)
    if context > 1010000:
        raise ValueError("context exceeds experiment scope")
    if trace["attention"] not in ("none_control", "selected_hypothesis", "full"):
        raise ValueError("unknown attention mode")
    tokens = trace["tokens"]
    if not isinstance(tokens, list) or not 1 <= len(tokens) <= 256:
        raise ValueError("trace needs 1..256 tokens")
    block_count = ceildiv(context, GEOMETRY["selection_block_tokens"])
    previous_ready = 0
    for ti, token in enumerate(tokens):
        strict_keys(token, ("token", "not_before_ns", "layers"))
        if type(token["token"]) is not int or token["token"] != ti:
            raise ValueError("token order is not contiguous")
        not_before = integer(token["not_before_ns"], "not_before_ns")
        if not_before < previous_ready:
            raise ValueError("noncausal token ready times")
        previous_ready = not_before
        if not isinstance(token["layers"], list) or len(token["layers"]) != 48:
            raise ValueError("native trace needs all 48 layers")
        for li, layer in enumerate(token["layers"]):
            strict_keys(layer, ("layer", "experts", "kv_blocks_by_query_head", "selection_delay_ns", "attention_compute_ns", "route_delay_ns", "expert_compute_ns"))
            if type(layer["layer"]) is not int or layer["layer"] != li:
                raise ValueError("noncausal layer order")
            for field in ("selection_delay_ns", "attention_compute_ns", "route_delay_ns", "expert_compute_ns"):
                integer(layer[field], field)
            experts = layer["experts"]
            if not isinstance(experts, list) or len(experts) != 8 or any(type(e) is not int or not 0 <= e < 256 for e in experts) or len(set(experts)) != 8:
                raise ValueError("native route needs 8 distinct expert IDs in [0,256)")
            heads = layer["kv_blocks_by_query_head"]
            full = li in GEOMETRY["full_attention_layers"]
            if trace["attention"] == "selected_hypothesis" and full:
                if not isinstance(heads, list) or len(heads) != 32:
                    raise ValueError("selected attention needs all 32 query heads")
                for blocks in heads:
                    if not isinstance(blocks, list) or not blocks or any(type(b) is not int or not 0 <= b < block_count for b in blocks) or len(blocks) != len(set(blocks)):
                        raise ValueError("malformed/duplicate/unknown KV block")
            elif heads != []:
                raise ValueError("KV selections only belong to selected full-attention layers")
    return trace


def kv_reads(layer, trace, page_bytes):
    """Union 16 query selections into cache blocks; PageService expands pages."""
    if layer["layer"] not in GEOMETRY["full_attention_layers"] or trace["attention"] == "none_control":
        return []
    block_tokens = GEOMETRY["selection_block_tokens"]
    block_bytes = block_tokens * GEOMETRY["kv_bytes_per_token_per_head"]
    if block_bytes % page_bytes:
        raise ValueError("storage page must divide the 128-token KV selection block")
    blocks = ceildiv(trace["context_tokens"], block_tokens)
    stride = ceildiv(1010000, block_tokens) * (block_bytes // page_bytes)
    # Stable disjoint address regions independent of request order/cache state.
    base = ceildiv(GEOMETRY["padded_expert_file_bytes"], page_bytes) * 48 * 256
    reads = []
    for head in range(2):
        selected = range(blocks) if trace["attention"] == "full" else sorted(set().union(*map(set, layer["kv_blocks_by_query_head"][head * 16:(head + 1) * 16])))
        for block in selected:
            valid_tokens = min(block_tokens, trace["context_tokens"] - block * block_tokens)
            size = valid_tokens * GEOMETRY["kv_bytes_per_token_per_head"]
            first = base + (layer["layer"] * 2 + head) * stride + block * (block_bytes // page_bytes)
            reads.append(Read(f"k:{layer['layer']}:{head}:{block}", "kv", size, first))
    return reads


def percentile(values, fraction):
    if not values:
        return 0
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)]


class Replay:
    """Single decode stream, immutable KV snapshot, stage-barrier execution.

    This schedule intentionally has no prefetch, route oracle, compute/storage
    overlap or KV writes. Its service time is not a hardware token/s prediction
    and not a universal latency lower bound. Resource busy-time bounds are
    independently reported as necessary conditions.
    """
    def __init__(self, config=DeviceConfig(), cache_bytes=8 * 2**30,
                 card_bytes=16 * 2**30, format="compact", audit=False):
        if format not in ("compact", "repo_padded"):
            raise ValueError("unknown expert format")
        self.c = config
        self.cache = Cache(cache_bytes)
        self.card_bytes = integer(card_bytes, "card_bytes", 1)
        self.format = format
        self.fast = FastBus(config.fast_bytes_per_ns)
        self.fast.audit = audit
        self.service = PageService(config, self.fast, audit)
        self.stages = []
        self.tokens = []
        self.now = 0
        self.expected_consumer = Counter()

    def stage(self, reads, now, token, layer, kind):
        if len({r.key for r in reads}) != len(reads):
            raise ValueError("duplicate demand object")
        if sum(r.size for r in reads) > self.cache.capacity:
            raise ValueError("pinned stage exceeds cache capacity; tiling is not modeled")
        # Pin all existing demanded entries first. Allocating a miss must never
        # evict another member of this very stage before it is pinned.
        existing = [r for r in reads if r.key in self.cache.entries]
        new = [r for r in reads if r.key not in self.cache.entries]
        misses = []
        hit_bytes = 0
        for r in existing + new:
            state = self.cache.acquire(r, now)
            if state == "miss":
                misses.append(r)
            elif state == "hit":
                hit_bytes += r.size
            else:
                raise AssertionError("sequential replay encountered pending prior stage")
        done = self.service.run(misses, now)
        for key, at in done.items():
            self.cache.mark_ready(key, at)
        ready = max([now] + list(done.values()))
        demand = sum(r.size for r in reads)
        # Repo files contain headers/alignment that need not be read by MACs.
        consume = len(reads) * GEOMETRY["padded_expert_consumed_bytes"] if kind == "expert" and self.format == "repo_padded" else demand
        finish = self.fast.transfer(ready, consume, "consume_" + kind)
        self.expected_consumer[kind] += consume
        for r in reads:
            self.cache.release(r.key, finish)
        self.stages.append(dict(token=token, layer=layer, kind=kind, request_ns=now,
                                ready_ns=ready, done_ns=finish, dependency_wait_ns=ready-now,
                                demand_bytes=demand, hit_bytes=hit_bytes, miss_bytes=demand-hit_bytes,
                                consume_bytes=consume, objects=len(reads)))
        return finish

    def run(self, trace, warmup_tokens=1):
        validate_trace(trace)
        integer(warmup_tokens, "warmup_tokens")
        if warmup_tokens >= len(trace["tokens"]):
            raise ValueError("measurement needs a token after warmup")
        if self.tokens:
            raise ValueError("Replay instances are single-use")
        g = GEOMETRY
        summary_bytes = (ceildiv(trace["context_tokens"], 128) * 2048 * 12
                         if trace["attention"] == "selected_hypothesis" else 0)
        permanent = g["resident_core_bytes"] + g["recurrent_state_bytes"] + g["conv_state_bytes"] + summary_bytes
        bridge = self.c.bridge_pages * self.c.page_bytes
        if permanent + self.cache.capacity + bridge > self.card_bytes:
            raise ValueError("permanent + cache + bridge exceeds assumed card capacity")
        for token in trace["tokens"]:
            ti = token["token"]
            start = max(self.now, token["not_before_ns"])
            before_service = self.service.stats.copy()
            before_fast = self.fast.bytes.copy()
            before_busy = self.fast.busy_ns
            now = self.fast.transfer(start, 1728, "input_embedding")
            for layer in token["layers"]:
                li = layer["layer"]
                full = li in g["full_attention_layers"]
                matrix = 78643200 if full else 88473600
                small = 6144 + (512 if full else 49408)
                now = self.fast.transfer(now, matrix * 9 // 16 + small * 2, "resident_attention")
                if not full:
                    now = self.fast.transfer(now, 2 * (4194304 + 98304), "recurrent_conv_rw")
                elif summary_bytes:
                    now = self.fast.transfer(now, summary_bytes // 12, "selection_summary")
                now += layer["selection_delay_ns"]
                now = self.stage(kv_reads(layer, trace, self.c.page_bytes), now, ti, li, "kv")
                now += layer["attention_compute_ns"]
                now = self.fast.transfer(now, (9437184 + 786432 + 3072) * 9 // 16, "resident_shared_router")
                now += layer["route_delay_ns"]
                size = g["compact_expert_bytes"] if self.format == "compact" else g["padded_expert_file_bytes"]
                expert_stride = ceildiv(g["padded_expert_file_bytes"], self.c.page_bytes)
                reads = [Read(f"e:{li}:{e}", "expert", size, (li * 256 + e) * expert_stride) for e in sorted(layer["experts"])]
                now = self.stage(reads, now, ti, li, "expert")
                now += layer["expert_compute_ns"]
            now = self.fast.transfer(now, 762839040 * 9 // 16 + 3072 * 2, "output_head")
            self.now = now
            stats = self.service.stats
            physical_e = stats["physical_expert_bytes"] - before_service["physical_expert_bytes"]
            physical_k = stats["physical_kv_bytes"] - before_service["physical_kv_bytes"]
            fast_bytes = sum(self.fast.bytes.values()) - sum(before_fast.values())
            # Necessary busy-time bounds, not the scheduled replay duration.
            channel_times = [(stats[f"channel_{ch}_data_ns"] - before_service[f"channel_{ch}_data_ns"],
                              stats[f"channel_{ch}_command_ns"] - before_service[f"channel_{ch}_command_ns"])
                             for ch in range(self.c.channels)]
            channel_bound = max((data + command if self.c.shared_command_data_bus else max(data, command))
                                for data, command in channel_times)
            array_bound = max(stats[f"die_{die}_array_ns"] - before_service[f"die_{die}_array_ns"]
                              for die in range(self.c.channels * self.c.dies_per_channel))
            fast_bound = self.fast.busy_ns - before_busy
            self.tokens.append(dict(token=ti, phase="warmup" if ti < warmup_tokens else "measurement",
                                    start_ns=start, done_ns=now, service_ns=now-start,
                                    physical_expert_bytes=physical_e, physical_kv_bytes=physical_k,
                                    fast_bytes=fast_bytes, channel_busy_bound_ns=channel_bound,
                                    die_array_busy_bound_ns=array_bound, fast_busy_bound_ns=fast_bound,
                                    necessary_resource_bound_ns=max(channel_bound, array_bound, fast_bound)))
        self.cache.check()
        self.service.check()
        assert self.fast.bytes["consume_expert"] == self.expected_consumer["expert"]
        assert self.fast.bytes["consume_kv"] == self.expected_consumer["kv"]
        assert self.fast.busy_ns <= self.now
        selected = [t for t in self.tokens if t["phase"] == "measurement"]
        stages = [s for s in self.stages if s["token"] >= warmup_tokens]
        n = len(selected)
        summary = dict(measured_tokens=n, warmup_tokens=warmup_tokens,
                       mean_service_ms=sum(t["service_ns"] for t in selected) / n / 1e6,
                       p50_service_ms=percentile([t["service_ns"] for t in selected], .5) / 1e6,
                       p95_service_ms=percentile([t["service_ns"] for t in selected], .95) / 1e6,
                       p99_service_ms=percentile([t["service_ns"] for t in selected], .99) / 1e6,
                       max_service_ms=max(t["service_ns"] for t in selected) / 1e6,
                       count_service_over_20ms=sum(t["service_ns"] > 20000000 for t in selected),
                       count_service_over_50ms=sum(t["service_ns"] > 50000000 for t in selected),
                       count_resource_bound_over_20ms=sum(t["necessary_resource_bound_ns"] > 20000000 for t in selected),
                       count_resource_bound_over_50ms=sum(t["necessary_resource_bound_ns"] > 50000000 for t in selected),
                       p50_dependency_wait_us=percentile([s["dependency_wait_ns"] for s in stages], .5) / 1e3,
                       p95_dependency_wait_us=percentile([s["dependency_wait_ns"] for s in stages], .95) / 1e3,
                       p99_dependency_wait_us=percentile([s["dependency_wait_ns"] for s in stages], .99) / 1e3,
                       mean_necessary_resource_ms=sum(t["necessary_resource_bound_ns"] for t in selected) / n / 1e6,
                       cache_peak_bytes=self.cache.peak, cache_capacity_bytes=self.cache.capacity,
                       permanent_bytes=permanent, bridge_allocated_bytes=bridge,
                       card_allocated_bytes=permanent+self.cache.capacity+bridge,
                       assumed_card_bytes=self.card_bytes, terminal_pending_pages=self.service.pending,
                       prefetch_bytes=0, prefetch_waste_bytes=0)
        for kind in ("expert", "kv"):
            demand = sum(s["demand_bytes"] for s in stages if s["kind"] == kind)
            hit = sum(s["hit_bytes"] for s in stages if s["kind"] == kind)
            summary[kind + "_byte_hit_fraction"] = hit / demand if demand else None
            summary[kind + "_demand_bytes_per_token"] = demand / n
            summary[kind + "_miss_file_bytes_per_token"] = (demand - hit) / n
            summary[kind + "_physical_bytes_per_token"] = sum(t[f"physical_{kind}_bytes"] for t in selected) / n
        for kind in ("expert", "kv"):
            missed = summary[kind + "_miss_file_bytes_per_token"]
            summary[kind + "_physical_to_miss_file_ratio"] = summary[kind + "_physical_bytes_per_token"] / missed if missed else None
        summary["fast_bytes_per_token"] = sum(t["fast_bytes"] for t in selected) / n
        return dict(summary=summary, tokens=self.tokens, stages=self.stages,
                    device=asdict(self.c), format=self.format,
                    storage_counters=dict(self.service.stats), fast_byte_counters=dict(self.fast.bytes),
                    cache_counters=dict(self.cache.stats), geometry=GEOMETRY)
