# LongLive Latency Profiling Infrastructure
# SPDX-License-Identifier: Apache-2.0
"""
Fine-grained profiling for LongLive video generation pipeline.

Captures:
- Per-timestep diffusion timing
- Per-layer attention kernel timing
- KV cache operations (update, rolling, recache)
- VAE encode/decode separately
- CPU-GPU synchronization points
- Memory usage over time
"""

import torch
import time
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Callable
from contextlib import contextmanager
from collections import defaultdict
import numpy as np


@dataclass
class TimingEvent:
    """Represents a single timing measurement."""
    name: str
    start_time: float = 0.0
    end_time: float = 0.0
    cuda_start: Optional[torch.cuda.Event] = None
    cuda_end: Optional[torch.cuda.Event] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def cpu_duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000

    def cuda_duration_ms(self) -> float:
        if self.cuda_start is not None and self.cuda_end is not None:
            try:
                return self.cuda_start.elapsed_time(self.cuda_end)
            except RuntimeError:
                return self.cpu_duration_ms
        return self.cpu_duration_ms


@dataclass
class MemorySnapshot:
    """Captures GPU memory state at a point in time."""
    timestamp: float
    allocated_gb: float
    reserved_gb: float
    free_gb: float
    peak_allocated_gb: float
    label: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class NullContext:
    """A no-op context manager for when profiling is disabled."""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass


class LongLiveProfiler:
    """
    Comprehensive profiler for LongLive video generation.

    Usage:
        profiler = LongLiveProfiler(enabled=True)

        with profiler.profile_region("inference"):
            with profiler.profile_region("diffusion", parent="inference"):
                # diffusion code
                pass

        stats = profiler.get_all_statistics()
        print(profiler.generate_report())
    """

    def __init__(self, enabled: bool = True, sync_cuda: bool = True):
        """
        Args:
            enabled: Whether profiling is active
            sync_cuda: Whether to synchronize CUDA for accurate timing
        """
        self.enabled = enabled
        self.sync_cuda = sync_cuda
        self.events: Dict[str, List[TimingEvent]] = defaultdict(list)
        self.memory_snapshots: List[MemorySnapshot] = []
        self.hierarchy: Dict[str, List[str]] = {}  # parent -> children
        self._active_events: Dict[str, TimingEvent] = {}
        self._start_time: Optional[float] = None

    def reset(self):
        """Clear all recorded events and snapshots."""
        self.events.clear()
        self.memory_snapshots.clear()
        self.hierarchy.clear()
        self._active_events.clear()
        self._start_time = None
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    @contextmanager
    def profile_region(self, name: str, parent: str = None, metadata: Dict = None):
        """
        Context manager for profiling a code region.

        Usage:
            with profiler.profile_region("diffusion_step", metadata={"timestep": 1000}):
                # code to profile
        """
        if not self.enabled:
            yield NullContext()
            return

        if self._start_time is None:
            self._start_time = time.perf_counter()

        cuda_available = torch.cuda.is_available()
        event = TimingEvent(
            name=name,
            start_time=time.perf_counter(),
            cuda_start=torch.cuda.Event(enable_timing=True) if cuda_available else None,
            cuda_end=torch.cuda.Event(enable_timing=True) if cuda_available else None,
            metadata=metadata or {}
        )

        if event.cuda_start is not None:
            event.cuda_start.record()

        self._active_events[name] = event

        if parent:
            if parent not in self.hierarchy:
                self.hierarchy[parent] = []
            if name not in self.hierarchy[parent]:
                self.hierarchy[parent].append(name)

        try:
            yield event
        finally:
            if event.cuda_end is not None:
                event.cuda_end.record()
                if self.sync_cuda:
                    torch.cuda.synchronize()

            event.end_time = time.perf_counter()
            self.events[name].append(event)
            if name in self._active_events:
                del self._active_events[name]

    def record_sync_point(self, name: str):
        """Record a CPU-GPU synchronization point."""
        if not self.enabled:
            return

        if torch.cuda.is_available():
            sync_start = time.perf_counter()
            torch.cuda.synchronize()
            sync_end = time.perf_counter()

            event = TimingEvent(
                name=f"sync:{name}",
                start_time=sync_start,
                end_time=sync_end,
                metadata={"type": "sync_point"}
            )
            self.events[f"sync:{name}"].append(event)

    def snapshot_memory(self, label: str = ""):
        """Capture current GPU memory state."""
        if not self.enabled or not torch.cuda.is_available():
            return

        try:
            stats = torch.cuda.memory_stats()
            snapshot = MemorySnapshot(
                timestamp=time.perf_counter() - (self._start_time or time.perf_counter()),
                allocated_gb=stats['allocated_bytes.all.current'] / (1024**3),
                reserved_gb=stats['reserved_bytes.all.current'] / (1024**3),
                free_gb=torch.cuda.mem_get_info()[0] / (1024**3),
                peak_allocated_gb=stats['allocated_bytes.all.peak'] / (1024**3),
                label=label
            )
            self.memory_snapshots.append(snapshot)
        except Exception as e:
            print(f"Warning: Could not capture memory snapshot: {e}")

    def get_statistics(self, event_name: str) -> Dict[str, float]:
        """
        Compute statistics for a named event.

        Returns:
            Dict with count, mean, std, p50, p95, p99, min, max in milliseconds
        """
        events = self.events.get(event_name, [])
        if not events:
            return {}

        durations = [e.cuda_duration_ms() for e in events]
        arr = np.array(durations)

        return {
            "count": len(arr),
            "mean_ms": float(np.mean(arr)),
            "std_ms": float(np.std(arr)),
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "p99_ms": float(np.percentile(arr, 99)),
            "min_ms": float(np.min(arr)),
            "max_ms": float(np.max(arr)),
            "total_ms": float(np.sum(arr)),
        }

    def get_all_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all recorded events."""
        return {name: self.get_statistics(name) for name in sorted(self.events.keys())}

    def get_event_durations(self, event_name: str) -> List[float]:
        """Get raw durations for an event."""
        events = self.events.get(event_name, [])
        return [e.cuda_duration_ms() for e in events]

    def compute_steady_state_inter_frame_latency(self, num_frame_per_block: int = 3, warmup_blocks: int = 2) -> float:
        """
        Compute steady-state inter-frame latency within the same batch.

        This is the time to generate one frame when:
        - KV cache is warm (no initial allocation)
        - No prompt switching
        - Processing middle blocks (not first or last)

        Returns:
            Latency in milliseconds per frame
        """
        block_events = []
        for name in self.events.keys():
            if name.startswith("block_"):
                block_events.extend(self.events[name])

        if len(block_events) <= warmup_blocks:
            return 0.0

        # Skip warmup blocks, use remaining
        steady_state_blocks = block_events[warmup_blocks:]
        block_times = [e.cuda_duration_ms() for e in steady_state_blocks]

        if not block_times:
            return 0.0

        avg_block_time = np.mean(block_times)
        return float(avg_block_time / num_frame_per_block)

    def compute_cross_batch_latency(self) -> float:
        """
        Compute worst-case inter-frame latency when starting a new batch.

        This includes:
        - KV cache initialization
        - Text encoding
        - First block processing (cold cache)

        Returns:
            Latency in milliseconds
        """
        total = 0.0

        init_stats = self.get_statistics("initialization")
        if init_stats:
            total += init_stats.get("mean_ms", 0)

        text_stats = self.get_statistics("text_encoding")
        if text_stats:
            total += text_stats.get("mean_ms", 0)

        block0_stats = self.get_statistics("block_0")
        if block0_stats:
            total += block0_stats.get("mean_ms", 0)

        return total

    def compute_prompt_switch_latency(self) -> float:
        """
        Compute latency when switching prompts mid-generation.

        Returns:
            Latency in milliseconds
        """
        total = 0.0

        recache_stats = self.get_statistics("kv_recache")
        if recache_stats:
            total += recache_stats.get("mean_ms", 0)

        crossattn_stats = self.get_statistics("crossattn_cache_reset")
        if crossattn_stats:
            total += crossattn_stats.get("mean_ms", 0)

        return total

    def generate_report(self, format: str = "text") -> str:
        """
        Generate a time budget breakdown report.

        Args:
            format: "text" for console output, "json" for structured data

        Returns:
            Report string
        """
        if format == "json":
            return json.dumps({
                "statistics": self.get_all_statistics(),
                "memory_snapshots": [s.to_dict() for s in self.memory_snapshots],
                "hierarchy": self.hierarchy,
            }, indent=2)

        return self._generate_text_report()

    def _generate_text_report(self) -> str:
        """Generate ASCII time budget visualization."""
        lines = []
        lines.append("=" * 80)
        lines.append("LONGLIVE LATENCY PROFILING REPORT")
        lines.append("=" * 80)

        stats = self.get_all_statistics()

        # Get total inference time
        total_ms = 0
        if "inference" in stats:
            total_ms = stats["inference"].get("total_ms", 0)
        else:
            # Sum up major phases
            for key in ["initialization", "diffusion", "vae_decode"]:
                if key in stats:
                    total_ms += stats[key].get("total_ms", 0)

        if total_ms == 0:
            total_ms = sum(s.get("total_ms", 0) for s in stats.values())

        lines.append(f"\nTotal Inference Time: {total_ms:.2f} ms")
        lines.append("-" * 80)

        # Major phases
        phases = [
            ("initialization", "Initialization"),
            ("diffusion", "Diffusion Loop"),
            ("vae_decode", "VAE Decode"),
        ]

        for phase_key, phase_name in phases:
            if phase_key in stats:
                phase_ms = stats[phase_key].get("mean_ms", 0) * stats[phase_key].get("count", 1)
                pct = (phase_ms / total_ms * 100) if total_ms > 0 else 0
                bar_len = int(pct / 2)
                bar = "#" * bar_len + "." * (50 - bar_len)
                lines.append(f"\n{phase_name}")
                lines.append(f"  [{bar}] {phase_ms:.2f} ms ({pct:.1f}%)")

        # Per-timestep breakdown
        lines.append("\n" + "-" * 80)
        lines.append("TIMESTEP BREAKDOWN")
        lines.append("-" * 80)

        for t in [1000, 750, 500, 250]:
            key = f"timestep_{t}"
            if key in stats:
                ts_stats = stats[key]
                lines.append(f"  Timestep {t}: {ts_stats.get('mean_ms', 0):.2f} ms "
                           f"(p95: {ts_stats.get('p95_ms', 0):.2f} ms)")

        # Block breakdown
        lines.append("\n" + "-" * 80)
        lines.append("BLOCK BREAKDOWN")
        lines.append("-" * 80)

        block_stats = [(k, v) for k, v in stats.items() if k.startswith("block_")]
        block_stats.sort(key=lambda x: int(x[0].split("_")[1]) if x[0].split("_")[1].isdigit() else 0)

        for block_name, block_stat in block_stats[:10]:  # Show first 10 blocks
            lines.append(f"  {block_name}: {block_stat.get('mean_ms', 0):.2f} ms")

        if len(block_stats) > 10:
            lines.append(f"  ... and {len(block_stats) - 10} more blocks")

        # KV Cache operations
        lines.append("\n" + "-" * 80)
        lines.append("KV CACHE OPERATIONS")
        lines.append("-" * 80)

        kv_ops = [
            ("kv_cache_init", "Initial Allocation"),
            ("kv_cache_update", "Updates"),
            ("kv_cache_rolling", "Rolling Window"),
            ("kv_recache", "Prompt Switch Recache"),
        ]

        for op_key, op_name in kv_ops:
            if op_key in stats:
                op_stats = stats[op_key]
                lines.append(f"  {op_name:25s}: {op_stats.get('mean_ms', 0):.3f} ms "
                           f"(count: {op_stats.get('count', 0)})")

        # Memory profile
        if self.memory_snapshots:
            lines.append("\n" + "-" * 80)
            lines.append("MEMORY PROFILE")
            lines.append("-" * 80)

            for snap in self.memory_snapshots:
                lines.append(f"  {snap.label:25s}: "
                           f"Alloc: {snap.allocated_gb:.2f} GB, "
                           f"Peak: {snap.peak_allocated_gb:.2f} GB")

        # Sync overhead
        sync_total = 0
        sync_events = [(k, v) for k, v in stats.items() if k.startswith("sync:")]
        if sync_events:
            lines.append("\n" + "-" * 80)
            lines.append("SYNCHRONIZATION OVERHEAD")
            lines.append("-" * 80)

            for key, s in sync_events:
                sync_ms = s.get("total_ms", 0)
                sync_total += sync_ms
                lines.append(f"  {key[5:]:25s}: {sync_ms:.3f} ms")

            lines.append(f"  {'Total sync overhead':25s}: {sync_total:.2f} ms")

        # Computed metrics
        lines.append("\n" + "-" * 80)
        lines.append("COMPUTED LATENCY METRICS")
        lines.append("-" * 80)

        steady_state = self.compute_steady_state_inter_frame_latency()
        cross_batch = self.compute_cross_batch_latency()
        prompt_switch = self.compute_prompt_switch_latency()

        lines.append(f"  Steady-state inter-frame: {steady_state:.2f} ms")
        lines.append(f"  Cross-batch latency:      {cross_batch:.2f} ms")
        lines.append(f"  Prompt-switch latency:    {prompt_switch:.2f} ms")

        lines.append("\n" + "=" * 80)

        return "\n".join(lines)

    def export_json(self, path: str):
        """Export full profiling data to JSON file."""
        data = {
            "statistics": self.get_all_statistics(),
            "memory_snapshots": [s.to_dict() for s in self.memory_snapshots],
            "hierarchy": self.hierarchy,
            "computed_metrics": {
                "steady_state_inter_frame_ms": self.compute_steady_state_inter_frame_latency(),
                "cross_batch_latency_ms": self.compute_cross_batch_latency(),
                "prompt_switch_latency_ms": self.compute_prompt_switch_latency(),
            }
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)


# Global profiler instance for convenience
_global_profiler: Optional[LongLiveProfiler] = None

def get_profiler() -> Optional[LongLiveProfiler]:
    """Get the global profiler instance."""
    return _global_profiler

def set_profiler(profiler: LongLiveProfiler):
    """Set the global profiler instance."""
    global _global_profiler
    _global_profiler = profiler

def profile_region(name: str, parent: str = None, metadata: Dict = None):
    """Convenience wrapper for global profiler."""
    if _global_profiler is not None:
        return _global_profiler.profile_region(name, parent, metadata)
    return NullContext()
