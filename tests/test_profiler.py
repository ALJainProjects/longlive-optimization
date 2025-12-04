# LongLive Profiler Tests
# SPDX-License-Identifier: Apache-2.0
"""Tests for the profiling infrastructure."""

import pytest
import torch
import time
import json
import tempfile
import os

from utils.profiler import (
    LongLiveProfiler,
    TimingEvent,
    MemorySnapshot,
    NullContext,
    get_profiler,
    set_profiler,
    profile_region,
)


class TestTimingEvent:
    """Tests for TimingEvent dataclass."""

    def test_cpu_duration(self):
        """Test CPU duration calculation."""
        event = TimingEvent(
            name="test",
            start_time=0.0,
            end_time=0.1,
        )
        assert abs(event.cpu_duration_ms - 100.0) < 0.01

    def test_cuda_duration_fallback(self):
        """Test fallback to CPU duration when CUDA events unavailable."""
        event = TimingEvent(
            name="test",
            start_time=0.0,
            end_time=0.05,
            cuda_start=None,
            cuda_end=None,
        )
        assert abs(event.cuda_duration_ms() - 50.0) < 0.01


class TestMemorySnapshot:
    """Tests for MemorySnapshot dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        snapshot = MemorySnapshot(
            timestamp=1.0,
            allocated_gb=4.0,
            reserved_gb=6.0,
            free_gb=2.0,
            peak_allocated_gb=5.0,
            label="test"
        )
        d = snapshot.to_dict()
        assert d["timestamp"] == 1.0
        assert d["allocated_gb"] == 4.0
        assert d["label"] == "test"


class TestNullContext:
    """Tests for NullContext."""

    def test_null_context_does_nothing(self):
        """Test that NullContext is a no-op."""
        ctx = NullContext()
        with ctx:
            x = 1 + 1
        assert x == 2


class TestLongLiveProfiler:
    """Tests for the main LongLiveProfiler class."""

    def test_init_defaults(self):
        """Test default initialization."""
        profiler = LongLiveProfiler()
        assert profiler.enabled is True
        assert profiler.sync_cuda is True
        assert len(profiler.events) == 0

    def test_init_disabled(self):
        """Test initialization with disabled profiling."""
        profiler = LongLiveProfiler(enabled=False)
        assert profiler.enabled is False

    def test_reset(self):
        """Test reset clears all data."""
        profiler = LongLiveProfiler()

        # Add some data
        with profiler.profile_region("test"):
            time.sleep(0.01)

        profiler.reset()

        assert len(profiler.events) == 0
        assert len(profiler.memory_snapshots) == 0
        assert len(profiler.hierarchy) == 0

    def test_profile_region_basic(self):
        """Test basic region profiling."""
        profiler = LongLiveProfiler(enabled=True, sync_cuda=False)

        with profiler.profile_region("test_region"):
            time.sleep(0.01)

        assert "test_region" in profiler.events
        assert len(profiler.events["test_region"]) == 1

    def test_profile_region_disabled(self):
        """Test that disabled profiler doesn't record."""
        profiler = LongLiveProfiler(enabled=False)

        with profiler.profile_region("test_region"):
            time.sleep(0.01)

        assert len(profiler.events) == 0

    def test_profile_region_with_parent(self):
        """Test hierarchical profiling."""
        profiler = LongLiveProfiler(enabled=True, sync_cuda=False)

        with profiler.profile_region("parent"):
            with profiler.profile_region("child", parent="parent"):
                time.sleep(0.01)

        assert "parent" in profiler.hierarchy
        assert "child" in profiler.hierarchy["parent"]

    def test_profile_region_with_metadata(self):
        """Test profiling with metadata."""
        profiler = LongLiveProfiler(enabled=True, sync_cuda=False)

        with profiler.profile_region("test", metadata={"timestep": 1000}):
            time.sleep(0.01)

        event = profiler.events["test"][0]
        assert event.metadata["timestep"] == 1000

    def test_multiple_invocations(self):
        """Test multiple invocations of same region."""
        profiler = LongLiveProfiler(enabled=True, sync_cuda=False)

        for _ in range(5):
            with profiler.profile_region("repeated"):
                time.sleep(0.001)

        assert len(profiler.events["repeated"]) == 5

    def test_record_sync_point(self):
        """Test sync point recording."""
        profiler = LongLiveProfiler(enabled=True, sync_cuda=False)

        if torch.cuda.is_available():
            profiler.record_sync_point("test_sync")
            assert "sync:test_sync" in profiler.events

    def test_snapshot_memory_disabled(self):
        """Test memory snapshot when disabled."""
        profiler = LongLiveProfiler(enabled=False)
        profiler.snapshot_memory("test")
        assert len(profiler.memory_snapshots) == 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_snapshot_memory_cuda(self):
        """Test memory snapshot with CUDA."""
        profiler = LongLiveProfiler(enabled=True)
        profiler.snapshot_memory("test_snapshot")
        assert len(profiler.memory_snapshots) == 1
        assert profiler.memory_snapshots[0].label == "test_snapshot"

    def test_get_statistics_empty(self):
        """Test statistics for non-existent event."""
        profiler = LongLiveProfiler()
        stats = profiler.get_statistics("nonexistent")
        assert stats == {}

    def test_get_statistics(self):
        """Test statistics calculation."""
        profiler = LongLiveProfiler(enabled=True, sync_cuda=False)

        for _ in range(10):
            with profiler.profile_region("stats_test"):
                time.sleep(0.005)

        stats = profiler.get_statistics("stats_test")

        assert stats["count"] == 10
        assert "mean_ms" in stats
        assert "std_ms" in stats
        assert "p50_ms" in stats
        assert "p95_ms" in stats
        assert "p99_ms" in stats
        assert "min_ms" in stats
        assert "max_ms" in stats

    def test_get_all_statistics(self):
        """Test getting statistics for all events."""
        profiler = LongLiveProfiler(enabled=True, sync_cuda=False)

        with profiler.profile_region("event_a"):
            time.sleep(0.001)
        with profiler.profile_region("event_b"):
            time.sleep(0.001)

        all_stats = profiler.get_all_statistics()

        assert "event_a" in all_stats
        assert "event_b" in all_stats

    def test_get_event_durations(self):
        """Test getting raw durations."""
        profiler = LongLiveProfiler(enabled=True, sync_cuda=False)

        for _ in range(3):
            with profiler.profile_region("duration_test"):
                time.sleep(0.01)

        durations = profiler.get_event_durations("duration_test")
        assert len(durations) == 3
        for d in durations:
            assert d >= 10.0  # At least 10ms

    def test_compute_steady_state_latency_no_blocks(self):
        """Test steady state computation with no block events."""
        profiler = LongLiveProfiler()
        latency = profiler.compute_steady_state_inter_frame_latency()
        assert latency == 0.0

    def test_compute_steady_state_latency(self):
        """Test steady state inter-frame latency computation."""
        profiler = LongLiveProfiler(enabled=True, sync_cuda=False)

        # Simulate block events
        for i in range(5):
            with profiler.profile_region(f"block_{i}"):
                time.sleep(0.03)  # 30ms per block

        latency = profiler.compute_steady_state_inter_frame_latency(
            num_frame_per_block=3,
            warmup_blocks=2
        )

        # Should be roughly 30ms / 3 = 10ms per frame
        assert latency > 0

    def test_compute_cross_batch_latency(self):
        """Test cross-batch latency computation."""
        profiler = LongLiveProfiler(enabled=True, sync_cuda=False)

        with profiler.profile_region("initialization"):
            time.sleep(0.01)
        with profiler.profile_region("text_encoding"):
            time.sleep(0.01)
        with profiler.profile_region("block_0"):
            time.sleep(0.01)

        latency = profiler.compute_cross_batch_latency()
        assert latency > 0

    def test_compute_prompt_switch_latency(self):
        """Test prompt switch latency computation."""
        profiler = LongLiveProfiler(enabled=True, sync_cuda=False)

        with profiler.profile_region("kv_recache"):
            time.sleep(0.02)
        with profiler.profile_region("crossattn_cache_reset"):
            time.sleep(0.01)

        latency = profiler.compute_prompt_switch_latency()
        assert latency > 0

    def test_generate_report_text(self):
        """Test text report generation."""
        profiler = LongLiveProfiler(enabled=True, sync_cuda=False)

        with profiler.profile_region("inference"):
            with profiler.profile_region("diffusion", parent="inference"):
                time.sleep(0.01)

        report = profiler.generate_report(format="text")

        assert "LONGLIVE LATENCY PROFILING REPORT" in report
        assert "inference" in report.lower() or "diffusion" in report.lower()

    def test_generate_report_json(self):
        """Test JSON report generation."""
        profiler = LongLiveProfiler(enabled=True, sync_cuda=False)

        with profiler.profile_region("test_event"):
            time.sleep(0.01)

        report = profiler.generate_report(format="json")

        data = json.loads(report)
        assert "statistics" in data
        assert "hierarchy" in data

    def test_export_json(self):
        """Test JSON export to file."""
        profiler = LongLiveProfiler(enabled=True, sync_cuda=False)

        with profiler.profile_region("export_test"):
            time.sleep(0.01)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            path = f.name

        try:
            profiler.export_json(path)
            assert os.path.exists(path)

            with open(path) as f:
                data = json.load(f)

            assert "statistics" in data
            assert "computed_metrics" in data
        finally:
            if os.path.exists(path):
                os.remove(path)


class TestGlobalProfiler:
    """Tests for global profiler functions."""

    def test_get_set_profiler(self):
        """Test global profiler get/set."""
        profiler = LongLiveProfiler()
        set_profiler(profiler)

        retrieved = get_profiler()
        assert retrieved is profiler

    def test_profile_region_global(self):
        """Test global profile_region function."""
        profiler = LongLiveProfiler(enabled=True, sync_cuda=False)
        set_profiler(profiler)

        with profile_region("global_test"):
            time.sleep(0.01)

        assert "global_test" in profiler.events

    def test_profile_region_no_profiler(self):
        """Test profile_region when no global profiler set."""
        set_profiler(None)

        # Should return NullContext and not raise
        with profile_region("no_profiler_test"):
            pass
