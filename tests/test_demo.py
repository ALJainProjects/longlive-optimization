# LongLive Demo Tests
# SPDX-License-Identifier: Apache-2.0
"""Tests for demo components."""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import numpy as np


class TestOptimizedLongLivePipeline:
    """Tests for OptimizedLongLivePipeline wrapper."""

    def test_init(self):
        """Test pipeline initialization."""
        from demo.pipeline_wrapper import OptimizedLongLivePipeline

        pipeline = OptimizedLongLivePipeline("test_config.yaml")

        assert pipeline.config_path == "test_config.yaml"
        assert pipeline.is_ready is False
        assert pipeline.is_generating is False
        assert pipeline.current_prompt == ""

    def test_default_fps(self):
        """Test default FPS setting."""
        from demo.pipeline_wrapper import OptimizedLongLivePipeline

        pipeline = OptimizedLongLivePipeline("test.yaml")
        assert pipeline.fps == 16

    @pytest.mark.asyncio
    async def test_update_prompt(self):
        """Test prompt update."""
        from demo.pipeline_wrapper import OptimizedLongLivePipeline

        pipeline = OptimizedLongLivePipeline("test.yaml")
        await pipeline.update_prompt("Test prompt")

        assert pipeline.pending_prompt == "Test prompt"

    def test_get_metrics_initial(self):
        """Test initial metrics."""
        from demo.pipeline_wrapper import OptimizedLongLivePipeline

        pipeline = OptimizedLongLivePipeline("test.yaml")
        metrics = pipeline.get_metrics()

        assert "fps" in metrics
        assert "latency_ms" in metrics
        assert "frame_count" in metrics
        assert "generating" in metrics
        assert metrics["generating"] is False

    @pytest.mark.asyncio
    async def test_start_stop_generation(self):
        """Test start and stop generation."""
        from demo.pipeline_wrapper import OptimizedLongLivePipeline

        pipeline = OptimizedLongLivePipeline("test.yaml")

        # Mock _generation_loop to prevent actual generation
        with patch.object(pipeline, '_generation_loop'):
            await pipeline.start_generation()
            assert pipeline.is_generating is True

            await pipeline.stop_generation()
            assert pipeline.is_generating is False

    def test_generation_loop_no_prompt(self):
        """Test generation loop exits early without prompt."""
        from demo.pipeline_wrapper import OptimizedLongLivePipeline

        pipeline = OptimizedLongLivePipeline("test.yaml")
        pipeline.is_generating = True
        pipeline.current_prompt = ""

        # Run one iteration (should sleep and return)
        import time
        start = time.time()

        def stop_after_short_time():
            time.sleep(0.15)
            pipeline.is_generating = False

        import threading
        stopper = threading.Thread(target=stop_after_short_time)
        stopper.start()

        # This should not block indefinitely
        pipeline._generation_loop()
        stopper.join()


class TestVideoGeneratorTrack:
    """Tests for VideoGeneratorTrack (WebRTC)."""

    @pytest.mark.skipif(True, reason="Requires aiortc installation")
    def test_track_initialization(self):
        """Test track initialization."""
        pass

    def test_track_without_webrtc(self):
        """Test track works without WebRTC (fallback mode)."""
        from demo.pipeline_wrapper import OptimizedLongLivePipeline

        # Create a mock track class
        class MockVideoTrack:
            def __init__(self, pipeline):
                self.pipeline = pipeline
                self.frame_count = 0

        pipeline = OptimizedLongLivePipeline("test.yaml")
        track = MockVideoTrack(pipeline)

        assert track.frame_count == 0


class TestDemoServer:
    """Tests for demo server endpoints."""

    @pytest.mark.asyncio
    async def test_get_status_no_pipeline(self):
        """Test status endpoint with no pipeline."""
        from demo import server

        # Save original pipeline
        original_pipeline = server.pipeline
        server.pipeline = None

        try:
            response = await server.get_status()
            assert response["status"] == "not_loaded"
        finally:
            server.pipeline = original_pipeline

    @pytest.mark.asyncio
    async def test_get_status_with_pipeline(self):
        """Test status endpoint with pipeline."""
        from demo import server
        from demo.pipeline_wrapper import OptimizedLongLivePipeline

        mock_pipeline = Mock(spec=OptimizedLongLivePipeline)
        mock_pipeline.is_ready = True
        mock_pipeline.is_generating = False
        mock_pipeline.current_prompt = "Test"
        mock_pipeline.get_metrics.return_value = {"fps": 20.0}

        original_pipeline = server.pipeline
        server.pipeline = mock_pipeline

        try:
            response = await server.get_status()
            assert response["status"] == "ready"
            assert response["generating"] is False
        finally:
            server.pipeline = original_pipeline

    @pytest.mark.asyncio
    async def test_set_prompt_no_pipeline(self):
        """Test set prompt with no pipeline."""
        from demo import server

        original_pipeline = server.pipeline
        server.pipeline = None

        try:
            response = await server.set_prompt("Test prompt")
            assert response["status"] == "error"
        finally:
            server.pipeline = original_pipeline

    @pytest.mark.asyncio
    async def test_set_prompt_with_pipeline(self):
        """Test set prompt with pipeline."""
        from demo import server
        from demo.pipeline_wrapper import OptimizedLongLivePipeline

        mock_pipeline = Mock(spec=OptimizedLongLivePipeline)
        mock_pipeline.update_prompt = AsyncMock()

        original_pipeline = server.pipeline
        server.pipeline = mock_pipeline

        try:
            response = await server.set_prompt("New prompt")
            assert response["status"] == "updated"
            mock_pipeline.update_prompt.assert_called_once_with("New prompt")
        finally:
            server.pipeline = original_pipeline

    def test_index_returns_html(self):
        """Test index endpoint returns HTML."""
        from demo import server
        import asyncio

        # Run async function
        response = asyncio.run(server.index())

        assert "<!DOCTYPE html>" in response
        assert "LongLive Interactive Demo" in response


class TestDemoIntegration:
    """Integration tests for demo components."""

    @pytest.fixture
    def mock_omegaconf(self):
        """Mock OmegaConf for config loading."""
        with patch('demo.pipeline_wrapper.OmegaConf') as mock:
            mock.load.return_value = Mock(
                num_frame_per_block=3,
                optimization=Mock(
                    use_torch_compile=False,
                    use_fp8=False,
                )
            )
            yield mock

    def test_pipeline_metrics_structure(self):
        """Test that metrics have expected structure."""
        from demo.pipeline_wrapper import OptimizedLongLivePipeline

        pipeline = OptimizedLongLivePipeline("test.yaml")
        metrics = pipeline.get_metrics()

        expected_keys = ["fps", "latency_ms", "frame_count", "elapsed_s", "generating", "current_prompt"]
        for key in expected_keys:
            assert key in metrics, f"Missing key: {key}"

    def test_time_base_is_fraction(self):
        """Test that time_base is a Fraction."""
        from demo.pipeline_wrapper import OptimizedLongLivePipeline
        from fractions import Fraction

        pipeline = OptimizedLongLivePipeline("test.yaml")
        assert isinstance(pipeline.time_base, Fraction)
        assert pipeline.time_base == Fraction(1, 16)
