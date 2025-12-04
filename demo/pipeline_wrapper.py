# LongLive Optimized Pipeline Wrapper
# SPDX-License-Identifier: Apache-2.0
"""
Wrapper for LongLive pipeline with streaming frame generation support.

Features:
- Async frame generation
- Prompt queue with atomic updates
- Pre-warming for reduced first-frame latency
- Metrics tracking
"""

import asyncio
import time
from typing import Optional, Dict, Any, AsyncGenerator
from queue import Queue
from threading import Thread, Lock
from fractions import Fraction

import torch
import numpy as np
from omegaconf import OmegaConf


class OptimizedLongLivePipeline:
    """
    Wrapper around LongLive pipeline for interactive streaming.
    """

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = None
        self.pipeline = None
        self.device = None

        # State
        self.is_ready = False
        self.is_generating = False
        self.current_prompt = ""
        self.pending_prompt = None

        # Threading
        self._lock = Lock()
        self._frame_queue: Queue = Queue(maxsize=10)
        self._generation_thread: Optional[Thread] = None
        self._stop_event = asyncio.Event()

        # Metrics
        self._frame_count = 0
        self._generation_start_time = 0
        self._last_frame_time = 0
        self._latency_samples = []

        # Video timing
        self.fps = 16
        self.time_base = Fraction(1, self.fps)

    async def initialize(self):
        """Initialize the pipeline with warmup."""
        self.config = OmegaConf.load(self.config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Run initialization in background to not block
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._init_pipeline)

        self.is_ready = True

    def _init_pipeline(self):
        """Initialize pipeline (runs in background thread)."""
        from pipeline.causal_inference import CausalInferencePipeline
        from utils.quantization import apply_torch_compile, apply_fp8_quantization

        print("Loading LongLive pipeline...")

        self.pipeline = CausalInferencePipeline(self.config, device=self.device)

        # Apply optimizations
        opt_config = getattr(self.config, 'optimization', OmegaConf.create({}))

        if getattr(opt_config, 'use_torch_compile', False):
            print("Applying torch.compile...")
            self.pipeline.generator.model = apply_torch_compile(
                self.pipeline.generator.model,
                mode=getattr(opt_config, 'compile_mode', 'reduce-overhead')
            )

        if getattr(opt_config, 'use_fp8', False):
            print("Applying FP8 quantization...")
            self.pipeline.generator.model = apply_fp8_quantization(
                self.pipeline.generator.model
            )

        # Move to device
        self.pipeline = self.pipeline.to(dtype=torch.bfloat16)
        self.pipeline.generator.to(device=self.device)
        self.pipeline.vae.to(device=self.device)

        # Warmup
        print("Running warmup inference...")
        self._run_warmup()

        print("Pipeline ready!")

    def _run_warmup(self, num_warmup_frames: int = 6):
        """Run warmup inference to trigger JIT compilation."""
        torch.manual_seed(0)
        noise = torch.randn(
            [1, num_warmup_frames, 16, 60, 104],
            device=self.device,
            dtype=torch.bfloat16
        )

        with torch.no_grad():
            _ = self.pipeline.inference(
                noise=noise,
                text_prompts=["Warmup prompt"],
                return_latents=False,
                profile=False
            )

        torch.cuda.synchronize()
        print(f"Warmup complete ({num_warmup_frames} frames)")

    async def update_prompt(self, prompt: str):
        """Update the generation prompt (thread-safe)."""
        with self._lock:
            self.pending_prompt = prompt
            print(f"Prompt queued: {prompt[:50]}...")

    async def start_generation(self):
        """Start continuous frame generation."""
        if self.is_generating:
            return

        self.is_generating = True
        self._stop_event.clear()
        self._generation_start_time = time.time()
        self._frame_count = 0

        # Start generation in background thread
        self._generation_thread = Thread(target=self._generation_loop, daemon=True)
        self._generation_thread.start()

        print("Generation started")

    async def stop_generation(self):
        """Stop frame generation."""
        self.is_generating = False
        self._stop_event.set()

        if self._generation_thread:
            self._generation_thread.join(timeout=5.0)
            self._generation_thread = None

        print("Generation stopped")

    def _generation_loop(self):
        """Background thread for continuous generation."""
        torch.manual_seed(int(time.time()))

        # Initialize noise for continuous generation
        batch_size = 1
        num_frames = self.config.num_frame_per_block  # Generate block-by-block

        while self.is_generating:
            # Check for prompt update
            with self._lock:
                if self.pending_prompt is not None:
                    self.current_prompt = self.pending_prompt
                    self.pending_prompt = None
                    # TODO: Trigger KV recache here

            if not self.current_prompt:
                time.sleep(0.1)
                continue

            # Generate a block of frames
            try:
                start_time = time.time()

                noise = torch.randn(
                    [batch_size, num_frames, 16, 60, 104],
                    device=self.device,
                    dtype=torch.bfloat16
                )

                with torch.no_grad():
                    video = self.pipeline.inference(
                        noise=noise,
                        text_prompts=[self.current_prompt],
                        return_latents=False,
                        profile=False
                    )

                # Extract frames and queue them
                video_np = (video[0] * 255).to(torch.uint8).cpu().numpy()

                for frame in video_np:
                    # frame is [C, H, W] in RGB
                    frame = frame.transpose(1, 2, 0)  # [H, W, C]

                    if not self._frame_queue.full():
                        self._frame_queue.put(frame)

                    self._frame_count += 1
                    self._last_frame_time = time.time()

                # Track latency
                elapsed = time.time() - start_time
                latency_per_frame = elapsed * 1000 / num_frames
                self._latency_samples.append(latency_per_frame)
                if len(self._latency_samples) > 100:
                    self._latency_samples.pop(0)

            except Exception as e:
                print(f"Generation error: {e}")
                time.sleep(0.5)

    async def get_next_frame(self) -> Optional[np.ndarray]:
        """Get the next generated frame (async)."""
        try:
            # Use asyncio to not block
            loop = asyncio.get_event_loop()
            frame = await loop.run_in_executor(None, self._frame_queue.get, True, 1.0)
            return frame
        except:
            return None

    def get_metrics(self) -> Dict[str, Any]:
        """Get current generation metrics."""
        elapsed = time.time() - self._generation_start_time if self._generation_start_time > 0 else 0
        fps = self._frame_count / elapsed if elapsed > 0 else 0

        avg_latency = np.mean(self._latency_samples) if self._latency_samples else 0

        return {
            "fps": fps,
            "latency_ms": avg_latency,
            "frame_count": self._frame_count,
            "elapsed_s": elapsed,
            "generating": self.is_generating,
            "current_prompt": self.current_prompt[:50] if self.current_prompt else "",
        }

    async def cleanup(self):
        """Cleanup resources."""
        await self.stop_generation()
        self.pipeline = None
        torch.cuda.empty_cache()
