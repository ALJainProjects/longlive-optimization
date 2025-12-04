# LongLive Advanced Optimizations
# SPDX-License-Identifier: Apache-2.0
"""
Advanced optimization techniques for LongLive video generation.

Includes:
- Quantized KV cache
- Streaming VAE decoder
- Token merging
- Prefix caching
- Triple buffering
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
import hashlib


# =============================================================================
# Quantized KV Cache
# =============================================================================

class QuantizedKVCache:
    """
    Memory-efficient KV cache using INT8 quantization.

    Reduces memory by 2x and bandwidth by ~50%.
    """

    def __init__(
        self,
        num_layers: int,
        max_seq_len: int,
        num_heads: int,
        head_dim: int,
        device: torch.device,
        quantization: str = "int8"  # "int8" or "fp8"
    ):
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device
        self.quantization = quantization

        # Determine dtype
        if quantization == "int8":
            self.quant_dtype = torch.int8
            self.max_val = 127.0
        elif quantization == "fp8":
            self.quant_dtype = torch.float8_e4m3fn
            self.max_val = 448.0  # FP8 E4M3 max
        else:
            raise ValueError(f"Unknown quantization: {quantization}")

        # Initialize quantized caches
        self.k_cache = torch.zeros(
            (num_layers, 1, max_seq_len, num_heads, head_dim),
            dtype=self.quant_dtype,
            device=device
        )
        self.v_cache = torch.zeros(
            (num_layers, 1, max_seq_len, num_heads, head_dim),
            dtype=self.quant_dtype,
            device=device
        )

        # Per-layer scales
        self.k_scales = torch.ones(num_layers, device=device)
        self.v_scales = torch.ones(num_layers, device=device)

        # Current positions
        self.positions = torch.zeros(num_layers, dtype=torch.long, device=device)

    def update(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache with new K, V and return full dequantized cache.

        Args:
            layer_idx: Layer index
            k: New keys [batch, seq_len, num_heads, head_dim]
            v: New values [batch, seq_len, num_heads, head_dim]

        Returns:
            Full dequantized K, V for attention
        """
        seq_len = k.size(1)
        pos = self.positions[layer_idx].item()

        # Quantize new K, V
        k_scale = k.abs().max() / self.max_val + 1e-8
        v_scale = v.abs().max() / self.max_val + 1e-8

        k_quant = (k / k_scale).to(self.quant_dtype)
        v_quant = (v / v_scale).to(self.quant_dtype)

        # Update scales (running average)
        alpha = 0.9
        self.k_scales[layer_idx] = alpha * self.k_scales[layer_idx] + (1 - alpha) * k_scale
        self.v_scales[layer_idx] = alpha * self.v_scales[layer_idx] + (1 - alpha) * v_scale

        # Store in cache
        end_pos = pos + seq_len
        if end_pos <= self.max_seq_len:
            self.k_cache[layer_idx, :, pos:end_pos] = k_quant
            self.v_cache[layer_idx, :, pos:end_pos] = v_quant
            self.positions[layer_idx] = end_pos
        else:
            # Rolling window
            self.k_cache[layer_idx, :, :-seq_len] = self.k_cache[layer_idx, :, seq_len:]
            self.v_cache[layer_idx, :, :-seq_len] = self.v_cache[layer_idx, :, seq_len:]
            self.k_cache[layer_idx, :, -seq_len:] = k_quant
            self.v_cache[layer_idx, :, -seq_len:] = v_quant

        # Dequantize for attention
        current_len = min(end_pos, self.max_seq_len)
        k_full = self.k_cache[layer_idx, :, :current_len].float() * self.k_scales[layer_idx]
        v_full = self.v_cache[layer_idx, :, :current_len].float() * self.v_scales[layer_idx]

        return k_full, v_full

    def reset(self):
        """Reset all caches."""
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.k_scales.fill_(1.0)
        self.v_scales.fill_(1.0)
        self.positions.zero_()

    def memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        k_size = self.k_cache.numel() * self.k_cache.element_size()
        v_size = self.v_cache.numel() * self.v_cache.element_size()
        return (k_size + v_size) / (1024 * 1024)


# =============================================================================
# Streaming VAE Decoder
# =============================================================================

class StreamingVAEDecoder:
    """
    Streaming VAE decoder that overlaps decode with generation.
    """

    def __init__(self, vae_model: nn.Module, device: torch.device):
        self.vae = vae_model
        self.device = device
        self.decode_stream = torch.cuda.Stream(device=device)
        self.pending_decode: Optional[torch.cuda.Event] = None
        self.decoded_frames: List[torch.Tensor] = []

    def submit_decode(self, latent: torch.Tensor):
        """
        Submit latent for async decoding.

        Args:
            latent: Latent tensor [batch, frames, channels, H, W]
        """
        # Record event on current stream
        current_event = torch.cuda.Event()
        current_event.record()

        # Wait on decode stream
        self.decode_stream.wait_event(current_event)

        with torch.cuda.stream(self.decode_stream):
            # Decode
            decoded = self.vae.decode_to_pixel(latent, use_cache=True)
            self.decoded_frames.append(decoded)

        # Record completion event
        self.pending_decode = torch.cuda.Event()
        self.pending_decode.record(self.decode_stream)

    def get_decoded(self, wait: bool = True) -> Optional[torch.Tensor]:
        """
        Get decoded frames.

        Args:
            wait: Whether to wait for completion

        Returns:
            Decoded frames or None if not ready
        """
        if self.pending_decode is None:
            return None

        if wait:
            self.pending_decode.synchronize()
        elif not self.pending_decode.query():
            return None

        if self.decoded_frames:
            return self.decoded_frames.pop(0)
        return None

    def is_ready(self) -> bool:
        """Check if decode is complete."""
        if self.pending_decode is None:
            return True
        return self.pending_decode.query()


# =============================================================================
# Token Merging
# =============================================================================

def merge_tokens(
    x: torch.Tensor,
    num_merges: int,
    mode: str = "mean"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Merge similar adjacent tokens to reduce sequence length.

    Args:
        x: Input tensor [batch, seq_len, dim]
        num_merges: Number of token pairs to merge
        mode: Merge mode ("mean", "first", "attention")

    Returns:
        merged: Merged tensor [batch, seq_len - num_merges, dim]
        merge_info: Info for unmerging if needed
    """
    batch, seq_len, dim = x.shape

    if num_merges <= 0 or seq_len <= num_merges:
        return x, None

    # Compute similarity between adjacent tokens
    similarity = F.cosine_similarity(x[:, :-1], x[:, 1:], dim=-1)  # [batch, seq_len-1]

    # Find most similar pairs
    _, merge_indices = similarity.topk(num_merges, dim=-1)  # [batch, num_merges]
    merge_indices = merge_indices.sort(dim=-1).values

    # Create merge mask
    merged_tokens = []
    merge_info = {"indices": merge_indices, "mode": mode}

    for b in range(batch):
        tokens = []
        skip_next = False
        merge_idx = 0

        for i in range(seq_len):
            if skip_next:
                skip_next = False
                continue

            if merge_idx < num_merges and i == merge_indices[b, merge_idx].item():
                # Merge this token with next
                if mode == "mean":
                    merged = (x[b, i] + x[b, i + 1]) / 2
                elif mode == "first":
                    merged = x[b, i]
                else:  # attention-weighted
                    weight = similarity[b, i].unsqueeze(-1)
                    merged = weight * x[b, i] + (1 - weight) * x[b, i + 1]

                tokens.append(merged)
                skip_next = True
                merge_idx += 1
            else:
                tokens.append(x[b, i])

        merged_tokens.append(torch.stack(tokens))

    return torch.stack(merged_tokens), merge_info


def unmerge_tokens(
    x: torch.Tensor,
    merge_info: Dict[str, Any],
    original_len: int
) -> torch.Tensor:
    """
    Restore merged tokens to original length.

    Args:
        x: Merged tensor
        merge_info: Info from merge_tokens
        original_len: Original sequence length

    Returns:
        Unmerged tensor
    """
    if merge_info is None:
        return x

    batch = x.shape[0]
    merge_indices = merge_info["indices"]
    num_merges = merge_indices.shape[-1]

    unmerged = []
    for b in range(batch):
        tokens = []
        src_idx = 0
        for i in range(original_len):
            if src_idx < num_merges and i == merge_indices[b, src_idx].item():
                # This was a merged position - duplicate
                tokens.append(x[b, i - src_idx])
                tokens.append(x[b, i - src_idx])
                src_idx += 1
            else:
                tokens.append(x[b, i - src_idx])

        unmerged.append(torch.stack(tokens[:original_len]))

    return torch.stack(unmerged)


# =============================================================================
# Prefix Caching
# =============================================================================

class PrefixCache:
    """
    Cache for reusable prefixes (e.g., frame-sink KV states).
    """

    def __init__(self, max_cache_size: int = 100):
        self.cache: Dict[str, Dict[str, torch.Tensor]] = {}
        self.access_count: Dict[str, int] = {}
        self.max_cache_size = max_cache_size

    def _compute_hash(self, tensor: torch.Tensor) -> str:
        """Compute hash of tensor content."""
        # Use first/last elements + shape for fast hash
        data = f"{tensor.shape}_{tensor.flatten()[:10].tolist()}_{tensor.flatten()[-10:].tolist()}"
        return hashlib.md5(data.encode()).hexdigest()[:16]

    def get(self, prefix: torch.Tensor) -> Optional[Dict[str, torch.Tensor]]:
        """
        Get cached KV for prefix.

        Args:
            prefix: Prefix tensor to look up

        Returns:
            Cached KV dict or None
        """
        key = self._compute_hash(prefix)
        if key in self.cache:
            self.access_count[key] = self.access_count.get(key, 0) + 1
            return self.cache[key]
        return None

    def put(self, prefix: torch.Tensor, kv: Dict[str, torch.Tensor]):
        """
        Cache KV for prefix.

        Args:
            prefix: Prefix tensor
            kv: KV dict to cache
        """
        key = self._compute_hash(prefix)

        # Evict LRU if at capacity
        if len(self.cache) >= self.max_cache_size:
            lru_key = min(self.access_count, key=self.access_count.get)
            del self.cache[lru_key]
            del self.access_count[lru_key]

        self.cache[key] = {k: v.clone() for k, v in kv.items()}
        self.access_count[key] = 1

    def clear(self):
        """Clear all cached entries."""
        self.cache.clear()
        self.access_count.clear()


# =============================================================================
# Triple Buffering
# =============================================================================

class TripleBuffer:
    """
    Triple buffer for overlapped generation, encoding, and display.

    Allows three stages of the pipeline to run concurrently.
    """

    def __init__(self, buffer_shape: Tuple[int, ...], dtype: torch.dtype, device: torch.device):
        self.buffers = [
            torch.zeros(buffer_shape, dtype=dtype, device=device)
            for _ in range(3)
        ]
        self.current = 0
        self.ready = [False, False, False]

    @property
    def generate_buffer(self) -> torch.Tensor:
        """Buffer for current generation."""
        return self.buffers[self.current]

    @property
    def encode_buffer(self) -> torch.Tensor:
        """Buffer for encoding (previous generation)."""
        return self.buffers[(self.current + 2) % 3]

    @property
    def display_buffer(self) -> torch.Tensor:
        """Buffer for display (previous encoding)."""
        return self.buffers[(self.current + 1) % 3]

    def rotate(self):
        """Rotate to next buffer position."""
        self.ready[self.current] = True
        self.current = (self.current + 1) % 3

    def is_encode_ready(self) -> bool:
        """Check if encode buffer has data."""
        return self.ready[(self.current + 2) % 3]

    def is_display_ready(self) -> bool:
        """Check if display buffer has data."""
        return self.ready[(self.current + 1) % 3]


# =============================================================================
# Async Noise Prefetcher
# =============================================================================

class AsyncNoisePrefetcher:
    """
    Prefetch random noise on a separate CUDA stream.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.prefetch_stream = torch.cuda.Stream(device=device)
        self.next_noise: Optional[torch.Tensor] = None
        self.prefetch_event: Optional[torch.cuda.Event] = None
        self.shape: Optional[Tuple[int, ...]] = None

    def prefetch(self, shape: Tuple[int, ...], dtype: torch.dtype = torch.bfloat16):
        """
        Start prefetching noise with given shape.

        Args:
            shape: Noise tensor shape
            dtype: Noise dtype
        """
        self.shape = shape

        with torch.cuda.stream(self.prefetch_stream):
            self.next_noise = torch.randn(shape, device=self.device, dtype=dtype)

        self.prefetch_event = torch.cuda.Event()
        self.prefetch_event.record(self.prefetch_stream)

    def get(self) -> torch.Tensor:
        """
        Get prefetched noise, waiting if necessary.

        Returns:
            Noise tensor
        """
        if self.prefetch_event is not None:
            self.prefetch_event.synchronize()

        noise = self.next_noise
        self.next_noise = None
        self.prefetch_event = None

        return noise

    def is_ready(self) -> bool:
        """Check if prefetch is complete."""
        if self.prefetch_event is None:
            return self.next_noise is not None
        return self.prefetch_event.query()


# =============================================================================
# Utility Functions
# =============================================================================

def apply_all_optimizations(
    pipeline,
    use_quantized_kv: bool = True,
    use_streaming_vae: bool = True,
    use_prefix_cache: bool = True,
    kv_quantization: str = "int8"
) -> Dict[str, Any]:
    """
    Apply all available optimizations to a pipeline.

    Args:
        pipeline: LongLive pipeline
        use_quantized_kv: Enable quantized KV cache
        use_streaming_vae: Enable streaming VAE
        use_prefix_cache: Enable prefix caching
        kv_quantization: KV cache quantization type

    Returns:
        Dict of optimization components
    """
    components = {}

    if use_quantized_kv:
        components["kv_cache"] = QuantizedKVCache(
            num_layers=pipeline.num_transformer_blocks,
            max_seq_len=pipeline.local_attn_size * pipeline.frame_seq_length,
            num_heads=12,
            head_dim=128,
            device=next(pipeline.generator.parameters()).device,
            quantization=kv_quantization
        )

    if use_streaming_vae:
        components["streaming_vae"] = StreamingVAEDecoder(
            pipeline.vae,
            device=next(pipeline.generator.parameters()).device
        )

    if use_prefix_cache:
        components["prefix_cache"] = PrefixCache(max_cache_size=50)

    return components
