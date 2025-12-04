# LongLive Advanced Optimizations Tests
# SPDX-License-Identifier: Apache-2.0
"""Tests for advanced optimization techniques."""

import pytest
import torch
import torch.nn as nn

from utils.advanced_optimizations import (
    QuantizedKVCache,
    StreamingVAEDecoder,
    merge_tokens,
    unmerge_tokens,
    PrefixCache,
    TripleBuffer,
    AsyncNoisePrefetcher,
    apply_all_optimizations,
)


class TestQuantizedKVCache:
    """Tests for quantized KV cache."""

    def test_init_int8(self, device):
        """Test INT8 cache initialization."""
        cache = QuantizedKVCache(
            num_layers=4,
            max_seq_len=128,
            num_heads=8,
            head_dim=64,
            device=device,
            quantization="int8"
        )

        assert cache.k_cache.dtype == torch.int8
        assert cache.v_cache.dtype == torch.int8
        assert cache.k_cache.shape == (4, 1, 128, 8, 64)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="FP8 requires CUDA")
    def test_init_fp8(self):
        """Test FP8 cache initialization."""
        device = torch.device("cuda")
        cache = QuantizedKVCache(
            num_layers=4,
            max_seq_len=128,
            num_heads=8,
            head_dim=64,
            device=device,
            quantization="fp8"
        )

        assert cache.quant_dtype == torch.float8_e4m3fn

    def test_invalid_quantization(self, device):
        """Test invalid quantization type."""
        with pytest.raises(ValueError, match="Unknown quantization"):
            QuantizedKVCache(
                num_layers=4,
                max_seq_len=128,
                num_heads=8,
                head_dim=64,
                device=device,
                quantization="invalid"
            )

    def test_update_basic(self, device):
        """Test basic cache update."""
        cache = QuantizedKVCache(
            num_layers=2,
            max_seq_len=64,
            num_heads=4,
            head_dim=32,
            device=device,
            quantization="int8"
        )

        # Create sample K, V
        k = torch.randn(1, 8, 4, 32, device=device)
        v = torch.randn(1, 8, 4, 32, device=device)

        k_out, v_out = cache.update(0, k, v)

        assert k_out.shape == (1, 8, 4, 32)
        assert v_out.shape == (1, 8, 4, 32)
        assert cache.positions[0] == 8

    def test_update_multiple_layers(self, device):
        """Test updates across multiple layers."""
        cache = QuantizedKVCache(
            num_layers=3,
            max_seq_len=64,
            num_heads=4,
            head_dim=32,
            device=device,
            quantization="int8"
        )

        for layer_idx in range(3):
            k = torch.randn(1, 4, 4, 32, device=device)
            v = torch.randn(1, 4, 4, 32, device=device)
            cache.update(layer_idx, k, v)

        assert cache.positions[0] == 4
        assert cache.positions[1] == 4
        assert cache.positions[2] == 4

    def test_update_rolling_window(self, device):
        """Test rolling window when cache is full."""
        cache = QuantizedKVCache(
            num_layers=1,
            max_seq_len=8,
            num_heads=2,
            head_dim=16,
            device=device,
            quantization="int8"
        )

        # Fill cache beyond max
        for _ in range(3):
            k = torch.randn(1, 4, 2, 16, device=device)
            v = torch.randn(1, 4, 2, 16, device=device)
            k_out, v_out = cache.update(0, k, v)

        # Cache should roll over
        assert k_out.shape[1] == 8  # max_seq_len

    def test_reset(self, device):
        """Test cache reset."""
        cache = QuantizedKVCache(
            num_layers=2,
            max_seq_len=32,
            num_heads=4,
            head_dim=16,
            device=device,
            quantization="int8"
        )

        k = torch.randn(1, 8, 4, 16, device=device)
        v = torch.randn(1, 8, 4, 16, device=device)
        cache.update(0, k, v)

        cache.reset()

        assert cache.positions[0] == 0
        assert torch.all(cache.k_cache == 0)

    def test_memory_usage(self, device):
        """Test memory usage calculation."""
        cache = QuantizedKVCache(
            num_layers=4,
            max_seq_len=128,
            num_heads=8,
            head_dim=64,
            device=device,
            quantization="int8"
        )

        mb = cache.memory_usage_mb()
        # 4 * 1 * 128 * 8 * 64 * 2 * 1 byte = 0.5 MB
        assert mb > 0
        assert mb < 1.0


class TestTokenMerging:
    """Tests for token merging functions."""

    def test_merge_tokens_basic(self, device):
        """Test basic token merging."""
        x = torch.randn(2, 16, 64, device=device)
        merged, info = merge_tokens(x, num_merges=2)

        assert merged.shape[1] == 14  # 16 - 2 = 14
        assert info is not None
        assert "indices" in info

    def test_merge_tokens_zero_merges(self, device):
        """Test with zero merges."""
        x = torch.randn(2, 16, 64, device=device)
        merged, info = merge_tokens(x, num_merges=0)

        assert merged.shape == x.shape
        assert info is None

    def test_merge_tokens_modes(self, device):
        """Test different merge modes."""
        x = torch.randn(1, 8, 32, device=device)

        for mode in ["mean", "first", "attention"]:
            merged, info = merge_tokens(x, num_merges=2, mode=mode)
            assert merged.shape[1] == 6
            assert info["mode"] == mode

    def test_merge_tokens_max_merges(self, device):
        """Test when num_merges equals sequence length."""
        x = torch.randn(1, 4, 32, device=device)
        merged, info = merge_tokens(x, num_merges=4)

        # Should return original when seq_len <= num_merges
        assert merged.shape == x.shape

    def test_unmerge_tokens_none_info(self, device):
        """Test unmerge with None info."""
        x = torch.randn(2, 16, 64, device=device)
        unmerged = unmerge_tokens(x, merge_info=None, original_len=16)

        assert unmerged.shape == x.shape

    def test_merge_unmerge_roundtrip(self, device):
        """Test merge followed by unmerge."""
        x = torch.randn(1, 8, 32, device=device)
        merged, info = merge_tokens(x, num_merges=2)
        unmerged = unmerge_tokens(merged, info, original_len=8)

        assert unmerged.shape == x.shape


class TestPrefixCache:
    """Tests for prefix caching."""

    def test_init(self):
        """Test cache initialization."""
        cache = PrefixCache(max_cache_size=10)
        assert len(cache.cache) == 0
        assert cache.max_cache_size == 10

    def test_put_get(self, device):
        """Test basic put and get."""
        cache = PrefixCache()
        prefix = torch.randn(2, 8, 64, device=device)
        kv = {"k": torch.randn(2, 8, 64, device=device), "v": torch.randn(2, 8, 64, device=device)}

        cache.put(prefix, kv)
        retrieved = cache.get(prefix)

        assert retrieved is not None
        assert "k" in retrieved
        assert "v" in retrieved

    def test_get_miss(self, device):
        """Test cache miss."""
        cache = PrefixCache()
        prefix = torch.randn(2, 8, 64, device=device)

        retrieved = cache.get(prefix)
        assert retrieved is None

    def test_eviction(self, device):
        """Test LRU eviction."""
        cache = PrefixCache(max_cache_size=2)

        # Add 3 items (should evict first)
        for i in range(3):
            prefix = torch.randn(1, 4, 32, device=device) * (i + 1)
            kv = {"k": torch.randn(1, 4, 32, device=device)}
            cache.put(prefix, kv)

        assert len(cache.cache) == 2

    def test_access_count(self, device):
        """Test access count tracking."""
        cache = PrefixCache()
        prefix = torch.randn(1, 4, 32, device=device)
        kv = {"k": torch.randn(1, 4, 32, device=device)}

        cache.put(prefix, kv)
        cache.get(prefix)
        cache.get(prefix)

        key = cache._compute_hash(prefix)
        assert cache.access_count[key] == 3  # 1 put + 2 gets

    def test_clear(self, device):
        """Test cache clearing."""
        cache = PrefixCache()
        prefix = torch.randn(1, 4, 32, device=device)
        kv = {"k": torch.randn(1, 4, 32, device=device)}

        cache.put(prefix, kv)
        cache.clear()

        assert len(cache.cache) == 0
        assert len(cache.access_count) == 0


class TestTripleBuffer:
    """Tests for triple buffering."""

    def test_init(self, device, dtype):
        """Test buffer initialization."""
        buffer = TripleBuffer(
            buffer_shape=(4, 64, 64),
            dtype=dtype,
            device=device
        )

        assert len(buffer.buffers) == 3
        assert buffer.current == 0
        assert not any(buffer.ready)

    def test_buffer_properties(self, device, dtype):
        """Test buffer property accessors."""
        buffer = TripleBuffer(
            buffer_shape=(4, 64, 64),
            dtype=dtype,
            device=device
        )

        gen_buf = buffer.generate_buffer
        enc_buf = buffer.encode_buffer
        disp_buf = buffer.display_buffer

        assert gen_buf.shape == (4, 64, 64)
        assert enc_buf.shape == (4, 64, 64)
        assert disp_buf.shape == (4, 64, 64)

    def test_rotate(self, device, dtype):
        """Test buffer rotation."""
        buffer = TripleBuffer(
            buffer_shape=(2, 32, 32),
            dtype=dtype,
            device=device
        )

        assert buffer.current == 0

        buffer.rotate()
        assert buffer.current == 1
        assert buffer.ready[0] is True

        buffer.rotate()
        assert buffer.current == 2
        assert buffer.ready[1] is True

        buffer.rotate()
        assert buffer.current == 0
        assert buffer.ready[2] is True

    def test_is_ready_methods(self, device, dtype):
        """Test ready state methods."""
        buffer = TripleBuffer(
            buffer_shape=(2, 32, 32),
            dtype=dtype,
            device=device
        )

        assert not buffer.is_encode_ready()
        assert not buffer.is_display_ready()

        buffer.rotate()

        # After one rotation, encode buffer should have previous data
        # but display buffer should not yet
        assert buffer.is_encode_ready()
        assert not buffer.is_display_ready()


class TestAsyncNoisePrefetcher:
    """Tests for async noise prefetching."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_prefetch_and_get(self):
        """Test prefetch and get on CUDA."""
        device = torch.device("cuda")
        prefetcher = AsyncNoisePrefetcher(device)

        shape = (2, 16, 64, 64)
        prefetcher.prefetch(shape, dtype=torch.float32)

        noise = prefetcher.get()

        assert noise is not None
        assert noise.shape == shape
        assert noise.device.type == "cuda"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_is_ready(self):
        """Test is_ready check."""
        device = torch.device("cuda")
        prefetcher = AsyncNoisePrefetcher(device)

        # Not ready before prefetch
        assert prefetcher.is_ready() is False

        prefetcher.prefetch((2, 16, 64, 64))

        # Wait for completion
        prefetcher.prefetch_event.synchronize()
        assert prefetcher.is_ready() is True

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_get_clears_state(self):
        """Test that get() clears internal state."""
        device = torch.device("cuda")
        prefetcher = AsyncNoisePrefetcher(device)

        prefetcher.prefetch((2, 8, 32, 32))
        _ = prefetcher.get()

        assert prefetcher.next_noise is None
        assert prefetcher.prefetch_event is None


@pytest.mark.skip(reason="Requires full pipeline initialization")
class TestApplyAllOptimizations:
    """Tests for the apply_all_optimizations utility."""

    def test_returns_components(self):
        """Test that components are returned."""
        # This would require a real pipeline
        pass


class TestStreamingVAEDecoder:
    """Tests for streaming VAE decoder."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_init(self):
        """Test decoder initialization."""
        device = torch.device("cuda")

        class MockVAE(nn.Module):
            def decode_to_pixel(self, x, use_cache=False):
                return x * 2

        vae = MockVAE()
        decoder = StreamingVAEDecoder(vae, device)

        assert decoder.vae is vae
        assert decoder.decode_stream is not None
        assert len(decoder.decoded_frames) == 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_is_ready_initial(self):
        """Test is_ready before any decode."""
        device = torch.device("cuda")

        class MockVAE(nn.Module):
            def decode_to_pixel(self, x, use_cache=False):
                return x * 2

        decoder = StreamingVAEDecoder(MockVAE(), device)
        assert decoder.is_ready() is True  # No pending decode

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_submit_and_get(self):
        """Test submit and get decode."""
        device = torch.device("cuda")

        class MockVAE(nn.Module):
            def decode_to_pixel(self, x, use_cache=False):
                return x * 2

        decoder = StreamingVAEDecoder(MockVAE(), device)

        latent = torch.randn(1, 3, 16, 64, 64, device=device)
        decoder.submit_decode(latent)

        result = decoder.get_decoded(wait=True)

        assert result is not None
        assert torch.allclose(result, latent * 2)
