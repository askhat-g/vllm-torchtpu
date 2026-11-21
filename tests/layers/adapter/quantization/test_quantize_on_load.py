# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for QUANTIZE_ON_LOAD_PREFIXES and should_quantize_on_load."""

from unittest.mock import MagicMock

import pytest
import torch
from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.linear import LinearBase

from vllm_torchtpu.layers.adapter.quantization.configs import (
    VllmQuantConfig,
    should_quantize_on_load,
)
from vllm_torchtpu.layers.adapter.quantization.fp8 import (
    VllmFp8Config,
    VllmFp8LinearMethodTPU,
    VllmFp8MoEMethodTPU,
)
from vllm_torchtpu.layers.adapter.quantization.unquantized import (
    VllmUnquantizedConfig,
    VllmUnquantizedFusedMoEMethod,
    VllmUnquantizedLinearMethod,
)


@pytest.fixture
def linear_layer():
    """A bare LinearBase instance for testing get_quant_method dispatch."""
    layer = LinearBase.__new__(LinearBase)
    torch.nn.Module.__init__(layer)
    layer.output_size = 32
    layer.input_size = 64
    return layer


@pytest.fixture
def moe_layer():
    """A bare RoutedExperts instance for testing get_quant_method dispatch."""
    layer = RoutedExperts.__new__(RoutedExperts)
    torch.nn.Module.__init__(layer)
    layer.moe_config = MagicMock()
    layer.moe_config.moe_parallel_config.use_ep = False
    layer.moe_config.has_bias = False
    return layer


@pytest.fixture(autouse=True)
def mock_vllm_config(monkeypatch):
    """Ensure VllmQuantConfig.vllm_config is set for get_linear_config."""
    monkeypatch.setattr(VllmQuantConfig, "vllm_config", MagicMock())


class TestShouldQuantizeOnLoad:
    def test_empty_env_returns_false(self, monkeypatch):
        monkeypatch.setenv("QUANTIZE_ON_LOAD_PREFIXES", "")
        assert not should_quantize_on_load("model.layers.0.self_attn.q_proj")

    def test_single_prefix_match(self, monkeypatch):
        monkeypatch.setenv("QUANTIZE_ON_LOAD_PREFIXES", "self_attn")
        assert should_quantize_on_load("model.layers.0.self_attn.q_proj")
        assert should_quantize_on_load("model.layers.0.self_attn.attn")
        assert not should_quantize_on_load("model.layers.0.mlp.gate_proj")

    def test_dot_bounded_prevents_partial_substring_match(self, monkeypatch):
        monkeypatch.setenv("QUANTIZE_ON_LOAD_PREFIXES", "attn")
        # "attn" should not match "self_attn" as a partial token
        assert not should_quantize_on_load("model.layers.0.self_attn.q_proj")
        # "attn" should match ".attn." as a dot-bounded token
        assert should_quantize_on_load("model.layers.0.self_attn.attn")

    def test_multiple_prefixes(self, monkeypatch):
        monkeypatch.setenv(
            "QUANTIZE_ON_LOAD_PREFIXES", "self_attn,shared_experts,layers.0.mlp"
        )
        assert should_quantize_on_load("model.layers.2.self_attn.o_proj")
        assert should_quantize_on_load("model.layers.5.shared_experts.down_proj")
        assert should_quantize_on_load("model.layers.0.mlp.gate_up_proj")
        assert not should_quantize_on_load("model.layers.1.mlp.gate_up_proj")


class TestUnquantizedConfigQuantizeOnLoadDispatch:
    def test_matching_linear_layer_uses_fp8_method(self, monkeypatch, linear_layer):
        monkeypatch.setenv("QUANTIZE_ON_LOAD_PREFIXES", "self_attn")
        cfg = VllmUnquantizedConfig()
        method = cfg.get_quant_method(
            linear_layer, prefix="model.layers.0.self_attn.q_proj"
        )
        assert isinstance(method, VllmFp8LinearMethodTPU)

    def test_kv_b_proj_is_skipped_for_linear_method(self, monkeypatch, linear_layer):
        """kv_b_proj must remain unquantized here because MLAAttention slices
        W_UK_T/W_UV and quantizes them separately.
        """
        monkeypatch.setenv("QUANTIZE_ON_LOAD_PREFIXES", "self_attn")
        cfg = VllmUnquantizedConfig()
        method = cfg.get_quant_method(
            linear_layer, prefix="model.layers.0.self_attn.kv_b_proj"
        )
        assert isinstance(method, VllmUnquantizedLinearMethod)

    def test_non_matching_linear_layer_uses_unquantized_method(
        self, monkeypatch, linear_layer
    ):
        monkeypatch.setenv("QUANTIZE_ON_LOAD_PREFIXES", "self_attn")
        cfg = VllmUnquantizedConfig()
        method = cfg.get_quant_method(
            linear_layer, prefix="model.layers.0.mlp.down_proj"
        )
        assert isinstance(method, VllmUnquantizedLinearMethod)

    def test_matching_moe_layer_uses_online_fp8_moe_method(
        self, monkeypatch, moe_layer
    ):
        monkeypatch.setenv("QUANTIZE_ON_LOAD_PREFIXES", "experts")
        cfg = VllmUnquantizedConfig()
        method = cfg.get_quant_method(moe_layer, prefix="model.layers.0.mlp.experts")
        assert isinstance(method, VllmFp8MoEMethodTPU)
        assert not method.quant_config.is_checkpoint_fp8_serialized
        assert method.weight_block_size is None

    def test_matching_moe_layer_with_bias_raises(self, monkeypatch, moe_layer):
        monkeypatch.setenv("QUANTIZE_ON_LOAD_PREFIXES", "experts")
        moe_layer.moe_config.has_bias = True
        cfg = VllmUnquantizedConfig()
        with pytest.raises(NotImplementedError, match="expert bias"):
            cfg.get_quant_method(moe_layer, prefix="model.layers.0.mlp.experts")

    def test_experts_prefix_leaves_shared_experts_and_attention_unquantized(
        self, monkeypatch, linear_layer
    ):
        monkeypatch.setenv("QUANTIZE_ON_LOAD_PREFIXES", "experts")
        cfg = VllmUnquantizedConfig()
        for prefix in (
            "model.layers.0.mlp.shared_experts.down_proj",
            "model.layers.0.self_attn.qkv_proj",
        ):
            method = cfg.get_quant_method(linear_layer, prefix=prefix)
            assert isinstance(method, VllmUnquantizedLinearMethod)

    def test_non_matching_moe_layer_uses_unquantized_method(
        self, monkeypatch, moe_layer
    ):
        monkeypatch.setenv("QUANTIZE_ON_LOAD_PREFIXES", "self_attn")
        cfg = VllmUnquantizedConfig()
        # CPU CI does not register torch's TPU device type.
        with set_current_vllm_config(
            VllmConfig(device_config=DeviceConfig(device="cpu"))
        ):
            method = cfg.get_quant_method(
                moe_layer, prefix="model.layers.0.mlp.experts"
            )
        assert isinstance(method, VllmUnquantizedFusedMoEMethod)


class TestDummyOnlineFp8MoEWeights:
    def test_dummy_load_allocates_fp8_buffers_directly_and_skips_quantization(
        self, monkeypatch, moe_layer
    ):
        fp8_mod = __import__(
            "vllm_torchtpu.layers.adapter.quantization.fp8", fromlist=["*"]
        )
        vllm_cfg = MagicMock()
        vllm_cfg.load_config.load_format = "dummy"
        cfg = VllmFp8Config(
            is_checkpoint_fp8_serialized=False, activation_scheme="dynamic"
        )
        cfg.vllm_config = vllm_cfg
        moe_layer.moe_config.experts_per_token = 8
        moe_layer.renormalize = True
        moe_layer.weight_loader = object()
        method = VllmFp8MoEMethodTPU(cfg, moe_layer.moe_config)
        method._tpu_topk_weights_float32 = False
        monkeypatch.setattr(
            fp8_mod,
            "_get_moe_quant_config",
            lambda _msg: ("float8_e4m3fn", torch.float8_e4m3fn, None),
        )
        monkeypatch.setattr(method, "_set_tpu_activation", lambda _l: "silu")
        quantize_spy = MagicMock(
            side_effect=AssertionError("should not quantize dummy weights")
        )
        monkeypatch.setattr(fp8_mod, "_quantize_bf16_moe_weights", quantize_spy)
        monkeypatch.setattr(
            fp8_mod.moe_routing, "register_experts_start_buffer", MagicMock()
        )
        monkeypatch.setattr(fp8_mod, "prebuild_fused_moe_kernel", MagicMock())
        fused_ep_mod = __import__(
            "vllm_torchtpu.layers.adapter.fused_moe_ep", fromlist=["*"]
        )
        monkeypatch.setattr(
            fused_ep_mod, "prebuild_fused_moe_ep", MagicMock(return_value=None)
        )

        method.create_weights(
            moe_layer,
            num_experts=4,
            hidden_size=64,
            intermediate_size_per_partition=130,
            params_dtype=torch.bfloat16,
        )
        assert moe_layer._tpu_dummy_fp8_initialized is True
        assert moe_layer.w13_weight.shape == (4, 64, 512)
        assert moe_layer.w13_weight.dtype == torch.float8_e4m3fn
        assert moe_layer.w13_weight.dummy_weight_value == 0
        assert moe_layer.w2_weight.shape == (4, 256, 64)
        assert moe_layer.w2_weight.dtype == torch.float8_e4m3fn
        assert moe_layer.w13_weight_scale_inv.shape == (4, 1, 1, 512)
        assert moe_layer.w13_weight_scale_inv.dtype == torch.float32
        assert moe_layer.w13_weight_scale_inv.dummy_weight_value == 1.0
        assert moe_layer.w13_weight.weight_loader is moe_layer.weight_loader

        method.process_weights_after_loading(moe_layer)
        quantize_spy.assert_not_called()

    def test_block_requantize_falls_back_to_bf16_dummy_weights(
        self, monkeypatch, moe_layer
    ):
        fp8_mod = __import__(
            "vllm_torchtpu.layers.adapter.quantization.fp8", fromlist=["*"]
        )
        vllm_cfg = MagicMock()
        vllm_cfg.load_config.load_format = "dummy"
        cfg = VllmFp8Config(
            is_checkpoint_fp8_serialized=False, activation_scheme="dynamic"
        )
        cfg.vllm_config = vllm_cfg
        method = VllmFp8MoEMethodTPU(cfg, moe_layer.moe_config)
        monkeypatch.setattr(
            fp8_mod,
            "_get_moe_quant_config",
            lambda _msg: ("float8_e4m3fn", torch.float8_e4m3fn, 128),
        )
        assert not method._create_dummy_online_fp8_weights(moe_layer, 4, 64, 130)
        assert not getattr(moe_layer, "_tpu_dummy_fp8_initialized", False)
