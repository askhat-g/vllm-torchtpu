# Copyright 2026 Google LLC
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
"""Unit and integration tests for the framework-free `megakernel.core.runtime`."""

from __future__ import annotations

import ast
import dataclasses
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import vllm_torchtpu.megakernel.core as mk
from tests.megakernel.core.megakernel_synth import TINY_TILES, synth_hybrid_spec
from vllm_torchtpu.megakernel.core import model_def, planner, profile
from vllm_torchtpu.megakernel.core import runtime as mk_rt
from vllm_torchtpu.megakernel.core.exec_spec import ExecSpec
from vllm_torchtpu.megakernel.core.geometry import ModelGeometry
from vllm_torchtpu.megakernel.core.model_def import (
    Every,
    FromHF,
    FromHFLayerTypes,
    LMHead,
    ModelDef,
    Schedule,
    find_model,
    register_model,
)
from vllm_torchtpu.megakernel.core.modules import GDN, GQA, Embedding, MoE, Norm
from vllm_torchtpu.megakernel.core.pool.config import CacheExec, with_cache
from vllm_torchtpu.megakernel.core.profile import DEFAULT_PROFILE, Profile
from vllm_torchtpu.megakernel.core.runtime import (
    BucketPolicy,
    JaxStepMetaOps,
    SentinelVariantPlan,
    WarmupStepPlan,
    align_cache_exec,
    bucket_exec_spec,
    build_warmup_plan,
    check_compile_hlo_mesh,
    check_post_mesh_eligibility,
    check_startup_eligibility,
    explicit_profile_settings,
    ir_required_buckets,
    pack_step_meta,
    parse_runtime_config,
    pool_schedule_info,
    resolve_driver_profile,
    resolve_step_positions,
    verify_compiled_step_ir,
    verify_ir_coverage,
    verify_sentinel_step,
)


@pytest.fixture()
def isolated_model_registry(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(model_def, "_MODEL_REGISTRY", dict(model_def._MODEL_REGISTRY))
    monkeypatch.setattr(model_def, "_ARCH_REGISTRY", dict(model_def._ARCH_REGISTRY))
    monkeypatch.setattr(
        model_def, "_MODEL_TYPE_REGISTRY", dict(model_def._MODEL_TYPE_REGISTRY)
    )
    monkeypatch.setattr(profile, "_REGISTRY", dict(profile._REGISTRY))
    monkeypatch.setattr(profile, "_MODEL_DEFAULTS", dict(profile._MODEL_DEFAULTS))


def test_runtime_package_ast_and_fresh_import_free_of_vllm_and_torch():
    """`megakernel/core/runtime/*.py` never imports `vllm`/`torch`/`os` and <= 400
    lines."""
    runtime_dir = Path(mk_rt.__file__).parent
    py_files = sorted(runtime_dir.glob("*.py"))
    assert len(py_files) == 10

    for py_path in py_files:
        src = py_path.read_text(encoding="utf-8")
        lines = src.splitlines()
        assert len(lines) <= 400, f"{py_path.name} has {len(lines)} lines (> 400)"
        assert "qwen" not in src.lower(), f"{py_path.name} mentions Qwen"
        tree = ast.parse(src, filename=str(py_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    assert top not in ("torch", "vllm", "os"), (
                        f"{py_path.name} imports {alias.name}"
                    )
            elif isinstance(node, ast.ImportFrom) and node.module:
                top = node.module.split(".")[0]
                assert top not in ("torch", "vllm", "os"), (
                    f"{py_path.name} imports from {node.module}"
                )

    code = (
        "import sys; "
        "sys.modules['torch'] = None; "
        "sys.modules['vllm'] = None; "
        "import vllm_torchtpu.megakernel.core.runtime as rt; "
        "assert hasattr(rt, 'BucketPolicy') and hasattr(rt, 'pack_step_meta')"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"stderr:\n{proc.stderr}"


def test_runtime_config_and_gpr_bucket_exec_spec():
    """Test `parse_runtime_config`, `resolve_driver_profile` with `BucketPolicy`,
    and `bucket_exec_spec` GPR rules (bucket parsing and selection:
    `test_bucket_policy.py`)."""
    prof = resolve_driver_profile(
        DEFAULT_PROFILE,
        fused_pool_io=False,
        fused_pool_io_max_b=16,
        fused_pool_io_min_b_hi=128,
        pool_gather=False,
    )
    assert prof.fused_pool_io is False
    assert prof.fused_pool_io_max_b == 16
    assert prof.fused_pool_io_min_b_hi == 128
    assert prof.pool_gather is False

    pol = BucketPolicy.resolve(
        prof,
        buckets=(8, 16),
        verify_buckets=(16, 32),
        speculative_tokens=0,
        buckets_explicit=True,
        verify_buckets_explicit=False,
    )
    assert pol.buckets == (8, 16)
    assert pol.verify_buckets == ()
    assert list(pol.iter_buckets()) == [(1, 8), (1, 16)]
    assert (pol.max_bucket, pol.max_verify_bucket) == (16, 0)
    assert pol.select(5, 1) == 8

    cfg = parse_runtime_config(
        {
            "enabled": True,
            "buckets": [4, 8],
            "verify_buckets": [16],
            "fused_pool_io": "auto",
            "pool_gather": True,
        }
    )
    assert cfg["enabled"] is True
    assert cfg["buckets"] == (4, 8)
    assert cfg["verify_buckets"] == (16,)
    owned = explicit_profile_settings(cfg)
    assert owned["buckets"] == (4, 8)
    assert owned["verify_buckets"] == (16,)
    assert owned["fused_pool_io"] == "auto"
    assert owned["pool_gather"] is True

    # BLOCKER 2 regression: Profile with pool_gather=True must normalize
    # True -> "eligible" via Profile.exec_for and never re-apply raw True on T=1.
    g = ModelGeometry(
        L=4,
        att_every=4,
        H=256,
        NH=4,
        NKV=1,
        HD=256,
        ROT=64,
        KH=2,
        VH=4,
        KD=128,
        VD=128,
        E=128,
        TOPK=2,
        I=256,
        IS=256,
        moe_tp=False,
        e_slots=0,
        theta=10_000_000.0,
        zero_centered_norm=True,
    )
    ms = synth_hybrid_spec(g, R=1)
    base_ex = ExecSpec(
        R=1,
        B=4,
        T=1,
        ctx=256,
        interpret=True,
        stack_e=False,
        tiles=dataclasses.replace(TINY_TILES, bank=512 << 10),
        cache=CacheExec(paged_kv=True, num_pages=96, max_pages_per_seq=2),
    )
    gpr_prof = Profile(name="test-gpr-true", numerics="approx", pool_gather=True)
    assert gpr_prof.exec_for(base_ex, B=4, T=1, g=g).cache.pool_gather == "eligible"
    ex_t1, use_fused_t1, use_gpr_t1 = bucket_exec_spec(
        ms,
        base_ex,
        gpr_prof,
        bucket_t=4,
        tokens_per_seq=1,
        fused_pool_io=False,
        fused_pool_io_max_b=gpr_prof.fused_pool_io_max_b,
        fused_pool_io_min_b_hi=gpr_prof.fused_pool_io_min_b_hi,
    )
    assert ex_t1.cache.pool_gather is False
    assert (use_fused_t1, use_gpr_t1) == (False, False)
    # planner.plan on T=1 must succeed (would raise if pool_gather=True)
    planner.plan(ms, ex_t1)

    ex_t4, use_fused_t4, use_gpr_t4 = bucket_exec_spec(
        ms,
        base_ex,
        gpr_prof,
        bucket_t=4,
        tokens_per_seq=4,
        fused_pool_io=False,
        fused_pool_io_max_b=gpr_prof.fused_pool_io_max_b,
        fused_pool_io_min_b_hi=gpr_prof.fused_pool_io_min_b_hi,
    )
    assert ex_t4.cache.pool_gather is True
    assert (use_fused_t4, use_gpr_t4) == (False, True)
    planner.plan(ms, ex_t4)

    # Explicit base_exec pool_gather="split" is preserved when profile pool_gather=None
    base_split = with_cache(base_ex, pool_gather="split")
    ex_split_t4, _, use_gpr_split = bucket_exec_spec(
        ms,
        base_split,
        DEFAULT_PROFILE,
        bucket_t=4,
        tokens_per_seq=4,
        fused_pool_io=False,
        fused_pool_io_max_b=DEFAULT_PROFILE.fused_pool_io_max_b,
        fused_pool_io_min_b_hi=DEFAULT_PROFILE.fused_pool_io_min_b_hi,
    )
    assert ex_split_t4.cache.pool_gather is False
    assert not use_gpr_split


def test_pack_step_meta_and_resolve_step_positions_jax():
    """Test `pack_step_meta` and `resolve_step_positions` over `JaxStepMetaOps`."""
    ops = JaxStepMetaOps()
    seq_lens = jnp.asarray([5, 10], dtype=jnp.int32)
    pos_t1 = resolve_step_positions(
        None, seq_lens, num_reqs=2, tokens_per_seq=1, like=seq_lens, ops=ops
    )
    np.testing.assert_array_equal(np.asarray(pos_t1), np.array([4, 9], dtype=np.int32))

    pos_t4 = resolve_step_positions(
        None, seq_lens, num_reqs=2, tokens_per_seq=4, like=seq_lens, ops=ops
    )
    np.testing.assert_array_equal(
        np.asarray(pos_t4), np.array([2, 3, 4, 5, 7, 8, 9, 10], dtype=np.int32)
    )

    # Short positions triggers fallback to seq_lens; M-RoPE 2D positions are preserved
    pos_short = jnp.asarray([4], dtype=jnp.int32)
    pos_from_short = resolve_step_positions(
        pos_short, seq_lens, num_reqs=2, tokens_per_seq=1, like=seq_lens, ops=ops
    )
    np.testing.assert_array_equal(
        np.asarray(pos_from_short), np.array([4, 9], dtype=np.int32)
    )
    pos_mrope = jnp.stack([pos_t4, pos_t4, pos_t4], axis=0)
    pos_from_mrope = resolve_step_positions(
        pos_mrope, seq_lens, num_reqs=2, tokens_per_seq=4, like=seq_lens, ops=ops
    )
    assert pos_from_mrope.shape == (8,)
    np.testing.assert_array_equal(np.asarray(pos_from_mrope), np.asarray(pos_t4))

    ids = jnp.asarray([11, 12, 13, 14, 21, 22, 23, 24], dtype=jnp.int32)
    bt_1d = jnp.asarray([100, 101, 200, 201], dtype=jnp.int32)
    s0 = jnp.asarray([1, 0], dtype=jnp.int32)  # seq 1 inactive (slot 0)
    c0 = jnp.asarray([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=jnp.int32)
    r0 = jnp.asarray([0, -1, 0, 0, 0, 0], dtype=jnp.int32)
    meta = pack_step_meta(
        ids,
        pos_mrope,
        bt_1d,
        ((s0, c0, r0), (s0, c0, r0), (s0, c0, r0)),
        bucket_t=8,
        tokens_per_seq=4,
        max_pages_per_seq=2,
        has_roff=True,
        use_ckpt_2d=True,
        ops=ops,
    )
    assert meta.ids_b.shape == (8,)
    assert meta.pos_b.shape == (8,)
    assert meta.pt_b.shape == (2, 2)
    assert meta.ckpt_b.shape == (3, 2, 4)
    assert meta.roff_b.shape == (3, 2)
    np.testing.assert_array_equal(
        np.asarray(meta.ids_b), np.array([11, 12, 13, 14, 0, 0, 0, 0], dtype=np.int32)
    )
    np.testing.assert_array_equal(
        np.asarray(meta.roff_b[:, 0]), np.array([-1, -1, -1], dtype=np.int32)
    )
    np.testing.assert_array_equal(
        np.asarray(meta.roff_b[:, 1]), np.array([0, 0, 0], dtype=np.int32)
    )

    with pytest.raises(ValueError, match="must be divisible by tokens_per_seq"):
        pack_step_meta(
            ids,
            pos_t4,
            bt_1d,
            ((s0, c0, r0), (s0, c0, r0), (s0, c0, r0)),
            bucket_t=6,
            tokens_per_seq=4,
            max_pages_per_seq=2,
            has_roff=False,
            use_ckpt_2d=False,
            ops=ops,
        )


_SYNTH_ROUTING_DEFAULT_HF_CONFIG: dict[str, Any] = {
    "architectures": ["SynthMoeForCausalLM"],
    "model_type": "synth_moe",
    "vocab_size": 256,
    "hidden_size": 4096,
    "num_hidden_layers": 60,
    "num_attention_heads": 32,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "partial_rotary_factor": 0.25,
    "rope_theta": 10_000_000.0,
    "rms_norm_eps": 1e-6,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 64,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "num_experts": 512,
    "num_experts_per_tok": 10,
    "moe_intermediate_size": 1024,
    "shared_expert_intermediate_size": 1024,
    "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 15,
}


class _SynthRoutingModel(ModelDef):
    """Synthetic hybrid `ModelDef` registering 7 HF architecture / model_type keys."""

    name = "synth_routing"
    hf_architectures = (
        "SynthMoeForConditionalGeneration",
        "SynthMoeForCausalLM",
        "SynthForConditionalGeneration",
        "SynthForCausalLM",
    )
    hf_model_type = ("synth_moe", "synth_moe_text", "synth_family")
    supported_tp_sizes = (8,)
    default_hf_config = _SYNTH_ROUTING_DEFAULT_HF_CONFIG
    geometry = FromHF(
        hidden="hidden_size",
        layers="num_hidden_layers",
        vocab="vocab_size",
        eps="rms_norm_eps",
        rope_theta="rope_theta",
        rope_partial=("partial_rotary_factor", 0.25),
    )
    attn = GQA.bind(
        prefix="self_attn",
        heads="num_attention_heads",
        kv_heads="num_key_value_heads",
        head_dim="head_dim",
        qk_norm=True,
        output_gate=True,
    )
    linear = GDN.bind(
        prefix="linear_attn",
        k_heads="linear_num_key_heads",
        v_heads="linear_num_value_heads",
        k_dim="linear_key_head_dim",
        v_dim="linear_value_head_dim",
        conv_kernel="linear_conv_kernel_dim",
    )
    moe = MoE.bind(
        prefix="mlp",
        experts="num_experts",
        top_k="num_experts_per_tok",
        inter="moe_intermediate_size",
        shared_inter="shared_expert_intermediate_size",
        shared_gate=True,
        moe_tp=False,
    )
    schedule = Schedule(
        mixer=FromHFLayerTypes(
            "layer_types",
            {"full_attention": attn, "linear_attention": linear},
        ),
        ffn=moe,
        pre_norm=Norm.bind("input_layernorm", zero_centered=True),
        post_norm=Norm.bind("post_attention_layernorm", zero_centered=True),
    )
    embed = Embedding("model.embed_tokens")
    final_norm = Norm.bind("model.norm", zero_centered=True)
    lm_head = LMHead("lm_head")


@pytest.mark.parametrize(
    ("arch", "model_type", "num_experts", "exp_found", "exp_eligible", "exp_reason"),
    [
        # 1-2. Canonical MoE model_types
        ("UnknownArch", "synth_moe", 512, True, True, "ok"),
        ("UnknownArch", "synth_moe_text", 512, True, True, "ok"),
        # 3. Shared family model_type (MoE vs dense disambiguated by validate_hf_config)
        ("UnknownArch", "synth_family", 512, True, True, "ok"),
        ("UnknownArch", "synth_family", 0, True, False, "non_moe_checkpoint"),
        # 4-5. Canonical MoE HF architectures
        ("SynthMoeForConditionalGeneration", "unknown", 512, True, True, "ok"),
        ("SynthMoeForCausalLM", "unknown", 512, True, True, "ok"),
        # 6-7. Shared family HF architectures (MoE vs dense)
        ("SynthForConditionalGeneration", "unknown", 512, True, True, "ok"),
        (
            "SynthForConditionalGeneration",
            "unknown",
            0,
            True,
            False,
            "non_moe_checkpoint",
        ),
        ("SynthForCausalLM", "unknown", 512, True, True, "ok"),
        ("SynthForCausalLM", "unknown", None, True, False, "non_moe_checkpoint"),
        # Unsupported model keys return None from find_model
        ("OtherForCausalLM", "other_arch", 512, False, False, ""),
        ("AltForCausalLM", "alt_arch", 512, False, False, ""),
    ],
)
def test_multi_key_routing_and_dense_rejection(
    isolated_model_registry,
    arch: str,
    model_type: str,
    num_experts: int | None,
    exp_found: bool,
    exp_eligible: bool,
    exp_reason: str,
):
    """Verify all 7 routing keys resolve to `_SynthRoutingModel` and reject dense."""
    register_model(
        name=_SynthRoutingModel.name,
        hf_architectures=_SynthRoutingModel.hf_architectures,
        hf_model_type=_SynthRoutingModel.hf_model_type,
        replace=True,
    )(_SynthRoutingModel)
    cfg = dict(_SynthRoutingModel.default_hf_config)
    cfg["architectures"] = [arch]
    cfg["model_type"] = model_type
    if num_experts is None:
        cfg.pop("num_experts", None)
    else:
        cfg["num_experts"] = num_experts

    found = find_model(cfg)
    if not exp_found:
        assert found is None
        return
    assert found is _SynthRoutingModel
    el = found.validate_hf_config(cfg)
    assert el.eligible is exp_eligible
    assert el.reason == exp_reason


def test_schedule_driven_validate_hf_config(isolated_model_registry):
    """`ModelDef.validate_hf_config` resolves `Schedule.ffn` across schedule forms."""
    valid_cfg = dict(_SynthRoutingModel.default_hf_config)
    assert _SynthRoutingModel.validate_hf_config(valid_cfg).eligible

    dense_cfg = {k: v for k, v in valid_cfg.items() if k != "num_experts"}
    res_dense = _SynthRoutingModel.validate_hf_config(dense_cfg)
    assert not res_dense.eligible and res_dense.reason == "non_moe_checkpoint"
    assert "_SynthRoutingModel supports MoE checkpoints only" in res_dense.message

    # Non-numeric num_experts (e.g. string "512") returns non_moe_checkpoint cleanly
    str_experts_cfg = {**valid_cfg, "num_experts": "512"}
    res_str = _SynthRoutingModel.validate_hf_config(str_experts_cfg)
    assert not res_str.eligible and res_str.reason == "non_moe_checkpoint"

    # Missing hidden_size still checks present geometry keys (e.g. head_dim)
    no_hidden_bad_hd = {"num_experts": 512, "head_dim": 64}
    res_hd = _SynthRoutingModel.validate_hf_config(no_hidden_bad_hd)
    assert not res_hd.eligible and res_hd.reason == "geometry_mismatch"
    assert "HD=64 (expected 256)" in res_hd.message

    custom_moe = MoE.bind(
        prefix="mlp",
        experts="routed_expert_count",
        top_k="num_experts_per_tok",
        inter="moe_intermediate_size",
        shared_inter="shared_expert_intermediate_size",
        shared_gate=True,
    )
    other_moe = MoE.bind(
        prefix="mlp",
        experts="alt_expert_count",
        top_k="num_experts_per_tok",
        inter="moe_intermediate_size",
        shared_inter="shared_expert_intermediate_size",
        shared_gate=True,
    )
    attn = GQA.bind(
        prefix="self_attn",
        heads="num_attention_heads",
        kv_heads="num_key_value_heads",
        head_dim="head_dim",
    )
    gdn = GDN.bind(
        prefix="linear_attn",
        k_heads="linear_num_key_heads",
        v_heads="linear_num_value_heads",
        k_dim="linear_key_head_dim",
        v_dim="linear_value_head_dim",
        conv_kernel="linear_conv_kernel_dim",
    )
    dummy_non_moe = Norm.bind("mlp_norm", zero_centered=True)

    base_synth_cfg = {
        **dense_cfg,
        "architectures": [],
        "num_hidden_layers": 4,
        "routed_expert_count": 128,
        "alt_expert_count": 64,
        "layer_types": ["linear_attention", "full_attention"] * 2,
        "ffn_types": ["dense", "moe", "dense", "moe"],
    }
    geom_spec = FromHF(
        hidden="hidden_size",
        layers="num_hidden_layers",
        eps="rms_norm_eps",
        rope_theta="rope_theta",
    )

    # 1. Schedule.ffn wrapped in Every(...)
    @register_model(name="synth_every_ffn", hf_model_type="synth_every_ffn")
    class SynthEveryFFN(ModelDef):
        default_hf_config = {**base_synth_cfg, "model_type": "synth_every_ffn"}
        geometry = geom_spec
        schedule = Schedule(
            mixer=Every(n=2, last=attn, rest=gdn),
            ffn=Every(n=2, last=custom_moe, rest=dummy_non_moe),
            pre_norm=Norm.bind("input_layernorm", zero_centered=True),
            post_norm=Norm.bind("post_attention_layernorm", zero_centered=True),
        )
        final_norm = Norm.bind("model.norm", zero_centered=True)

    assert SynthEveryFFN.validate_hf_config(SynthEveryFFN.default_hf_config).eligible
    assert not SynthEveryFFN.validate_hf_config(
        {**SynthEveryFFN.default_hf_config, "routed_expert_count": 0}
    ).eligible

    # 2. Schedule.ffn wrapped in FromHFLayerTypes(...)
    class SynthLayerTypesFFN(ModelDef):
        default_hf_config = {**base_synth_cfg, "model_type": "synth_lh_ffn"}
        geometry = geom_spec
        schedule = Schedule(
            mixer=Every(n=2, last=attn, rest=gdn),
            ffn=FromHFLayerTypes(
                "ffn_types",
                {"dense": dummy_non_moe, "moe": custom_moe},
            ),
            pre_norm=Norm.bind("input_layernorm", zero_centered=True),
            post_norm=Norm.bind("post_attention_layernorm", zero_centered=True),
        )
        final_norm = Norm.bind("model.norm", zero_centered=True)

    assert SynthLayerTypesFFN.validate_hf_config(
        SynthLayerTypesFFN.default_hf_config
    ).eligible
    assert not SynthLayerTypesFFN.validate_hf_config(
        {**SynthLayerTypesFFN.default_hf_config, "routed_expert_count": None}
    ).eligible

    # 3. Schedule.ffn as a per-layer sequence + Schedule.overrides with a second MoE key
    @register_model(name="synth_seq_ov_ffn", hf_model_type="synth_seq_ov_ffn")
    class SynthSeqOverridesFFN(ModelDef):
        default_hf_config = {**base_synth_cfg, "model_type": "synth_seq_ov_ffn"}
        geometry = geom_spec
        schedule = Schedule(
            mixer=Every(n=2, last=attn, rest=gdn),
            ffn=[dummy_non_moe, custom_moe, dummy_non_moe, dummy_non_moe],
            pre_norm=Norm.bind("input_layernorm", zero_centered=True),
            post_norm=Norm.bind("post_attention_layernorm", zero_centered=True),
            overrides={3: {"ffn": other_moe}},
        )
        final_norm = Norm.bind("model.norm", zero_centered=True)

    assert SynthSeqOverridesFFN.validate_hf_config(
        SynthSeqOverridesFFN.default_hf_config
    ).eligible
    assert not SynthSeqOverridesFFN.validate_hf_config(
        {**SynthSeqOverridesFFN.default_hf_config, "routed_expert_count": 0}
    ).eligible
    assert not SynthSeqOverridesFFN.validate_hf_config(
        {**SynthSeqOverridesFFN.default_hf_config, "alt_expert_count": 0}
    ).eligible


def test_checks_hlo_ir_warmup_and_sentinel():
    """Test `check_compile_hlo_mesh`, `verify_compiled_step_ir`, and IR coverage."""
    cpu_mesh = jax.sharding.Mesh(np.asarray(jax.devices("cpu")[:2]), ("tp",))
    check_compile_hlo_mesh(cpu_mesh, True)
    fake_mesh = types.SimpleNamespace(devices=np.asarray([object(), object()]))
    with pytest.raises(mk.MegakernelConfigError, match="requires a mesh of real"):
        check_compile_hlo_mesh(fake_mesh, True)

    good_ir = """
    module @test {
      func.func public @main(
        %arg0: tensor<64x2x64x4x128xf8E4M3FN> {tf.aliasing_output = 1 : i32},
        %arg1: tensor<64x2x64x4x128xf8E4M3FN> {tf.aliasing_output = 2 : i32}
      ) -> (
        tensor<8x1024xbf16>,
        tensor<64x2x64x4x128xf8E4M3FN>,
        tensor<64x2x64x4x128xf8E4M3FN>
      ) {
        %0 = stablehlo.custom_call @tpu_custom_call(%arg0, %arg1)
          : (tensor<64x2x64x4x128xf8E4M3FN>, tensor<64x2x64x4x128xf8E4M3FN>)
          -> tensor<8x1024xbf16>
        return
      }
    }
    """
    ok_ir, msg_ir = verify_compiled_step_ir(
        good_ir, num_pages=64, page_size=128, kvh=1, hd=256, num_pools=2
    )
    assert ok_ir, msg_ir

    bad_ir = good_ir.replace("{tf.aliasing_output = 2 : i32}", "")
    ok_bad, msg_bad = verify_compiled_step_ir(
        bad_ir, num_pages=64, page_size=128, kvh=1, hd=256, num_pools=2
    )
    assert not ok_bad and "expected >= 2 donated pool arguments" in msg_bad

    assert ir_required_buckets(
        [(1, 8), (1, 16), (4, 16)],
        {(1, 8): True, (1, 16): False, (4, 16): True},
        is_eager=False,
    ) == [(1, 8), (1, 16), (4, 16)]
    assert ir_required_buckets([(1, 8)], {(1, 8): True}, is_eager=True) == [(1, 8)]
    assert ir_required_buckets([(1, 8)], {(1, 8): False}, is_eager=True) == []

    # verify_ir_coverage tests
    verify_ir_coverage([(1, 8)], {(1, 8): True}, {(1, 8)}, is_eager=False)
    verify_ir_coverage([], {}, set(), is_eager=False)
    with pytest.raises(RuntimeError, match="boom"):
        verify_ir_coverage(
            [(1, 8)], {(1, 8): True}, {(1, 8)}, is_eager=False, ir_error="boom"
        )
    with pytest.raises(RuntimeError, match="did not run for buckets"):
        verify_ir_coverage(
            [(1, 8), (1, 16)],
            {(1, 8): True, (1, 16): True},
            {(1, 8)},
            is_eager=False,
        )

    # pool_schedule_info raises MegakernelLayoutError when cache_backend is absent
    with pytest.raises(mk.MegakernelLayoutError, match="has no pool backend layout"):
        pool_schedule_info(types.SimpleNamespace(cache_backend=None))


def test_no_kw_rejects_legacy_and_unknown_kwargs():
    """Runtime API functions reject legacy or unknown keyword arguments."""
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        check_startup_eligibility(  # type: ignore[call-arg]
            num_layers=60,
            expected_num_layers=60,
            tp_size=8,
            supported_tp_sizes=(8,),
            unified_kv_layout=True,
        )
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        check_startup_eligibility(  # type: ignore[call-arg]
            num_layers=60,
            expected_num_layers=60,
            tp_size=8,
            supported_tp_sizes=(8,),
            supports_mm_inputs=False,
        )
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        check_startup_eligibility(  # type: ignore[call-arg]
            num_layers=60,
            expected_num_layers=60,
            tp_size=8,
            supported_tp_sizes=(8,),
            unknown_flag=True,
        )
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        check_post_mesh_eligibility(  # type: ignore[call-arg]
            tp_size=8,
            num_layers=60,
        )
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        resolve_driver_profile(DEFAULT_PROFILE, unknown_flag=True)  # type: ignore[call-arg]

    g = ModelGeometry(
        L=4,
        att_every=4,
        H=256,
        NH=4,
        NKV=1,
        HD=256,
        ROT=64,
        KH=2,
        VH=4,
        KD=128,
        VD=128,
        E=128,
        TOPK=2,
        I=256,
        IS=256,
        theta=10_000_000.0,
        zero_centered_norm=True,
    )
    ms = synth_hybrid_spec(g, R=1)
    base_ex = ExecSpec(R=1, B=4, T=1, ctx=256, interpret=True, stack_e=False)
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        bucket_exec_spec(  # type: ignore[call-arg]
            ms,
            base_ex,
            DEFAULT_PROFILE,
            bucket_t=4,
            tokens_per_seq=1,
            pool_gather=True,
        )
    with pytest.raises(TypeError, match="missing 1 required keyword-only argument"):
        align_cache_exec(  # type: ignore[call-arg]
            base_ex, DEFAULT_PROFILE, g, num_pages=64, max_pages_per_seq=2
        )
    with pytest.raises(TypeError, match="missing .* required keyword-only argument"):
        verify_compiled_step_ir(  # type: ignore[call-arg]
            "module {}", num_pages=64, kvh=1
        )
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        build_warmup_plan(  # type: ignore[call-arg]
            ms, ((1, 4),), {}, {}, num_pages=64, max_pages_per_seq=2, unknown=1
        )
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        verify_sentinel_step(  # type: ignore[call-arg]
            np.zeros((2, 4)),
            [],
            [],
            [],
            [],
            tokens_per_seq=1,
            bucket_t=2,
            variant_key=(1, False, False),
            unknown=1,
        )

    # `build_warmup_plan` takes `(T, B)` `bucket_keys` directly: default keys
    # match the pre-generalization `WarmupStepPlan` list byte-for-byte, and
    # `T=8` verify keys produce plans for `(8, b)` with `spv = spp * 8`.
    pol_def = BucketPolicy((8, 16), (16, 32))
    default_keys = list(pol_def.iter_buckets())
    fio_def = {(1, 8): True, (1, 16): False, (4, 16): True, (4, 32): False}
    gpr_def = {(1, 8): False, (1, 16): True, (4, 16): False, (4, 32): False}
    steps_def, warmed_def, n_save_def = build_warmup_plan(
        ms,
        default_keys,
        fio_def,
        gpr_def,
        num_pages=256,
        max_pages_per_seq=4,
        ckpt_window=4,
        runner_tok_pads=(16, 32),
        req_block_pairs=((8, 4), (16, 4)),
        is_eager=False,
    )
    expected_default_steps = [
        WarmupStepPlan(
            1, 8, 16, 8, 4, 4, SentinelVariantPlan((1, True, False), 1, 3, 15, 1, True)
        ),
        WarmupStepPlan(1, 8, 16, 16, 4, 4, None),
        WarmupStepPlan(
            1,
            16,
            16,
            8,
            4,
            4,
            SentinelVariantPlan((1, False, True), 2, 15, 75, 1, True),
        ),
        WarmupStepPlan(1, 16, 16, 16, 4, 4, None),
        WarmupStepPlan(
            4,
            16,
            16,
            8,
            4,
            4,
            SentinelVariantPlan((4, True, False), 3, 27, 135, 4, True),
        ),
        WarmupStepPlan(4, 16, 16, 16, 4, 4, None),
        WarmupStepPlan(
            4,
            32,
            32,
            8,
            4,
            4,
            SentinelVariantPlan((4, False, False), 4, 39, 195, 4, True),
        ),
        WarmupStepPlan(4, 32, 32, 16, 4, 4, None),
    ]
    assert steps_def == expected_default_steps
    assert repr(steps_def) == repr(expected_default_steps)
    assert warmed_def == default_keys and n_save_def == 255

    keys_t8 = [(1, 8), (8, 16), (8, 32)]
    fio_t8 = {(1, 8): True, (8, 16): True, (8, 32): False}
    gpr_t8 = {(1, 8): False, (8, 16): False, (8, 32): False}
    steps_t8, warmed_t8, n_save_t8 = build_warmup_plan(
        ms,
        keys_t8,
        fio_t8,
        gpr_t8,
        num_pages=512,
        max_pages_per_seq=4,
        ckpt_window=4,
        is_eager=False,
    )
    assert warmed_t8 == keys_t8
    assert [(s.tokens_per_seq, s.bucket_t, s.req_pad, s.c_cols) for s in steps_t8] == [
        (1, 8, 8, 4),
        (8, 16, 2, 8),
        (8, 32, 4, 8),
    ]
    assert [s.sentinel for s in steps_t8] == [
        SentinelVariantPlan((1, True, False), 1, 3, 15, 1, True),
        SentinelVariantPlan((8, True, False), 2, 27, 135, 8, True),
        SentinelVariantPlan((8, False, False), 3, 51, 255, 8, True),
    ]
    assert n_save_t8 == 375
