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
"""Model-free unit tests for `cache_common`, `cache_hlo`, and `cache_flat`."""

from __future__ import annotations

from typing import Any

import jax
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from tests.megakernel.core.megakernel_fakes import TINY_TILES
from tests.megakernel.core.megakernel_synth import SynthHybrid, synth_hybrid_spec
from vllm_torchtpu.megakernel.core import (
    _builtins,
    cache_common,
    cache_flat,
    cache_hlo,
    fused_io_hook,
    planner,
    pool,
)
from vllm_torchtpu.megakernel.core.errors import MegakernelConfigError
from vllm_torchtpu.megakernel.core.exec_spec import ExecSpec
from vllm_torchtpu.megakernel.core.ops import F32
from vllm_torchtpu.megakernel.core.pool.config import CacheExec
from vllm_torchtpu.megakernel.core.reference import spec_reference_step
from vllm_torchtpu.megakernel.core.weights.eplb import (
    default_eplb_cores,
    encode_eplb_ecopy,
)
from vllm_torchtpu.megakernel.core.weights.spec_inputs import (
    pack_spec_inputs,
    random_spec_inputs,
)


def _make_paged_ex(B: int, T: int, R: int = 2, *, gpr: Any = False) -> ExecSpec:
    tiles = TINY_TILES.with_modules({"gqa": {"kv_blk": 128, "nkv": 2}})
    return ExecSpec(
        B=B,
        T=T,
        R=R,
        ctx=256,
        stack_e=True,
        tiles=tiles,
        interpret=True,
        cache=CacheExec(paged_kv=True, num_pages=64, pool_gather=gpr),
    )


def test_flat_pooled_step_matches_reference_and_hlo_checks():
    """`make_flat_pooled_step` matches oracle and `check_pooled_step_hlo` checks HLO."""
    R = 2
    spec = synth_hybrid_spec(L=2, att_every=2, R=R)
    assert spec.cache_backend is not None
    lay = spec.cache_backend.layout
    ex = _make_paged_ex(B=4, T=1, R=R)
    mesh = Mesh(np.array(jax.devices("cpu")[:R]), ("tp",))
    rep, tp = NamedSharding(mesh, P()), NamedSharding(mesh, P("tp"))

    pos, hid, w_log, st_log = random_spec_inputs(jax.random.PRNGKey(21), spec, ex)
    pos_mk, hid_mk, w_mk, _ = pack_spec_inputs(
        pos, hid, w_log, st_log, spec, ex, mesh=mesh
    )
    pt, ckpt, roff, _ = pool.allocate_test_layout(lay, ex)
    raw = pool.states_to_pool(lay, ex, st_log, pt, ckpt, read_offsets=roff)
    raw_sh = tuple(jax.device_put(p, tp) for p in raw)
    pt_sh = jax.device_put(pt, rep)
    ckpt_sh = jax.device_put(ckpt, rep)
    roff_sh = jax.device_put(roff, rep)

    flat_fn, _, _, donated = cache_flat.make_flat_pooled_step(
        mesh, spec, ex, donate_pools=False
    )
    assert donated == ()
    w_keys = planner.plan(spec, ex).w_names
    w_flat = [w_mk[k] for k in w_keys]
    flat_outs = flat_fn(pos_mk, hid_mk, *w_flat, *raw_sh, pt_sh, ckpt_sh, roff_sh)
    jax.block_until_ready(flat_outs[0])

    ref_hid, _ = spec_reference_step(
        spec,
        spec.config,
        ex,
        pos,
        hid,
        w_log,
        st_log,
        paged_kv=True,
        pool_io=True,
    )
    out_hid_r = flat_outs[0].reshape(R, ex.M, spec.config.H)[:, :4]
    np.testing.assert_allclose(
        out_hid_r[0].astype(F32), ref_hid.astype(F32), rtol=3e-2, atol=3e-2
    )

    hlo = flat_fn.lower(
        pos_mk, hid_mk, *w_flat, *raw_sh, pt_sh, ckpt_sh, roff_sh
    ).as_text()
    ok, msg = cache_hlo.check_pooled_step_hlo(hlo, layout=lay)
    assert ok, msg

    bad_hlo = (
        hlo + "\n%bad = f8e4m3fn[1,64,2,64,4,128] copy(f8e4m3fn[1,64,2,64,4,128] %p)"
    )
    ok_bad, msg_bad = cache_hlo.check_pooled_step_hlo(bad_hlo, layout=lay)
    assert not ok_bad and "full-pool" in msg_bad.lower()

    g_eplb = SynthHybrid(L=2, E=16, moe_tp=False, e_slots=10)
    e_cores = default_eplb_cores(g_eplb, R=2)
    assert len(e_cores) == 2 and len(e_cores[0]) == 2 and len(e_cores[0][0]) == 10
    ecopy = encode_eplb_ecopy(g_eplb, 2, e_cores)
    assert ecopy.shape == (2, 16)


def test_should_fuse_pool_io_without_registered_hook(monkeypatch):
    """`should_fuse_pool_io` checks `fused_io_available` for `auto` and `True`."""
    spec = synth_hybrid_spec(L=2, att_every=2, R=2)
    ex = _make_paged_ex(B=8, T=1, R=2)
    monkeypatch.setattr(_builtins, "_LOADED", set(_builtins.BUILTINS))
    monkeypatch.setattr(fused_io_hook, "_IMPL", None)
    assert not fused_io_hook.fused_io_available()
    assert cache_common.should_fuse_pool_io(spec, ex, fused_pool_io="auto") is False
    assert cache_common.should_fuse_pool_io(spec, ex, fused_pool_io=False) is False
    with pytest.raises(MegakernelConfigError, match="fused pool I/O"):
        cache_common.should_fuse_pool_io(spec, ex, fused_pool_io=True)
