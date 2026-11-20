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
"""Per-core hardware specifications, constants catalog, and 3D topologies."""

from __future__ import annotations

import json
import math
from collections.abc import KeysView, Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Protocol, TypedDict, Unpack, runtime_checkable

from .constants import (
    BF16_BYTES,
    BIN_KIBI,
    BIN_MEBI,
    BITS_PER_BYTE,
    CHIP_COUNT_BRACKET,
    CHIP_SPEC_SCHEMA,
    DCN_BRACKET_GBIT_PER_S,
    DEFAULT_CHIPS_PER_HOST,
    DEFAULT_CORES_PER_CHIP,
    DEFAULT_D2D_BW,
    DEFAULT_DCN_BW,
    DEFAULT_DCN_HOST_GBIT_PER_S,
    DEFAULT_DMA_GRANULE_BYTES,
    DEFAULT_GPU_MEMORY_UTILIZATION,
    DEFAULT_ICI_HOP_LATENCY_S,
    DEFAULT_ICI_LINK_BW,
    DEFAULT_ICI_LINKS_PER_CHIP,
    DEFAULT_MXU_LHS_ROWS_BF16,
    DEFAULT_MXU_LHS_ROWS_FP8,
    DEFAULT_MXU_M_TILE,
    DEFAULT_PCIE_BW,
    DEFAULT_VPU_LANES,
    DEFAULT_VPU_SUBLANES,
    DEFAULT_VPU_TO_BF16_RATIO,
    FALLBACK_TPU_SPECS,
    GPU_MEM_UTIL_BRACKET,
    ICI_HOP_LATENCY_BRACKET_S,
    MIN_TORUS_DIM,
    MXU_M_TILE_BRACKET,
    PAGE_TABLE_BRACKET_BYTES,
    PAGE_TABLE_BYTE_BUDGET,
    PCIE_BRACKET_BW,
    PERCENT_SCALE,
    PHYSICAL_ALIGNMENT_ELEMS,
    RPA_PAGE_SIZE_BRACKET_TOKENS,
    RPA_PAGE_SIZE_TOKENS,
    SUBCHANNEL_BLOCK_K_FLOOR,
    TOPOLOGY_AXES,
    UNIT_GIGA,
    UNIT_KILO,
    UNIT_MEGA,
    UNIT_MILLI,
    UNIT_TERA,
    VALID_CHIP_PROVENANCE,
    VPU_TO_BF16_RATIO_BRACKET,
)

_TORUS_AXES_LEN: int = 3
# Reserve half of 1 MiB SMEM for KV page tables and half for compiler/kernel scratch
# (1_048_576 B // 2 == 524_288 B PAGE_TABLE_BYTE_BUDGET).
_SMEM_PAGE_TABLE_DIVISOR = 2


@runtime_checkable
class _ChipVersionProto(Protocol):
    """Structural protocol for JAX ChipVersion physical core metadata."""

    @property
    def num_physical_tensor_cores_per_chip(self) -> int:
        """Return physical TensorCores per TPU chip."""
        ...


@runtime_checkable
class _SparseCoreProto(Protocol):
    """Structural protocol for JAX TpuInfo SparseCore descriptor."""

    @property
    def dma_granule_size_bytes(self) -> int:
        """Return SparseCore DMA transfer granule size in bytes."""
        ...


class _TpuInfoProto(Protocol):
    """Structural protocol for JAX get_tpu_info_for_chip result."""

    @property
    def hbm_capacity_bytes(self) -> int:
        """Return per-core HBM capacity in bytes."""
        ...

    @property
    def mem_bw_bytes_per_second(self) -> float:
        """Return per-core HBM bandwidth in bytes per second."""
        ...

    @property
    def bf16_ops_per_second(self) -> float:
        """Return per-core BF16 peak FLOP/s."""
        ...

    @property
    def fp8_ops_per_second(self) -> float:
        """Return per-core FP8 peak FLOP/s."""
        ...

    @property
    def vmem_capacity_bytes(self) -> int:
        """Return per-core scratchpad capacity in bytes."""
        ...


@runtime_checkable
class _TpuInfoTilingProto(Protocol):
    """Optional MXU and VPU tiling attributes exposed by JAX TpuInfo."""

    @property
    def mxu_column_size(self) -> int:
        """Return MXU systolic array column count."""
        ...

    @property
    def num_lanes(self) -> int:
        """Return vector register lane count."""
        ...

    @property
    def num_sublanes(self) -> int:
        """Return vector register sublane count."""
        ...


@runtime_checkable
class _TpuInfoSmemProto(Protocol):
    """Optional SMEM capacity attribute exposed by JAX TpuInfo."""

    @property
    def smem_capacity_bytes(self) -> int:
        """Return per-core scalar memory capacity in bytes."""
        ...


@runtime_checkable
class _TpuInfoIntOpsProto(Protocol):
    """Optional integer peak rate attributes exposed by JAX TpuInfo."""

    @property
    def int8_ops_per_second(self) -> float:
        """Return per-core INT8 peak op/s."""
        ...

    @property
    def int4_ops_per_second(self) -> float:
        """Return per-core INT4 peak op/s."""
        ...


@runtime_checkable
class _TpuInfoSparseCoreProto(Protocol):
    """Optional SparseCore descriptor attribute exposed by JAX TpuInfo."""

    @property
    def sparse_core(self) -> object:
        """Return SparseCore descriptor or None when absent."""
        ...


class _PltpuProto(Protocol):
    """Structural protocol for jax.experimental.pallas.tpu module."""

    def get_tpu_info_for_chip(
        self, version: object, cores_per_chip: int
    ) -> _TpuInfoProto:
        """Query per-device TPU hardware limits from JAX."""
        ...


try:
    from jax.experimental.pallas import tpu as _pltpu
except ImportError:
    pltpu: _PltpuProto | None = None
    _TPU_7X_VERSION: object = "tpu7x"
else:
    pltpu = _pltpu
    _TPU_7X_VERSION = _pltpu.ChipVersion.TPU_7X

try:
    import jax as _jax
except ImportError:
    JAX_VERSION: str = "unavailable"
else:
    JAX_VERSION = _jax.__version__


@dataclass(frozen=True, slots=True)
class ConstantSpec:
    """Metadata for one hardware or solver constant in the roofline catalog."""

    value: float | int
    unit: str
    scope: str
    kind: str
    source: str
    bracket: tuple[float, float] | None = None
    display: str = ""
    description: str = ""


@dataclass(frozen=True, slots=True)
class ChipOverrides:
    """Optional ICI, DCN, PCIe, tile, and capability overrides for ChipSpec."""

    ici_link_bw: float = DEFAULT_ICI_LINK_BW
    d2d_bw: float | None = None
    ici_bw: float | None = None
    ici_hop_latency_s: float = DEFAULT_ICI_HOP_LATENCY_S
    cores_per_chip: int | None = None
    mxu_m_tile: int | None = None
    dcn_host_gbit_per_s: float = DEFAULT_DCN_HOST_GBIT_PER_S
    chips_per_host: int = DEFAULT_CHIPS_PER_HOST
    dcn_bw: float | None = None
    pcie_bw: float = DEFAULT_PCIE_BW
    vpu_flops: float | None = None
    hbm_bytes: int | None = None
    hbm_bw: float | None = None
    bf16_flops: float | None = None
    fp8_flops: float | None = None
    int8_flops: float | None = None
    fp4_flops: float | None = None
    int4_flops: float | None = None
    vmem_bytes: int | None = None
    smem_page_table_bytes: int | None = None
    ici_links_per_chip: int | None = None
    min_torus_dim: int | None = None
    vpu_lanes: int | None = None
    vpu_sublanes: int | None = None
    vpu_to_bf16_ratio: float | None = None
    mxu_lhs_rows_bf16: int | None = None
    mxu_lhs_rows_fp8: int | None = None
    physical_alignment_elems: int | None = None
    dma_granule_bytes: int | None = None
    subchannel_block_k_floor: int | None = None
    has_sparse_core: bool | None = None
    provenance: str | None = None


@dataclass(frozen=True, slots=True)
class ChipSpec:
    """Hardware ceiling parameters for one TensorCore and its parent chip/host."""

    name: str
    hbm_bytes: int
    hbm_bw: float
    bf16_flops: float
    fp8_flops: float
    vmem_bytes: int
    ici_bw: float = math.inf
    cores_per_chip: int = DEFAULT_CORES_PER_CHIP
    ici_link_bw: float = DEFAULT_ICI_LINK_BW
    d2d_bw: float = DEFAULT_D2D_BW
    ici_hop_latency_s: float = DEFAULT_ICI_HOP_LATENCY_S
    mxu_m_tile: int = DEFAULT_MXU_M_TILE
    dcn_host_gbit_per_s: float = DEFAULT_DCN_HOST_GBIT_PER_S
    chips_per_host: int = DEFAULT_CHIPS_PER_HOST
    dcn_bw: float = DEFAULT_DCN_BW
    pcie_bw: float = DEFAULT_PCIE_BW
    vpu_flops: float = 1.155e15 * DEFAULT_VPU_TO_BF16_RATIO
    mxu_lhs_rows_bf16: int = DEFAULT_MXU_LHS_ROWS_BF16
    mxu_lhs_rows_fp8: int = DEFAULT_MXU_LHS_ROWS_FP8
    int8_flops: float = 0.0
    fp4_flops: float = 0.0
    int4_flops: float = 0.0
    ici_links_per_chip: int = DEFAULT_ICI_LINKS_PER_CHIP
    min_torus_dim: int = MIN_TORUS_DIM
    vpu_lanes: int = DEFAULT_VPU_LANES
    vpu_sublanes: int = DEFAULT_VPU_SUBLANES
    physical_alignment_elems: int = PHYSICAL_ALIGNMENT_ELEMS
    smem_page_table_bytes: int = PAGE_TABLE_BYTE_BUDGET
    dma_granule_bytes: int = DEFAULT_DMA_GRANULE_BYTES
    subchannel_block_k_floor: int = SUBCHANNEL_BLOCK_K_FLOOR
    has_sparse_core: bool = True
    provenance: str = "public_spec"

    def _fallback_matmul_peak(
        self, weight_bytes: float, act: str
    ) -> tuple[str, float, bool]:
        """Select FP8 or BF16 fallback ceiling without native fast-path."""
        unpack = weight_bytes < 1.0
        if weight_bytes < BF16_BYTES and act == "a8" and self.fp8_flops > 0.0:
            return ("fp8", self.fp8_flops, unpack)
        return ("bf16", self.bf16_flops, unpack)

    def matmul_peak_kind(
        self,
        weight_bytes: float,
        act: str = "a8",
        *,
        weight_dtype: str = "",
    ) -> tuple[str, float, bool]:
        """Select MXU precision bucket, peak rate, and unpack requirement."""
        dtype_norm = weight_dtype.strip().lower()
        is_fp4 = (
            dtype_norm in ("nvfp4", "mxfp4", "fp4")
            or "fp4" in dtype_norm
            or (not dtype_norm and weight_bytes < 1.0)
        )
        result: tuple[str, float, bool] | None = None
        if "int4" in dtype_norm:
            if self.int4_flops > 0.0 and act == "a8":
                result = ("int4", self.int4_flops, False)
        elif is_fp4:
            if self.fp4_flops > 0.0 and act == "a8":
                result = ("fp4", self.fp4_flops, False)
        elif "int8" in dtype_norm:
            if self.int8_flops > 0.0 and act == "a8":
                result = ("int8", self.int8_flops, False)
            else:
                result = ("bf16", self.bf16_flops, False)
        if result is not None:
            return result
        return self._fallback_matmul_peak(weight_bytes, act)

    def catalog(self) -> dict[str, dict[str, object]]:
        """Return the structured constants catalog for this ChipSpec."""
        return chip_catalog(self)


def _format_byte_count(v: float) -> str:
    """Format byte count using GB, MiB, KiB, or B units."""
    if v >= UNIT_GIGA:
        return f"{v / UNIT_GIGA:.1f} GB"
    if v >= BIN_MEBI:
        return f"{v / BIN_MEBI:.1f} MiB"
    if v >= BIN_KIBI:
        return f"{v / BIN_KIBI:.1f} KiB"
    return f"{int(v)} B"


def _format_bandwidth_rate(v: float) -> str:
    """Format bandwidth in TB/s, GB/s, MB/s, or B/s."""
    if v >= UNIT_TERA:
        return f"{v / UNIT_TERA:.2f} TB/s"
    if v >= UNIT_GIGA:
        return f"{v / UNIT_GIGA:.1f} GB/s"
    if v >= UNIT_MEGA:
        return f"{v / UNIT_MEGA:.1f} MB/s"
    return f"{v:.1f} B/s"


def _format_flops_rate(v: float) -> str:
    """Format compute throughput in TFLOP/s, GFLOP/s, or FLOP/s."""
    tflops = v / UNIT_TERA
    if tflops >= UNIT_KILO and tflops != int(tflops):
        return f"{tflops:,.1f} TFLOP/s"
    if tflops >= 1.0:
        return f"{tflops:,.0f} TFLOP/s"
    if v >= UNIT_GIGA:
        return f"{v / UNIT_GIGA:,.0f} GFLOP/s"
    return f"{v:,.0f} FLOP/s"


def _format_duration_s(v: float) -> str:
    """Format duration in seconds using us, ms, or s units."""
    if v < UNIT_MILLI:
        return f"{v * UNIT_MEGA:.1f} us"
    if v < 1.0:
        return f"{v * UNIT_KILO:.1f} ms"
    return f"{v:.2f} s"


def format_constant_display(value: float, unit: str) -> str:
    """Format numeric constants into clean human-readable strings."""
    numeric_value = float(value)
    dispatch = {
        "B": _format_byte_count,
        "B/s": _format_bandwidth_rate,
        "FLOP/s": _format_flops_rate,
        "s": _format_duration_s,
    }
    if unit in dispatch:
        return dispatch[unit](numeric_value)
    if unit == "Gbit/s":
        return f"{numeric_value:.1f} Gbit/s"
    if unit == "ratio":
        return f"{numeric_value * PERCENT_SCALE:.0f}% ({numeric_value:.2f})"
    if unit in ("cores", "chips", "rows", "tokens"):
        return f"{int(value)} {unit}"
    return f"{value} {unit}".strip()


def _spec_dict(spec: ConstantSpec) -> dict[str, object]:
    """Convert ConstantSpec to an untyped object dictionary without Any."""
    return dict(asdict(spec).items())


def chip_catalog(chip: ChipSpec) -> dict[str, dict[str, object]]:
    """Build structured constants catalog with value, unit, scope, and source."""
    host_chips = float(max(1, chip.chips_per_host))
    dcn_low = (DCN_BRACKET_GBIT_PER_S[0] * UNIT_GIGA / BITS_PER_BYTE) / host_chips
    dcn_high = (DCN_BRACKET_GBIT_PER_S[1] * UNIT_GIGA / BITS_PER_BYTE) / host_chips
    src_prefix = f"{chip.name} ({chip.provenance})"
    entries: dict[str, ConstantSpec] = {
        "hbm_bytes": ConstantSpec(
            value=chip.hbm_bytes,
            unit="B",
            scope="core",
            kind="fact",
            source=f"{src_prefix}: per-core HBM capacity",
            display=format_constant_display(chip.hbm_bytes, "B"),
            description="High-Bandwidth Memory (HBM) physical capacity per TensorCore",
        ),
        "hbm_bw": ConstantSpec(
            value=chip.hbm_bw,
            unit="B/s",
            scope="core",
            kind="fact",
            source=f"{src_prefix}: per-core HBM bandwidth",
            display=format_constant_display(chip.hbm_bw, "B/s"),
            description="Peak HBM memory bandwidth per TensorCore",
        ),
        "bf16_flops": ConstantSpec(
            value=chip.bf16_flops,
            unit="FLOP/s",
            scope="core",
            kind="fact",
            source=f"{src_prefix}: per-core BF16 MXU peak",
            display=format_constant_display(chip.bf16_flops, "FLOP/s"),
            description="Peak BF16 matrix-multiply throughput per TensorCore",
        ),
        "fp8_flops": ConstantSpec(
            value=chip.fp8_flops,
            unit="FLOP/s",
            scope="core",
            kind="fact",
            source=f"{src_prefix}: per-core FP8 MXU peak",
            display=format_constant_display(chip.fp8_flops, "FLOP/s"),
            description="Peak FP8 matrix-multiply throughput per TensorCore",
        ),
        "vmem_bytes": ConstantSpec(
            value=chip.vmem_bytes,
            unit="B",
            scope="core",
            kind="fact",
            source=f"{src_prefix}: per-core VMEM scratchpad capacity",
            display=format_constant_display(chip.vmem_bytes, "B"),
            description="On-chip Vector Memory (VMEM / SRAM) capacity per TensorCore",
        ),
        "cores_per_chip": ConstantSpec(
            value=chip.cores_per_chip,
            unit="cores",
            scope="chip",
            kind="fact",
            source=f"{src_prefix}: physical TensorCores per chip",
            display=format_constant_display(chip.cores_per_chip, "cores"),
            description="Physical TensorCores sharing a single TPU chip",
        ),
        "ici_link_bw": ConstantSpec(
            value=chip.ici_link_bw,
            unit="B/s",
            scope="link",
            kind="derived",
            source=f"{src_prefix}: unidirectional ICI link bandwidth",
            display=format_constant_display(chip.ici_link_bw, "B/s"),
            description=(
                "Unidirectional bandwidth per optical Inter-Chip Interconnect "
                "(ICI) link"
            ),
        ),
        "d2d_bw": ConstantSpec(
            value=chip.d2d_bw,
            unit="B/s",
            scope="chip",
            kind="derived",
            source=(
                f"{src_prefix}: dedicated bridge approximated numerically as "
                f"{chip.ici_links_per_chip} * ici_link_bw"
                + (" (cloud.google.com/tpu/docs/tpu7x)" if chip.name == "tpu7x" else "")
            ),
            display=format_constant_display(chip.d2d_bw, "B/s"),
            description=(
                "Dedicated on-chip die-to-die bridge bandwidth (approximated "
                f"numerically as {chip.ici_links_per_chip} * ici_link_bw)"
            ),
        ),
        "ici_bw": ConstantSpec(
            value=chip.ici_bw,
            unit="B/s",
            scope="core",
            kind="derived",
            source=(
                f"{src_prefix}: {chip.ici_links_per_chip} * ici_link_bw / "
                "cores_per_chip per-core ICI share"
                + (" (cloud.google.com/tpu/docs/tpu7x)" if chip.name == "tpu7x" else "")
            ),
            display=format_constant_display(chip.ici_bw, "B/s"),
            description=(
                "Per-TensorCore fair share of aggregate ICI interconnect bandwidth"
            ),
        ),
        "dcn_bw": ConstantSpec(
            value=chip.dcn_bw,
            unit="B/s",
            scope="chip",
            kind="derived",
            source=(
                f"{src_prefix}: dcn_host_gbit_per_s * 1e9 / 8 / "
                f"{chip.chips_per_host} chips"
            ),
            bracket=(dcn_low, dcn_high),
            display=(
                f"{chip.dcn_bw / UNIT_GIGA:.1f} GB/s "
                f"({chip.dcn_bw * BITS_PER_BYTE / UNIT_GIGA:.0f} Gbit/s)"
            ),
            description=(
                "Per-chip slice of datacenter network (DCN) host NIC bandwidth "
                "for disaggregated KV transfer"
            ),
        ),
        "ici_hop_latency_s": ConstantSpec(
            value=chip.ici_hop_latency_s,
            unit="s",
            scope="link",
            kind="assumed",
            source=(
                f"{src_prefix}: assumed per-hop latency used by serial estimate only, "
                "never by hard floor"
            ),
            bracket=ICI_HOP_LATENCY_BRACKET_S,
            display=format_constant_display(chip.ici_hop_latency_s, "s"),
            description=(
                "Unloaded point-to-point transmission latency per physical ICI "
                "torus hop"
            ),
        ),
        "pcie_bw": ConstantSpec(
            value=chip.pcie_bw,
            unit="B/s",
            scope="chip",
            kind="assumed",
            source=(
                f"{src_prefix}: host-to-device PCIe bandwidth "
                "(effective rate uncalibrated)"
            ),
            bracket=PCIE_BRACKET_BW,
            display=format_constant_display(chip.pcie_bw, "B/s"),
            description="Peak PCIe host-to-device interconnect bandwidth per chip",
        ),
        "mxu_m_tile": ConstantSpec(
            value=chip.mxu_m_tile,
            unit="rows",
            scope="core",
            kind="assumed",
            source=(
                f"{src_prefix}: assumed MoE kernel M-tile multiple on "
                f"{chip.mxu_m_tile}-wide MXU"
            ),
            bracket=MXU_M_TILE_BRACKET,
            display=format_constant_display(chip.mxu_m_tile, "rows"),
            description=(
                "Hardware Matrix Multiply Unit (MXU) systolic array row granularity "
                "tile floor"
            ),
        ),
        "dcn_host_gbit_per_s": ConstantSpec(
            value=chip.dcn_host_gbit_per_s,
            unit="Gbit/s",
            scope="host",
            kind="assumed",
            source=(
                f"{src_prefix}: host NIC bandwidth across "
                f"{chip.chips_per_host} chips per host"
            ),
            bracket=DCN_BRACKET_GBIT_PER_S,
            display=format_constant_display(chip.dcn_host_gbit_per_s, "Gbit/s"),
            description=(
                "Total host-level network interface card (NIC) bandwidth shared "
                "across host chips"
            ),
        ),
        "chips_per_host": ConstantSpec(
            value=chip.chips_per_host,
            unit="chips",
            scope="host",
            kind="assumed",
            source=f"{src_prefix}: {chip.chips_per_host} chips per host tray",
            bracket=CHIP_COUNT_BRACKET,
            display=format_constant_display(chip.chips_per_host, "chips"),
            description=(
                "Number of TPU chips co-located on a single host tray / CPU subsystem"
            ),
        ),
        "page_table_byte_budget": ConstantSpec(
            value=chip.smem_page_table_bytes,
            unit="B",
            scope="core",
            kind="assumed",
            source=(f"{src_prefix}: on-chip SMEM page-table scratch budget per core"),
            bracket=PAGE_TABLE_BRACKET_BYTES,
            display=format_constant_display(chip.smem_page_table_bytes, "B"),
            description=(
                "Reserved on-chip fast memory scratchpad budget for Pallas "
                "block-table lookups"
            ),
        ),
        "rpa_page_size_tokens": ConstantSpec(
            value=RPA_PAGE_SIZE_TOKENS,
            unit="tokens",
            scope="core",
            kind="assumed",
            source=(
                "Assumed default RPA KV page granularity constant "
                "(64 tokens; vllm_torchtpu.roofline.constants)"
            ),
            bracket=RPA_PAGE_SIZE_BRACKET_TOKENS,
            display=format_constant_display(RPA_PAGE_SIZE_TOKENS, "tokens"),
            description=(
                "Paged KV-cache token page granularity for Ring-Paged-Attention (RPA)"
            ),
        ),
        "default_gpu_memory_utilization": ConstantSpec(
            value=DEFAULT_GPU_MEMORY_UTILIZATION,
            unit="ratio",
            scope="core",
            kind="config",
            source=("Roofline default 0.92 (vLLM CacheConfig default is 0.90)"),
            bracket=GPU_MEM_UTIL_BRACKET,
            display=format_constant_display(DEFAULT_GPU_MEMORY_UTILIZATION, "ratio"),
            description=(
                "Default peak HBM memory allocation fraction reserved for weights "
                "and KV cache"
            ),
        ),
        "vpu_flops": ConstantSpec(
            value=chip.vpu_flops,
            unit="FLOP/s",
            scope="core",
            kind="derived",
            source=(
                f"{src_prefix}: vector processing throughput per TensorCore "
                f"({chip.vpu_lanes}x{chip.vpu_sublanes} vector layout)"
            ),
            bracket=(
                chip.bf16_flops * VPU_TO_BF16_RATIO_BRACKET[0],
                chip.bf16_flops * VPU_TO_BF16_RATIO_BRACKET[1],
            ),
            display=format_constant_display(chip.vpu_flops, "FLOP/s"),
            description=(
                "Peak vector processing unit (VPU) arithmetic throughput per TensorCore"
            ),
        ),
        "mxu_lhs_rows_bf16": ConstantSpec(
            value=chip.mxu_lhs_rows_bf16,
            unit="rows",
            scope="core",
            kind="fact",
            source=f"{src_prefix}: MXU BF16 LHS sublane tiling floor",
            display=format_constant_display(chip.mxu_lhs_rows_bf16, "rows"),
            description=(
                "Hardware Matrix Multiply Unit (MXU) systolic array "
                "LHS row floor for BF16"
            ),
        ),
        "mxu_lhs_rows_fp8": ConstantSpec(
            value=chip.mxu_lhs_rows_fp8,
            unit="rows",
            scope="core",
            kind="fact",
            source=f"{src_prefix}: MXU FP8 LHS sublane tiling floor",
            display=format_constant_display(chip.mxu_lhs_rows_fp8, "rows"),
            description=(
                "Hardware Matrix Multiply Unit (MXU) systolic array "
                "LHS row floor for FP8"
            ),
        ),
        "dma_granule_bytes": ConstantSpec(
            value=chip.dma_granule_bytes,
            unit="B",
            scope="core",
            kind="fact",
            source=f"{src_prefix}: SparseCore DMA granule size",
            display=format_constant_display(chip.dma_granule_bytes, "B"),
            description=("Minimum SparseCore DMA transfer alignment granule in bytes"),
        ),
        "has_sparse_core": ConstantSpec(
            value=chip.has_sparse_core,
            unit="",
            scope="chip",
            kind="fact",
            source=f"{src_prefix}: SparseCore hardware capability",
            display="true" if chip.has_sparse_core else "false",
            description="Whether the chip includes dedicated SparseCore offload units",
        ),
    }
    return {
        entry_name: _spec_dict(constant_spec)
        for entry_name, constant_spec in entries.items()
    }


@dataclass(frozen=True, slots=True)
class _JaxChipCapabilities:
    """Raw hardware capability fields read from JAX or stdlib fallback tables."""

    hbm_bytes: int
    hbm_bw: float
    bf16_flops: float
    fp8_flops: float
    vmem_bytes: int
    cores_per_chip: int
    int8_flops: float
    int4_flops: float
    mxu_m_tile: int
    vpu_lanes: int
    vpu_sublanes: int
    smem_page_table_bytes: int
    dma_granule_bytes: int
    has_sparse_core: bool
    provenance: str


def _read_jax_capabilities(name: str, version: object) -> _JaxChipCapabilities:
    """Read per-TensorCore and chip capabilities from JAX or stdlib fallback."""
    if pltpu is None or isinstance(version, str):
        if name not in FALLBACK_TPU_SPECS:
            msg = f"Unknown fallback TPU chip {name!r}"
            raise KeyError(msg)
        hbm_b, hbm_bw, bf16, fp8_flops, vmem_b = FALLBACK_TPU_SPECS[name]
        return _JaxChipCapabilities(
            hbm_bytes=hbm_b,
            hbm_bw=hbm_bw,
            bf16_flops=bf16,
            fp8_flops=fp8_flops,
            vmem_bytes=vmem_b,
            cores_per_chip=DEFAULT_CORES_PER_CHIP,
            int8_flops=0.0,
            int4_flops=0.0,
            mxu_m_tile=DEFAULT_MXU_M_TILE,
            vpu_lanes=DEFAULT_VPU_LANES,
            vpu_sublanes=DEFAULT_VPU_SUBLANES,
            smem_page_table_bytes=PAGE_TABLE_BYTE_BUDGET,
            dma_granule_bytes=DEFAULT_DMA_GRANULE_BYTES,
            has_sparse_core=True,
            provenance="public_spec",
        )
    # Querying with logical cores_per_chip=1 returns per-TensorCore HBM,
    # bandwidth, MXU peaks, and VMEM; ChipOverrides supplies ICI/DCN/PCIe rates.
    info = pltpu.get_tpu_info_for_chip(version, 1)
    cores = (
        int(version.num_physical_tensor_cores_per_chip)
        if isinstance(version, _ChipVersionProto)
        and version.num_physical_tensor_cores_per_chip > 0
        else DEFAULT_CORES_PER_CHIP
    )
    mxu_tile = DEFAULT_MXU_M_TILE
    vpu_lanes = DEFAULT_VPU_LANES
    vpu_sublanes = DEFAULT_VPU_SUBLANES
    if isinstance(info, _TpuInfoTilingProto):
        mxu_tile = info.mxu_column_size if info.mxu_column_size > 0 else mxu_tile
        vpu_lanes = info.num_lanes if info.num_lanes > 0 else vpu_lanes
        vpu_sublanes = info.num_sublanes if info.num_sublanes > 0 else vpu_sublanes
    smem_budget = PAGE_TABLE_BYTE_BUDGET
    if (
        isinstance(info, _TpuInfoSmemProto)
        and info.smem_capacity_bytes >= _SMEM_PAGE_TABLE_DIVISOR
    ):
        smem_budget = int(info.smem_capacity_bytes) // _SMEM_PAGE_TABLE_DIVISOR
    int8_flops = 0.0
    int4_flops = 0.0
    if isinstance(info, _TpuInfoIntOpsProto):
        int8_flops = float(info.int8_ops_per_second)
        int4_flops = float(info.int4_ops_per_second)
    has_sparse_core = True
    dma_granule = DEFAULT_DMA_GRANULE_BYTES
    if isinstance(info, _TpuInfoSparseCoreProto):
        sparse_core = info.sparse_core
        has_sparse_core = sparse_core is not None
        if (
            isinstance(sparse_core, _SparseCoreProto)
            and sparse_core.dma_granule_size_bytes > 0
        ):
            dma_granule = int(sparse_core.dma_granule_size_bytes)
    return _JaxChipCapabilities(
        hbm_bytes=int(info.hbm_capacity_bytes),
        hbm_bw=float(info.mem_bw_bytes_per_second),
        bf16_flops=float(info.bf16_ops_per_second),
        fp8_flops=float(info.fp8_ops_per_second),
        vmem_bytes=int(info.vmem_capacity_bytes),
        cores_per_chip=cores,
        int8_flops=int8_flops,
        int4_flops=int4_flops,
        mxu_m_tile=mxu_tile,
        vpu_lanes=vpu_lanes,
        vpu_sublanes=vpu_sublanes,
        smem_page_table_bytes=smem_budget,
        dma_granule_bytes=dma_granule,
        has_sparse_core=has_sparse_core,
        provenance="jax_tpu_info",
    )


def _build_chip_from_jax(
    name: str,
    version: object,
    overrides_cfg: ChipOverrides,
) -> ChipSpec:
    """Assemble ChipSpec by merging JAX capabilities with explicit overrides."""
    caps = _read_jax_capabilities(name, version)
    eff_cores = (
        overrides_cfg.cores_per_chip
        if overrides_cfg.cores_per_chip is not None
        else caps.cores_per_chip
    )
    eff_links = (
        overrides_cfg.ici_links_per_chip
        if overrides_cfg.ici_links_per_chip is not None
        else DEFAULT_ICI_LINKS_PER_CHIP
    )
    eff_d2d_bw = (
        overrides_cfg.d2d_bw
        if overrides_cfg.d2d_bw is not None
        else float(eff_links) * overrides_cfg.ici_link_bw
    )
    eff_ici_bw = (
        overrides_cfg.ici_bw
        if overrides_cfg.ici_bw is not None
        else (float(eff_links) * overrides_cfg.ici_link_bw / float(max(1, eff_cores)))
    )
    eff_dcn_bw = (
        overrides_cfg.dcn_bw
        if overrides_cfg.dcn_bw is not None
        else (overrides_cfg.dcn_host_gbit_per_s * UNIT_GIGA / BITS_PER_BYTE)
        / float(max(1, overrides_cfg.chips_per_host))
    )
    eff_bf16 = (
        overrides_cfg.bf16_flops
        if overrides_cfg.bf16_flops is not None
        else caps.bf16_flops
    )
    eff_ratio = (
        overrides_cfg.vpu_to_bf16_ratio
        if overrides_cfg.vpu_to_bf16_ratio is not None
        else DEFAULT_VPU_TO_BF16_RATIO
    )
    eff_vpu = (
        overrides_cfg.vpu_flops
        if overrides_cfg.vpu_flops is not None
        else eff_bf16 * eff_ratio
    )
    return ChipSpec(
        name=name,
        hbm_bytes=(
            overrides_cfg.hbm_bytes
            if overrides_cfg.hbm_bytes is not None
            else caps.hbm_bytes
        ),
        hbm_bw=(
            overrides_cfg.hbm_bw if overrides_cfg.hbm_bw is not None else caps.hbm_bw
        ),
        bf16_flops=eff_bf16,
        fp8_flops=(
            overrides_cfg.fp8_flops
            if overrides_cfg.fp8_flops is not None
            else caps.fp8_flops
        ),
        vmem_bytes=(
            overrides_cfg.vmem_bytes
            if overrides_cfg.vmem_bytes is not None
            else caps.vmem_bytes
        ),
        cores_per_chip=eff_cores,
        ici_link_bw=overrides_cfg.ici_link_bw,
        d2d_bw=eff_d2d_bw,
        ici_bw=eff_ici_bw,
        ici_hop_latency_s=overrides_cfg.ici_hop_latency_s,
        mxu_m_tile=(
            overrides_cfg.mxu_m_tile
            if overrides_cfg.mxu_m_tile is not None
            else caps.mxu_m_tile
        ),
        dcn_host_gbit_per_s=overrides_cfg.dcn_host_gbit_per_s,
        chips_per_host=overrides_cfg.chips_per_host,
        dcn_bw=eff_dcn_bw,
        pcie_bw=overrides_cfg.pcie_bw,
        vpu_flops=eff_vpu,
        mxu_lhs_rows_bf16=(
            overrides_cfg.mxu_lhs_rows_bf16
            if overrides_cfg.mxu_lhs_rows_bf16 is not None
            else DEFAULT_MXU_LHS_ROWS_BF16
        ),
        mxu_lhs_rows_fp8=(
            overrides_cfg.mxu_lhs_rows_fp8
            if overrides_cfg.mxu_lhs_rows_fp8 is not None
            else DEFAULT_MXU_LHS_ROWS_FP8
        ),
        int8_flops=(
            overrides_cfg.int8_flops
            if overrides_cfg.int8_flops is not None
            else caps.int8_flops
        ),
        fp4_flops=(
            overrides_cfg.fp4_flops if overrides_cfg.fp4_flops is not None else 0.0
        ),
        int4_flops=(
            overrides_cfg.int4_flops
            if overrides_cfg.int4_flops is not None
            else caps.int4_flops
        ),
        ici_links_per_chip=eff_links,
        min_torus_dim=(
            overrides_cfg.min_torus_dim
            if overrides_cfg.min_torus_dim is not None
            else MIN_TORUS_DIM
        ),
        vpu_lanes=(
            overrides_cfg.vpu_lanes
            if overrides_cfg.vpu_lanes is not None
            else caps.vpu_lanes
        ),
        vpu_sublanes=(
            overrides_cfg.vpu_sublanes
            if overrides_cfg.vpu_sublanes is not None
            else caps.vpu_sublanes
        ),
        physical_alignment_elems=(
            overrides_cfg.physical_alignment_elems
            if overrides_cfg.physical_alignment_elems is not None
            else PHYSICAL_ALIGNMENT_ELEMS
        ),
        smem_page_table_bytes=(
            overrides_cfg.smem_page_table_bytes
            if overrides_cfg.smem_page_table_bytes is not None
            else caps.smem_page_table_bytes
        ),
        dma_granule_bytes=(
            overrides_cfg.dma_granule_bytes
            if overrides_cfg.dma_granule_bytes is not None
            else caps.dma_granule_bytes
        ),
        subchannel_block_k_floor=(
            overrides_cfg.subchannel_block_k_floor
            if overrides_cfg.subchannel_block_k_floor is not None
            else SUBCHANNEL_BLOCK_K_FLOOR
        ),
        has_sparse_core=(
            overrides_cfg.has_sparse_core
            if overrides_cfg.has_sparse_core is not None
            else caps.has_sparse_core
        ),
        provenance=(
            overrides_cfg.provenance
            if overrides_cfg.provenance is not None
            else caps.provenance
        ),
    )


class _FromJaxOverrides(TypedDict, total=False):
    """Optional legacy keyword overrides accepted by from_jax."""

    cores_per_chip: int | None
    ici_link_bw: float | None
    ici_bw: float | None
    ici_hop_latency_s: float | None


def from_jax(
    name: str,
    version: object,
    overrides: ChipOverrides | None = None,
    **legacy_kwargs: Unpack[_FromJaxOverrides],
) -> ChipSpec:
    """Construct ChipSpec from JAX TPU info table or stdlib fallback table."""
    overrides_cfg = overrides or ChipOverrides()
    cores_per_chip = legacy_kwargs.get("cores_per_chip")
    ici_link_bw = legacy_kwargs.get("ici_link_bw")
    ici_bw = legacy_kwargs.get("ici_bw")
    ici_hop_latency_s = legacy_kwargs.get("ici_hop_latency_s")
    if (
        cores_per_chip is not None
        or ici_link_bw is not None
        or ici_bw is not None
        or ici_hop_latency_s is not None
    ):
        overrides_cfg = replace(
            overrides_cfg,
            cores_per_chip=(
                cores_per_chip
                if cores_per_chip is not None
                else overrides_cfg.cores_per_chip
            ),
            ici_link_bw=(
                ici_link_bw if ici_link_bw is not None else overrides_cfg.ici_link_bw
            ),
            ici_bw=ici_bw if ici_bw is not None else overrides_cfg.ici_bw,
            ici_hop_latency_s=(
                ici_hop_latency_s
                if ici_hop_latency_s is not None
                else overrides_cfg.ici_hop_latency_s
            ),
        )
    return _build_chip_from_jax(name, version, overrides_cfg)


def count_hosts(
    num_chips: int,
    chip: ChipSpec | None = None,
    *,
    chips_per_host: int | None = None,
) -> int:
    """Return physical host count for ``num_chips`` using ceiling division."""
    per_host = (
        chips_per_host
        if chips_per_host is not None
        else (chip.chips_per_host if chip is not None else DEFAULT_CHIPS_PER_HOST)
    )
    per_host = max(1, per_host)
    return max(1, (num_chips + per_host - 1) // per_host)


@dataclass(frozen=True, slots=True)
class SliceTopology:
    """Physical 3D chip slice topology and torus wraparound rules."""

    name: str
    chip_axes: tuple[int, int, int]
    cores_per_chip: int = DEFAULT_CORES_PER_CHIP
    wrapped_axes: tuple[bool, bool, bool] | None = None
    min_torus_dim: int = MIN_TORUS_DIM

    def __post_init__(self) -> None:
        """Validate 3D axis and wraparound tuple lengths."""
        if len(self.chip_axes) != _TORUS_AXES_LEN:
            msg = (
                f"chip_axes must have length {_TORUS_AXES_LEN}, "
                f"got {len(self.chip_axes)}"
            )
            raise ValueError(msg)
        if self.wrapped_axes is not None and len(self.wrapped_axes) != _TORUS_AXES_LEN:
            msg = (
                f"wrapped_axes must have length {_TORUS_AXES_LEN}, "
                f"got {len(self.wrapped_axes)}"
            )
            raise ValueError(msg)

    @property
    def num_chips(self) -> int:
        """Total number of physical chips in the 3D topology."""
        axis_x, axis_y, axis_z = self.chip_axes
        return axis_x * axis_y * axis_z

    @property
    def num_cores(self) -> int:
        """Total number of TensorCores across all chips in the topology."""
        return self.num_chips * self.cores_per_chip

    @property
    def has_torus_wraparound(self) -> bool:
        """True when physical chip axes have torus wraparounds."""
        if self.wrapped_axes is not None:
            return any(self.wrapped_axes)
        # Google Cloud TPU7x doc slice table (https://cloud.google.com/tpu/docs/tpu7x):
        # "AxBxC where A, B, and C are multiples of 4" by default (overridable via
        # min_torus_dim).
        return all(
            dimension_size >= self.min_torus_dim
            and dimension_size % self.min_torus_dim == 0
            for dimension_size in self.chip_axes
        )

    def axis_wrapped(self, axis: int, submesh_extent: int) -> bool:
        """True when a collective spans the full extent of a wrapped physical axis."""
        if self.wrapped_axes is not None:
            return (
                self.wrapped_axes[axis]
                and submesh_extent == self.chip_axes[axis]
                and submesh_extent > 1
            )
        return (
            self.has_torus_wraparound
            and submesh_extent == self.chip_axes[axis]
            and submesh_extent >= self.min_torus_dim
        )


CHIPS: dict[str, ChipSpec] = {
    "tpu7x": from_jax(
        "tpu7x",
        _TPU_7X_VERSION,
        ChipOverrides(),
    ),
}

# Live read-only set view over CHIPS so register_chip stays synchronized globally.
SUPPORTED_CHIPS: KeysView[str] = CHIPS.keys()

TOPOLOGIES: dict[str, SliceTopology] = {
    name: SliceTopology(name, axes) for name, axes in TOPOLOGY_AXES.items()
}

CHIP_TOPOLOGIES: dict[str, dict[str, SliceTopology]] = {
    "tpu7x": TOPOLOGIES,
}


def topologies_for_chip(
    chip: str | ChipSpec,
    *,
    fallback: Mapping[str, SliceTopology] | None = None,
) -> dict[str, SliceTopology]:
    """Return the topology map for a chip name or ChipSpec."""
    chip_name = chip.name if isinstance(chip, ChipSpec) else chip
    if chip_name in CHIP_TOPOLOGIES:
        return dict(CHIP_TOPOLOGIES[chip_name])
    if fallback is not None:
        return dict(fallback)
    if isinstance(chip, ChipSpec):
        return {
            topo_name: replace(
                topo,
                cores_per_chip=chip.cores_per_chip,
                min_torus_dim=chip.min_torus_dim,
            )
            for topo_name, topo in TOPOLOGIES.items()
        }
    return dict(TOPOLOGIES)


def register_chip(
    chip: ChipSpec,
    topologies: Mapping[str, SliceTopology],
) -> None:
    """Register a ChipSpec and topology map for global and SUPPORTED_CHIPS lookup."""
    if not chip.name:
        msg = "ChipSpec.name must be non-empty"
        raise ValueError(msg)
    if not topologies:
        msg = f"Chip {chip.name!r} requires at least one SliceTopology"
        raise ValueError(msg)
    CHIPS[chip.name] = chip
    CHIP_TOPOLOGIES[chip.name] = dict(topologies)


def _read_spec_mapping(
    source: str | Path | Mapping[str, object],
) -> Mapping[str, object]:
    """Load a chip specification JSON path or validate an in-memory mapping."""
    if isinstance(source, (str, Path)):
        loaded = json.loads(Path(source).read_text(encoding="utf-8"))
        if not isinstance(loaded, Mapping):
            msg = "Chip spec JSON root must be an object"
            raise TypeError(msg)
        return loaded
    return source


def _parse_spec_int(
    raw: Mapping[str, object],
    key: str,
    *,
    default: int | None = None,
) -> int:
    """Extract a strictly positive integer from a chip spec mapping."""
    val = raw.get(key)
    if val is None:
        if default is not None:
            return default
        msg = f"Chip spec missing required field {key!r}"
        raise ValueError(msg)
    if isinstance(val, bool) or not isinstance(val, int) or val <= 0:
        msg = f"Chip spec field {key!r} must be a positive integer, got {val!r}"
        raise ValueError(msg)
    return int(val)


def _parse_spec_float(
    raw: Mapping[str, object],
    key: str,
    *,
    default: float | None = None,
    allow_zero: bool = False,
) -> float:
    """Extract a positive (or non-negative) finite float from a chip spec."""
    val = raw.get(key)
    if val is None:
        if default is not None:
            return default
        msg = f"Chip spec missing required field {key!r}"
        raise ValueError(msg)
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        msg = f"Chip spec field {key!r} must be numeric, got {val!r}"
        raise TypeError(msg)
    num = float(val)
    if not math.isfinite(num) or num < 0.0 or (not allow_zero and num <= 0.0):
        bound = ">= 0" if allow_zero else "> 0"
        msg = f"Chip spec field {key!r} must be finite and {bound}, got {val!r}"
        raise ValueError(msg)
    return num


def _parse_spec_bool(
    raw: Mapping[str, object],
    key: str,
    *,
    default: bool,
) -> bool:
    """Extract a boolean capability flag from a chip spec mapping."""
    val = raw.get(key)
    if val is None:
        return default
    if not isinstance(val, bool):
        msg = f"Chip spec field {key!r} must be a boolean, got {val!r}"
        raise TypeError(msg)
    return val


def _parse_one_topology(
    entry: object,
    cores_per_chip: int,
    min_torus_dim: int = MIN_TORUS_DIM,
) -> SliceTopology:
    """Parse one topology entry dict into a SliceTopology instance."""
    if not isinstance(entry, Mapping):
        msg = f"Chip spec topology entry must be an object, got {entry!r}"
        raise TypeError(msg)
    topo_name = entry.get("name")
    if not isinstance(topo_name, str) or not topo_name.strip():
        msg = f"Chip spec topology requires non-empty string 'name', got {topo_name!r}"
        raise ValueError(msg)
    axes_raw = entry.get("chip_axes")
    if (
        not isinstance(axes_raw, (list, tuple))
        or len(axes_raw) != _TORUS_AXES_LEN
        or any(
            isinstance(a, bool) or not isinstance(a, int) or a <= 0 for a in axes_raw
        )
    ):
        msg = (
            f"Topology {topo_name!r} 'chip_axes' must be 3 positive ints, "
            f"got {axes_raw!r}"
        )
        raise ValueError(msg)
    wrapped_raw = entry.get("wrapped_axes")
    wrapped_axes: tuple[bool, bool, bool] | None = None
    if wrapped_raw is not None:
        if (
            not isinstance(wrapped_raw, (list, tuple))
            or len(wrapped_raw) != _TORUS_AXES_LEN
            or any(not isinstance(w, bool) for w in wrapped_raw)
        ):
            msg = (
                f"Topology {topo_name!r} 'wrapped_axes' must be 3 bools, "
                f"got {wrapped_raw!r}"
            )
            raise ValueError(msg)
        wrapped_axes = (
            bool(wrapped_raw[0]),
            bool(wrapped_raw[1]),
            bool(wrapped_raw[2]),
        )
    return SliceTopology(
        name=topo_name,
        chip_axes=(int(axes_raw[0]), int(axes_raw[1]), int(axes_raw[2])),
        cores_per_chip=cores_per_chip,
        wrapped_axes=wrapped_axes,
        min_torus_dim=min_torus_dim,
    )


def _parse_spec_topologies(
    raw: Mapping[str, object],
    cores_per_chip: int,
    base_name: str | None,
    min_torus_dim: int = MIN_TORUS_DIM,
) -> dict[str, SliceTopology]:
    """Parse topology definitions or derive defaults when omitted."""
    topos_raw = raw.get("topologies")
    if topos_raw is None:
        if base_name is not None:
            return {
                t_name: replace(
                    topo,
                    cores_per_chip=cores_per_chip,
                    min_torus_dim=min_torus_dim,
                )
                for t_name, topo in topologies_for_chip(base_name).items()
            }
        return {
            "1x1": SliceTopology(
                name="1x1",
                chip_axes=(1, 1, 1),
                cores_per_chip=cores_per_chip,
                wrapped_axes=(False, False, False),
                min_torus_dim=min_torus_dim,
            )
        }
    if not isinstance(topos_raw, (list, tuple)) or not topos_raw:
        msg = "Chip spec 'topologies' must be a non-empty list when provided"
        raise ValueError(msg)
    parsed = [
        _parse_one_topology(entry, cores_per_chip, min_torus_dim) for entry in topos_raw
    ]
    return {topo.name: topo for topo in parsed}


def _resolve_base_chip(raw: Mapping[str, object]) -> tuple[str | None, ChipSpec | None]:
    """Validate optional 'base' chip reference and enforce schema/provenance."""
    schema = raw.get("schema_version")
    if schema != CHIP_SPEC_SCHEMA:
        msg = (
            f"Unsupported chip spec schema_version {schema!r}; "
            f"expected {CHIP_SPEC_SCHEMA!r}"
        )
        raise ValueError(msg)
    provenance = raw.get("provenance")
    if not isinstance(provenance, str) or provenance not in VALID_CHIP_PROVENANCE:
        allowed = sorted(VALID_CHIP_PROVENANCE)
        msg = f"Invalid chip spec provenance {provenance!r}; expected one of {allowed}"
        raise ValueError(msg)
    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        msg = f"Chip spec field 'name' must be a non-empty string, got {name!r}"
        raise ValueError(msg)
    base_raw = raw.get("base")
    if base_raw is None:
        return None, None
    if not isinstance(base_raw, str) or base_raw not in CHIPS:
        msg = f"Unknown base chip {base_raw!r} in chip spec {name!r}"
        raise ValueError(msg)
    return base_raw, CHIPS[base_raw]


def _resolve_spec_vpu_flops(
    raw: Mapping[str, object],
    bf16_flops: float,
    base_chip: ChipSpec | None,
) -> float:
    """Resolve and validate per-core VPU FLOP/s for a chip spec."""
    if "vpu_flops" in raw and raw["vpu_flops"] is not None:
        return _parse_spec_float(raw, "vpu_flops", allow_zero=False)
    if "vpu_to_bf16_ratio" in raw and raw["vpu_to_bf16_ratio"] is not None:
        ratio = _parse_spec_float(raw, "vpu_to_bf16_ratio", allow_zero=False)
        return bf16_flops * ratio
    if base_chip is not None and base_chip.bf16_flops > 0.0:
        ratio = base_chip.vpu_flops / base_chip.bf16_flops
        return bf16_flops * ratio
    return bf16_flops * DEFAULT_VPU_TO_BF16_RATIO


def _build_spec_chip(
    raw: Mapping[str, object],
    base_chip: ChipSpec | None,
) -> ChipSpec:
    """Build a validated ChipSpec instance from a parsed JSON mapping."""
    name = str(raw["name"]).strip()
    provenance = str(raw["provenance"])
    hbm_bytes = _parse_spec_int(
        raw, "hbm_bytes", default=base_chip.hbm_bytes if base_chip else None
    )
    hbm_bw = _parse_spec_float(
        raw, "hbm_bw", default=base_chip.hbm_bw if base_chip else None
    )
    bf16_flops = _parse_spec_float(
        raw, "bf16_flops", default=base_chip.bf16_flops if base_chip else None
    )
    vmem_bytes = _parse_spec_int(
        raw, "vmem_bytes", default=base_chip.vmem_bytes if base_chip else None
    )
    cores_per_chip = _parse_spec_int(
        raw, "cores_per_chip", default=base_chip.cores_per_chip if base_chip else None
    )
    chips_per_host = _parse_spec_int(
        raw, "chips_per_host", default=base_chip.chips_per_host if base_chip else None
    )
    fp8_flops = _parse_spec_float(
        raw,
        "fp8_flops",
        default=base_chip.fp8_flops if base_chip else 0.0,
        allow_zero=True,
    )
    int8_flops = _parse_spec_float(
        raw,
        "int8_flops",
        default=base_chip.int8_flops if base_chip else 0.0,
        allow_zero=True,
    )
    fp4_flops = _parse_spec_float(
        raw,
        "fp4_flops",
        default=base_chip.fp4_flops if base_chip else 0.0,
        allow_zero=True,
    )
    int4_flops = _parse_spec_float(
        raw,
        "int4_flops",
        default=base_chip.int4_flops if base_chip else 0.0,
        allow_zero=True,
    )
    ici_links_per_chip = _parse_spec_int(
        raw,
        "ici_links_per_chip",
        default=base_chip.ici_links_per_chip
        if base_chip
        else DEFAULT_ICI_LINKS_PER_CHIP,
    )
    ici_link_bw = _parse_spec_float(
        raw,
        "ici_link_bw",
        default=base_chip.ici_link_bw if base_chip else DEFAULT_ICI_LINK_BW,
    )
    has_d2d_input = (
        raw.get("ici_links_per_chip") is not None or raw.get("ici_link_bw") is not None
    )
    d2d_default = (
        base_chip.d2d_bw
        if base_chip is not None and not has_d2d_input
        else float(ici_links_per_chip) * ici_link_bw
    )
    d2d_bw = _parse_spec_float(
        raw,
        "d2d_bw",
        default=d2d_default,
    )
    has_ici_input = has_d2d_input or raw.get("cores_per_chip") is not None
    ici_default = (
        base_chip.ici_bw
        if base_chip is not None and not has_ici_input
        else float(ici_links_per_chip) * ici_link_bw / float(cores_per_chip)
    )
    ici_bw = _parse_spec_float(
        raw,
        "ici_bw",
        default=ici_default,
    )
    dcn_host_gbit_per_s = _parse_spec_float(
        raw,
        "dcn_host_gbit_per_s",
        default=(
            base_chip.dcn_host_gbit_per_s if base_chip else DEFAULT_DCN_HOST_GBIT_PER_S
        ),
    )
    has_dcn_input = (
        raw.get("dcn_host_gbit_per_s") is not None
        or raw.get("chips_per_host") is not None
    )
    dcn_default = (
        base_chip.dcn_bw
        if base_chip is not None and not has_dcn_input
        else (dcn_host_gbit_per_s * UNIT_GIGA / BITS_PER_BYTE) / float(chips_per_host)
    )
    dcn_bw = _parse_spec_float(
        raw,
        "dcn_bw",
        default=dcn_default,
    )
    vpu_flops = _resolve_spec_vpu_flops(raw, bf16_flops, base_chip)
    return ChipSpec(
        name=name,
        hbm_bytes=hbm_bytes,
        hbm_bw=hbm_bw,
        bf16_flops=bf16_flops,
        fp8_flops=fp8_flops,
        vmem_bytes=vmem_bytes,
        ici_bw=ici_bw,
        cores_per_chip=cores_per_chip,
        ici_link_bw=ici_link_bw,
        d2d_bw=d2d_bw,
        ici_hop_latency_s=_parse_spec_float(
            raw,
            "ici_hop_latency_s",
            default=(
                base_chip.ici_hop_latency_s if base_chip else DEFAULT_ICI_HOP_LATENCY_S
            ),
            allow_zero=True,
        ),
        mxu_m_tile=_parse_spec_int(
            raw,
            "mxu_m_tile",
            default=base_chip.mxu_m_tile if base_chip else DEFAULT_MXU_M_TILE,
        ),
        dcn_host_gbit_per_s=dcn_host_gbit_per_s,
        chips_per_host=chips_per_host,
        dcn_bw=dcn_bw,
        pcie_bw=_parse_spec_float(
            raw,
            "pcie_bw",
            default=base_chip.pcie_bw if base_chip else DEFAULT_PCIE_BW,
        ),
        vpu_flops=vpu_flops,
        mxu_lhs_rows_bf16=_parse_spec_int(
            raw,
            "mxu_lhs_rows_bf16",
            default=(
                base_chip.mxu_lhs_rows_bf16 if base_chip else DEFAULT_MXU_LHS_ROWS_BF16
            ),
        ),
        mxu_lhs_rows_fp8=_parse_spec_int(
            raw,
            "mxu_lhs_rows_fp8",
            default=(
                base_chip.mxu_lhs_rows_fp8 if base_chip else DEFAULT_MXU_LHS_ROWS_FP8
            ),
        ),
        int8_flops=int8_flops,
        fp4_flops=fp4_flops,
        int4_flops=int4_flops,
        ici_links_per_chip=ici_links_per_chip,
        min_torus_dim=_parse_spec_int(
            raw,
            "min_torus_dim",
            default=base_chip.min_torus_dim if base_chip else MIN_TORUS_DIM,
        ),
        vpu_lanes=_parse_spec_int(
            raw,
            "vpu_lanes",
            default=base_chip.vpu_lanes if base_chip else DEFAULT_VPU_LANES,
        ),
        vpu_sublanes=_parse_spec_int(
            raw,
            "vpu_sublanes",
            default=base_chip.vpu_sublanes if base_chip else DEFAULT_VPU_SUBLANES,
        ),
        physical_alignment_elems=_parse_spec_int(
            raw,
            "physical_alignment_elems",
            default=(
                base_chip.physical_alignment_elems
                if base_chip
                else PHYSICAL_ALIGNMENT_ELEMS
            ),
        ),
        smem_page_table_bytes=_parse_spec_int(
            raw,
            "smem_page_table_bytes",
            default=(
                base_chip.smem_page_table_bytes if base_chip else PAGE_TABLE_BYTE_BUDGET
            ),
        ),
        dma_granule_bytes=_parse_spec_int(
            raw,
            "dma_granule_bytes",
            default=(
                base_chip.dma_granule_bytes if base_chip else DEFAULT_DMA_GRANULE_BYTES
            ),
        ),
        subchannel_block_k_floor=_parse_spec_int(
            raw,
            "subchannel_block_k_floor",
            default=(
                base_chip.subchannel_block_k_floor
                if base_chip
                else SUBCHANNEL_BLOCK_K_FLOOR
            ),
        ),
        has_sparse_core=_parse_spec_bool(
            raw,
            "has_sparse_core",
            default=base_chip.has_sparse_core if base_chip else True,
        ),
        provenance=provenance,
    )


def load_chip_spec(
    source: str | Path | Mapping[str, object],
) -> tuple[ChipSpec, dict[str, SliceTopology]]:
    """Validate and load a ChipSpec and topology map from JSON path or mapping."""
    raw = _read_spec_mapping(source)
    base_name, base_chip = _resolve_base_chip(raw)
    chip = _build_spec_chip(raw, base_chip)
    topologies = _parse_spec_topologies(
        raw, chip.cores_per_chip, base_name, chip.min_torus_dim
    )
    return chip, topologies
