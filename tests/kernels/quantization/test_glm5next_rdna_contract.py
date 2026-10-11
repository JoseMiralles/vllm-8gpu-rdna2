# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for the gfx1030 Glm5Next path.

The HIP objects are not built in this environment. These tests lock the
source contract (current stream, no packed bf16 dot, no WMMA/MFMA, no
hardware fp8 conversion) and the SM100 family alias.
"""

import re
from pathlib import Path

import pytest
import torch

from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import (
    _awq_uint8_qzeros_to_rdna2,
)
from vllm.platforms.interface import DeviceCapability

REPO = Path(__file__).resolve().parents[3]
KERNEL_DIR = REPO / "csrc" / "rocm" / "rdna" / "glm5next"
_COMMENT = re.compile(r"/\*.*?\*/|//.*?$", re.S | re.M)


def _code(path: Path) -> str:
    return _COMMENT.sub("", path.read_text())


def test_gfx1030_capability_aliases_sm100_family_bucket():
    cap = DeviceCapability(major=10, minor=3)
    assert cap.to_int() == 103
    assert cap.to_int() // 10 == 100 // 10


def test_rdna_family_check_is_not_sm100(monkeypatch: pytest.MonkeyPatch):
    from vllm.platforms import rocm

    monkeypatch.setattr(rocm, "on_gfx10x", lambda: True)
    monkeypatch.setattr(rocm, "on_rdna", lambda: False)
    monkeypatch.setattr(
        rocm.RocmPlatform,
        "get_device_capability",
        classmethod(lambda cls, device_id=0: DeviceCapability(10, 3)),
    )
    assert rocm.RocmPlatform.is_device_capability_family(100) is False
    assert rocm.RocmPlatform.is_device_capability_family(90) is False

    monkeypatch.setattr(rocm, "on_gfx10x", lambda: False)
    monkeypatch.setattr(rocm, "on_rdna", lambda: True)
    monkeypatch.setattr(
        rocm.RocmPlatform,
        "get_device_capability",
        classmethod(lambda cls, device_id=0: DeviceCapability(11, 0)),
    )
    assert rocm.RocmPlatform.is_device_capability_family(110) is False

    # CDNA keeps the generic family test.
    monkeypatch.setattr(rocm, "on_gfx10x", lambda: False)
    monkeypatch.setattr(rocm, "on_rdna", lambda: False)
    monkeypatch.setattr(
        rocm.RocmPlatform,
        "get_device_capability",
        classmethod(lambda cls, device_id=0: DeviceCapability(9, 0)),
    )
    assert rocm.RocmPlatform.is_device_capability_family(90) is True
    assert rocm.RocmPlatform.is_device_capability_family(100) is False


def test_glm5next_kernels_match_gfx1030_contract():
    sources = sorted(KERNEL_DIR.glob("*.cu"))
    assert sources, f"no kernels in {KERNEL_DIR}"
    cmake = (REPO / "CMakeLists.txt").read_text()
    gfx1030 = cmake.split('if(VLLM_GPU_ARCHES MATCHES "gfx1030")', 1)[1]
    gfx1030 = gfx1030.split("endif()", 1)[0]
    gfx1100 = cmake.split('if(VLLM_GPU_ARCHES MATCHES "gfx1100")', 1)[1]
    gfx1100 = gfx1100.split("endif()", 1)[0]
    for path in sources:
        code = _code(path)
        assert "at::cuda::getCurrentCUDAStream()" in code
        assert "getCurrentHIPStream" not in code
        assert "hip/hip_fp8.h" not in code
        assert "v_wmma" not in code
        assert "v_mfma" not in code
        assert "v_dot2_f32_bf16" not in code
        assert "0xffffffffffffffff" not in code
        for launch in re.findall(r"<<<[^;]*?>>>", code):
            assert "stream" in launch, path.name
        rel = path.relative_to(REPO).as_posix()
        assert rel in gfx1030
        assert rel not in gfx1100


def test_awq_moe_zeros_apply_gptqv1_offset():
    # [E, N/2, G] = [1, 4, 1], columns 1..8 packed low-nibble-even.
    raw = torch.tensor([0x21, 0x43, 0x65, 0x87], dtype=torch.uint8)
    qz = raw.view(1, 4, 1)
    packed = _awq_uint8_qzeros_to_rdna2(qz)
    assert packed.shape == (1, 1, 1)
    assert packed.dtype == torch.int32
    expected = 0
    for nibble in range(8):
        expected |= nibble << (4 * nibble)
    assert int(packed.reshape(-1)[0].item()) == expected
