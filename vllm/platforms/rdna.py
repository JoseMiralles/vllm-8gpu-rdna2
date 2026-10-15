# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gate for RDNA-specific behavior in shared (non-ROCm-only) code paths.

Hot paths traced by Dynamo should read the result once at module import
(``_ON_RDNA = on_rdna_family()``) instead of calling these per forward.
"""

_ON_RDNA_FAMILY: bool | None = None
_ON_RDNA2: bool | None = None


def on_rdna_family() -> bool:
    """ROCm on RDNA2 (gfx10x) or RDNA3/4 (gfx11/gfx12, excluding CDNA gfx1250).

    Never imports ``vllm.platforms.rocm`` on other platforms.
    """
    global _ON_RDNA_FAMILY
    if _ON_RDNA_FAMILY is None:
        from vllm.platforms import current_platform

        if current_platform.is_rocm():
            from vllm.platforms.rocm import on_gfx10x, on_rdna

            _ON_RDNA_FAMILY = on_gfx10x() or on_rdna()
        else:
            _ON_RDNA_FAMILY = False
    return _ON_RDNA_FAMILY


def on_rdna2() -> bool:
    """ROCm on RDNA2 (gfx10x)."""
    global _ON_RDNA2
    if _ON_RDNA2 is None:
        from vllm.platforms import current_platform

        if current_platform.is_rocm():
            from vllm.platforms.rocm import on_gfx10x

            _ON_RDNA2 = on_gfx10x()
        else:
            _ON_RDNA2 = False
    return _ON_RDNA2


def on_rdna2_hip() -> bool:
    """True iff the opt-in RDNA2 HIP kernel set is enabled on this process.

    The gfx1030 HIP ports are opt-in (``VLLM_RDNA_ALL_HIP=1``), matching the
    other RDNA flags on this branch: with the flag unset gfx10x runs the Triton
    kernels it ran before the port, so a deploy that does not opt in sees no
    behaviour change.
    """
    from vllm.envs import VLLM_RDNA_ALL_HIP

    return on_rdna2() and VLLM_RDNA_ALL_HIP
