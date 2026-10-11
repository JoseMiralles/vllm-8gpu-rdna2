# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Wrappers and fake impls for the Glm5Next gfx1030 HIP ops."""

import contextlib

import torch

with contextlib.suppress(ImportError):
    import vllm._rocm_C  # noqa: F401

try:
    from torch.library import register_fake
except ImportError:  # older torch
    from torch.library import impl_abstract as register_fake


def _has_op(name: str) -> bool:
    # hasattr is unreliable on torch op namespaces (dir() only shows "name").
    try:
        getattr(torch.ops._rocm_C, name)
    except (AttributeError, RuntimeError):
        return False
    return True


def glm5_kda_slot_is_empty(slot: int, num_slots: int) -> bool:
    """Return whether a KDA state slot must not be read or written.

    Matches ``glm5_kda_slot_is_empty`` in ``glm5_kda_slot.cuh`` and
    ``fused_recurrent_kda`` (``state_idx <= 0``). ``NULL_BLOCK_ID`` is 0
    and ``PAD_SLOT_ID`` is -1.

    Args:
        slot: State index from the scheduler.
        num_slots: Length of the recurrent-state buffer.

    Returns:
        True when the slot is padding, the null block, or out of range.
    """
    return int(slot) <= 0 or int(slot) >= int(num_slots)


def has_glm5_kda_recurrent() -> bool:
    return _has_op("glm5_kda_recurrent_rdna2")


def glm5_kda_recurrent(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state_indices: torch.Tensor,
    ssm_state: torch.Tensor,
    out: torch.Tensor,
    lower_bound: float,
) -> None:
    """In-place post-conv KDA recurrence. ``out`` and ``ssm_state`` are written.

    Args:
        q: Post-conv query, fp16 or bf16, ``[1, T, H, 128]`` or ``[T, H, 128]``.
        k: Post-conv key, same shape and dtype as ``q``.
        v: Post-conv value, same shape and dtype as ``q``.
        g: Raw forget pre-activation, same shape and dtype as ``q``.
        beta: Raw beta, ``[1, T, H]`` or ``[T, H]``.
        a_log: Per-head fp32 ``A_log``.
        dt_bias: fp32 ``[H * 128]``.
        state_indices: int32 slots, length ``T``. Slot 0 (NULL_BLOCK_ID)
            and negative slots (PAD_SLOT_ID is -1) are left untouched.
        ssm_state: fp32 ``[slots, H, 128, 128]`` in ``[V, K]`` order.
        out: Pre-norm output, same shape and dtype as ``q``.
        lower_bound: Bounded-gate floor.
    """
    torch.ops._rocm_C.glm5_kda_recurrent_rdna2(
        q,
        k,
        v,
        g,
        beta,
        a_log,
        dt_bias,
        state_indices,
        ssm_state,
        out,
        lower_bound,
    )


if has_glm5_kda_recurrent():

    @register_fake("_rocm_C::glm5_kda_recurrent_rdna2")
    def _glm5_kda_recurrent_fake(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        a_log: torch.Tensor,
        dt_bias: torch.Tensor,
        state_indices: torch.Tensor,
        ssm_state: torch.Tensor,
        out: torch.Tensor,
        lower_bound: float,
    ) -> None:
        return None
