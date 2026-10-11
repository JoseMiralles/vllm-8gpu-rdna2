# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Glm5Next kernels for AMD RDNA (gfx1030).

Thin wrappers over ``_rocm_C`` ops built from ``csrc/rocm/rdna/glm5next/``.
The model calls ``glm5_kda_recurrent`` on plain decode. The other ops are
registered for the same ISA and are not substituted for upstream Triton KDA,
the fp8 indexer cache, or latent MLA.
"""
