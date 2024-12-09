// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/builtins/ukernel/arch/riscv_64/common_riscv_64.h"
#include "iree/builtins/ukernel/arch/riscv_64/mmt4d_riscv_64_internal.h"


IREE_UK_ATTRIBUTE_ALWAYS_INLINE static inline float
extract_element_from_vfloat32m1_t(vfloat32m1_t lhs, int pos, size_t vl) {
  vfloat32m1_t lhs_slided = __riscv_vslidedown_vx_f32m1(lhs, pos , vl);
  return __riscv_vfmv_f_s_f32m1_f32(lhs_slided);
}

IREE_UK_ATTRIBUTE_ALWAYS_INLINE static inline void
iree_uk_mmt4d_tile_f32f32f32_1x8x1_to_8x8x1_riscv_64(
    void* IREE_UK_RESTRICT out_tile, const void* IREE_UK_RESTRICT lhs_panel,
    const void* IREE_UK_RESTRICT rhs_panel,
    const iree_uk_mmt4d_params_t* params, int M0) {
  IREE_UK_ASSERT(M0 >= 1 && M0 <= 8 && iree_uk_is_po2_u32(M0));
  const float* IREE_UK_RESTRICT lhs_ptr = lhs_panel;
  const float* IREE_UK_RESTRICT rhs_ptr = rhs_panel;
  float* IREE_UK_RESTRICT out_ptr = out_tile;

  vfloat32m1_t acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7;
  size_t vlmax =  __riscv_vsetvlmax_e32m1();

  if (M0 == 1) {
    if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
      acc0 =  __riscv_vle32_v_f32m1(out_ptr, vlmax);
    } else {
      acc0 =  __riscv_vfmv_v_f_f32m1(0.0, vlmax);
    }
    for (int k = 0; k < params->K; ++k) {
      vfloat32m1_t rhs = __riscv_vle32_v_f32m1(rhs_ptr, vlmax);
      rhs_ptr += 8;
      float lhs = *lhs_ptr++;
      acc0 = __riscv_vfmacc_vf_f32m1(acc0, lhs, rhs, vlmax);

    }
    __riscv_vse32_v_f32m1(out_ptr, acc0, vlmax);
  }

  if (M0 == 2) {
    if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
      acc0 =  __riscv_vle32_v_f32m1(out_ptr, vlmax);
      acc1 =  __riscv_vle32_v_f32m1(out_ptr + 8 , vlmax);
    } else {
      acc0 =  __riscv_vfmv_v_f_f32m1(0.0, vlmax);
      acc1 =  __riscv_vfmv_v_f_f32m1(0.0, vlmax);
    }
    for (int k = 0; k < params->K; ++k) {
      vfloat32m1_t rhs = __riscv_vle32_v_f32m1(rhs_ptr, vlmax);
      rhs_ptr += 8;
      //float lhs = *lhs_ptr++;
      //float lhs2 = *lhs_ptr++;
      vfloat32m1_t lhs = __riscv_vle32_v_f32m1(lhs_ptr, 2);
      float lhs0 =  extract_element_from_vfloat32m1_t(lhs, 0, vlmax);
      float lhs1 =  extract_element_from_vfloat32m1_t(lhs, 1, vlmax);
      lhs_ptr += 2;
      acc0 = __riscv_vfmacc_vf_f32m1(acc0, lhs0, rhs, vlmax);
      acc1 = __riscv_vfmacc_vf_f32m1(acc1, lhs1, rhs, vlmax);
    }
    __riscv_vse32_v_f32m1(out_ptr, acc0, vlmax);
    __riscv_vse32_v_f32m1(out_ptr + 8, acc1, vlmax);
  }
  if (M0 == 4) {
    if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
      acc0 =  __riscv_vle32_v_f32m1(out_ptr, vlmax);
      acc1 =  __riscv_vle32_v_f32m1(out_ptr + 8 , vlmax);
      acc2 =  __riscv_vle32_v_f32m1(out_ptr + 16, vlmax);
      acc3 =  __riscv_vle32_v_f32m1(out_ptr + 24 , vlmax);
    } else {
      acc0 =  __riscv_vfmv_v_f_f32m1(0.0, vlmax);
      acc1 =  __riscv_vfmv_v_f_f32m1(0.0, vlmax);
      acc2 =  __riscv_vfmv_v_f_f32m1(0.0, vlmax);
      acc3 =  __riscv_vfmv_v_f_f32m1(0.0, vlmax);
    }
    for (int k = 0; k < params->K; ++k) {
      vfloat32m1_t rhs = __riscv_vle32_v_f32m1(rhs_ptr, vlmax);
      rhs_ptr += 8;
      // float lhs = *lhs_ptr++;
      // float lhs2 = *lhs_ptr++;
      // float lhs3 = *lhs_ptr++;
      // float lhs4 = *lhs_ptr++;
      vfloat32m1_t lhs = __riscv_vle32_v_f32m1(lhs_ptr, 4);
      float lhs0 =  extract_element_from_vfloat32m1_t(lhs, 0, vlmax);
      float lhs1 =  extract_element_from_vfloat32m1_t(lhs, 1, vlmax);
      float lhs2 =  extract_element_from_vfloat32m1_t(lhs, 2, vlmax);
      float lhs3 =  extract_element_from_vfloat32m1_t(lhs, 3, vlmax);
      lhs_ptr += 4;
      acc0 = __riscv_vfmacc_vf_f32m1(acc0, lhs0, rhs, vlmax);
      acc1 = __riscv_vfmacc_vf_f32m1(acc1, lhs1, rhs, vlmax);
      acc2 = __riscv_vfmacc_vf_f32m1(acc2, lhs2, rhs, vlmax);
      acc3 = __riscv_vfmacc_vf_f32m1(acc3, lhs3, rhs, vlmax);
    }
    __riscv_vse32_v_f32m1(out_ptr, acc0, vlmax);
    __riscv_vse32_v_f32m1(out_ptr + 8, acc1, vlmax);
    __riscv_vse32_v_f32m1(out_ptr + 16, acc2, vlmax);
    __riscv_vse32_v_f32m1(out_ptr + 24, acc3, vlmax);
  }
  if (M0 == 8) {
    if (params->flags & IREE_UK_FLAG_MMT4D_ACCUMULATE) {
      acc0 =  __riscv_vle32_v_f32m1(out_ptr, vlmax);
      acc1 =  __riscv_vle32_v_f32m1(out_ptr + 8 , vlmax);
      acc2 =  __riscv_vle32_v_f32m1(out_ptr + 16, vlmax);
      acc3 =  __riscv_vle32_v_f32m1(out_ptr + 24 , vlmax);
      acc4 =  __riscv_vle32_v_f32m1(out_ptr + 32, vlmax);
      acc5 =  __riscv_vle32_v_f32m1(out_ptr + 40, vlmax);
      acc6 =  __riscv_vle32_v_f32m1(out_ptr + 48, vlmax);
      acc7 =  __riscv_vle32_v_f32m1(out_ptr + 56 , vlmax);
    } else {
      acc0 =  __riscv_vfmv_v_f_f32m1(0.0, vlmax);
      acc1 =  __riscv_vfmv_v_f_f32m1(0.0, vlmax);
      acc2 =  __riscv_vfmv_v_f_f32m1(0.0, vlmax);
      acc3 =  __riscv_vfmv_v_f_f32m1(0.0, vlmax);
      acc4 =  __riscv_vfmv_v_f_f32m1(0.0, vlmax);
      acc5 =  __riscv_vfmv_v_f_f32m1(0.0, vlmax);
      acc6 =  __riscv_vfmv_v_f_f32m1(0.0, vlmax);
      acc7 =  __riscv_vfmv_v_f_f32m1(0.0, vlmax);

    }
    for (int k = 0; k < params->K; ++k) {
      vfloat32m1_t rhs = __riscv_vle32_v_f32m1(rhs_ptr, vlmax);
      rhs_ptr += 8;
      // use vector load and extract value instead
      // same logic follows for M0=4 and M0=8
      // float lhs = *lhs_ptr++;
      // float lhs2 = *lhs_ptr++;
      // float lhs3 = *lhs_ptr++;
      // float lhs4 = *lhs_ptr++;
      // float lhs5 = *lhs_ptr++;
      // float lhs6 = *lhs_ptr++;
      // float lhs7 = *lhs_ptr++;
      // float lhs8 = *lhs_ptr++;
      vfloat32m1_t lhs = __riscv_vle32_v_f32m1(lhs_ptr, 8);
      float lhs0 =  extract_element_from_vfloat32m1_t(lhs, 0, vlmax);
      float lhs1 =  extract_element_from_vfloat32m1_t(lhs, 1, vlmax);
      float lhs2 =  extract_element_from_vfloat32m1_t(lhs, 2, vlmax);
      float lhs3 =  extract_element_from_vfloat32m1_t(lhs, 3, vlmax);
      float lhs4 =  extract_element_from_vfloat32m1_t(lhs, 4, vlmax);
      float lhs5 =  extract_element_from_vfloat32m1_t(lhs, 5, vlmax);
      float lhs6 =  extract_element_from_vfloat32m1_t(lhs, 6, vlmax);
      float lhs7 =  extract_element_from_vfloat32m1_t(lhs, 7, vlmax);
      lhs_ptr += 8;
      acc0 = __riscv_vfmacc_vf_f32m1(acc0, lhs0, rhs, vlmax);
      acc1 = __riscv_vfmacc_vf_f32m1(acc1, lhs1, rhs, vlmax);
      acc2 = __riscv_vfmacc_vf_f32m1(acc2, lhs2, rhs, vlmax);
      acc3 = __riscv_vfmacc_vf_f32m1(acc3, lhs3, rhs, vlmax);
      acc4 = __riscv_vfmacc_vf_f32m1(acc4, lhs4, rhs, vlmax);
      acc5 = __riscv_vfmacc_vf_f32m1(acc5, lhs5, rhs, vlmax);
      acc6 = __riscv_vfmacc_vf_f32m1(acc6, lhs6, rhs, vlmax);
      acc7 = __riscv_vfmacc_vf_f32m1(acc7, lhs7, rhs, vlmax);
    }
    __riscv_vse32_v_f32m1(out_ptr, acc0, vlmax);
    __riscv_vse32_v_f32m1(out_ptr + 8, acc1, vlmax);
    __riscv_vse32_v_f32m1(out_ptr + 16, acc2, vlmax);
    __riscv_vse32_v_f32m1(out_ptr + 24, acc3, vlmax);
    __riscv_vse32_v_f32m1(out_ptr + 32, acc4, vlmax);
    __riscv_vse32_v_f32m1(out_ptr + 40, acc5, vlmax);
    __riscv_vse32_v_f32m1(out_ptr + 48, acc6, vlmax);
    __riscv_vse32_v_f32m1(out_ptr + 56, acc7, vlmax);
  }
}

IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_f32f32f32_1x8x1_to_8x8x1_riscv_64,
    iree_uk_mmt4d_tile_f32f32f32_1x8x1_riscv_64, 1)
IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_f32f32f32_1x8x1_to_8x8x1_riscv_64,
    iree_uk_mmt4d_tile_f32f32f32_2x8x1_riscv_64, 2)
IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_f32f32f32_1x8x1_to_8x8x1_riscv_64,
    iree_uk_mmt4d_tile_f32f32f32_4x8x1_riscv_64, 4)
IREE_UK_MMT4D_TILE_FUNC_IMPL_FOR_M0(
    iree_uk_mmt4d_tile_f32f32f32_1x8x1_to_8x8x1_riscv_64,
    iree_uk_mmt4d_tile_f32f32f32_8x8x1_riscv_64, 8)
