// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BUILTINS_UKERNEL_ARCH_RISCV_64_COMMON_RISCV_64_H_
#define IREE_BUILTINS_UKERNEL_ARCH_RISCV_64_COMMON_RISCV_64_H_

#include <riscv_vector.h>

#include "iree/builtins/ukernel/common.h"
#include "iree/schemas/cpu_data.h"

#if defined(IREE_DEVICE_STANDALONE)
// Standalone builds (e.g. bitcode) use our own Clang, supporting everything.
// PLACEHOLDER
#else
// Compiling with the system toolchain. Include the configured header.
#include "iree/builtins/ukernel/arch/riscv_64/config_riscv_64.h"
#endif

static inline bool iree_uk_cpu_riscv_64(const iree_uk_uint64_t* cpu_data) {
  (void)cpu_data;
  return true;
}

#endif  // IREE_BUILTINS_UKERNEL_ARCH_ARM_64_COMMON_ARM_64_H_
