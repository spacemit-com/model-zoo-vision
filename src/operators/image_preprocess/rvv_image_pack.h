/*
 * Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef RVV_IMAGE_PACK_H
#define RVV_IMAGE_PACK_H

#if defined(__riscv_vector)
#include <array>
#include <cstddef>
#include <cstdint>

#include <riscv_vector.h>

namespace vision_operators {
namespace detail {

// Pack only the content of one interleaved three-channel uint8 row. Callers
// supply destination row pointers, so source stride, plane stride, offsets and
// padding stay outside this primitive. Mean/scales are in output-channel order.
// swap_rb exchanges the first and third source channels (BGR <-> RGB).
// Keep mul/sub/mul separate to preserve the reference rounding points. Explicit
// division pipelines must keep using the CPU LUT path instead of this helper.
inline void pack_u8c3_to_f32_planes_rvv(
    const uint8_t* source, int width, bool swap_rb,
    float* first, float* second, float* third,
    const std::array<float, 3>& mean,
    const std::array<float, 3>& output_scale,
    const std::array<float, 3>& input_scale = {1.0F, 1.0F, 1.0F})
{
    while (width > 0) {
        const size_t vl = __riscv_vsetvl_e8m1(width);
        const auto pixels = __riscv_vlseg3e8_v_u8m1x3(source, vl);
        const auto c0 = __riscv_vget_v_u8m1x3_u8m1(pixels, 0);
        const auto c1 = __riscv_vget_v_u8m1x3_u8m1(pixels, 1);
        const auto c2 = __riscv_vget_v_u8m1x3_u8m1(pixels, 2);
        float* destinations[] = {first, second, third};
        for (int channel = 0; channel < 3; ++channel) {
            const auto values = channel == 1 ? c1 :
                ((channel == 0) != swap_rb ? c0 : c2);
            auto floats = __riscv_vfwcvt_f_xu_v_f32m4(
                __riscv_vzext_vf2_u16m2(values, vl), vl);
            if (input_scale[channel] != 1.0F) {
                floats = __riscv_vfmul_vf_f32m4(floats, input_scale[channel], vl);
            }
            floats = __riscv_vfsub_vf_f32m4(floats, mean[channel], vl);
            floats = __riscv_vfmul_vf_f32m4(floats, output_scale[channel], vl);
            __riscv_vse32_v_f32m4(destinations[channel], floats, vl);
        }
        source += vl * 3;
        first += vl;
        second += vl;
        third += vl;
        width -= static_cast<int>(vl);
    }
}

}  // namespace detail
}  // namespace vision_operators
#endif  // defined(__riscv_vector)

#endif  // RVV_IMAGE_PACK_H
