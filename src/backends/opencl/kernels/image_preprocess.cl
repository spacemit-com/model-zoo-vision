#define COMMON_ARGUMENTS                                                   \
    int input_width, int input_height, int output_width, int output_height,\
    float src_x, float src_y, float src_width, float src_height,           \
    int dst_x, int dst_y, int dst_width, int dst_height,                   \
    int output_rgb, int interpolation,                                     \
    float4 mean, float4 scale, float4 padding

#define PROCESS_PIXEL(SAMPLE_EXPRESSION, OUTPUT_TYPE, STORE_FUNCTION)       \
    int x = get_global_id(0);                                               \
    int y = get_global_id(1);                                               \
    if (x >= output_width || y >= output_height) return;                    \
    float3 value = padding.xyz;                                             \
    if (x >= dst_x && x < dst_x + dst_width &&                              \
        y >= dst_y && y < dst_y + dst_height) {                             \
        float2 source = source_coordinate(                                  \
            x, y, dst_x, dst_y, dst_width, dst_height,                      \
            src_x, src_y, src_width, src_height);                           \
        if (interpolation == 1) source = floor(source + 0.5f);              \
        value = SAMPLE_EXPRESSION;                                          \
        value = output_rgb ? value : value.zyx;                             \
    }                                                                       \
    value = (value - mean.xyz) * scale.xyz;                                 \
    int index = y * output_width + x;                                       \
    int plane = output_width * output_height;                               \
    STORE_FUNCTION(output, index, plane, value)

#define PROCESS_BGR_IDENTITY_GROUP(STORE4_FUNCTION, STORE_FUNCTION)         \
    int x = get_global_id(0) * 4;                                           \
    int y = get_global_id(1);                                               \
    if (x >= output_width || y >= output_height) return;                    \
    int count = min(4, output_width - x);                                   \
    int plane = output_width * output_height;                               \
    int index = y * output_width + x;                                       \
    int content_right = dst_x + input_width;                                \
    int content_bottom = dst_y + input_height;                              \
    if (count == 4 && (index & 3) == 0 && (plane & 3) == 0 &&               \
        (y < dst_y || y >= content_bottom ||                                \
        x + 3 < dst_x || x >= content_right)) {                             \
        float4 first = (float4)(                                            \
            (padding.x - mean.x) * scale.x);                                \
        float4 second = (float4)(                                           \
            (padding.y - mean.y) * scale.y);                                \
        float4 third = (float4)(                                            \
            (padding.z - mean.z) * scale.z);                                \
        STORE4_FUNCTION(                                                    \
            output, index, plane, first, second, third);                    \
        return;                                                             \
    }                                                                       \
    int input_offset = (y - dst_y) * input_stride +                         \
        (x - dst_x) * 3;                                                    \
    if (count == 4 && (index & 3) == 0 && (plane & 3) == 0 &&               \
        x >= dst_x && x + 3 < content_right && y >= dst_y &&                \
        y < content_bottom && (input_offset & 3) == 0) {                    \
        float4 red;                                                         \
        float4 green;                                                       \
        float4 blue;                                                        \
        read_bgr_buffer_pixels4(                                            \
            input, input_stride, x - dst_x, y - dst_y,                     \
            &red, &green, &blue);                                           \
        float4 first = red;                                                 \
        float4 third = blue;                                                \
        if (!output_rgb) {                                                  \
            first = blue;                                                   \
            third = red;                                                    \
        }                                                                   \
        first = (first - mean.x) * scale.x;                                 \
        green = (green - mean.y) * scale.y;                                 \
        third = (third - mean.z) * scale.z;                                 \
        STORE4_FUNCTION(                                                    \
            output, index, plane, first, green, third);                     \
        return;                                                             \
    }                                                                       \
    for (int lane = 0; lane < count; ++lane) {                              \
        int output_x = x + lane;                                            \
        float3 value = padding.xyz;                                         \
        if (output_x >= dst_x && output_x < content_right &&                \
            y >= dst_y && y < content_bottom) {                             \
            value = read_bgr_buffer_pixel(                                  \
                input, input_stride, input_width, input_height,             \
                (int2)(output_x - dst_x, y - dst_y));                       \
            value = output_rgb ? value : value.zyx;                         \
        }                                                                   \
        value = (value - mean.xyz) * scale.xyz;                             \
        STORE_FUNCTION(output, index + lane, plane, value);                 \
    }

#define PROCESS_NV12_IDENTITY_GROUP(STORE4_FUNCTION, STORE_FUNCTION)        \
    int x = get_global_id(0) * 4;                                           \
    int y = get_global_id(1);                                               \
    if (x >= output_width || y >= output_height) return;                    \
    int count = min(4, output_width - x);                                   \
    int plane = output_width * output_height;                               \
    int index = y * output_width + x;                                       \
    int content_right = dst_x + input_width;                                \
    int content_bottom = dst_y + input_height;                              \
    if (count == 4 && (index & 3) == 0 && (plane & 3) == 0 &&               \
        (y < dst_y || y >= content_bottom ||                                \
        x + 3 < dst_x || x >= content_right)) {                             \
        float4 first = (float4)(                                            \
            (padding.x - mean.x) * scale.x);                                \
        float4 second = (float4)(                                           \
            (padding.y - mean.y) * scale.y);                                \
        float4 third = (float4)(                                            \
            (padding.z - mean.z) * scale.z);                                \
        STORE4_FUNCTION(                                                    \
            output, index, plane, first, second, third);                    \
        return;                                                             \
    }                                                                       \
    int input_x = x - dst_x;                                                \
    int input_y = y - dst_y;                                                \
    if (count == 4 && (index & 3) == 0 && (plane & 3) == 0 &&               \
        x >= dst_x && x + 3 < content_right && y >= dst_y &&                \
        y < content_bottom && (input_x & 1) == 0) {                         \
        float4 red;                                                         \
        float4 green;                                                       \
        float4 blue;                                                        \
        read_nv12_pixels4_even(                                             \
            y_image, uv_image, input_x, input_y,                           \
            &red, &green, &blue);                                           \
        float4 first = red;                                                 \
        float4 third = blue;                                                \
        if (!output_rgb) {                                                  \
            first = blue;                                                   \
            third = red;                                                    \
        }                                                                   \
        first = (first - mean.x) * scale.x;                                 \
        green = (green - mean.y) * scale.y;                                 \
        third = (third - mean.z) * scale.z;                                 \
        STORE4_FUNCTION(                                                    \
            output, index, plane, first, green, third);                     \
        return;                                                             \
    }                                                                       \
    for (int lane = 0; lane < count; ++lane) {                              \
        int output_x = x + lane;                                            \
        float3 value = padding.xyz;                                         \
        if (output_x >= dst_x && output_x < content_right &&                \
            y >= dst_y && y < content_bottom) {                             \
            value = read_nv12_pixel(                                        \
                y_image, uv_image,                                          \
                (int2)(output_x - dst_x, y - dst_y));                       \
            value = output_rgb ? value : value.zyx;                         \
        }                                                                   \
        value = (value - mean.xyz) * scale.xyz;                             \
        STORE_FUNCTION(output, index + lane, plane, value);                 \
    }

__kernel void preprocess_nv12_images_f32(
    read_only image2d_t y_image,
    read_only image2d_t uv_image,
    __global float* output,
    COMMON_ARGUMENTS)
{
    PROCESS_PIXEL(
        sample_nv12_images(
            y_image, uv_image, source.x, source.y, interpolation),
        float,
        store_float_output);
}

__kernel void preprocess_nv12_images_fast_f32(
    read_only image2d_t y_image,
    read_only image2d_t uv_image,
    __global float* output,
    COMMON_ARGUMENTS)
{
    PROCESS_PIXEL(
        sample_nv12_images_fast(
            y_image, uv_image, source.x, source.y, interpolation),
        float,
        store_float_output);
}

__kernel void preprocess_nv12_images_identity_f32(
    read_only image2d_t y_image,
    read_only image2d_t uv_image,
    __global float* output,
    int input_width,
    int input_height,
    int output_width,
    int output_height,
    int dst_x,
    int dst_y,
    int output_rgb,
    float4 mean,
    float4 scale,
    float4 padding)
{
    PROCESS_NV12_IDENTITY_GROUP(
        store_float4_output,
        store_float_output);
}

__kernel void preprocess_bgr_buffer_f32(
    __global const uchar* input,
    int input_stride,
    __global float* output,
    COMMON_ARGUMENTS)
{
    PROCESS_PIXEL(
        sample_bgr_buffer(
            input,
            input_stride,
            input_width,
            input_height,
            source.x,
            source.y,
            interpolation),
        float,
        store_float_output);
}

__kernel void preprocess_bgr_buffer_identity_f32(
    __global const uchar* input,
    int input_stride,
    __global float* output,
    int input_width,
    int input_height,
    int output_width,
    int output_height,
    int dst_x,
    int dst_y,
    int output_rgb,
    float4 mean,
    float4 scale,
    float4 padding)
{
    PROCESS_BGR_IDENTITY_GROUP(
        store_float4_output,
        store_float_output);
}

#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void preprocess_nv12_images_f16(
    read_only image2d_t y_image,
    read_only image2d_t uv_image,
    __global half* output,
    COMMON_ARGUMENTS)
{
    PROCESS_PIXEL(
        sample_nv12_images(
            y_image, uv_image, source.x, source.y, interpolation),
        half,
        store_half_output);
}

__kernel void preprocess_nv12_images_fast_f16(
    read_only image2d_t y_image,
    read_only image2d_t uv_image,
    __global half* output,
    COMMON_ARGUMENTS)
{
    PROCESS_PIXEL(
        sample_nv12_images_fast(
            y_image, uv_image, source.x, source.y, interpolation),
        half,
        store_half_output);
}

__kernel void preprocess_nv12_images_identity_f16(
    read_only image2d_t y_image,
    read_only image2d_t uv_image,
    __global half* output,
    int input_width,
    int input_height,
    int output_width,
    int output_height,
    int dst_x,
    int dst_y,
    int output_rgb,
    float4 mean,
    float4 scale,
    float4 padding)
{
    PROCESS_NV12_IDENTITY_GROUP(
        store_half4_output,
        store_half_output);
}

__kernel void preprocess_bgr_buffer_f16(
    __global const uchar* input,
    int input_stride,
    __global half* output,
    COMMON_ARGUMENTS)
{
    PROCESS_PIXEL(
        sample_bgr_buffer(
            input,
            input_stride,
            input_width,
            input_height,
            source.x,
            source.y,
            interpolation),
        half,
        store_half_output);
}

__kernel void preprocess_bgr_buffer_identity_f16(
    __global const uchar* input,
    int input_stride,
    __global half* output,
    int input_width,
    int input_height,
    int output_width,
    int output_height,
    int dst_x,
    int dst_y,
    int output_rgb,
    float4 mean,
    float4 scale,
    float4 padding)
{
    PROCESS_BGR_IDENTITY_GROUP(
        store_half4_output,
        store_half_output);
}
#endif
