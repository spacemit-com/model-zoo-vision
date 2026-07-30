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
#endif
