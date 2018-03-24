#ifndef _ROIOFFSET_POOLING_KERNEL
#define _ROIOFFSET_POOLING_KERNEL

#ifdef __cplusplus
extern "C" {
#endif

int ROIOffsetPoolForwardLauncher(
    const float* bottom_data, const float spatial_scale, const int num_rois, const int height,
    const int width, const int channels, const int pooled_height,
    const int pooled_width, const float* bottom_rois, const float* offsets,
    float* top_data, int* argmax_data, cudaStream_t stream);


int ROIOffsetPoolBackwardLauncher(const float* top_diff, const float spatial_scale, const int batch_size, const int num_rois,
    const int height, const int width, const int channels, const int pooled_height,
    const int pooled_width, const float* bottom_rois, const float* offsets,
    float* bottom_diff, float* offset_diff, const int* argmax_data, const float* bottom_data, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif

