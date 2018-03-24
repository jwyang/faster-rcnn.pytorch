#include <THC/THC.h>
#include <math.h>
#include "roioffset_pooling_kernel.h"

extern THCState *state;

int roioffset_pooling_forward_cuda(int pooled_height, int pooled_width, float spatial_scale,
                        THCudaTensor * features, THCudaTensor * rois, THCudaTensor * offset, THCudaTensor * output, THCudaIntTensor * argmax)
{
    // Grab the input tensor
    float * data_flat = THCudaTensor_data(state, features);
    float * rois_flat = THCudaTensor_data(state, rois);
    float * offset_flat = THCudaTensor_data(state, offset);

    float * output_flat = THCudaTensor_data(state, output);
    int * argmax_flat = THCudaIntTensor_data(state, argmax);

    // Number of ROIs
    int num_rois = THCudaTensor_size(state, rois, 0);
    int size_rois = THCudaTensor_size(state, rois, 1);
    if (size_rois != 5)
    {
        return 0;
    }

    // batch size
    // int batch_size = THCudaTensor_size(state, features, 0);
    // if (batch_size != 1)
    // {
    //     return 0;
    // }
    // data height
    int data_height = THCudaTensor_size(state, features, 2);
    // data width
    int data_width = THCudaTensor_size(state, features, 3);
    // Number of channels
    int num_channels = THCudaTensor_size(state, features, 1);

    cudaStream_t stream = THCState_getCurrentStream(state);

    ROIOffsetPoolForwardLauncher(
        data_flat, spatial_scale, num_rois, data_height,
        data_width, num_channels, pooled_height,
        pooled_width, rois_flat, offset_flat,
        output_flat, argmax_flat, stream);

    return 1;
}

int roioffset_pooling_backward_cuda(int pooled_height, int pooled_width, float spatial_scale,
                        THCudaTensor * top_grad, THCudaTensor * rois, THCudaTensor * offsets, THCudaTensor * bottom_grad, THCudaTensor * offset_grad, THCudaIntTensor * argmax, THCudaTensor * features)
{
    // Grab the input tensor
    float * top_grad_flat = THCudaTensor_data(state, top_grad);
    float * rois_flat = THCudaTensor_data(state, rois);
    float * offset_flat = THCudaTensor_data(state, offsets);
    float * data_flat = THCudaTensor_data(state, features);

    float * bottom_grad_flat = THCudaTensor_data(state, bottom_grad);
    float * offset_grad_flat = THCudaTensor_data(state, offset_grad);
    int * argmax_flat = THCudaIntTensor_data(state, argmax);

    // Number of ROIs
    int num_rois = THCudaTensor_size(state, rois, 0);
    int size_rois = THCudaTensor_size(state, rois, 1);
    if (size_rois != 5)
    {
        return 0;
    }

    // batch size
    int batch_size = THCudaTensor_size(state, bottom_grad, 0);
    // if (batch_size != 1)
    // {
    //     return 0;
    // }
    // data height
    int data_height = THCudaTensor_size(state, bottom_grad, 2);
    // data width
    int data_width = THCudaTensor_size(state, bottom_grad, 3);
    // Number of channels
    int num_channels = THCudaTensor_size(state, bottom_grad, 1);

    cudaStream_t stream = THCState_getCurrentStream(state);
    ROIOffsetPoolBackwardLauncher(
        top_grad_flat, spatial_scale, batch_size, num_rois, data_height,
        data_width, num_channels, pooled_height,
        pooled_width, rois_flat, offset_flat,
        bottom_grad_flat, offset_grad_flat, argmax_flat, data_flat, stream);

    return 1;
}
