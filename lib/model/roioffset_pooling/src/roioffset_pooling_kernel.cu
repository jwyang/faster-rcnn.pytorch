// #ifdef __cplusplus
// extern "C" {
// #endif

#include <stdio.h>
#include <vector>
#include <math.h>
#include <float.h>
#include "roioffset_pooling_kernel.h"


#define DIVUP(m, n) ((m) / (m) + ((m) % (n) > 0))

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

__device__ float index_bilinear(const float* data, const int base_index, const int height, const int width, float* oy, float* ox)
{
    // transform into data dimensions
    float d_oy = (*oy) * height;
    float d_ox = (*ox) * width;
    
    if (d_ox > width-1 - (base_index % width)) {
        d_ox = width-1 - (base_index % width);
        *ox = d_ox / width;
    } else if (d_ox < -(base_index % width)) {
        d_ox = -(base_index % width);
        *ox = d_ox / width;
    }
    
    int area = width * height;
    float oy_max = floorf( (area-1 - (base_index % area)) / width );
    if (d_oy > floorf( (area-1 - (base_index % area)) / width )) {
        d_oy = floorf( (area-1 - (base_index % area)) / width );
        *oy = d_oy / height;
    } else if (d_oy < -floorf( (base_index % area) / width )) {
        d_oy = -floorf( (base_index % area) / width );
        *oy = d_oy / height;
    }
    
    if (d_ox > width-1 - (base_index % width)) {
        d_ox = width-1 - (base_index % width);
    } else if (d_ox < -(base_index % width)) {
        d_ox = -(base_index % width);
    }
    
    int cx = (int) ceilf(d_ox);
    int fx = (int) floorf(d_ox);
    int cy = (int) ceilf(d_oy);
    int fy = (int) floorf(d_oy);
    
    int i00 = (int) (base_index + fy * width + fx);
    int i01 = (int) (base_index + fy * width + cx);
    int i10 = (int) (base_index + cy * width + fx);
    int i11 = (int) (base_index + cy * width + cx);
    
    float out = 0;
    out = (1-(d_oy - fy)) * (1-(d_ox - fx)) * (data[i00]);
    out += (1-(d_oy - fy)) * (1-(cx - d_ox)) * (data[i01]);
    out += (1-(cy - d_oy)) * (1-(d_ox - fx)) * (data[i10]);
    out += (1-(cy - d_oy)) * (1-(cx - d_ox)) * (data[i11]);
    
    out /= powf( 2, (fx == cx) + (fy == cy) );
    return out;

}


__global__ void ROIOffsetPoolForward(const int nthreads, const float* bottom_data,
    const float spatial_scale, const int height, const int width,
    const int channels, const int pooled_height, const int pooled_width,
    const float* bottom_rois, const float* offsets, float* top_data, int* argmax_data)
{
    CUDA_KERNEL_LOOP(index, nthreads)
    {
        // (n, c, ph, pw) is an element in the pooled output
        // int n = index;
        // int pw = n % pooled_width;
        // n /= pooled_width;
        // int ph = n % pooled_height;
        // n /= pooled_height;
        // int c = n % channels;
        // n /= channels;
        int pw = index % pooled_width;
        int ph = (index / pooled_width) % pooled_height;
        int c  = (index / pooled_width / pooled_height) % channels;
        int n  = index / pooled_width / pooled_height / channels;
        
        // bottom_rois += n * 5;
        int roi_batch_ind = bottom_rois[n * 5 + 0];
        int roi_start_w = round(bottom_rois[n * 5 + 1] * spatial_scale);
        int roi_start_h = round(bottom_rois[n * 5 + 2] * spatial_scale);
        int roi_end_w = round(bottom_rois[n * 5 + 3] * spatial_scale);
        int roi_end_h = round(bottom_rois[n * 5 + 4] * spatial_scale);

        // Force malformed ROIs to be 1x1
        int roi_width = fmaxf(roi_end_w - roi_start_w + 1, 1);
        int roi_height = fmaxf(roi_end_h - roi_start_h + 1, 1);
        float bin_size_h = (float)(roi_height) / (float)(pooled_height);
        float bin_size_w = (float)(roi_width) / (float)(pooled_width);
     
        int hstart = (int)(floorf((float)(ph) * bin_size_h));
        int wstart = (int)(floorf((float)(pw) * bin_size_w));
        int hend = (int)(ceilf((float)(ph + 1) * bin_size_h));
        int wend = (int)(ceilf((float)(pw + 1) * bin_size_w));

        // Add roi offsets and clip to input boundaries
        hstart = fminf(fmaxf(hstart + roi_start_h, 0), height);
        hend = fminf(fmaxf(hend + roi_start_h, 0), height);
        wstart = fminf(fmaxf(wstart + roi_start_w, 0), width);
        wend = fminf(fmaxf(wend + roi_start_w, 0), width);
        bool is_empty = (hend <= hstart) || (wend <= wstart);

        // Define an empty pooling region to be zero
        float maxval = is_empty ? 0 : -FLT_MAX;
        // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
        int maxidx = -1;

        int bottom_data_batch_offset = roi_batch_ind * channels * height * width;
        int bottom_data_offset = bottom_data_batch_offset + c * height * width;

        // Add learned offsets
        float* ox = (float*) offsets+2*index;
        float* oy = (float*) offsets+2*index+1;
        
        for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
                int bottom_index = h * width + w;
                float val = index_bilinear(bottom_data, bottom_data_offset + bottom_index, height, width, oy, ox);
                if (val > maxval) {
                    maxval = val;
                    maxidx = bottom_data_offset + bottom_index;
                }
            }
        }
        top_data[index] = maxval;
        if (argmax_data != NULL)
            argmax_data[index] = maxidx;
    }
}

int ROIOffsetPoolForwardLauncher(
    const float* bottom_data, const float spatial_scale, const int num_rois, const int height,
    const int width, const int channels, const int pooled_height,
    const int pooled_width, const float* bottom_rois, const float* offset,
    float* top_data, int* argmax_data, cudaStream_t stream)
{
    const int kThreadsPerBlock = 1024;
    int output_size = num_rois * pooled_height * pooled_width * channels;
    cudaError_t err;

    ROIOffsetPoolForward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(
      output_size, bottom_data, spatial_scale, height, width, channels, pooled_height,
      pooled_width, bottom_rois, offset, top_data, argmax_data);

    // dim3 blocks(DIVUP(output_size, kThreadsPerBlock),
    //             DIVUP(output_size, kThreadsPerBlock));
    // dim3 threads(kThreadsPerBlock);
    //
    // ROIPoolForward<<<blocks, threads, 0, stream>>>(
    //   output_size, bottom_data, spatial_scale, height, width, channels, pooled_height,
    //   pooled_width, bottom_rois, top_data, argmax_data);

    err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }

    return 1;
}


__global__ void ROIOffsetPoolBackward_feature(const int nthreads, const float* top_diff,
    const int* argmax_data, const int num_rois, const float spatial_scale,
    const int height, const int width, const int channels,
    const int pooled_height, const int pooled_width, float* bottom_diff,
    const float* bottom_rois, const float* offsets) {
    CUDA_1D_KERNEL_LOOP(index, nthreads)
    {
    
        // Iterate over input feature map
        
        int n = index;
        int w = n % width;
        n /= width;
        int h = n % height;
        n /= height;
        int c = n % channels;
        n /= channels;
        
        float gradient = 0;
        // Accumulate gradient over all ROIs that pooled this element
        
        for (int roi_n = 0; roi_n < num_rois; ++roi_n)
        {
            if (n != bottom_rois[roi_n*5 + 0]) {
                continue;
            }
            
            int base_offset = roi_n*pooled_height*pooled_width*channels + c*pooled_height*pooled_width;
            for (int a = 0; a < pooled_height * pooled_width; ++a)
            {
                int i = base_offset + a;
                
                float d_ox = offsets[2*i] * width;
                float d_oy = offsets[2*i+1] * height;

                float w_diff = abs(w-d_ox - argmax_data[i]%width);
                float h_diff = abs(h-d_oy - (argmax_data[i]/width)%height);

                if (w_diff < 1 && h_diff < 1)
                {
                    gradient += top_diff[i] * (1-w_diff) * (1-h_diff) / powf( 2, ((w_diff == 0) + (h_diff == 0)) );
                }
            }
        }
        bottom_diff[index] = gradient;
        
  }
}

__device__ int sgn(const float x)
{
    if (x > 0) {
        return 1;
    } else if (x == 0) {
        return 0;
    } else {
        return -1;
    }
}

__global__ void ROIOffsetPoolBackward_offset(const int nthreads, const float* top_diff,
    const int* argmax_data, const int num_rois, const float spatial_scale,
    const int height, const int width, const int channels,
    const int pooled_height, const int pooled_width, float* offset_diff,
    const float* bottom_rois, const float* offsets, const float* bottom_data) {
    CUDA_1D_KERNEL_LOOP(index, nthreads)
    {        
    
        // Iterate over output pool
        
        // Accumulate gradient over all elements that contributed to this bin
        
        float* ox = (float*) offsets+2*index;
        float* oy = (float*) offsets+2*index+1;
        
        // transform into data dimensions
        float d_oy = (*oy) * height;
        float d_ox = (*ox) * width;
                
        int cx = (int) ceilf(d_ox);
        int fx = (int) floorf(d_ox);
        int cy = (int) ceilf(d_oy);
        int fy = (int) floorf(d_oy);

        int base_index = argmax_data[index];
        int i00 = (int) (base_index + fy * width + fx);
        int i01 = (int) (base_index + fy * width + cx);
        int i10 = (int) (base_index + cy * width + fx);
        int i11 = (int) (base_index + cy * width + cx);
        
        float grad_y = 0;
        grad_y += (1 - (d_ox - fx)) * sgn(d_oy - fy) * bottom_data[i00];
        grad_y += (1 - (cx - d_ox)) * sgn(d_oy - fy) * bottom_data[i01];
        grad_y += (1 - (d_ox - fx)) * sgn(cy - d_oy) * bottom_data[i10];
        grad_y += (1 - (cx - d_ox)) * sgn(cy - d_oy) * bottom_data[i11];
        
        float grad_x = 0;
        grad_x += (1 - (d_oy - fy)) * sgn(d_ox - fx) * bottom_data[i00];
        grad_x += (1 - (cy - d_oy)) * sgn(d_ox - fx) * bottom_data[i10];
        grad_x += (1 - (d_oy - fy)) * sgn(cx - d_ox) * bottom_data[i01];
        grad_x += (1 - (cy - d_oy)) * sgn(cx - d_ox) * bottom_data[i11];
        
        offset_diff[2*index] = top_diff[index] * grad_x / powf( 2, ((fx == cx) + (fy == cy)) );
        offset_diff[2*index+1] = top_diff[index] * grad_y / powf( 2, ((fx == cx) + (fy == cy)) );
       
  }
}


int ROIOffsetPoolBackwardLauncher(const float* top_diff, const float spatial_scale, const int batch_size, const int num_rois,
    const int height, const int width, const int channels, const int pooled_height,
    const int pooled_width, const float* bottom_rois, const float* offsets,
    float* bottom_diff, float* offset_diff, const int* argmax_data, const float* bottom_data, cudaStream_t stream)
{
    const int kThreadsPerBlock = 1024;
    int output_size = batch_size * height * width * channels;
    cudaError_t err;

    ROIOffsetPoolBackward_feature<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(
      output_size, top_diff, argmax_data, num_rois, spatial_scale, height, width, channels, pooled_height,
      pooled_width, bottom_diff, bottom_rois, offsets);

    err = cudaGetLastError();
    
    if(cudaSuccess != err)
    {
        fprintf( stderr, "ROIOffsetPoolBackward_feature: cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }
    
    output_size = num_rois * pooled_height * pooled_width * channels;
    
    ROIOffsetPoolBackward_offset<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(
      output_size, top_diff, argmax_data, num_rois, spatial_scale, height, width, channels, pooled_height,
      pooled_width, offset_diff, bottom_rois, offsets, bottom_data);

    err = cudaGetLastError();
    
    if(cudaSuccess != err)
    {
        fprintf( stderr, "ROIOffsetPoolBackward_offset: cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }
    

    return 1;
}

