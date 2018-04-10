#include <TH/TH.h>
#include <math.h>
#include <omp.h>


void ROIAlignForwardCpu(const float* bottom_data, const float spatial_scale, const int num_rois,
                     const int height, const int width, const int channels,
                     const int aligned_height, const int aligned_width, const float * bottom_rois,
                     float* top_data);

void ROIAlignBackwardCpu(const float* top_diff, const float spatial_scale, const int num_rois,
                     const int height, const int width, const int channels,
                     const int aligned_height, const int aligned_width, const float * bottom_rois,
                     float* top_data);

int roi_align_forward(int aligned_height, int aligned_width, float spatial_scale,
                     THFloatTensor * features, THFloatTensor * rois, THFloatTensor * output)
{
    //Grab the input tensor
    float * data_flat = THFloatTensor_data(features);
    float * rois_flat = THFloatTensor_data(rois);

    float * output_flat = THFloatTensor_data(output);

    // Number of ROIs
    int num_rois = THFloatTensor_size(rois, 0);
    int size_rois = THFloatTensor_size(rois, 1);
    if (size_rois != 5)
    {
        return 0;
    }

    // data height
    int data_height = THFloatTensor_size(features, 2);
    // data width
    int data_width = THFloatTensor_size(features, 3);
    // Number of channels
    int num_channels = THFloatTensor_size(features, 1);

    // do ROIAlignForward
    ROIAlignForwardCpu(data_flat, spatial_scale, num_rois, data_height, data_width, num_channels,
            aligned_height, aligned_width, rois_flat, output_flat);

    return 1;
}

int roi_align_backward(int aligned_height, int aligned_width, float spatial_scale,
                       THFloatTensor * top_grad, THFloatTensor * rois, THFloatTensor * bottom_grad)
{
    //Grab the input tensor
    float * top_grad_flat = THFloatTensor_data(top_grad);
    float * rois_flat = THFloatTensor_data(rois);

    float * bottom_grad_flat = THFloatTensor_data(bottom_grad);

    // Number of ROIs
    int num_rois = THFloatTensor_size(rois, 0);
    int size_rois = THFloatTensor_size(rois, 1);
    if (size_rois != 5)
    {
        return 0;
    }

    // batch size
    int batch_size = THFloatTensor_size(bottom_grad, 0);
    // data height
    int data_height = THFloatTensor_size(bottom_grad, 2);
    // data width
    int data_width = THFloatTensor_size(bottom_grad, 3);
    // Number of channels
    int num_channels = THFloatTensor_size(bottom_grad, 1);

    // do ROIAlignBackward
    ROIAlignBackwardCpu(top_grad_flat, spatial_scale, num_rois, data_height,
            data_width, num_channels, aligned_height, aligned_width, rois_flat, bottom_grad_flat);

    return 1;
}

void ROIAlignForwardCpu(const float* bottom_data, const float spatial_scale, const int num_rois,
                     const int height, const int width, const int channels,
                     const int aligned_height, const int aligned_width, const float * bottom_rois,
                     float* top_data)
{
    const int output_size = num_rois * aligned_height * aligned_width * channels;

    #pragma omp parallel for 
    for (int idx = 0; idx < output_size; ++idx)
    {
        // (n, c, ph, pw) is an element in the aligned output
        int pw = idx % aligned_width;
        int ph = (idx / aligned_width) % aligned_height;
        int c = (idx / aligned_width / aligned_height) % channels;
        int n = idx / aligned_width / aligned_height / channels;

        float roi_batch_ind = bottom_rois[n * 5 + 0];
        float roi_start_w = bottom_rois[n * 5 + 1] * spatial_scale;
        float roi_start_h = bottom_rois[n * 5 + 2] * spatial_scale;
        float roi_end_w = bottom_rois[n * 5 + 3] * spatial_scale;
        float roi_end_h = bottom_rois[n * 5 + 4] * spatial_scale;

        // Force malformed ROI to be 1x1
        float roi_width = fmaxf(roi_end_w - roi_start_w + 1., 0.);
        float roi_height = fmaxf(roi_end_h - roi_start_h + 1., 0.);
        float bin_size_h = roi_height / (aligned_height - 1.);
        float bin_size_w = roi_width / (aligned_width - 1.);

        float h = (float)(ph) * bin_size_h + roi_start_h;
        float w = (float)(pw) * bin_size_w + roi_start_w;

        int hstart = fminf(floor(h), height - 2);
        int wstart = fminf(floor(w), width - 2);

        int img_start = roi_batch_ind * channels * height * width;

        // bilinear interpolation
        if (h < 0 || h >= height || w < 0 || w >= width)
        {
            top_data[idx] = 0.;
        }
        else
        {
            float h_ratio = h - (float)(hstart);
            float w_ratio = w - (float)(wstart);
            int upleft = img_start + (c * height + hstart) * width + wstart;
            int upright = upleft + 1;
            int downleft = upleft + width;
            int downright = downleft + 1;

            top_data[idx] = bottom_data[upleft] * (1. - h_ratio) * (1. - w_ratio)
                + bottom_data[upright] * (1. - h_ratio) * w_ratio
                + bottom_data[downleft] * h_ratio * (1. - w_ratio)
                + bottom_data[downright] * h_ratio * w_ratio;
        }
    }
}

void ROIAlignBackwardCpu(const float* top_diff, const float spatial_scale, const int num_rois,
                     const int height, const int width, const int channels,
                     const int aligned_height, const int aligned_width, const float * bottom_rois,
                     float* bottom_diff)
{
    const int output_size = num_rois * aligned_height * aligned_width * channels;

    #pragma omp parallel for 
    for (int idx = 0; idx < output_size; ++idx)
    {
        // (n, c, ph, pw) is an element in the aligned output
        int pw = idx % aligned_width;
        int ph = (idx / aligned_width) % aligned_height;
        int c = (idx / aligned_width / aligned_height) % channels;
        int n = idx / aligned_width / aligned_height / channels;

        float roi_batch_ind = bottom_rois[n * 5 + 0];
        float roi_start_w = bottom_rois[n * 5 + 1] * spatial_scale;
        float roi_start_h = bottom_rois[n * 5 + 2] * spatial_scale;
        float roi_end_w = bottom_rois[n * 5 + 3] * spatial_scale;
        float roi_end_h = bottom_rois[n * 5 + 4] * spatial_scale;

        // Force malformed ROI to be 1x1
        float roi_width = fmaxf(roi_end_w - roi_start_w + 1., 0.);
        float roi_height = fmaxf(roi_end_h - roi_start_h + 1., 0.);
        float bin_size_h = roi_height / (aligned_height - 1.);
        float bin_size_w = roi_width / (aligned_width - 1.);

        float h = (float)(ph) * bin_size_h + roi_start_h;
        float w = (float)(pw) * bin_size_w + roi_start_w;

        int hstart = fminf(floor(h), height - 2);
        int wstart = fminf(floor(w), width - 2);

        int img_start = roi_batch_ind * channels * height * width;

        // bilinear interpolation
        if (h < 0 || h >= height || w < 0 || w >= width)
        {
            float h_ratio = h - (float)(hstart);
            float w_ratio = w - (float)(wstart);
            int upleft = img_start + (c * height + hstart) * width + wstart;
            int upright = upleft + 1;
            int downleft = upleft + width;
            int downright = downleft + 1;

            bottom_diff[upleft] += top_diff[idx] * (1. - h_ratio) * (1. - w_ratio);
            bottom_diff[upright] += top_diff[idx] * (1. - h_ratio) *  w_ratio;
            bottom_diff[downleft] += top_diff[idx] * h_ratio * (1. - w_ratio);
            bottom_diff[downright] += top_diff[idx] * h_ratio * w_ratio;
        }
    }
}
