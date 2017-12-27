#include <THC/THC.h>
#include <stdbool.h>
#include <stdio.h>
#include "nms_cuda_kernel.h"

extern THCState *state;

int nms_cuda(THCudaTensor *keep_out, THCudaTensor *num_out,
             THCudaTensor *boxes_host, THCudaTensor *nms_overlap_thresh) {

    // printf("start to run nsm kernel\n");
    // printf("size 0: %d\n", THCudaTensor_stride(state, boxes_host, 0));
    // float *boxes = THCudaTensor_data(state, boxes_host);
    // printf("threshold: %f\n", boxes[0]);
    // float *thresh = THCudaTensor_data(state, nms_overlap_thresh);
    // printf("threshold: %f\n", thresh[0]);
    float *keep_out_data = THCudaTensor_data(state, keep_out);
    float *boxes_host_data = THCudaTensor_data(state, boxes_host);

    float *num_out_data = THCudaTensor_data(state, num_out);
    float *nms_overlap_thresh_data = THCudaTensor_data(state, nms_overlap_thresh);

    printf("get all data pointer from pytorch");

    nms_cuda_kernel(keep_out_data, num_out_data, boxes_host_data,
                   (int)(boxes_host->size[0]), (int)(boxes_host->size[1]),
                   nms_overlap_thresh_data[0]);

    return 1;
}

// int test(THCudaTensor *input) {
//   printf("run test\n");
//   float *input_data = THCudaTensor_data(state, input);
//   // printf("data value: %d\n", input_data[0]);
//   return 1;
// }
