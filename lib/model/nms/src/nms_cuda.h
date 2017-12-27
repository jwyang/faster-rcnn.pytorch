// #include <THC/THC.h>
// //
// struct THCState;

int nms_cuda(THCudaTensor *keep_out, THCudaTensor *num_out,
             THCudaTensor *boxes_host, THCudaTensor *nms_overlap_thresh);
// int test(THCudaTensor *input);
