// int nms_cuda(THCudaTensor *keep_out, THCudaTensor *num_out,
//             THCudaTensor *boxes_host, THCudaTensor *nms_overlap_thresh);

int nms_cuda(THCudaIntTensor *keep_out, THCudaTensor *boxes_host,
             THCudaIntTensor *num_out, float nms_overlap_thresh);
