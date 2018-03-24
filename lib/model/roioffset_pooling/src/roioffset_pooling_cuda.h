int roioffset_pooling_forward_cuda(int pooled_height, int pooled_width, float spatial_scale,
                        THCudaTensor * features, THCudaTensor * rois, THCudaTensor * offset, THCudaTensor * output, THCudaIntTensor * argmax);

int roioffset_pooling_backward_cuda(int pooled_height, int pooled_width, float spatial_scale,
                        THCudaTensor * top_grad, THCudaTensor * rois, THCudaTensor * offsets, THCudaTensor * bottom_grad, THCudaTensor * offset_grad, THCudaIntTensor * argmax, THCudaTensor * features);