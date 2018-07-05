int roi_align_forward(int aligned_height, int aligned_width, float spatial_scale,
                      THFloatTensor * features, THFloatTensor * rois, THFloatTensor * output);

int roi_align_backward(int aligned_height, int aligned_width, float spatial_scale,
                      THFloatTensor * top_grad, THFloatTensor * rois, THFloatTensor * bottom_grad);
