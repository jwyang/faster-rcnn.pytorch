#ifdef __cplusplus
extern "C" {
#endif

void nms_cuda_compute(int* keep_out, int *num_out, float* boxes_host, int boxes_num,
          int boxes_dim, float nms_overlap_thresh);

#ifdef __cplusplus
}
#endif
