// Bilinear sampling is done in BHWD (coalescing is not obvious in BDHW)
// we assume BHWD format in inputImages
// we assume BHW(YX) format on grids

int BilinearSamplerBHWD_updateOutput_cuda(THCudaTensor *inputImages, THCudaTensor *grids, THCudaTensor *output, int *);

int BilinearSamplerBHWD_updateGradInput_cuda(THCudaTensor *inputImages, THCudaTensor *grids, THCudaTensor *gradInputImages,
                                        THCudaTensor *gradGrids, THCudaTensor *gradOutput, int *);

int BilinearSamplerBHWD_updateGradInputOnlyGrid_cuda(THCudaTensor *inputImages, THCudaTensor *grids,
                                        THCudaTensor *gradGrids, THCudaTensor *gradOutput, int *);
