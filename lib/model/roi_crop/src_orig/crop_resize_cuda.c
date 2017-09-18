#include <THC/THC.h>
#include <stdbool.h>
#include <stdio.h>
#include "crop_resize_cuda_kernel.h"

#define real float

// this symbol will be resolved automatically from PyTorch libs
extern THCState *state;

// Bilinear sampling is done in BHWD (coalescing is not obvious in BDHW)
// we assume BHWD format in inputImages
// we assume BHW(YX) format on grids

int BilinearSamplerBHWD_updateOutput_cuda(THCudaTensor *inputImages, THCudaTensor *grids, THCudaTensor *output)
{
//  THCState *state = getCutorchState(L);
//  THCudaTensor *inputImages = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
//  THCudaTensor *grids = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");
//  THCudaTensor *output = (THCudaTensor *)luaT_checkudata(L, 4, "torch.CudaTensor");

  int success = 0;
  success = BilinearSamplerBHWD_updateOutput_cuda_kernel(output->size[2],
                                               output->size[1],
                                               output->size[0],
                                               THCudaTensor_size(state, inputImages, 3),
                                               THCudaTensor_size(state, inputImages, 1),
                                               THCudaTensor_size(state, inputImages, 2),
                                               THCudaTensor_size(state, inputImages, 0),
                                               THCudaTensor_size(state, output, 2),
                                               THCudaTensor_data(state, inputImages),
                                               THCudaTensor_stride(state, inputImages, 0),
                                               THCudaTensor_stride(state, inputImages, 3),
                                               THCudaTensor_stride(state, inputImages, 1),
                                               THCudaTensor_stride(state, inputImages, 2),
                                               THCudaTensor_data(state, grids),
                                               THCudaTensor_stride(state, grids, 0),
                                               THCudaTensor_stride(state, grids, 3),
                                               THCudaTensor_stride(state, grids, 1),
                                               THCudaTensor_stride(state, grids, 2),
                                               THCudaTensor_data(state, output),
                                               THCudaTensor_stride(state, output, 0),
                                               THCudaTensor_stride(state, output, 3),
                                               THCudaTensor_stride(state, output, 1),
                                               THCudaTensor_stride(state, output, 2),
                                               THCState_getCurrentStream(state));

  //check for errors
  if (!success) {
    THError("aborting");
  }
  return 1;
}

int BilinearSamplerBHWD_updateGradInput_cuda(THCudaTensor *inputImages, THCudaTensor *grids, THCudaTensor *gradInputImages,
                                        THCudaTensor *gradGrids, THCudaTensor *gradOutput)
{
//  THCState *state = getCutorchState(L);
//  THCudaTensor *inputImages = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
//  THCudaTensor *grids = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");
//  THCudaTensor *gradInputImages = (THCudaTensor *)luaT_checkudata(L, 4, "torch.CudaTensor");
//  THCudaTensor *gradGrids = (THCudaTensor *)luaT_checkudata(L, 5, "torch.CudaTensor");
//  THCudaTensor *gradOutput = (THCudaTensor *)luaT_checkudata(L, 6, "torch.CudaTensor");

  int success = 0;
  success = BilinearSamplerBHWD_updateGradInput_cuda_kernel(gradOutput->size[2],
                                                  gradOutput->size[1],
                                                  gradOutput->size[0],
                                                  THCudaTensor_size(state, inputImages, 3),
                                                  THCudaTensor_size(state, inputImages, 1),
                                                  THCudaTensor_size(state, inputImages, 2),
                                                  THCudaTensor_size(state, inputImages, 0),
                                                  THCudaTensor_size(state, gradOutput, 2),
                                                  THCudaTensor_data(state, inputImages),
                                                  THCudaTensor_stride(state, inputImages, 0),
                                                  THCudaTensor_stride(state, inputImages, 3),
                                                  THCudaTensor_stride(state, inputImages, 1),
                                                  THCudaTensor_stride(state, inputImages, 2),
                                                  THCudaTensor_data(state, grids),
                                                  THCudaTensor_stride(state, grids, 0),
                                                  THCudaTensor_stride(state, grids, 3),
                                                  THCudaTensor_stride(state, grids, 1),
                                                  THCudaTensor_stride(state, grids, 2),
                                                  THCudaTensor_data(state, gradInputImages),
                                                  THCudaTensor_stride(state, gradInputImages, 0),
                                                  THCudaTensor_stride(state, gradInputImages, 3),
                                                  THCudaTensor_stride(state, gradInputImages, 1),
                                                  THCudaTensor_stride(state, gradInputImages, 2),
                                                  THCudaTensor_data(state, gradGrids),
                                                  THCudaTensor_stride(state, gradGrids, 0),
                                                  THCudaTensor_stride(state, gradGrids, 3),
                                                  THCudaTensor_stride(state, gradGrids, 1),
                                                  THCudaTensor_stride(state, gradGrids, 2),
                                                  THCudaTensor_data(state, gradOutput),
                                                  THCudaTensor_stride(state, gradOutput, 0),
                                                  THCudaTensor_stride(state, gradOutput, 3),
                                                  THCudaTensor_stride(state, gradOutput, 1),
                                                  THCudaTensor_stride(state, gradOutput, 2),
                                                  THCState_getCurrentStream(state));

  //check for errors
  if (!success) {
    THError("aborting");
  }
  return 1;
}

int BilinearSamplerBHWD_updateGradInputOnlyGrid_cuda(THCudaTensor *inputImages, THCudaTensor *grids,
                                        THCudaTensor *gradGrids, THCudaTensor *gradOutput)
{
//  THCState *state = getCutorchState(L);
//  THCudaTensor *inputImages = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
//  THCudaTensor *grids = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");
//  THCudaTensor *gradGrids = (THCudaTensor *)luaT_checkudata(L, 5, "torch.CudaTensor");
//  THCudaTensor *gradOutput = (THCudaTensor *)luaT_checkudata(L, 6, "torch.CudaTensor");

  int success = 0;
  success = BilinearSamplerBHWD_updateGradInputOnlyGrid_cuda_kernel(
                                                  gradOutput->size[2],
                                                  gradOutput->size[1],
                                                  gradOutput->size[0],
                                                  THCudaTensor_size(state, inputImages, 3),
                                                  THCudaTensor_size(state, inputImages, 1),
                                                  THCudaTensor_size(state, inputImages, 2),
                                                  THCudaTensor_size(state, inputImages, 0),
                                                  THCudaTensor_size(state, gradOutput, 2),
                                                  THCudaTensor_data(state, inputImages),
                                                  THCudaTensor_stride(state, inputImages, 0),
                                                  THCudaTensor_stride(state, inputImages, 3),
                                                  THCudaTensor_stride(state, inputImages, 1),
                                                  THCudaTensor_stride(state, inputImages, 2),
                                                  THCudaTensor_data(state, grids),
                                                  THCudaTensor_stride(state, grids, 0),
                                                  THCudaTensor_stride(state, grids, 3),
                                                  THCudaTensor_stride(state, grids, 1),
                                                  THCudaTensor_stride(state, grids, 2),
                                                  THCudaTensor_data(state, gradGrids),
                                                  THCudaTensor_stride(state, gradGrids, 0),
                                                  THCudaTensor_stride(state, gradGrids, 3),
                                                  THCudaTensor_stride(state, gradGrids, 1),
                                                  THCudaTensor_stride(state, gradGrids, 2),
                                                  THCudaTensor_data(state, gradOutput),
                                                  THCudaTensor_stride(state, gradOutput, 0),
                                                  THCudaTensor_stride(state, gradOutput, 3),
                                                  THCudaTensor_stride(state, gradOutput, 1),
                                                  THCudaTensor_stride(state, gradOutput, 2),
                                                  THCState_getCurrentStream(state));

  //check for errors
  if (!success) {
    THError("aborting");
  }
  return 1;
}




