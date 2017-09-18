#include <stdbool.h>
#include <stdio.h>
#include "crop_resize_cuda_kernel.h"

#define real float

// Bilinear sampling is done in BHWD (coalescing is not obvious in BDHW)
// we assume BHWD format in inputImages
// we assume BHW(YX) format on grids

__device__ void getTopLeft(float x, int width, int& point, float& weight)
{
   /* for interpolation :
      stores in point and weight :
      - the x-coordinate of the pixel on the left (or y-coordinate of the upper pixel)
      - the weight for interpolating
   */

   float xcoord = (x + 1) * (width - 1) / 2;
   point = floor(xcoord);
   weight = 1 - (xcoord - point);
}

__device__ bool between(int value, int lowerBound, int upperBound)
{
   return (value >= lowerBound && value <= upperBound);
}

__device__ void sumReduceShMem(volatile float s[])
{
   /* obviously only works for 32 elements */
   /* sums up a shared memory array of 32 elements, stores it in s[0] */
   /* whole warp can then read first element (broadcasting) */
   if(threadIdx.x<16) { s[threadIdx.x] = s[threadIdx.x] + s[threadIdx.x+16]; }
   if(threadIdx.x<8) { s[threadIdx.x] = s[threadIdx.x] + s[threadIdx.x+8]; }
   if(threadIdx.x<4) { s[threadIdx.x] = s[threadIdx.x] + s[threadIdx.x+4]; }
   if(threadIdx.x<2) { s[threadIdx.x] = s[threadIdx.x] + s[threadIdx.x+2]; }
   if(threadIdx.x<1) { s[threadIdx.x] = s[threadIdx.x] + s[threadIdx.x+1]; }
}

__global__ void bilinearSamplingFromGrid(float* inputImages_data, int inputImages_strideBatch, int inputImages_strideChannels, int inputImages_strideHeight, int inputImages_strideWidth,
                                         float* grids_data, int grids_strideBatch, int grids_strideYX, int grids_strideHeight, int grids_strideWidth,
                                         float* output_data, int output_strideBatch, int output_strideChannels, int output_strideHeight, int output_strideWidth,
                                         int inputImages_channels, int inputImages_height, int inputImages_width, int output_width, int roiPerImage)
{
   // each (32,16) block 16 output pixels (for coalescing the grid read)
   // x,y = coordinates (xOut = blockIdx.x*16+blockDim.y+threadIdx.y)
   // z = batch index
   // threadIdx.x : used for features (coalescing is trivial)

   const int xOut = blockIdx.x*blockDim.y+threadIdx.y;
   const bool withinImageBounds = xOut < output_width;
   const bool withinGridBounds = blockIdx.x*blockDim.y + threadIdx.x / 2 < output_width;
   const int yOut = blockIdx.y;
   const int width = inputImages_width;
   const int height = inputImages_height;

   const int b = blockIdx.z;

   const int b_input = blockIdx.z / roiPerImage;

   float yf,xf;

   __shared__ float gridData[32];
   if (threadIdx.y==0 && withinGridBounds)
   {
      gridData[threadIdx.x] = grids_data[b*grids_strideBatch + yOut*grids_strideHeight + xOut*grids_strideWidth + threadIdx.x];
   }
   __syncthreads();
   if(!withinImageBounds) return;
   yf = gridData[threadIdx.y*2];
   xf = gridData[threadIdx.y*2+1];

   int yInTopLeft, xInTopLeft;
   float yWeightTopLeft, xWeightTopLeft;
   getTopLeft(xf, inputImages_width, xInTopLeft, xWeightTopLeft);
   getTopLeft(yf, inputImages_height, yInTopLeft, yWeightTopLeft);

   const int outAddress = output_strideBatch * b + output_strideHeight * yOut + output_strideWidth * xOut;

   const int inTopLeftAddress = inputImages_strideBatch * b_input + inputImages_strideHeight * yInTopLeft + inputImages_strideWidth * xInTopLeft;
   const int inTopRightAddress = inTopLeftAddress + inputImages_strideWidth;
   const int inBottomLeftAddress = inTopLeftAddress + inputImages_strideHeight;
   const int inBottomRightAddress = inBottomLeftAddress + inputImages_strideWidth;

   float v=0;
   float inTopLeft=0;
   float inTopRight=0;
   float inBottomLeft=0;
   float inBottomRight=0;

   bool topLeftIsIn = between(xInTopLeft, 0, width-1) && between(yInTopLeft, 0, height-1);
   bool topRightIsIn = between(xInTopLeft+1, 0, width-1) && between(yInTopLeft, 0, height-1);
   bool bottomLeftIsIn = between(xInTopLeft, 0, width-1) && between(yInTopLeft+1, 0, height-1);
   bool bottomRightIsIn = between(xInTopLeft+1, 0, width-1) && between(yInTopLeft+1, 0, height-1);

   // interpolation happens here
   for(int t=0; t<inputImages_channels; t++)
  //  for(int t=threadIdx.x; t<inputImages_channels; t+= blockDim.x)
   {
      if (!topLeftIsIn && !topRightIsIn && !bottomLeftIsIn && !bottomRightIsIn)
        continue;

      if(topLeftIsIn) inTopLeft = inputImages_data[inTopLeftAddress + t];
      if(topRightIsIn) inTopRight = inputImages_data[inTopRightAddress + t];
      if(bottomLeftIsIn) inBottomLeft = inputImages_data[inBottomLeftAddress + t];
      if(bottomRightIsIn) inBottomRight = inputImages_data[inBottomRightAddress + t];

      v = xWeightTopLeft * yWeightTopLeft * inTopLeft
        + (1 - xWeightTopLeft) * yWeightTopLeft * inTopRight
        + xWeightTopLeft * (1 - yWeightTopLeft) * inBottomLeft
        + (1 - xWeightTopLeft) * (1 - yWeightTopLeft) * inBottomRight;

      output_data[outAddress + t] = v;
   }
}

template<bool onlyGrid> __global__ void backwardBilinearSampling(float* inputImages_data, int inputImages_strideBatch, int inputImages_strideChannels, int inputImages_strideHeight, int inputImages_strideWidth,
                                         float* gradInputImages_data, int gradInputImages_strideBatch, int gradInputImages_strideChannels, int gradInputImages_strideHeight, int gradInputImages_strideWidth,
                                         float* grids_data, int grids_strideBatch, int grids_strideYX, int grids_strideHeight, int grids_strideWidth,
                                         float* gradGrids_data, int gradGrids_strideBatch, int gradGrids_strideYX, int gradGrids_strideHeight, int gradGrids_strideWidth,
                                         float* gradOutput_data, int gradOutput_strideBatch, int gradOutput_strideChannels, int gradOutput_strideHeight, int gradOutput_strideWidth,
                                         int inputImages_channels, int inputImages_height, int inputImages_width, int gradOutput_width, int roiPerImage)
{
   // each (32,16) block 16 output pixels (for coalescing the grid read)
   // x,y = coordinates
   // z = batch index
   // threads : used for features

   const int xOut = blockIdx.x*blockDim.y+threadIdx.y;
   const bool withinImageBounds = xOut < gradOutput_width;
   const bool withinGridBounds = blockIdx.x*blockDim.y + threadIdx.x / 2 < gradOutput_width;

   const int yOut = blockIdx.y;
   const int width = inputImages_width;
   const int height = inputImages_height;

   const int b = blockIdx.z;

   // compute the index for input image
   const int b_input = blockIdx.z / roiPerImage;

   float yf,xf;

   __shared__ float gridData[32];
   if (threadIdx.y==0 && withinGridBounds)
   {
      gridData[threadIdx.x] = grids_data[b*grids_strideBatch + yOut*grids_strideHeight + xOut*grids_strideWidth + threadIdx.x];
   }
   __syncthreads();

   if(withinImageBounds)
   {
      yf = gridData[threadIdx.y*2];
      xf = gridData[threadIdx.y*2+1];

      int yInTopLeft, xInTopLeft;
      float yWeightTopLeft, xWeightTopLeft;
      getTopLeft(xf, inputImages_width, xInTopLeft, xWeightTopLeft);
      getTopLeft(yf, inputImages_height, yInTopLeft, yWeightTopLeft);

      const int inTopLeftAddress = inputImages_strideBatch * b_input + inputImages_strideHeight * yInTopLeft + inputImages_strideWidth * xInTopLeft;
      const int inTopRightAddress = inTopLeftAddress + inputImages_strideWidth;
      const int inBottomLeftAddress = inTopLeftAddress + inputImages_strideHeight;
      const int inBottomRightAddress = inBottomLeftAddress + inputImages_strideWidth;

      const int gradInputImagesTopLeftAddress = gradInputImages_strideBatch * b_input + gradInputImages_strideHeight * yInTopLeft + gradInputImages_strideWidth * xInTopLeft;
      const int gradInputImagesTopRightAddress = gradInputImagesTopLeftAddress + gradInputImages_strideWidth;
      const int gradInputImagesBottomLeftAddress = gradInputImagesTopLeftAddress + gradInputImages_strideHeight;
      const int gradInputImagesBottomRightAddress = gradInputImagesBottomLeftAddress + gradInputImages_strideWidth;

      const int gradOutputAddress = gradOutput_strideBatch * b + gradOutput_strideHeight * yOut + gradOutput_strideWidth * xOut;

      float topLeftDotProduct = 0;
      float topRightDotProduct = 0;
      float bottomLeftDotProduct = 0;
      float bottomRightDotProduct = 0;

      bool topLeftIsIn = between(xInTopLeft, 0, width-1) && between(yInTopLeft, 0, height-1);
      bool topRightIsIn = between(xInTopLeft+1, 0, width-1) && between(yInTopLeft, 0, height-1);
      bool bottomLeftIsIn = between(xInTopLeft, 0, width-1) && between(yInTopLeft+1, 0, height-1);
      bool bottomRightIsIn = between(xInTopLeft+1, 0, width-1) && between(yInTopLeft+1, 0, height-1);

      /*
         In that loop we accumulate
         - gradients into the gradInputImages array with atomic adds
         - we compute the dot product that we need for the grid gradient
      */
      for(int t=0; t<inputImages_channels; t++)
      // for(int t=threadIdx.x; t<inputImages_channels; t+= blockDim.x)
      {
         float gradOutValue = gradOutput_data[gradOutputAddress + t];
         // bool between(int value, int lowerBound, int upperBound)
         if(topLeftIsIn)
         {
            float inTopLeft = inputImages_data[inTopLeftAddress + t];
            topLeftDotProduct += inTopLeft * gradOutValue;
            if(!onlyGrid) atomicAdd(&gradInputImages_data[gradInputImagesTopLeftAddress + t], xWeightTopLeft * yWeightTopLeft * gradOutValue);
         }

         if(topRightIsIn)
         {
            float inTopRight = inputImages_data[inTopRightAddress + t];
            topRightDotProduct += inTopRight * gradOutValue;
            if(!onlyGrid) atomicAdd(&gradInputImages_data[gradInputImagesTopRightAddress + t], (1 - xWeightTopLeft) * yWeightTopLeft * gradOutValue);
         }

         if(bottomLeftIsIn)
         {
            float inBottomLeft = inputImages_data[inBottomLeftAddress + t];
            bottomLeftDotProduct += inBottomLeft * gradOutValue;
            if(!onlyGrid) atomicAdd(&gradInputImages_data[gradInputImagesBottomLeftAddress + t], xWeightTopLeft * (1 - yWeightTopLeft) * gradOutValue);
         }

         if(bottomRightIsIn)
         {
            float inBottomRight = inputImages_data[inBottomRightAddress + t];
            bottomRightDotProduct += inBottomRight * gradOutValue;
            if(!onlyGrid) atomicAdd(&gradInputImages_data[gradInputImagesBottomRightAddress + t], (1 - xWeightTopLeft) * (1 - yWeightTopLeft) * gradOutValue);
         }
      }

      /*
         Here we reduce the dot product and compute the grid gradient before writing it.
      */

      /* could do shuffles and use no shmem at all but cuda arch is 2.0 */
      __shared__ volatile float __shmem[16][32];
      __shmem[threadIdx.y][threadIdx.x] = topLeftDotProduct;
      sumReduceShMem(__shmem[threadIdx.y]);
      topLeftDotProduct = __shmem[threadIdx.y][0];

      __shmem[threadIdx.y][threadIdx.x] = topRightDotProduct;
      sumReduceShMem(__shmem[threadIdx.y]);
      topRightDotProduct = __shmem[threadIdx.y][0];

      __shmem[threadIdx.y][threadIdx.x] = bottomLeftDotProduct;
      sumReduceShMem(__shmem[threadIdx.y]);
      bottomLeftDotProduct = __shmem[threadIdx.y][0];

      __shmem[threadIdx.y][threadIdx.x] = bottomRightDotProduct;
      sumReduceShMem(__shmem[threadIdx.y]);
      bottomRightDotProduct = __shmem[threadIdx.y][0];

      yf = - xWeightTopLeft * topLeftDotProduct + xWeightTopLeft * bottomLeftDotProduct - (1-xWeightTopLeft) * topRightDotProduct + (1-xWeightTopLeft) * bottomRightDotProduct;
      xf = - yWeightTopLeft * topLeftDotProduct + yWeightTopLeft * topRightDotProduct - (1-yWeightTopLeft) * bottomLeftDotProduct + (1-yWeightTopLeft) * bottomRightDotProduct;

      if(threadIdx.x==0)
      {
         gridData[threadIdx.y*2] = yf * (inputImages_height-1) / 2;
         gridData[threadIdx.y*2+1] = xf * (inputImages_width-1) / 2;
      }
   }// must put a big if condition in order not to hang at __syncthreads()...
   __syncthreads();

   if(threadIdx.y==0 && withinGridBounds)
       gradGrids_data[b*gradGrids_strideBatch + yOut*gradGrids_strideHeight + xOut*gradGrids_strideWidth + threadIdx.x] = gridData[threadIdx.x];
}


#ifdef __cplusplus
extern "C" {
#endif

int BilinearSamplerBHWD_updateOutput_cuda_kernel(/*output->size[2]*/int sz1,
                                                 /*output->size[1]*/int sz2,
                                                 /*output->size[0]*/int sz3,
                                                 /*THCudaTensor_size(state, inputImages, 3)*/int ic,
                                                 /*THCudaTensor_size(state, inputImages, 1)*/int ih,
                                                 /*THCudaTensor_size(state, inputImages, 2)*/int iw,
                                                 /*THCudaTensor_size(state, inputImages, 0)*/int ib,
                                                 /*THCudaTensor_size(state, output, 2)*/int ow,
                                                 /*THCudaTensor *inputImages*/float *inputImages, int isb, int isc, int ish, int isw,
                                                 /*THCudaTensor *grids*/float *grids, int gsb, int gsc, int gsh, int gsw,
                                                 /*THCudaTensor *output*/float *output, int osb, int osc, int osh, int osw,
                                                 /*THCState_getCurrentStream(state)*/cudaStream_t stream)
{
   //dim3 blocks((output->size[2]+15)/16, output->size[1], output->size[0]);
   dim3 blocks((sz1+15)/16, sz2, sz3);
   dim3 threads(32,16);

   int roiPerImage = sz3 / ib;
  //  printf("sz3: %d\n", sz3);
  //  printf("isb: %d\n", ib);
  //  printf("roiPerImage: %d\n", roiPerImage);
   /* assume BHWD */
   bilinearSamplingFromGrid <<< blocks, threads, 0, /*THCState_getCurrentStream(state)*/stream >>> (
                                                    /*THCudaTensor_data(state, inputImages)*/inputImages,
                                                    /*THCudaTensor_stride(state, inputImages, 0)*/isb,
                                                    /*THCudaTensor_stride(state, inputImages, 3)*/isc,
                                                    /*THCudaTensor_stride(state, inputImages, 1)*/ish,
                                                    /*THCudaTensor_stride(state, inputImages, 2)*/isw,
                                                    /*THCudaTensor_data(state, grids)*/grids,
                                                    /*THCudaTensor_stride(state, grids, 0)*/gsb,
                                                    /*THCudaTensor_stride(state, grids, 3)*/gsc,
                                                    /*THCudaTensor_stride(state, grids, 1)*/gsh,
                                                    /*THCudaTensor_stride(state, grids, 2)*/gsw,
                                                    /*THCudaTensor_data(state, output)*/output,
                                                    /*THCudaTensor_stride(state, output, 0)*/osb,
                                                    /*THCudaTensor_stride(state, output, 3)*/osc,
                                                    /*THCudaTensor_stride(state, output, 1)*/osh,
                                                    /*THCudaTensor_stride(state, output, 2)*/osw,
                                                    /*THCudaTensor_size(state, inputImages, 3)*/ic,
                                                    /*THCudaTensor_size(state, inputImages, 1)*/ih,
                                                    /*THCudaTensor_size(state, inputImages, 2)*/iw,
                                                    /*THCudaTensor_size(state, output, 2)*/ow,
                                                    roiPerImage);


  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in BilinearSampler.updateOutput: %s\n", cudaGetErrorString(err));
    //THError("aborting");
    return 0;
  }
  return 1;
}

int BilinearSamplerBHWD_updateGradInput_cuda_kernel(/*gradOutput->size[2]*/int sz1,
                                                    /*gradOutput->size[1]*/int sz2,
                                                    /*gradOutput->size[0]*/int sz3,
                                                    /*THCudaTensor_size(state, inputImages, 3)*/int ic,
                                                    /*THCudaTensor_size(state, inputImages, 1)*/int ih,
                                                    /*THCudaTensor_size(state, inputImages, 2)*/int iw,
                                                    /*THCudaTensor_size(state, inputImages, 0)*/int ib,
                                                    /*THCudaTensor_size(state, gradOutput, 2)*/int gow,
                                                    /*THCudaTensor *inputImages*/float *inputImages, int isb, int isc, int ish, int isw,
                                                    /*THCudaTensor *grids*/float *grids, int gsb, int gsc, int gsh, int gsw,
                                                    /*THCudaTensor *gradInputImages*/float *gradInputImages, int gisb, int gisc, int gish, int gisw,
                                                    /*THCudaTensor *gradGrids*/float *gradGrids, int ggsb, int ggsc, int ggsh, int ggsw,
                                                    /*THCudaTensor *gradOutput*/float *gradOutput, int gosb, int gosc, int gosh, int gosw,
                                                    /*THCState_getCurrentStream(state)*/cudaStream_t stream)
{
//  THCState *state = getCutorchState(L);
//  THCudaTensor *inputImages = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
//  THCudaTensor *grids = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");
//  THCudaTensor *gradInputImages = (THCudaTensor *)luaT_checkudata(L, 4, "torch.CudaTensor");
//  THCudaTensor *gradGrids = (THCudaTensor *)luaT_checkudata(L, 5, "torch.CudaTensor");
//  THCudaTensor *gradOutput = (THCudaTensor *)luaT_checkudata(L, 6, "torch.CudaTensor");

   //dim3 blocks((gradOutput->size[2]+15)/16, gradOutput->size[1], gradOutput->size[0]);
   dim3 blocks((sz1+15)/16, sz2, sz3);
   dim3 threads(32,16);

   int roiPerImage = sz3 / ib;
  //  printf("sz3: %d\n", sz3);
  //  printf("isb: %d\n", ib);
  //  printf("roiPerImage: %d\n", roiPerImage);

   backwardBilinearSampling <false> <<< blocks, threads, 0, /*THCState_getCurrentStream(state)*/stream >>> (
                                                      /*THCudaTensor_data(state, inputImages)*/inputImages,
                                                      /*THCudaTensor_stride(state, inputImages, 0)*/isb,
                                                      /*THCudaTensor_stride(state, inputImages, 3)*/isc,
                                                      /*THCudaTensor_stride(state, inputImages, 1)*/ish,
                                                      /*THCudaTensor_stride(state, inputImages, 2)*/isw,
                                                      /*THCudaTensor_data(state, gradInputImages)*/gradInputImages,
                                                      /*THCudaTensor_stride(state, gradInputImages, 0)*/gisb,
                                                      /*THCudaTensor_stride(state, gradInputImages, 3)*/gisc,
                                                      /*THCudaTensor_stride(state, gradInputImages, 1)*/gish,
                                                      /*THCudaTensor_stride(state, gradInputImages, 2)*/gisw,
                                                      /*THCudaTensor_data(state, grids)*/grids,
                                                      /*THCudaTensor_stride(state, grids, 0)*/gsb,
                                                      /*THCudaTensor_stride(state, grids, 3)*/gsc,
                                                      /*THCudaTensor_stride(state, grids, 1)*/gsh,
                                                      /*THCudaTensor_stride(state, grids, 2)*/gsw,
                                                      /*THCudaTensor_data(state, gradGrids)*/gradGrids,
                                                      /*THCudaTensor_stride(state, gradGrids, 0)*/ggsb,
                                                      /*THCudaTensor_stride(state, gradGrids, 3)*/ggsc,
                                                      /*THCudaTensor_stride(state, gradGrids, 1)*/ggsh,
                                                      /*THCudaTensor_stride(state, gradGrids, 2)*/ggsw,
                                                      /*THCudaTensor_data(state, gradOutput)*/gradOutput,
                                                      /*THCudaTensor_stride(state, gradOutput, 0)*/gosb,
                                                      /*THCudaTensor_stride(state, gradOutput, 3)*/gosc,
                                                      /*THCudaTensor_stride(state, gradOutput, 1)*/gosh,
                                                      /*THCudaTensor_stride(state, gradOutput, 2)*/gosw,
                                                      /*THCudaTensor_size(state, inputImages, 3)*/ic,
                                                      /*THCudaTensor_size(state, inputImages, 1)*/ih,
                                                      /*THCudaTensor_size(state, inputImages, 2)*/iw,
                                                      /*THCudaTensor_size(state, gradOutput, 2)*/gow,
                                                      roiPerImage);



  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in BilinearSampler.updateGradInput: %s\n", cudaGetErrorString(err));
    //THError("aborting");
    return 0;
  }
  return 1;
}

int BilinearSamplerBHWD_updateGradInputOnlyGrid_cuda_kernel(
                                        /*gradOutput->size[2]*/int sz1,
                                        /*gradOutput->size[1]*/int sz2,
                                        /*gradOutput->size[0]*/int sz3,
                                        /*THCudaTensor_size(state, inputImages, 3)*/int ic,
                                        /*THCudaTensor_size(state, inputImages, 1)*/int ih,
                                        /*THCudaTensor_size(state, inputImages, 2)*/int iw,
                                        /*THCudaTensor_size(state, inputImages, 0)*/int ib,
                                        /*THCudaTensor_size(state, gradOutput, 2)*/int gow,
                                        /*THCudaTensor *inputImages*/float *inputImages, int isb, int isc, int ish, int isw,
                                        /*THCudaTensor *grids*/float *grids, int gsb, int gsc, int gsh, int gsw,
                                        /*THCudaTensor *gradGrids*/float *gradGrids, int ggsb, int ggsc, int ggsh, int ggsw,
                                        /*THCudaTensor *gradOutput*/float *gradOutput, int gosb, int gosc, int gosh, int gosw,
                                        /*THCState_getCurrentStream(state)*/cudaStream_t stream)
{
//  THCState *state = getCutorchState(L);
//  THCudaTensor *inputImages = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
//  THCudaTensor *grids = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");
//  THCudaTensor *gradGrids = (THCudaTensor *)luaT_checkudata(L, 5, "torch.CudaTensor");
//  THCudaTensor *gradOutput = (THCudaTensor *)luaT_checkudata(L, 6, "torch.CudaTensor");

   //dim3 blocks((gradOutput->size[2]+15)/16, gradOutput->size[1], gradOutput->size[0]);
   dim3 blocks((sz1+15)/16, sz2, sz3);
   dim3 threads(32,16);

   int roiPerImage = sz3 / ib;

   backwardBilinearSampling <true> <<< blocks, threads, 0, /*THCState_getCurrentStream(state)*/stream >>> (
                                                      /*THCudaTensor_data(state, inputImages)*/inputImages,
                                                      /*THCudaTensor_stride(state, inputImages, 0)*/isb,
                                                      /*THCudaTensor_stride(state, inputImages, 3)*/isc,
                                                      /*THCudaTensor_stride(state, inputImages, 1)*/ish,
                                                      /*THCudaTensor_stride(state, inputImages, 2)*/isw,
                                                      0,
                                                      0,
                                                      0,
                                                      0,
                                                      0,
                                                      /*THCudaTensor_data(state, grids)*/grids,
                                                      /*THCudaTensor_stride(state, grids, 0)*/gsb,
                                                      /*THCudaTensor_stride(state, grids, 3)*/gsc,
                                                      /*THCudaTensor_stride(state, grids, 1)*/gsh,
                                                      /*THCudaTensor_stride(state, grids, 2)*/gsw,
                                                      /*THCudaTensor_data(state, gradGrids)*/gradGrids,
                                                      /*THCudaTensor_stride(state, gradGrids, 0)*/ggsb,
                                                      /*THCudaTensor_stride(state, gradGrids, 3)*/ggsc,
                                                      /*THCudaTensor_stride(state, gradGrids, 1)*/ggsh,
                                                      /*THCudaTensor_stride(state, gradGrids, 2)*/ggsw,
                                                      /*THCudaTensor_data(state, gradOutput)*/gradOutput,
                                                      /*THCudaTensor_stride(state, gradOutput, 0)*/gosb,
                                                      /*THCudaTensor_stride(state, gradOutput, 3)*/gosc,
                                                      /*THCudaTensor_stride(state, gradOutput, 1)*/gosh,
                                                      /*THCudaTensor_stride(state, gradOutput, 2)*/gosw,
                                                      /*THCudaTensor_size(state, inputImages, 3)*/ic,
                                                      /*THCudaTensor_size(state, inputImages, 1)*/ih,
                                                      /*THCudaTensor_size(state, inputImages, 2)*/iw,
                                                      /*THCudaTensor_size(state, gradOutput, 2)*/gow,
                                                      roiPerImage);



  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in BilinearSampler.updateGradInput: %s\n", cudaGetErrorString(err));
    //THError("aborting");
    return 0;
  }
  return 1;
}

#ifdef __cplusplus
}
#endif
