#include <TH/TH.h>
#include <stdbool.h>
#include <stdio.h>

#define real float

int BilinearSamplerBHWD_updateOutput(THFloatTensor *inputImages, THFloatTensor *grids, THFloatTensor *output)
{

  int batchsize = THFloatTensor_size(inputImages, 0);
  int inputImages_height = THFloatTensor_size(inputImages, 1);
  int inputImages_width = THFloatTensor_size(inputImages, 2);
  int output_height = THFloatTensor_size(output, 1);
  int output_width = THFloatTensor_size(output, 2);
  int inputImages_channels = THFloatTensor_size(inputImages, 3);

  int output_strideBatch = THFloatTensor_stride(output, 0);
  int output_strideHeight = THFloatTensor_stride(output, 1);
  int output_strideWidth = THFloatTensor_stride(output, 2);

  int inputImages_strideBatch = THFloatTensor_stride(inputImages, 0);
  int inputImages_strideHeight = THFloatTensor_stride(inputImages, 1);
  int inputImages_strideWidth = THFloatTensor_stride(inputImages, 2);

  int grids_strideBatch = THFloatTensor_stride(grids, 0);
  int grids_strideHeight = THFloatTensor_stride(grids, 1);
  int grids_strideWidth = THFloatTensor_stride(grids, 2);

  real *inputImages_data, *output_data, *grids_data;
  inputImages_data = THFloatTensor_data(inputImages);
  output_data = THFloatTensor_data(output);
  grids_data = THFloatTensor_data(grids);

  int b, yOut, xOut;

  for(b=0; b < batchsize; b++)
  {
    for(yOut=0; yOut < output_height; yOut++)
    {
      for(xOut=0; xOut < output_width; xOut++)
      {
        //read the grid
        real yf = grids_data[b*grids_strideBatch + yOut*grids_strideHeight + xOut*grids_strideWidth];
        real xf = grids_data[b*grids_strideBatch + yOut*grids_strideHeight + xOut*grids_strideWidth + 1];

        // get the weights for interpolation
        int yInTopLeft, xInTopLeft;
        real yWeightTopLeft, xWeightTopLeft;

        real xcoord = (xf + 1) * (inputImages_width - 1) / 2;
        xInTopLeft = floor(xcoord);
        xWeightTopLeft = 1 - (xcoord - xInTopLeft);

        real ycoord = (yf + 1) * (inputImages_height - 1) / 2;
        yInTopLeft = floor(ycoord);
        yWeightTopLeft = 1 - (ycoord - yInTopLeft);



        const int outAddress = output_strideBatch * b + output_strideHeight * yOut + output_strideWidth * xOut;
        const int inTopLeftAddress = inputImages_strideBatch * b + inputImages_strideHeight * yInTopLeft + inputImages_strideWidth * xInTopLeft;
        const int inTopRightAddress = inTopLeftAddress + inputImages_strideWidth;
        const int inBottomLeftAddress = inTopLeftAddress + inputImages_strideHeight;
        const int inBottomRightAddress = inBottomLeftAddress + inputImages_strideWidth;

        real v=0;
        real inTopLeft=0;
        real inTopRight=0;
        real inBottomLeft=0;
        real inBottomRight=0;

        // we are careful with the boundaries
        bool topLeftIsIn = xInTopLeft >= 0 && xInTopLeft <= inputImages_width-1 && yInTopLeft >= 0 && yInTopLeft <= inputImages_height-1;
        bool topRightIsIn = xInTopLeft+1 >= 0 && xInTopLeft+1 <= inputImages_width-1 && yInTopLeft >= 0 && yInTopLeft <= inputImages_height-1;
        bool bottomLeftIsIn = xInTopLeft >= 0 && xInTopLeft <= inputImages_width-1 && yInTopLeft+1 >= 0 && yInTopLeft+1 <= inputImages_height-1;
        bool bottomRightIsIn = xInTopLeft+1 >= 0 && xInTopLeft+1 <= inputImages_width-1 && yInTopLeft+1 >= 0 && yInTopLeft+1 <= inputImages_height-1;

        int t;
        // interpolation happens here
        for(t=0; t<inputImages_channels; t++)
        {
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
    }
  }

  return 1;
}



int BilinearSamplerBHWD_updateGradInput(THFloatTensor *inputImages, THFloatTensor *grids, THFloatTensor *gradInputImages,
                                        THFloatTensor *gradGrids, THFloatTensor *gradOutput)
{
  bool onlyGrid=false;

  int batchsize = THFloatTensor_size(inputImages, 0);
  int inputImages_height = THFloatTensor_size(inputImages, 1);
  int inputImages_width = THFloatTensor_size(inputImages, 2);
  int gradOutput_height = THFloatTensor_size(gradOutput, 1);
  int gradOutput_width = THFloatTensor_size(gradOutput, 2);
  int inputImages_channels = THFloatTensor_size(inputImages, 3);

  int gradOutput_strideBatch = THFloatTensor_stride(gradOutput, 0);
  int gradOutput_strideHeight = THFloatTensor_stride(gradOutput, 1);
  int gradOutput_strideWidth = THFloatTensor_stride(gradOutput, 2);

  int inputImages_strideBatch = THFloatTensor_stride(inputImages, 0);
  int inputImages_strideHeight = THFloatTensor_stride(inputImages, 1);
  int inputImages_strideWidth = THFloatTensor_stride(inputImages, 2);

  int gradInputImages_strideBatch = THFloatTensor_stride(gradInputImages, 0);
  int gradInputImages_strideHeight = THFloatTensor_stride(gradInputImages, 1);
  int gradInputImages_strideWidth = THFloatTensor_stride(gradInputImages, 2);

  int grids_strideBatch = THFloatTensor_stride(grids, 0);
  int grids_strideHeight = THFloatTensor_stride(grids, 1);
  int grids_strideWidth = THFloatTensor_stride(grids, 2);

  int gradGrids_strideBatch = THFloatTensor_stride(gradGrids, 0);
  int gradGrids_strideHeight = THFloatTensor_stride(gradGrids, 1);
  int gradGrids_strideWidth = THFloatTensor_stride(gradGrids, 2);

  real *inputImages_data, *gradOutput_data, *grids_data, *gradGrids_data, *gradInputImages_data;
  inputImages_data = THFloatTensor_data(inputImages);
  gradOutput_data = THFloatTensor_data(gradOutput);
  grids_data = THFloatTensor_data(grids);
  gradGrids_data = THFloatTensor_data(gradGrids);
  gradInputImages_data = THFloatTensor_data(gradInputImages);

  int b, yOut, xOut;

  for(b=0; b < batchsize; b++)
  {
    for(yOut=0; yOut < gradOutput_height; yOut++)
    {
      for(xOut=0; xOut < gradOutput_width; xOut++)
      {
        //read the grid
        real yf = grids_data[b*grids_strideBatch + yOut*grids_strideHeight + xOut*grids_strideWidth];
        real xf = grids_data[b*grids_strideBatch + yOut*grids_strideHeight + xOut*grids_strideWidth + 1];

        // get the weights for interpolation
        int yInTopLeft, xInTopLeft;
        real yWeightTopLeft, xWeightTopLeft;

        real xcoord = (xf + 1) * (inputImages_width - 1) / 2;
        xInTopLeft = floor(xcoord);
        xWeightTopLeft = 1 - (xcoord - xInTopLeft);

        real ycoord = (yf + 1) * (inputImages_height - 1) / 2;
        yInTopLeft = floor(ycoord);
        yWeightTopLeft = 1 - (ycoord - yInTopLeft);


        const int inTopLeftAddress = inputImages_strideBatch * b + inputImages_strideHeight * yInTopLeft + inputImages_strideWidth * xInTopLeft;
        const int inTopRightAddress = inTopLeftAddress + inputImages_strideWidth;
        const int inBottomLeftAddress = inTopLeftAddress + inputImages_strideHeight;
        const int inBottomRightAddress = inBottomLeftAddress + inputImages_strideWidth;

        const int gradInputImagesTopLeftAddress = gradInputImages_strideBatch * b + gradInputImages_strideHeight * yInTopLeft + gradInputImages_strideWidth * xInTopLeft;
        const int gradInputImagesTopRightAddress = gradInputImagesTopLeftAddress + gradInputImages_strideWidth;
        const int gradInputImagesBottomLeftAddress = gradInputImagesTopLeftAddress + gradInputImages_strideHeight;
        const int gradInputImagesBottomRightAddress = gradInputImagesBottomLeftAddress + gradInputImages_strideWidth;

        const int gradOutputAddress = gradOutput_strideBatch * b + gradOutput_strideHeight * yOut + gradOutput_strideWidth * xOut;

        real topLeftDotProduct = 0;
        real topRightDotProduct = 0;
        real bottomLeftDotProduct = 0;
        real bottomRightDotProduct = 0;

        // we are careful with the boundaries
        bool topLeftIsIn = xInTopLeft >= 0 && xInTopLeft <= inputImages_width-1 && yInTopLeft >= 0 && yInTopLeft <= inputImages_height-1;
        bool topRightIsIn = xInTopLeft+1 >= 0 && xInTopLeft+1 <= inputImages_width-1 && yInTopLeft >= 0 && yInTopLeft <= inputImages_height-1;
        bool bottomLeftIsIn = xInTopLeft >= 0 && xInTopLeft <= inputImages_width-1 && yInTopLeft+1 >= 0 && yInTopLeft+1 <= inputImages_height-1;
        bool bottomRightIsIn = xInTopLeft+1 >= 0 && xInTopLeft+1 <= inputImages_width-1 && yInTopLeft+1 >= 0 && yInTopLeft+1 <= inputImages_height-1;

        int t;

        for(t=0; t<inputImages_channels; t++)
        {
           real gradOutValue = gradOutput_data[gradOutputAddress + t];
           if(topLeftIsIn)
           {
              real inTopLeft = inputImages_data[inTopLeftAddress + t];
              topLeftDotProduct += inTopLeft * gradOutValue;
              if(!onlyGrid) gradInputImages_data[gradInputImagesTopLeftAddress + t] += xWeightTopLeft * yWeightTopLeft * gradOutValue;
           }

           if(topRightIsIn)
           {
              real inTopRight = inputImages_data[inTopRightAddress + t];
              topRightDotProduct += inTopRight * gradOutValue;
              if(!onlyGrid) gradInputImages_data[gradInputImagesTopRightAddress + t] += (1 - xWeightTopLeft) * yWeightTopLeft * gradOutValue;
           }

           if(bottomLeftIsIn)
           {
              real inBottomLeft = inputImages_data[inBottomLeftAddress + t];
              bottomLeftDotProduct += inBottomLeft * gradOutValue;
              if(!onlyGrid) gradInputImages_data[gradInputImagesBottomLeftAddress + t] += xWeightTopLeft * (1 - yWeightTopLeft) * gradOutValue;
           }

           if(bottomRightIsIn)
           {
              real inBottomRight = inputImages_data[inBottomRightAddress + t];
              bottomRightDotProduct += inBottomRight * gradOutValue;
              if(!onlyGrid) gradInputImages_data[gradInputImagesBottomRightAddress + t] += (1 - xWeightTopLeft) * (1 - yWeightTopLeft) * gradOutValue;
           }
        }

        yf = - xWeightTopLeft * topLeftDotProduct + xWeightTopLeft * bottomLeftDotProduct - (1-xWeightTopLeft) * topRightDotProduct + (1-xWeightTopLeft) * bottomRightDotProduct;
        xf = - yWeightTopLeft * topLeftDotProduct + yWeightTopLeft * topRightDotProduct - (1-yWeightTopLeft) * bottomLeftDotProduct + (1-yWeightTopLeft) * bottomRightDotProduct;

        gradGrids_data[b*gradGrids_strideBatch + yOut*gradGrids_strideHeight + xOut*gradGrids_strideWidth] = yf * (inputImages_height-1) / 2;
        gradGrids_data[b*gradGrids_strideBatch + yOut*gradGrids_strideHeight + xOut*gradGrids_strideWidth + 1] = xf * (inputImages_width-1) / 2;

      }
    }
  }

  return 1;
}


int BilinearSamplerBCHW_updateOutput(THFloatTensor *inputImages, THFloatTensor *grids, THFloatTensor *output)
{

  int batchsize = THFloatTensor_size(inputImages, 0);
  int inputImages_height = THFloatTensor_size(inputImages, 2);
  int inputImages_width = THFloatTensor_size(inputImages, 3);
  
  int output_height = THFloatTensor_size(output, 2);
  int output_width = THFloatTensor_size(output, 3);
  int inputImages_channels = THFloatTensor_size(inputImages, 1);

  int output_strideBatch = THFloatTensor_stride(output, 0);
  int output_strideHeight = THFloatTensor_stride(output, 2);
  int output_strideWidth = THFloatTensor_stride(output, 3);  
  int output_strideChannel = THFloatTensor_stride(output, 1);
    
  int inputImages_strideBatch = THFloatTensor_stride(inputImages, 0);
  int inputImages_strideHeight = THFloatTensor_stride(inputImages, 2);
  int inputImages_strideWidth = THFloatTensor_stride(inputImages, 3);
  int inputImages_strideChannel = THFloatTensor_stride(inputImages, 1);

  int grids_strideBatch = THFloatTensor_stride(grids, 0);
  int grids_strideHeight = THFloatTensor_stride(grids, 2);
  int grids_strideWidth = THFloatTensor_stride(grids, 3);
  int grids_strideChannel = THFloatTensor_stride(grids, 1);


  real *inputImages_data, *output_data, *grids_data;
  inputImages_data = THFloatTensor_data(inputImages);
  output_data = THFloatTensor_data(output);
  grids_data = THFloatTensor_data(grids);

  int b, yOut, xOut;

  for(b=0; b < batchsize; b++)
  {
    for(yOut=0; yOut < output_height; yOut++)
    {
      for(xOut=0; xOut < output_width; xOut++)
      {
        //read the grid
        
        real xf = grids_data[b*grids_strideBatch + yOut*grids_strideHeight + xOut*grids_strideWidth + grids_strideChannel];
        real yf = grids_data[b*grids_strideBatch + yOut*grids_strideHeight + xOut*grids_strideWidth];

        // get the weights for interpolation
        int yInTopLeft, xInTopLeft;
        real yWeightTopLeft, xWeightTopLeft;

        real xcoord = (xf + 1) * (inputImages_width - 1) / 2;
        xInTopLeft = floor(xcoord);
        xWeightTopLeft = 1 - (xcoord - xInTopLeft);

        real ycoord = (yf + 1) * (inputImages_height - 1) / 2;
        yInTopLeft = floor(ycoord);
        yWeightTopLeft = 1 - (ycoord - yInTopLeft);



        const int outAddress = output_strideBatch * b + output_strideHeight * yOut + output_strideWidth * xOut;
        const int inTopLeftAddress = inputImages_strideBatch * b + inputImages_strideHeight * yInTopLeft + inputImages_strideWidth * xInTopLeft;
        const int inTopRightAddress = inTopLeftAddress + inputImages_strideWidth;
        const int inBottomLeftAddress = inTopLeftAddress + inputImages_strideHeight;
        const int inBottomRightAddress = inBottomLeftAddress + inputImages_strideWidth;

        real v=0;
        real inTopLeft=0;
        real inTopRight=0;
        real inBottomLeft=0;
        real inBottomRight=0;

        // we are careful with the boundaries
        bool topLeftIsIn = xInTopLeft >= 0 && xInTopLeft <= inputImages_width-1 && yInTopLeft >= 0 && yInTopLeft <= inputImages_height-1;
        bool topRightIsIn = xInTopLeft+1 >= 0 && xInTopLeft+1 <= inputImages_width-1 && yInTopLeft >= 0 && yInTopLeft <= inputImages_height-1;
        bool bottomLeftIsIn = xInTopLeft >= 0 && xInTopLeft <= inputImages_width-1 && yInTopLeft+1 >= 0 && yInTopLeft+1 <= inputImages_height-1;
        bool bottomRightIsIn = xInTopLeft+1 >= 0 && xInTopLeft+1 <= inputImages_width-1 && yInTopLeft+1 >= 0 && yInTopLeft+1 <= inputImages_height-1;

        int t;
        // interpolation happens here
        for(t=0; t<inputImages_channels; t++)
        {
           if(topLeftIsIn) inTopLeft = inputImages_data[inTopLeftAddress + t * inputImages_strideChannel];
           if(topRightIsIn) inTopRight = inputImages_data[inTopRightAddress + t * inputImages_strideChannel];
           if(bottomLeftIsIn) inBottomLeft = inputImages_data[inBottomLeftAddress + t * inputImages_strideChannel];
           if(bottomRightIsIn) inBottomRight = inputImages_data[inBottomRightAddress + t * inputImages_strideChannel];

           v = xWeightTopLeft * yWeightTopLeft * inTopLeft
             + (1 - xWeightTopLeft) * yWeightTopLeft * inTopRight
             + xWeightTopLeft * (1 - yWeightTopLeft) * inBottomLeft
             + (1 - xWeightTopLeft) * (1 - yWeightTopLeft) * inBottomRight;

           output_data[outAddress + t * output_strideChannel] = v;
        }

      }
    }
  }

  return 1;
}



int BilinearSamplerBCHW_updateGradInput(THFloatTensor *inputImages, THFloatTensor *grids, THFloatTensor *gradInputImages,
                                        THFloatTensor *gradGrids, THFloatTensor *gradOutput)
{
  bool onlyGrid=false;

  int batchsize = THFloatTensor_size(inputImages, 0);
  int inputImages_height = THFloatTensor_size(inputImages, 2);
  int inputImages_width = THFloatTensor_size(inputImages, 3);
  int gradOutput_height = THFloatTensor_size(gradOutput, 2);
  int gradOutput_width = THFloatTensor_size(gradOutput, 3);
  int inputImages_channels = THFloatTensor_size(inputImages, 1);

  int gradOutput_strideBatch = THFloatTensor_stride(gradOutput, 0);
  int gradOutput_strideHeight = THFloatTensor_stride(gradOutput, 2);
  int gradOutput_strideWidth = THFloatTensor_stride(gradOutput, 3);
  int gradOutput_strideChannel = THFloatTensor_stride(gradOutput, 1);

  int inputImages_strideBatch = THFloatTensor_stride(inputImages, 0);
  int inputImages_strideHeight = THFloatTensor_stride(inputImages, 2);
  int inputImages_strideWidth = THFloatTensor_stride(inputImages, 3);
  int inputImages_strideChannel = THFloatTensor_stride(inputImages, 1);
    
  int gradInputImages_strideBatch = THFloatTensor_stride(gradInputImages, 0);
  int gradInputImages_strideHeight = THFloatTensor_stride(gradInputImages, 2);
  int gradInputImages_strideWidth = THFloatTensor_stride(gradInputImages, 3);
  int gradInputImages_strideChannel = THFloatTensor_stride(gradInputImages, 1);

  int grids_strideBatch = THFloatTensor_stride(grids, 0);
  int grids_strideHeight = THFloatTensor_stride(grids, 2);
  int grids_strideWidth = THFloatTensor_stride(grids, 3);
  int grids_strideChannel = THFloatTensor_stride(grids, 1);

  int gradGrids_strideBatch = THFloatTensor_stride(gradGrids, 0);
  int gradGrids_strideHeight = THFloatTensor_stride(gradGrids, 2);
  int gradGrids_strideWidth = THFloatTensor_stride(gradGrids, 3);
  int gradGrids_strideChannel = THFloatTensor_stride(gradGrids, 1);

  real *inputImages_data, *gradOutput_data, *grids_data, *gradGrids_data, *gradInputImages_data;
  inputImages_data = THFloatTensor_data(inputImages);
  gradOutput_data = THFloatTensor_data(gradOutput);
  grids_data = THFloatTensor_data(grids);
  gradGrids_data = THFloatTensor_data(gradGrids);
  gradInputImages_data = THFloatTensor_data(gradInputImages);

  int b, yOut, xOut;

  for(b=0; b < batchsize; b++)
  {
    for(yOut=0; yOut < gradOutput_height; yOut++)
    {
      for(xOut=0; xOut < gradOutput_width; xOut++)
      {
        //read the grid
        real xf = grids_data[b*grids_strideBatch + yOut*grids_strideHeight + xOut*grids_strideWidth + grids_strideChannel];
        real yf = grids_data[b*grids_strideBatch + yOut*grids_strideHeight + xOut*grids_strideWidth];
        
        // get the weights for interpolation
        int yInTopLeft, xInTopLeft;
        real yWeightTopLeft, xWeightTopLeft;

        real xcoord = (xf + 1) * (inputImages_width - 1) / 2;
        xInTopLeft = floor(xcoord);
        xWeightTopLeft = 1 - (xcoord - xInTopLeft);

        real ycoord = (yf + 1) * (inputImages_height - 1) / 2;
        yInTopLeft = floor(ycoord);
        yWeightTopLeft = 1 - (ycoord - yInTopLeft);


        const int inTopLeftAddress = inputImages_strideBatch * b + inputImages_strideHeight * yInTopLeft + inputImages_strideWidth * xInTopLeft;
        const int inTopRightAddress = inTopLeftAddress + inputImages_strideWidth;
        const int inBottomLeftAddress = inTopLeftAddress + inputImages_strideHeight;
        const int inBottomRightAddress = inBottomLeftAddress + inputImages_strideWidth;

        const int gradInputImagesTopLeftAddress = gradInputImages_strideBatch * b + gradInputImages_strideHeight * yInTopLeft + gradInputImages_strideWidth * xInTopLeft;
        const int gradInputImagesTopRightAddress = gradInputImagesTopLeftAddress + gradInputImages_strideWidth;
        const int gradInputImagesBottomLeftAddress = gradInputImagesTopLeftAddress + gradInputImages_strideHeight;
        const int gradInputImagesBottomRightAddress = gradInputImagesBottomLeftAddress + gradInputImages_strideWidth;

        const int gradOutputAddress = gradOutput_strideBatch * b + gradOutput_strideHeight * yOut + gradOutput_strideWidth * xOut;

        real topLeftDotProduct = 0;
        real topRightDotProduct = 0;
        real bottomLeftDotProduct = 0;
        real bottomRightDotProduct = 0;

        // we are careful with the boundaries
        bool topLeftIsIn = xInTopLeft >= 0 && xInTopLeft <= inputImages_width-1 && yInTopLeft >= 0 && yInTopLeft <= inputImages_height-1;
        bool topRightIsIn = xInTopLeft+1 >= 0 && xInTopLeft+1 <= inputImages_width-1 && yInTopLeft >= 0 && yInTopLeft <= inputImages_height-1;
        bool bottomLeftIsIn = xInTopLeft >= 0 && xInTopLeft <= inputImages_width-1 && yInTopLeft+1 >= 0 && yInTopLeft+1 <= inputImages_height-1;
        bool bottomRightIsIn = xInTopLeft+1 >= 0 && xInTopLeft+1 <= inputImages_width-1 && yInTopLeft+1 >= 0 && yInTopLeft+1 <= inputImages_height-1;

        int t;

        for(t=0; t<inputImages_channels; t++)
        {
           real gradOutValue = gradOutput_data[gradOutputAddress + t * gradOutput_strideChannel];
           if(topLeftIsIn)
           {
              real inTopLeft = inputImages_data[inTopLeftAddress + t * inputImages_strideChannel];
              topLeftDotProduct += inTopLeft * gradOutValue;
              if(!onlyGrid) gradInputImages_data[gradInputImagesTopLeftAddress + t * gradInputImages_strideChannel] += xWeightTopLeft * yWeightTopLeft * gradOutValue;
           }

           if(topRightIsIn)
           {
              real inTopRight = inputImages_data[inTopRightAddress + t * inputImages_strideChannel];
              topRightDotProduct += inTopRight * gradOutValue;
              if(!onlyGrid) gradInputImages_data[gradInputImagesTopRightAddress + t * gradInputImages_strideChannel] += (1 - xWeightTopLeft) * yWeightTopLeft * gradOutValue;
           }

           if(bottomLeftIsIn)
           {
              real inBottomLeft = inputImages_data[inBottomLeftAddress + t * inputImages_strideChannel];
              bottomLeftDotProduct += inBottomLeft * gradOutValue;
              if(!onlyGrid) gradInputImages_data[gradInputImagesBottomLeftAddress + t * gradInputImages_strideChannel] += xWeightTopLeft * (1 - yWeightTopLeft) * gradOutValue;
           }

           if(bottomRightIsIn)
           {
              real inBottomRight = inputImages_data[inBottomRightAddress + t * inputImages_strideChannel];
              bottomRightDotProduct += inBottomRight * gradOutValue;
              if(!onlyGrid) gradInputImages_data[gradInputImagesBottomRightAddress + t * gradInputImages_strideChannel] += (1 - xWeightTopLeft) * (1 - yWeightTopLeft) * gradOutValue;
           }
        }

        xf = - yWeightTopLeft * topLeftDotProduct + yWeightTopLeft * topRightDotProduct - (1-yWeightTopLeft) * bottomLeftDotProduct + (1-yWeightTopLeft) * bottomRightDotProduct;
          
        yf = - xWeightTopLeft * topLeftDotProduct + xWeightTopLeft * bottomLeftDotProduct - (1-xWeightTopLeft) * topRightDotProduct + (1-xWeightTopLeft) * bottomRightDotProduct;
        

        gradGrids_data[b*gradGrids_strideBatch + yOut*gradGrids_strideHeight + xOut*gradGrids_strideWidth + gradGrids_strideChannel] = xf * (inputImages_width-1) / 2;
          
        gradGrids_data[b*gradGrids_strideBatch + yOut*gradGrids_strideHeight + xOut*gradGrids_strideWidth] = yf * (inputImages_height-1) / 2;
        

      }
    }
  }

  return 1;
}


