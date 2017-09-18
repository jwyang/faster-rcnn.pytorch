int BilinearSamplerBHWD_updateOutput(THFloatTensor *inputImages, THFloatTensor *grids, THFloatTensor *output);

int BilinearSamplerBHWD_updateGradInput(THFloatTensor *inputImages, THFloatTensor *grids, THFloatTensor *gradInputImages,
                                        THFloatTensor *gradGrids, THFloatTensor *gradOutput);



int BilinearSamplerBCHW_updateOutput(THFloatTensor *inputImages, THFloatTensor *grids, THFloatTensor *output);

int BilinearSamplerBCHW_updateGradInput(THFloatTensor *inputImages, THFloatTensor *grids, THFloatTensor *gradInputImages,
                                        THFloatTensor *gradGrids, THFloatTensor *gradOutput);
