#ifdef __cplusplus
extern "C" {
#endif


int BilinearSamplerBHWD_updateOutput_cuda_kernel(/*output->size[3]*/int oc,
                                                 /*output->size[2]*/int ow,
                                                 /*output->size[1]*/int oh,
                                                 /*output->size[0]*/int ob,
                                                 /*THCudaTensor_size(state, inputImages, 3)*/int ic,
                                                 /*THCudaTensor_size(state, inputImages, 1)*/int ih,
                                                 /*THCudaTensor_size(state, inputImages, 2)*/int iw,
                                                 /*THCudaTensor_size(state, inputImages, 0)*/int ib,
                                                 /*THCudaTensor *inputImages*/float *inputImages, int isb, int isc, int ish, int isw,
                                                 /*THCudaTensor *grids*/float *grids, int gsb, int gsc, int gsh, int gsw,
                                                 /*THCudaTensor *output*/float *output, int osb, int osc, int osh, int osw,
                                                 /*THCState_getCurrentStream(state)*/cudaStream_t stream);

int BilinearSamplerBHWD_updateGradInput_cuda_kernel(/*gradOutput->size[3]*/int goc,
                                                    /*gradOutput->size[2]*/int gow,
                                                    /*gradOutput->size[1]*/int goh,
                                                    /*gradOutput->size[0]*/int gob,
                                                    /*THCudaTensor_size(state, inputImages, 3)*/int ic,
                                                    /*THCudaTensor_size(state, inputImages, 1)*/int ih,
                                                    /*THCudaTensor_size(state, inputImages, 2)*/int iw,
                                                    /*THCudaTensor_size(state, inputImages, 0)*/int ib,
                                                    /*THCudaTensor *inputImages*/float *inputImages, int isb, int isc, int ish, int isw,
                                                    /*THCudaTensor *grids*/float *grids, int gsb, int gsc, int gsh, int gsw,
                                                    /*THCudaTensor *gradInputImages*/float *gradInputImages, int gisb, int gisc, int gish, int gisw,
                                                    /*THCudaTensor *gradGrids*/float *gradGrids, int ggsb, int ggsc, int ggsh, int ggsw,
                                                    /*THCudaTensor *gradOutput*/float *gradOutput, int gosb, int gosc, int gosh, int gosw,
                                                    /*THCState_getCurrentStream(state)*/cudaStream_t stream);


#ifdef __cplusplus
}
#endif
