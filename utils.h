#ifndef __UTILS_H__
#define __UTILS_H__

#include "commons.h"
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <stdlib.h>
#include <assert.h>


#define wbCheck(stmt)  do {                                               \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
        printf( "Failed to run stmt %d \n", __LINE__);                    \
        printf( "Got CUDA error ...  %s \n", cudaGetErrorString(err));    \
        return -1;                                                        \
    }                                                                     \
} while(0)


cv::Mat cscThresholding (const cv::Mat& rgbInput);
cv::Mat processCanny (const cv::Mat& gray);
cv::Mat regionOfInterest(cv::Mat& img);


struct _ConvParams {
    int kernel_width;
    int kernel_height;
    int kernel_radius_x;
    int kernel_radius_y;
    int tile_width;
    int w_x;
    int w_y;
};


#define clamp(x) (min(max((x), 0.0), 1.0))


__global__ void conv2d (float *image,
                        const float* __restrict__ weights,
                        float *out,
                        int channels,
                        int width,
                        int height);

#endif