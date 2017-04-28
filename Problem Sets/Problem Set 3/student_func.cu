/* Udacity Homework 3
   HDR Tone-mapping
  Background HDR
  ==============
  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  
  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.
  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.
  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.
  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.
  Background Chrominance-Luminance
  ================================
  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.
  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.
  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.
  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  
  Tone-mapping
  ============
  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.
  Example
  -------
  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9
  histo with 3 bins: [4 7 3]
  cdf : [4 11 14]
  Your task is to calculate this cumulative distribution by following these
  steps.
*/

#include <limits.h>
#include <float.h>
#include <math.h>
#include <stdio.h>

#include "utils.h"

#define THREADS_PER_BLOCK 1024
#define NUM_EXAMPLES 100
#define MIN 0
#define MAX 1

__global__
void histogramKernel(unsigned int* deviceBins, 
                     const float* deviceInBuffer, 
                     const int binCount, 
                     const float lumMin, 
                     const float lumMax, 
                     const int bufferSize) 
{  
    int bufferIndex = threadIdx.x + blockDim.x * blockIdx.x;
    
    // handle out of bounds situations
    if (bufferIndex >= bufferSize)
    {
        return;
    }
    float range = lumMax - lumMin;
    int binIndex = ((deviceInBuffer[bufferIndex] - lumMin) / range) * binCount;
    atomicAdd(&deviceBins[binIndex], 1);
}

__global__ 
void scanKernel(unsigned int* deviceBins, int bufferSize) 
{
    int bufferIndex = threadIdx.x + blockDim.x * blockIdx.x;
    
    if (bufferIndex >= bufferSize)
    {
        return;
    }
    
    for (int s = 1; s <= bufferSize; s *= 2) 
    {
        int spot = bufferIndex - s; 
        unsigned int val = 0;

        if (spot >= 0)
        {
          val = deviceBins[spot];
        }
        __syncthreads();

        if (spot >= 0)
        {
          deviceBins[bufferIndex] += val;
        }
        __syncthreads();
    }
}

__global__
void reduceMinMaxKernel(const float* const deviceInBuffer, 
                        float* deviceOutBuffer, 
                        const size_t bufferSize, 
                        int minOrMax) 
{
    extern __shared__ float sharedBuffer[];
    
    int blockSize = blockDim.x;
    int blockIndex = blockIdx.x;
    int threadIndex = threadIdx.x; 
    int bufferIndex = threadIndex + blockSize * blockIndex;
    
    // copy this block's portion of the input buffer into the shared buffer
    if (bufferIndex < bufferSize) 
    {
        sharedBuffer[threadIndex] = deviceInBuffer[bufferIndex];
    } 
    else
    {
        // handle out-of-bounds situations
        if (minOrMax == MIN)
        {
            sharedBuffer[threadIndex] = FLT_MAX;
        }
        else
        {
            sharedBuffer[threadIndex] = -FLT_MAX;
        }
    }
    __syncthreads();
       
    // min & max reduce
    for (unsigned int s = blockSize / 2; s > 0; s /= 2) 
    {
        if (threadIndex < s) 
        {
            if (minOrMax == MIN) 
            {
                sharedBuffer[threadIndex] = min(sharedBuffer[threadIndex], sharedBuffer[threadIndex + s]);
            } 
            else 
            {
                sharedBuffer[threadIndex] = max(sharedBuffer[threadIndex], sharedBuffer[threadIndex + s]);
            }
        }
        __syncthreads();
    }
    
    // write the min/max value of this block's portion into the output buffer
    if (threadIndex == 0) 
    {
        deviceOutBuffer[blockIndex] = sharedBuffer[0];
    }
}

static int calculateNumBlocks(int bufferSize) 
{
    return (int) ceil((float) bufferSize / (float) THREADS_PER_BLOCK) + 1;
}

static float reduceMinMax(const float* const deviceInBuffer, const size_t bufferSize, int minOrMax) 
{
    size_t currentBufferSize = bufferSize;
    float* deviceInputTemp;
    float* deviceOutputTemp;
    
    // copy the input buffer to a temporary buffer
    checkCudaErrors(cudaMalloc(&deviceInputTemp, sizeof(float) * bufferSize));    
    checkCudaErrors(cudaMemcpy(deviceInputTemp, deviceInBuffer, sizeof(float) * bufferSize, cudaMemcpyDeviceToDevice));
    
    const int sharedMemorySize = sizeof(float) * THREADS_PER_BLOCK;
    
    // keep reducing until the entire thing fits into a single block
    while (1) 
    {
        int numBlocks = calculateNumBlocks(currentBufferSize);
        checkCudaErrors(cudaMalloc(&deviceOutputTemp, sizeof(float) * numBlocks));
        reduceMinMaxKernel<<<numBlocks, THREADS_PER_BLOCK, sharedMemorySize>>>
                            (deviceInputTemp, deviceOutputTemp, currentBufferSize, minOrMax);
        cudaDeviceSynchronize(); 
        checkCudaErrors(cudaGetLastError());
            
        // move the current input to the output, and clear the last input if necessary
        checkCudaErrors(cudaFree(deviceInputTemp));
        deviceInputTemp = deviceOutputTemp;
        
        if (currentBufferSize < THREADS_PER_BLOCK) 
        {
            break;
        }
        else
        {
            // update current buffer size
            currentBufferSize = calculateNumBlocks(currentBufferSize);
        }
    }
    
    // transfer the min/max value to host memory and return
    float hostResult;
    cudaMemcpy(&hostResult, deviceOutputTemp, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(deviceOutputTemp);
    return hostResult;
}

void your_histogram_and_prefixsum(const float* const deviceLogLum,
                                  unsigned int* const deviceCdf,
                                  float &minLogLum,
                                  float &maxLogLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
    // REDUCE
    const size_t imageSize = numRows * numCols;
    minLogLum = reduceMinMax(deviceLogLum, imageSize, 0);
    maxLogLum = reduceMinMax(deviceLogLum, imageSize, 1);
    
    printf("Min: %f\n", minLogLum);
    printf("Max: %f\n", maxLogLum);
    printf("Number of Bins: %d\n", numBins);
    
    // HISTOGRAM
    unsigned int* deviceBins;
    size_t histogramSize = sizeof(unsigned int) * numBins;
    checkCudaErrors(cudaMalloc(&deviceBins, histogramSize));    
    checkCudaErrors(cudaMemset(deviceBins, 0, histogramSize));
    dim3 numBlocksHistogram(calculateNumBlocks(imageSize));
    histogramKernel<<<numBlocksHistogram, THREADS_PER_BLOCK>>>
                     (deviceBins, deviceLogLum, numBins, minLogLum, maxLogLum, imageSize);
    cudaDeviceSynchronize(); 
    checkCudaErrors(cudaGetLastError());

    // display some examples of histogram
    unsigned int hostOutBuffer[NUM_EXAMPLES];
    cudaMemcpy(&hostOutBuffer, deviceBins, sizeof(unsigned int) * NUM_EXAMPLES, cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < NUM_EXAMPLES; i++)
    {
        printf("hist out %d\n", hostOutBuffer[i]);
    }

    // SCAN
    dim3 numBlocksScan(calculateNumBlocks(numBins));
    scanKernel<<<numBlocksScan, THREADS_PER_BLOCK>>>(deviceBins, numBins);
    cudaDeviceSynchronize(); 
    checkCudaErrors(cudaGetLastError());
    
    // display some examples of CDF
    cudaMemcpy(&hostOutBuffer, deviceBins, sizeof(unsigned int) * NUM_EXAMPLES, cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < NUM_EXAMPLES; i++)
    {
        printf("cdf out %d\n", hostOutBuffer[i]);
    }
    cudaMemcpy(deviceCdf, deviceBins, histogramSize, cudaMemcpyDeviceToDevice);
    checkCudaErrors(cudaFree(deviceBins));
}
