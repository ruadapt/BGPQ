#ifndef MODELS_FIFO_CUH
#define MODELS_FIFO_CUH

#include <functional>
#include <iostream>
#include "util.cuh"
#include "buffer.cuh"
#include "gc.cuh"
#include "datastructure.hpp"
#include "knapsackKernel.cuh"

using namespace std;
__global__ void oneBufferApplication(Buffer<KnapsackItem> *buffer, int batchSize, 
                            int *weight, int *benefit, float *benefitPerWeight,
                            int capacity, int inputSize,
                            int *globalBenefit, int *activeCount,
                            int *gc_flag, int gc_threshold,
                            int *explored_nodes,
                            bool init_flag = true)
{
    extern __shared__ int smem[];
    KnapsackItem *delItem = (KnapsackItem *)&smem[0];
    KnapsackItem *insItem = (KnapsackItem *)&delItem[batchSize];
    int *delSize = (int *)&insItem[2 * batchSize];
    int *insSize = (int *)&delSize[1];

    int smemOffset = (sizeof(KnapsackItem) * batchSize * 3 +
                      sizeof(int) * 2) / sizeof(int);

    bool init = init_flag;

    if (threadIdx.x == 0) {
        *delSize = 0;
    }
    __syncthreads();

    if (init && blockIdx.x == 0 && threadIdx.x == 0) {
        *delSize = 1;
        delItem[0].first = 0;
        delItem[0].second = 0;
        delItem[0].third = -1;
        delItem[0].fourth = 0;
    }
    __syncthreads();

    while(1) {
        if (!init) {
            buffer->deleteFromBuffer(delItem, *delSize, smemOffset);
        }
        __syncthreads();
        init = false;

        if (threadIdx.x == 0) {
#ifdef PRINT_DEBUG
            printf("b %d r %d w %d e %d\n", *buffer->begPos, *buffer->readPos, *buffer->writePos, *buffer->endPos);
            printf("thread %d delete items %d\n", blockIdx.x, *delSize);
            /*for (int i = 0; i < *delSize; i++) {*/
                /*printf("%d %d | ", delItem[i].first, delItem[i].second);*/
            /*}*/
            printf("\n");
#endif
            *insSize = 0;
        }
        __syncthreads();

        if (*delSize > 0) {
            appKernelWrapper(weight, benefit, benefitPerWeight,
                      globalBenefit, inputSize, capacity,
                      explored_nodes,
                      delItem, delSize,
                      insItem, insSize);
        }
        __syncthreads();
#ifdef PRINT_DEBUG
        if (threadIdx.x == 0) {
            printf("b %d r %d w %d e %d\n", *buffer->begPos, *buffer->readPos, *buffer->writePos, *buffer->endPos);
            printf("thread %d insert items %d\n", blockIdx.x, *insSize);
            /*for (int i = 0; i < *insSize; i++) {*/
                /*printf("%d %d | ", insItem[i].first, insItem[i].second);*/
            /*}*/
            /*printf("\n");*/
        }
        __syncthreads();
#endif

        if (*insSize > 0) {
            buffer->insertToBuffer(insItem, *insSize, smemOffset);
        }
        __syncthreads();

        /*for (int batchOffset = 0; batchOffset < *insSize; batchOffset += batchSize) {*/
            /*int partialSize = min(batchSize, *insSize - batchOffset);*/
            /*buffer->insertion(insItem + batchOffset, partialSize, smemOffset);*/
            /*__syncthreads();*/
        /*}*/

        if (threadIdx.x == 0) {
            atomicAdd(activeCount, (*insSize - *delSize));
            if (atomicAdd(activeCount, 0) == 0) {
                *delSize = -1;
            }
#ifdef PERF_DEBUG 
            atomicAdd(explored_nodes, *insSize);
//            if (blockIdx.x == 0) printf("explored %d\n", *explored_nodes);
#endif
        }
        __syncthreads();
        if (*delSize == -1) break;
        __syncthreads();
    }
}

void onebuffer(int *weight, int *benefit, float *benefitPerWeight,
             int *max_benefit, int capacity, int inputSize,
             int batchNum, int batchSize, int blockNum, int blockSize,
             int gc_threshold)
{

    /* prepare buffer data */
    Buffer<KnapsackItem> buffer(batchNum * batchSize);
    Buffer<KnapsackItem> *d_buffer;
    cudaMalloc((void **)&d_buffer, sizeof(Buffer<KnapsackItem>));
    cudaMemcpy(d_buffer, &buffer, sizeof(Buffer<KnapsackItem>), cudaMemcpyHostToDevice);


    size_t smemOffset = sizeof(KnapsackItem) * batchSize * 3 +
                        sizeof(int) * 2 +
                        sizeof(KnapsackItem) +
                        sizeof(int) * 2;

    bool init_flag = true;
    int *gc_flag;
    cudaMalloc((void **)&gc_flag, sizeof(int));
    cudaMemset(gc_flag, 0, sizeof(int));
    int *activeCount;
    cudaMalloc((void **)&activeCount, sizeof(int));
    int initActiveCount = 1;
    cudaMemcpy(activeCount, &initActiveCount, sizeof(int), cudaMemcpyHostToDevice);
    int *explored_nodes;
    int h_explored_nodes = 0;
    cudaMalloc((void **)&explored_nodes, sizeof(int));
    cudaMemset(explored_nodes, 0, sizeof(int));

	struct timeval startTime, endTime;
	setTime(&startTime);

    oneBufferApplication<<<blockNum, blockSize, smemOffset>>>(d_buffer, batchSize, 
                                                     weight, benefit, benefitPerWeight,
                                                     capacity, inputSize,
                                                     buffer.globalBenefit, activeCount,
                                                     gc_flag, gc_threshold,
                                                     explored_nodes,
                                                     init_flag);
    cudaDeviceSynchronize();

    setTime(&endTime);
    cudaMemcpy(&h_explored_nodes, explored_nodes, sizeof(int), cudaMemcpyDeviceToHost);
#ifdef PERF_DEBUG
    cout << getTime(&startTime, &endTime) << endl;
    cout << "explored nodes: " << h_explored_nodes << ". buffer usage: ";
    buffer.printBufferPtr();
#else
    cout << getTime(&startTime, &endTime) << endl;
#endif

    cudaMemcpy((int *)max_benefit, buffer.globalBenefit, sizeof(int), cudaMemcpyDeviceToDevice);
    cudaFree(d_buffer); d_buffer = NULL;
    cudaFree(gc_flag); gc_flag = NULL;
    cudaFree(activeCount); activeCount = NULL;
}

#endif
