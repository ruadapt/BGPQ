#ifndef MODELS_PQ_CUH
#define MODELS_PQ_CUH

#include <functional>
#include <iostream>
#include "util.cuh"
#include "heap.cuh"
#include "gc.cuh"
#include "datastructure.hpp"
#include "knapsackKernel.cuh"

//#define RESERVED

using namespace std;

__global__ void oneHeapApplication(Heap<KnapsackItem> *heap, int batchSize, 
                            int *weight, int *benefit, float *benefitPerWeight,
                            int capacity, int inputSize,
                            int *globalBenefit, int *activeCount,
                            int *gc_flag, int gc_threshold,
                            unsigned long long int *explored_nodes,
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
        heap->tbstate[blockIdx.x] = 1;
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

//    int iter = 0;
    while(1) {
        __syncthreads();
        if (!init && heap->deleteRoot(delItem, *delSize) == true) {
            __syncthreads();
            heap->deleteUpdate(smemOffset);
        }
        __syncthreads();
        init = false;
        int trueDelSize = *delSize;

        if (threadIdx.x == 0) {
#ifdef PRINT_DEBUG
            printf("thread %d delete items %d\n", blockIdx.x, *delSize);
            for (int i = 0; i < *delSize; i++) {
                printf("%d %d | ", delItem[i].first, delItem[i].second);
            }
            printf("\n");
#endif
            *insSize = 0;
        }
        __syncthreads();
        if (trueDelSize > 0) {
#ifndef RESERVED
            __syncthreads();
            while (1)
            {
                appKernel(weight, benefit, benefitPerWeight,
                          globalBenefit, inputSize, capacity,
                          delItem, delSize,
                          insItem, insSize);
                __syncthreads();
                if (*insSize >= batchSize || *insSize == 0) break;
                __syncthreads();
                if (threadIdx.x == 0) {
                    atomicAdd(explored_nodes, (unsigned long long int)*insSize);
                    *delSize = *insSize;
                    *insSize = 0;
#ifdef PRINT_DEBUG
                    printf("inblock: thread %d insert %d items %d batches in heap\n", 
                    blockIdx.x, *delSize, *heap->batchCount);
                    for (int i = 0; i < *delSize; i++) {
                        printf("%d %d %d %d| ", 
                        insItem[i].first, insItem[i].second, insItem[i].third, insItem[i].fourth);
                    }
                    printf("\n");
#endif
                }
                __syncthreads();
                for (int i = threadIdx.x; i < *delSize; i += blockDim.x) {
                    delItem[i] = insItem[i];
                }
                __syncthreads();
            }
#else
            appKernel(weight, benefit, benefitPerWeight,
                      globalBenefit, inputSize, capacity,
                      delItem, delSize,
                      insItem, insSize);
            __syncthreads();
            if (threadIdx.x == 0) {
                atomicAdd(explored_nodes, *insSize);
            }
            __syncthreads();
#endif
        }
        __syncthreads();
#ifdef PRINT_DEBUG
        if (threadIdx.x == 0) {
            printf("thread %d insert items %d\n", blockIdx.x, *insSize);
            for (int i = 0; i < *insSize; i++) {
                printf("%d %d | ", insItem[i].first, insItem[i].second);
            }
            printf("\n");
        }
        __syncthreads();
#endif
        if (*insSize > 0) {
            __syncthreads();
//            if (threadIdx.x == 0) {
//                atomicAdd(explored_nodes, *insSize);
//            }
//            __syncthreads();
            for (int batchOffset = 0; batchOffset < *insSize; batchOffset += batchSize) {
                int partialSize = min(batchSize, *insSize - batchOffset);
                heap->insertion(insItem + batchOffset, partialSize, smemOffset);
                __syncthreads();
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            atomicAdd(activeCount, (*insSize - trueDelSize));
            if (atomicAdd(activeCount, 0) == 0) {
                *delSize = -1;
            }
            /**delSize = heap->ifTerminate();*/
            if (*heap->batchCount > gc_threshold) {
                *gc_flag  = 1;
            }
#ifdef PRINT_DEBUG
            printf("delSize: %d gcflag %d\n", *delSize, *gc_flag);
#endif
        }
        __syncthreads();
//       #printf("iter %d sync threadId: %d\n", ++iter, threadIdx.x);
//        if (iter == 100) return;
        if (*delSize == -1 || *gc_flag == 1) break;
    }
}

void oneheap(int *weight, int *benefit, float *benefitPerWeight,
             int *max_benefit, int capacity, int inputSize,
             int batchNum, int batchSize, int blockNum, int blockSize,
             int gc_threshold)
{

    /* prepare heap data */
    Heap<KnapsackItem> heap(batchNum, batchSize);

    Heap<KnapsackItem> *d_heap;
    cudaMalloc((void **)&d_heap, sizeof(Heap<KnapsackItem>));
    cudaMemcpy(d_heap, &heap, sizeof(Heap<KnapsackItem>), cudaMemcpyHostToDevice);

    size_t smemOffset = sizeof(KnapsackItem) * batchSize * 3 +
                        sizeof(int) * 2 +
                        sizeof(KnapsackItem) +
                        5 * batchSize * sizeof(KnapsackItem);

    bool init_flag = true;
    int *gc_flag;
    cudaMalloc((void **)&gc_flag, sizeof(int));
    cudaMemset(gc_flag, 0, sizeof(int));
    int *activeCount;
    cudaMalloc((void **)&activeCount, sizeof(int));
    int initActiveCount = 1;
    cudaMemcpy(activeCount, &initActiveCount, sizeof(int), cudaMemcpyHostToDevice);

	struct timeval startTime, endTime;
	setTime(&startTime);
    unsigned long long int *explored_nodes;
    cudaMalloc((void **)&explored_nodes, sizeof(unsigned long long int));
    cudaMemset(explored_nodes, 0, sizeof(unsigned long long int));
#ifdef PERF_DEBUG
    struct timeval appStartTime, appEndTime;
    double appTime = 0;
    struct timeval gcStartTime, gcEndTime;
    double gcTime = 0;
#endif

    unsigned long long int h_explored_nodes = 0;
    while (1) {
#ifdef PERF_DEBUG
        setTime(&appStartTime);
#endif
        oneHeapApplication<<<blockNum, blockSize, smemOffset>>>(d_heap, batchSize, 
                                                         weight, benefit, benefitPerWeight,
                                                         capacity, inputSize,
                                                         heap.globalBenefit, activeCount,
                                                         gc_flag, gc_threshold,
                                                         explored_nodes,
                                                         init_flag);
        cudaDeviceSynchronize();
#ifdef PERF_DEBUG
        setTime(&appEndTime);
        appTime += getTime(&appStartTime, &appEndTime);
        cudaMemcpy(&heap, d_heap, sizeof(Heap<KnapsackItem>), cudaMemcpyDeviceToHost);
        int batchCount = 0;
        cudaMemcpy(&batchCount, heap.batchCount, sizeof(int), cudaMemcpyDeviceToHost);
        int cur_benefit = 0;
        cudaMemcpy(&cur_benefit, heap.globalBenefit, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_explored_nodes, explored_nodes, sizeof(int), cudaMemcpyDeviceToHost);
        cout << appTime << " " << batchCount << " " << cur_benefit << " " << h_explored_nodes << " "; 
        fflush(stdout);
#endif
        /*int app_terminate = 0;*/
        /*cudaMemcpy(&app_terminate, heap.terminate, sizeof(int), cudaMemcpyDeviceToHost);*/
        /*if (app_terminate) break;*/
#ifdef PERF_DEBUG
        setTime(&gcStartTime);
#endif
        // garbage collection
        invalidFilter(heap, d_heap, batchSize,
                      weight, benefit, benefitPerWeight,
                      capacity, inputSize, heap.globalBenefit);
#ifdef PERF_DEBUG
        setTime(&gcEndTime);
        gcTime += getTime(&gcStartTime, &gcEndTime);
        cudaMemcpy(&batchCount, heap.batchCount, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&cur_benefit, heap.globalBenefit, sizeof(int), cudaMemcpyDeviceToHost);
        cout << gcTime << " " << batchCount << " " << cur_benefit << " "; 
#endif
        // reset gc flag
        cudaMemset(gc_flag, 0, sizeof(int));
        int tmpItemCount = heap.itemCount();
        if (tmpItemCount == 0) break;
        cudaMemcpy(activeCount, &tmpItemCount, sizeof(int), cudaMemcpyHostToDevice);
        init_flag = false;
    }

#ifdef PERF_DEBUG
    cout << endl;
#endif
    cudaMemcpy(&h_explored_nodes, explored_nodes, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
	setTime(&endTime);
//	cout << getTime(&startTime, &endTime) << " " << h_explored_nodes << " ";
    int curBenefit = 0;
    cudaMemcpy(&curBenefit, heap.globalBenefit, sizeof(int), cudaMemcpyDeviceToHost);
    printf("%d,%d,%.f\n",curBenefit,h_explored_nodes,getTime(&startTime, &endTime));
    cudaMemcpy((int *)max_benefit, heap.globalBenefit, sizeof(int), cudaMemcpyDeviceToDevice);
    cudaFree(d_heap); d_heap = NULL;
    cudaFree(gc_flag); gc_flag = NULL;
    cudaFree(activeCount); activeCount = NULL;
}

#endif
