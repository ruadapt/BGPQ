#ifndef MODELS_CUH
#define MODELS_CUH

#include <functional>
#include <iostream>
#include "util.cuh"
#include "heap.cuh"
#include "buffer.cuh"
#include "gc.cuh"
#include "datastructure.hpp"
#include "knapsackKernel.cuh"
#include "knapsack_util.cuh"

using namespace std;

__device__ int explored_flag;

__global__ void DEBUG_INIT() {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        explored_flag = 0;
    }
}

__global__ void oneBufferApplicationMixed(Buffer<KnapsackItem> *buffer, int batchSize, 
                            int *weight, int *benefit, float *benefitPerWeight,
                            int capacity, int inputSize,
                            int *globalBenefit, int *activeCount,
                            int *gc_flag, int gc_threshold, int switch_counter, int global_max_benefit,
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
	int previousBenefit = *globalBenefit;
    int counter = 0;

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

        if (threadIdx.x == 0) {
            atomicAdd(activeCount, (*insSize - *delSize));
            if (atomicAdd(activeCount, 0) == 0) {
                *delSize = -1;
            }
            if (*globalBenefit == global_max_benefit && atomicCAS(&explored_flag, 0, 1) == 0) {
                printf("%d ", *explored_nodes);
            }
			if (*insSize > 0 && previousBenefit < *globalBenefit) {
                counter++;
                if (counter == switch_counter)
    				*gc_flag = 2;
            } else {
                counter = 0;
            }
			previousBenefit = *globalBenefit;
        }
        __syncthreads();
        if (*delSize == -1 || *gc_flag != 0) break;
        __syncthreads();
    }
}

__global__ void oneHeapApplicationMixed(Heap<KnapsackItem> *heap, int batchSize, 
                            int *weight, int *benefit, float *benefitPerWeight,
                            int capacity, int inputSize,
                            int *globalBenefit, int *activeCount,
                            int *gc_flag, int gc_threshold, int switch_counter, int global_max_benefit,
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
	int previousBenefit = *globalBenefit;
    int counter = 0;
	
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

    while(1) {

        if (!init && heap->deleteRoot(delItem, *delSize) == true) {
            __syncthreads();
            heap->deleteUpdate(smemOffset);
        }
        __syncthreads();
        init = false;

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

        if (*delSize > 0) {
            appKernel(weight, benefit, benefitPerWeight,
                      globalBenefit, inputSize, capacity,
                      delItem, delSize,
                      insItem, insSize);
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
            if (threadIdx.x == 0) {
                atomicAdd(explored_nodes, *insSize);
            }
            __syncthreads();
            for (int batchOffset = 0; batchOffset < *insSize; batchOffset += batchSize) {
                int partialSize = min(batchSize, *insSize - batchOffset);
                heap->insertion(insItem + batchOffset, partialSize, smemOffset);
                __syncthreads();
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            atomicAdd(activeCount, (*insSize - *delSize));
            if (atomicAdd(activeCount, 0) == 0) {
                *delSize = -1; // TODO: use one global flag for termination
            }
			/* we only check if max benefit has been updated when we insert something */
			if (*insSize > 0 && previousBenefit == *globalBenefit) {
                if (*globalBenefit == global_max_benefit && atomicCAS(&explored_flag, 0, 1) == 0) {
                    printf("%d ", *explored_nodes);
                }
                counter++;
                if (counter == switch_counter)
	    			// TODO(chenyh) use a universal termination flag which contains different type
    				atomicExch(gc_flag, 2); // we use gc_flag as termination type flag.
            } else {
                counter = 0;
            }
			previousBenefit = *globalBenefit;
            if (*heap->batchCount > gc_threshold) {
                atomicCAS(gc_flag, 0, 1);
            }
        }
        __syncthreads();
        if (*delSize == -1 || *gc_flag != 0) break;
    }
}

void oneheapfifoswitch(int *weight, int *benefit, float *benefitPerWeight,
             int *max_benefit, int capacity, int inputSize,
             int batchNum, int batchSize, int blockNum, int blockSize,
             int gc_threshold, int switch_counter, int global_max_benefit)
{
    /* prepare heap data */
    Heap<KnapsackItem> heap(batchNum, batchSize);
    Heap<KnapsackItem> *d_heap;
    cudaMalloc((void **)&d_heap, sizeof(Heap<KnapsackItem>));
    cudaMemcpy(d_heap, &heap, sizeof(Heap<KnapsackItem>), cudaMemcpyHostToDevice);
	
	/* prepare fifo queue (buffer) */
	Buffer<KnapsackItem> buffer(batchSize * batchNum);
	Buffer<KnapsackItem> *d_buffer;
	cudaMalloc((void **)&d_buffer, sizeof(Buffer<KnapsackItem>));
	cudaMemcpy(d_buffer, &buffer, sizeof(Buffer<KnapsackItem>), cudaMemcpyHostToDevice);	

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
	
	int h_explored_nodes = 0;
    int *explored_nodes;
    cudaMalloc((void **)&explored_nodes, sizeof(int));
    cudaMemset(explored_nodes, 0, sizeof(int));

    int switch_number = 0;
    int gc_number = 0;
	
	int remainItemCount = INT_MAX;
    int tmpItemCountBefore, tmpItemCountAfter;
    int currentBenefit = 0;
    
    struct timeval heapStartTime, heapEndTime;
    struct timeval gcStartTime, gcEndTime;
    struct timeval bufferStartTime, bufferEndTime;
    double heapTime = 0, gcTime = 0, bufferTime = 0;

#ifdef PERF_DEBUG
    DEBUG_INIT<<<1, 1>>>();
    cudaDeviceSynchronize();
#endif
	
	while (remainItemCount != 0) {
		/* ============= switch to the fifo queue mode ============= */
#ifdef PERF_DEBUG
        cout << "start reset heap...";
#endif
		if (init_flag == false) {
            /* reset heap */
            heap.reset();
            /* update benefit */
            cudaMemcpy(heap.globalBenefit, &currentBenefit, sizeof(int), cudaMemcpyHostToDevice);
			/* move items from buffer to heap */
            unsigned long long int begPos;
            cudaMemcpy(&begPos, buffer.begPos, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
			HeapInsert<<<32, blockSize, smemOffset>>>(d_heap, buffer.bufferItems + begPos, remainItemCount, batchSize);
			cudaDeviceSynchronize();
            int new_gc_threshold = remainItemCount / batchSize / 10 + remainItemCount / batchSize;
            gc_threshold = gc_threshold > new_gc_threshold ? gc_threshold : new_gc_threshold;
		}
#ifdef PERF_DEBUG
        cout << "done.. \nnew gc threshold " << gc_threshold << "\n";
        heap.checkInsertHeap();
#endif
		/* heap-based knapsack */
		while (remainItemCount != 0) {
            setTime(&heapStartTime);
            oneHeapApplicationMixed<<<blockNum, blockSize, smemOffset>>>(d_heap, batchSize, 
															 weight, benefit, benefitPerWeight,
															 capacity, inputSize,
															 heap.globalBenefit, activeCount,
															 gc_flag, gc_threshold, switch_counter, global_max_benefit,
															 explored_nodes,
															 init_flag);
			cudaDeviceSynchronize();
            setTime(&heapEndTime);
            heapTime += getTime(&heapStartTime, &heapEndTime);
            setTime(&gcStartTime);
            tmpItemCountBefore = heap.itemCount();
 #ifdef PERF_DEBUG
           cout << "before gc: " << heap.itemCount() << " | " << heap.itemCount() / batchSize << " ";
#endif
			init_flag = false;

			// garbage collection
			invalidFilter(heap, d_heap, batchSize,
						  weight, benefit, benefitPerWeight,
						  capacity, inputSize, heap.globalBenefit);
            gc_number++;
            remainItemCount = heap.itemCount();
            cudaMemcpy(&currentBenefit, heap.globalBenefit, sizeof(int), cudaMemcpyDeviceToHost);
			// check if we should go to buffer
			int h_gc_flag;
			cudaMemcpy(&h_gc_flag, gc_flag, sizeof(int), cudaMemcpyDeviceToHost);
            tmpItemCountAfter = heap.itemCount();
            if (tmpItemCountAfter > tmpItemCountBefore / 2) gc_threshold *= 2;
            setTime(&gcEndTime);
            gcTime += getTime(&gcStartTime, &gcEndTime);
#ifdef PERF_DEBUG
            cout << "after gc: " << heap.itemCount() << " | " << heap.itemCount() / batchSize
                << " benefit: " << currentBenefit << " gc flag: " << h_gc_flag << endl;
#endif

			// reset gc flag
			cudaMemset(gc_flag, 0, sizeof(int));
			// reset activeCount
			cudaMemcpy(activeCount, &remainItemCount, sizeof(int), cudaMemcpyHostToDevice);
			if (h_gc_flag == 2 /* go to buffer */) break;
		}
#ifdef PERF_DEBUG
        cout << "items from heap: " << remainItemCount << " global benefit: " << currentBenefit << endl;
        cout << "heap time: " << getTime(&heapStartTime, &heapEndTime) 
            << " gc time: " << getTime(&gcStartTime, &gcEndTime) << endl;
#endif
		
		if (remainItemCount == 0) break;
#ifdef PERF_DEBUG 
        cout << "start reset buffer ...";
#endif
		/* ============= switch to the fifo queue mode ============= */
		/* move items from heap to buffer */
		unsigned long long int heap_item_count;
		heapDataToArray<KnapsackItem>(heap, buffer.bufferItems, heap_item_count);
        cudaMemset(buffer.readPos, 0, sizeof(unsigned long long int));
		cudaMemcpy(buffer.writePos, &heap_item_count, sizeof(unsigned long long int), cudaMemcpyHostToDevice);
        cudaMemset(buffer.begPos, 0, sizeof(unsigned long long int));
		cudaMemcpy(buffer.endPos, &heap_item_count, sizeof(unsigned long long int), cudaMemcpyHostToDevice);
		cudaMemcpy(buffer.globalBenefit, &currentBenefit, sizeof(int), cudaMemcpyHostToDevice);	
        switch_number++;
#ifdef PERF_DEBUG
        cout << "done\n";
#endif
        setTime(&bufferStartTime);
		oneBufferApplicationMixed<<<32, blockSize, smemOffset>>>(d_buffer, batchSize, 
														 weight, benefit, benefitPerWeight,
														 capacity, inputSize,
														 buffer.globalBenefit, activeCount,
														 gc_flag, gc_threshold, switch_counter, global_max_benefit,
														 explored_nodes,
														 init_flag);
		cudaDeviceSynchronize();
        setTime(&bufferEndTime);
        bufferTime += getTime(&bufferStartTime, &bufferEndTime);
		remainItemCount = buffer.getBufferSize();

#ifdef PERF_DEBUG
        cout << "items from buffer: " << remainItemCount << " global benefit: " << currentBenefit << endl;
        cout << "fifo time: " << getTime(&bufferStartTime, &bufferEndTime) << endl;
        setTime(&gcStartTime);
#endif
        invalidFilterBuffer(buffer, d_buffer, batchSize,
                weight, benefit, benefitPerWeight,
                capacity, inputSize,
                buffer.globalBenefit);
        gc_number++;
		remainItemCount = buffer.getBufferSize();
#ifdef PERF_DEBUG
        setTime(&gcEndTime);
        gcTime += getTime(&gcStartTime, &gcEndTime);
        cout << "gc time: " << getTime(&gcStartTime, &gcEndTime) << " after gc: " << remainItemCount << endl;
//        buffer.printBufferPtr();
#endif
		// reset gc flag
		cudaMemset(gc_flag, 0, sizeof(int));
		// reset activeCount
		cudaMemcpy(activeCount, &remainItemCount, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(&currentBenefit, buffer.globalBenefit, sizeof(int), cudaMemcpyDeviceToHost);
    }

    setTime(&endTime);
    cudaMemcpy(&h_explored_nodes, explored_nodes, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(max_benefit, &currentBenefit, sizeof(int), cudaMemcpyHostToDevice);
#ifdef PERF_DEBUG
    cout << "heap time: " << heapTime << " gc time: " << gcTime << " fifo time: " << bufferTime << " pure total time: ";
    cout << heapTime + gcTime + bufferTime << endl;
    cout << getTime(&startTime, &endTime) << " " << h_explored_nodes << endl;
#else
    printf("%d,%d,%.f,%.f,%.f\n",currentBenefit,h_explored_nodes,heapTime+gcTime,bufferTime,heapTime+gcTime+bufferTime);
/*
    cout << heapTime << " " << gcTime << " " << bufferTime << " " 
        << getTime(&startTime, &endTime) << " " << h_explored_nodes << " "
        << heapTime + gcTime + bufferTime << " "
        << switch_number << " " << gc_number << " ";
*/
#endif
    cudaFree(d_heap); d_heap = NULL;
	cudaFree(d_buffer); d_buffer = NULL;
    cudaFree(gc_flag); gc_flag = NULL;
    cudaFree(activeCount); activeCount = NULL;
}

#endif
