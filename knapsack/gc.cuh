#ifndef GC_CUH
#define GC_CUH

#include "buffer.cuh"
#include "heap.cuh"
#include "datastructure.hpp"

#include "knapsackKernel.cuh"

__global__ void garbageCollectionBuffer(Buffer<KnapsackItem> *buffer, int batchSize,
                       unsigned long long int begPos, unsigned long long int endPos, unsigned long long int buffer_capacity,
                        int *weight, int *benefit, float *benefitPerWeight,
                        int capacity, int inputSize,
                        int *max_benefit,
                        KnapsackItem *insItems, unsigned long long int *insSize)
{
    for (unsigned long long int i = begPos + blockIdx.x * blockDim.x + threadIdx.x; i < endPos; i += gridDim.x * blockDim.x) {
        int oldBenefit = -(buffer->bufferItems[i % buffer_capacity]).first;
        int oldWeight = (buffer->bufferItems[i % buffer_capacity]).second;
        short oldIndex = (buffer->bufferItems[i % buffer_capacity]).third;

        int _bound = ComputeBound(oldBenefit, oldWeight, oldIndex, inputSize, capacity, weight, benefit);
        if (oldBenefit + _bound > *max_benefit) {
            unsigned long long int index = atomicAdd(insSize, 1);
            insItems[index] = buffer->bufferItems[i % buffer_capacity];
        }
    }
}

__global__ void garbageCollection(Heap<KnapsackItem> *heap, int batchSize, int batchCount,
                        int *weight, int *benefit, float *benefitPerWeight,
                        int capacity, int inputSize,
                        int *max_benefit, int k,
                        KnapsackItem *insItems, int *insSize)
{
    KnapsackItem dummy_item(INIT_LIMITS, 0, 0, 0);
    // TODO now only support batchsize == blockSize
    /*handle parital batch first*/
    if (blockIdx.x == 0) {
        for (int i = threadIdx.x; i < *heap->partialBufferSize; i += blockDim.x) {
            int oldBenefit = -(heap->heapItems[i]).first;
            int oldWeight = (heap->heapItems[i]).second;
            short oldIndex = (heap->heapItems[i]).third;

            int _bound = ComputeBound(oldBenefit, oldWeight, oldIndex, inputSize, capacity, weight, benefit);
            if (oldBenefit + _bound > *max_benefit) {
                int index = atomicAdd(insSize, 1);
                insItems[index] = heap->heapItems[i];
            }
            heap->heapItems[i] = dummy_item;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            *heap->partialBufferSize = 0;
        }
        __syncthreads();
    }
    int startBatchIdx = batchCount - k >= 0 ? batchCount - k + 1 : 1;
    for (int batchIndex = startBatchIdx + blockIdx.x; batchIndex <= batchCount; batchIndex += gridDim.x) {
        if(threadIdx.x == 0) {
            heap->status[batchIndex] = AVAIL;
        }

        int i = batchIndex * batchSize + threadIdx.x;
        int oldBenefit = -(heap->heapItems[i]).first;
        int oldWeight = (heap->heapItems[i]).second;
        short oldIndex = (heap->heapItems[i]).third;

        int _bound = ComputeBound(oldBenefit, oldWeight, oldIndex, inputSize, capacity, weight, benefit);
		if (oldBenefit + _bound > *max_benefit) {
            int index = atomicAdd(insSize, 1);
            insItems[index] = heap->heapItems[i];
		}
        heap->heapItems[i] = dummy_item;
    }

    if(!threadIdx.x && !blockIdx.x){
        *heap->batchCount = startBatchIdx - 1;
    }
    __syncthreads();

}

__global__ void HeapInsert(Heap<KnapsackItem> *heap, KnapsackItem *insItems, int insSize, int batchSize){	
	if(insSize == 0)
		return;
	int batchNeed = (insSize + batchSize - 1) / batchSize;
    for (int i = blockIdx.x; i < batchNeed; i += gridDim.x) {
        int size = min(batchSize, insSize - i * batchSize);
        heap->insertion(insItems + i * batchSize,
                        size, 0);
        __syncthreads();
    }
	
}

void invalidFilterBuffer(Buffer<KnapsackItem> &buffer, Buffer<KnapsackItem> *d_buffer, int batchSize,
                    int *weight, int *benefit, float *benefitPerWeight,
                    int capacity, int inputSize,
                    int *max_benefit, int k = 10240)
{
    unsigned long long int old_begPos, old_endPos, buffer_capacity;
    cudaMemcpy(&old_begPos, buffer.begPos, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&old_endPos, buffer.endPos, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
    buffer_capacity = buffer.capacity;

    KnapsackItem *insItems;
    unsigned long long int *insSize;
    cudaMalloc((void **)&insItems, (old_endPos - old_begPos) * sizeof(KnapsackItem));
    cudaMalloc((void **)&insSize, sizeof(unsigned long long int));
    unsigned long long int h_insSize = 0;
    cudaMemcpy(insSize, &h_insSize, sizeof(unsigned long long int), cudaMemcpyHostToDevice);

    int blockNum = 32;
    int blockSize = batchSize;
    /*size_t smemOffset = sizeof(KnapsackItem) * batchSize * 3 +*/
                        /*sizeof(int) * 2 +*/
                        /*sizeof(KnapsackItem) +*/
                        /*5 * batchSize * sizeof(KnapsackItem);*/

    garbageCollectionBuffer<<<blockNum, blockSize>>>(d_buffer, batchSize,
                                                old_begPos, old_endPos, buffer_capacity,
                                                weight, benefit, benefitPerWeight,
                                                capacity, inputSize,
                                                max_benefit,
                                                insItems, insSize);
    cudaDeviceSynchronize();

    cudaMemcpy(&h_insSize, insSize, sizeof(int), cudaMemcpyDeviceToHost);

    cudaMemset(buffer.readPos, 0, sizeof(unsigned long long int));
    cudaMemset(buffer.begPos, 0, sizeof(unsigned long long int));
    cudaMemcpy(buffer.writePos, &h_insSize, sizeof(unsigned long long int), cudaMemcpyHostToDevice);
    cudaMemcpy(buffer.endPos, &h_insSize, sizeof(unsigned long long int), cudaMemcpyHostToDevice);

    cudaMemcpy(buffer.bufferItems, insItems, sizeof(KnapsackItem) * h_insSize, cudaMemcpyDeviceToDevice);

    cudaFree(insItems); insItems = NULL;
    cudaFree(insSize); insSize = NULL;

}

void invalidFilter(Heap<KnapsackItem> &heap, Heap<KnapsackItem> *d_heap, int batchSize,
                    int *weight, int *benefit, float *benefitPerWeight,
                    int capacity, int inputSize,
                    int *max_benefit, int k = 10240)
{
    int batchCount = 0;
    cudaMemcpy(&batchCount, heap.batchCount, sizeof(int), cudaMemcpyDeviceToHost);
    KnapsackItem *insItems;
    int *insSize;
    cudaMalloc((void **)&insItems, (batchCount + 1) * batchSize * sizeof(KnapsackItem));
    cudaMalloc((void **)&insSize, sizeof(int));
    cudaMemset(insSize, 0, sizeof(int));

    int blockNum = 32;
    int blockSize = batchSize;

    size_t smemOffset = sizeof(KnapsackItem) * batchSize * 3 +
                        sizeof(int) * 2 +
                        sizeof(KnapsackItem) +
                        5 * batchSize * sizeof(KnapsackItem);

    garbageCollection<<<blockNum, blockSize>>>(d_heap, batchSize, batchCount,
                                                weight, benefit, benefitPerWeight,
                                                capacity, inputSize,
                                                max_benefit, k,
                                                insItems, insSize);
    cudaDeviceSynchronize();

    int h_insSize;
    cudaMemcpy(&h_insSize, insSize, sizeof(int), cudaMemcpyDeviceToHost);

    HeapInsert<<<blockNum, blockSize, smemOffset>>>(d_heap, insItems, h_insSize, batchSize);
    cudaDeviceSynchronize();

    cudaFree(insItems); insItems = NULL;
    cudaFree(insSize); insSize = NULL;

}

/*void invalidFilter2Heap(Heap<KnapsackItem> heap1, Heap<KnapsackItem> heap2, */
                        /*Heap<KnapsackItem> *d_heap1, Heap<KnapsackItem> *d_heap2, int batchSize,*/
                        /*int *weight, int *benefit, float *benefitPerWeight,*/
                        /*int capacity, int inputSize, int *max_benefit,*/
                        /*int expandFlag, int gcThreshold, int k)*/
/*{*/
    /*if (expandFlag == 0) {*/
        /*int batchCount = heap1.nodeCount();*/
        /*if (batchCount > gcThreshold) {*/
            /*cout << "gc..." << batchCount;*/
            /*invalidFilter(heap1, d_heap1, batchSize, batchCount,*/
                          /*weight, benefit, benefitPerWeight,*/
                          /*capacity, inputSize, max_benefit, 1024000);*/
            /*cout << heap2.nodeCount();*/
        /*}*/
    /*}*/
    /*else if (expandFlag == 1) {*/
        /*int batchCount = heap2.nodeCount();*/
        /*if (batchCount > gcThreshold) {*/
            /*cout << "gc..." << batchCount;*/
            /*invalidFilter(heap2, d_heap2, batchSize, batchCount,*/
                          /*weight, benefit, benefitPerWeight,*/
                          /*capacity, inputSize, max_benefit, 1024000);*/
            /*cout << heap2.nodeCount();*/
        /*}*/
    /*}*/
/*}*/



#endif
