#ifndef GC_CUH
#define GC_CUH

//#include "buffer.cuh"
#include "heap.cuh"

#include "pq_model.cuh"

namespace astar {

/*__global__ void garbageCollectionBuffer(Buffer<HeapItem> *buffer, int batchSize,*/
                       /*unsigned long long int begPos, unsigned long long int endPos, unsigned long long int buffer_capacity,*/
                        /*int *weight, int *benefit, float *benefitPerWeight,*/
                        /*int capacity, int inputSize,*/
                        /*int *max_benefit,*/
                        /*HeapItem *insItems, unsigned long long int *insSize)*/
/*{*/
    /*for (unsigned long long int i = begPos + blockIdx.x * blockDim.x + threadIdx.x; i < endPos; i += gridDim.x * blockDim.x) {*/
        /*int oldBenefit = -(buffer->bufferItems[i % buffer_capacity]).first;*/
        /*int oldWeight = (buffer->bufferItems[i % buffer_capacity]).second;*/
        /*short oldIndex = (buffer->bufferItems[i % buffer_capacity]).third;*/

        /*int _bound = ComputeBound(oldBenefit, oldWeight, oldIndex, inputSize, capacity, weight, benefit);*/
        /*if (oldBenefit + _bound > *max_benefit) {*/
            /*unsigned long long int index = atomicAdd(insSize, 1);*/
            /*insItems[index] = buffer->bufferItems[i % buffer_capacity];*/
        /*}*/
    /*}*/
/*}*/

/*void invalidFilterBuffer(Buffer<HeapItem> &buffer, Buffer<HeapItem> *d_buffer, int batchSize,*/
                    /*int *weight, int *benefit, float *benefitPerWeight,*/
                    /*int capacity, int inputSize,*/
                    /*int *max_benefit, int k = 10240)*/
/*{*/
    /*unsigned long long int old_begPos, old_endPos, buffer_capacity;*/
    /*cudaMemcpy(&old_begPos, buffer.begPos, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);*/
    /*cudaMemcpy(&old_endPos, buffer.endPos, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);*/
    /*buffer_capacity = buffer.capacity;*/

    /*HeapItem *insItems;*/
    /*unsigned long long int *insSize;*/
    /*cudaMalloc((void **)&insItems, (old_endPos - old_begPos) * sizeof(HeapItem));*/
    /*cudaMalloc((void **)&insSize, sizeof(unsigned long long int));*/
    /*unsigned long long int h_insSize = 0;*/
    /*cudaMemcpy(insSize, &h_insSize, sizeof(unsigned long long int), cudaMemcpyHostToDevice);*/

    /*int blockNum = 32;*/
    /*int blockSize = batchSize;*/
    /*[>size_t smemOffset = sizeof(HeapItem) * batchSize * 3 +<]*/
                        /*[>sizeof(int) * 2 +<]*/
                        /*[>sizeof(HeapItem) +<]*/
                        /*[>5 * batchSize * sizeof(HeapItem);<]*/

    /*garbageCollectionBuffer<<<blockNum, blockSize>>>(d_buffer, batchSize,*/
                                                /*old_begPos, old_endPos, buffer_capacity,*/
                                                /*weight, benefit, benefitPerWeight,*/
                                                /*capacity, inputSize,*/
                                                /*max_benefit,*/
                                                /*insItems, insSize);*/
    /*cudaDeviceSynchronize();*/

    /*cudaMemcpy(&h_insSize, insSize, sizeof(int), cudaMemcpyDeviceToHost);*/

    /*cudaMemset(buffer.readPos, 0, sizeof(unsigned long long int));*/
    /*cudaMemset(buffer.begPos, 0, sizeof(unsigned long long int));*/
    /*cudaMemcpy(buffer.writePos, &h_insSize, sizeof(unsigned long long int), cudaMemcpyHostToDevice);*/
    /*cudaMemcpy(buffer.endPos, &h_insSize, sizeof(unsigned long long int), cudaMemcpyHostToDevice);*/

    /*cudaMemcpy(buffer.bufferItems, insItems, sizeof(HeapItem) * h_insSize, cudaMemcpyDeviceToDevice);*/

    /*cudaFree(insItems); insItems = NULL;*/
    /*cudaFree(insSize); insSize = NULL;*/

/*}*/

template <class HeapItem>
__global__ void garbageCollection(Heap<HeapItem> *heap, uint32_t batchSize, uint32_t batchCount,
                                  AppItem *app_item,
                                  HeapItem *insItems, uint32_t *insSize)
{
    uint32_t *dist = app_item->d_dist;
    HeapItem dummy_item;
    // TODO now only support batchsize == blockSize
    //handle parital batch first
    if (blockIdx.x == 0) {
        for (int i = threadIdx.x; i < *heap->partialBatchSize; i += blockDim.x) {
            HeapItem item = heap->heapItems[i];
            uint32_t item_node = item.node_;
            uint32_t item_f = item.f_;
            uint32_t item_g = item_f - app_item->H(item_node);

            if (item_g == dist[item_node]) {
                int index = atomicAdd(insSize, 1);
                insItems[index] = heap->heapItems[i];
            }
            heap->heapItems[i] = dummy_item;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            *heap->partialBatchSize = 0;
        }
        __syncthreads();
    }
    for (int batchIndex = 1 + blockIdx.x; batchIndex <= batchCount; batchIndex += gridDim.x) {
        if (threadIdx.x == 0) {
            heap->status[batchIndex] = 0; /* UNUSED in td or AVAIL in bu */
            heap->tbstate[blockIdx.x] = 1;
        }

        int i = batchIndex * batchSize + threadIdx.x;
        HeapItem item = heap->heapItems[i];
        uint32_t item_node = item.node_;
        uint32_t item_f = item.f_;
        uint32_t item_g = item_f - app_item->H(item_node);

        if (item_g == dist[item_node]) {
            int index = atomicAdd(insSize, 1);
            insItems[index] = heap->heapItems[i];
        }
        heap->heapItems[i] = dummy_item;
    }

    if(!threadIdx.x && !blockIdx.x){
        *heap->batchCount = 0;
        *heap->terminate = 0;
    }
    __syncthreads();

}

template <class HeapItem>
__global__ void HeapInsert(Heap<HeapItem> *heap, HeapItem *insItems, uint32_t *insSize, uint32_t batchSize)
{	
	if(*insSize == 0)
		return;
	int batchNeed = (*insSize + batchSize - 1) / batchSize;
    for (int i = blockIdx.x; i < batchNeed; i += gridDim.x) {
        int size = min(batchSize, *insSize - i * batchSize);
        heap->insertion(insItems + i * batchSize,
                        size, 0);
        __syncthreads();
    }
	
}

template <class HeapItem>
void invalidFilter(PQModel<HeapItem> &model, Heap<HeapItem> &heap) {
    uint32_t batchSize = model.batch_size_;
    uint32_t blockNum = 16;
    uint32_t blockSize = model.block_size_;

    Heap<HeapItem> *d_heap = model.d_heap_;
    // this space should be preallocated (but may waste some gpu space)
    uint32_t batchCount = 0;
    cudaMemcpy(&batchCount, heap.batchCount, sizeof(uint32_t), cudaMemcpyDeviceToHost);

    HeapItem *insItems;
    uint32_t *insSize;
    cudaMalloc((void **)&insItems, (batchCount + 1) * batchSize * sizeof(HeapItem));
    cudaMalloc((void **)&insSize, sizeof(uint32_t));
    cudaMemset(insSize, 0, sizeof(uint32_t));

    uint32_t smemOffset = sizeof(HeapItem) * batchSize * 3 +
                        sizeof(uint32_t) * 2 +
                        sizeof(HeapItem) +
                        5 * batchSize * sizeof(HeapItem);

    garbageCollection<HeapItem><<<blockNum, blockSize>>>(d_heap, batchSize, batchCount,
                                                         model.d_app_item_,
                                                         insItems, insSize);
    cudaDeviceSynchronize();

    HeapInsert<HeapItem><<<blockNum, blockSize, smemOffset>>>(d_heap, insItems, insSize, batchSize);
    cudaDeviceSynchronize();

    cudaFree(insItems); insItems = NULL;
    cudaFree(insSize); insSize = NULL;

}

/*void invalidFilter2Heap(Heap<HeapItem> heap1, Heap<HeapItem> heap2, */
                        /*Heap<HeapItem> *d_heap1, Heap<HeapItem> *d_heap2, int batchSize,*/
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


} // astar

#endif
