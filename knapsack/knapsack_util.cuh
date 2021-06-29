#ifndef KNAPSACK_UTIL_CUH
#define KNAPSACK_UTIL_CUH

#include "heap.cuh"

template <typename K = int>
void heapDataToArray(Heap<K> &heap, K *array, unsigned long long int &number) {
    int pSize = 0, bCount = 0;
    int bSize = heap.batchSize;
    cudaMemcpy(&pSize, heap.partialBufferSize, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&bCount, heap.batchCount, sizeof(int), cudaMemcpyDeviceToHost);

    /*K *tmp = new K[bSize * (bCount + 1)];*/
    /*for (int i = 0; i < bSize * (bCount + 1); i++) {*/
        /*tmp[i] = INIT_LIMITS;*/
    /*}*/
    /*cudaMemcpy(heap.heapItems, tmp, sizeof(K) * bSize * (bCount + 1), cudaMemcpyHostToDevice);*/
    /*delete []tmp;*/
    /*cudaMemset(heap.status, AVAIL, sizeof(int) * (bCount + 1));*/
    /*cudaMemset(heap.batchCount, 0, sizeof(int));*/
    /*cudaMemset(heap.partialBufferSize, 0, sizeof(int));*/

    cudaMemcpy(array, heap.heapItems, sizeof(K) * pSize, cudaMemcpyDeviceToDevice);
    cudaMemcpy(array + pSize, heap.heapItems + bSize, sizeof(K) * bCount * bSize, cudaMemcpyDeviceToDevice);
    number = pSize + bCount * bSize;
}

#endif
