#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <time.h>
#include <queue>
#include <algorithm>
#include <functional>

#include <heap.cuh>
#include <cmath>
#include "util.hpp"

using namespace std;

__global__ void concurrentKernel(Heap<int> *heap, int *items, int initSize, int testSize, int batchSize) {

    for (int i = blockIdx.x; i < testSize / batchSize; i += gridDim.x) {
        heap->insertion(items + initSize + i * batchSize,
                        batchSize, 0);
        __syncthreads();
        if (heap->deleteRoot(items, batchSize) == true) {
            __syncthreads();

            heap->deleteUpdate(0);
        }
        __syncthreads();
    }
}



__global__ void insertKernel(Heap<int> *heap, 
                             int *items, 
                             int arraySize, 
                             int batchSize) {

//    batchSize /= 3;
    int batchNeed = arraySize / batchSize;
    // insertion
    for (int i = blockIdx.x; i < batchNeed; i += gridDim.x) {
        // insert items to buffer
        heap->insertion(items + i * batchSize,
                        batchSize, 0);
        __syncthreads();
    }
}

__global__ void deleteKernel(Heap<int> *heap, int *items, int arraySize, int batchSize) {

    int batchNeed = arraySize / batchSize;
    int size = 0;
    // deletion
    for (int i = blockIdx.x; i < batchNeed; i += gridDim.x) {

        // delete items from heap
        if (heap->deleteRoot(items, size) == true) {
            __syncthreads();

            heap->deleteUpdate(0);
        }
        __syncthreads();
    }
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__);  }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


int main(int argc, char *argv[]) {

    if (argc != 5) {
        cout << argv[0] << " [test type] [# init in M] [# keys in M] [keyType: 0:random 1:ascend 2:descend]\n";
        return -1;
    }

    srand(time(NULL));

    int batchSize = 1024;
    int testType = atoi(argv[1]);
    int initSize = atoi(argv[2]) * 1000000;
    int _initSize = atoi(argv[2]);
    initSize = (initSize + batchSize - 1) / batchSize * batchSize;
    int arrayNum = atoi(argv[3]) * 1000000;
    int _arrayNum = arrayNum / 1000000;
    arrayNum = (arrayNum + batchSize - 1) / batchSize * batchSize;
    int keyType = atoi(argv[4]);

    int batchNum = ((initSize + arrayNum) / batchSize + 1) * 2;
    while (batchNum <= (initSize + arrayNum) / batchSize)
        batchNum *= 2;
    batchNum *= 2;
    batchNum = 512 * 1024;
    int blockNum = 32;
    int blockSize = 512;

    struct timeval startTime;
    struct timeval endTime;
    double insertTime, deleteTime;
    int *oriItems = new int[initSize + arrayNum];
#ifdef ENABLE_SEQ 
    int *h_tItems = new int[arrayNum];
#endif
    for (int i = 0; i < initSize + arrayNum; ++i) {
        oriItems[i] = rand() % INT_MAX;
#ifdef ENABLE_SEQ
        h_tItems[i] = oriItems[i];
#endif
    }

    if (keyType == 1) {
        std::sort(oriItems, oriItems + initSize + arrayNum);
    } else if (keyType == 2) {
        std::sort(oriItems, oriItems + initSize + arrayNum, std::greater<int>());
    }

#ifdef ENABLE_SEQ
    std::sort(h_tItems, h_tItems + arrayNum);
#endif

    // bitonic heap sort
    Heap<int> h_heap(batchNum * 2, batchSize, INT_MAX);

    int *heapItems;
    Heap<int> *d_heap;

    cudaMalloc((void **)&heapItems, sizeof(int) * (arrayNum + initSize));
    cudaMemcpy(heapItems, oriItems, sizeof(int) * (arrayNum + initSize), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_heap, sizeof(Heap<int>));
    cudaMemcpy(d_heap, &h_heap, sizeof(Heap<int>), cudaMemcpyHostToDevice);

    int smemSize = batchSize * 3 * sizeof(int);
    smemSize += (blockSize + 1) * sizeof(int) + 2 * batchSize * sizeof(int);

    if (testType == 0) {
        // concurrent insertion
        setTime(&startTime);

        insertKernel<<<blockNum, blockSize, smemSize>>>(d_heap, heapItems, arrayNum, batchSize);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        setTime(&endTime);
        insertTime = getTime(&startTime, &endTime);

        // concurrent deletion
        setTime(&startTime);

        deleteKernel<<<blockNum, blockSize, smemSize>>>(d_heap, heapItems, arrayNum, batchSize);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        setTime(&endTime);
        deleteTime = getTime(&startTime, &endTime);
        /*cout << "BGPQ delete: " << deleteTime << "ms\n";*/
        /*cout << insertTime + deleteTime << " ms\n";*/

        printf("%s,insdel,%d,%dM,%.f,%.f,%.f\n",
                argv[0] == std::string("./BGPQ_T") ? "BGPQ_T" : "BGPQ_B",
                keyType,_arrayNum,insertTime,deleteTime,insertTime+deleteTime);
#ifdef ENABLE_SEQ
        cudaMemcpy(oriItems, heapItems, arrayNum * sizeof(int), cudaMemcpyDeviceToHost);
        for (int i = 0; i < arrayNum; i++) {
            if (oriItems[i] != h_tItems[i]) {
                printf("Error %d seq %d BGPQ %d\n", i, h_tItems[i], oriItems[i]);
                exit(1);
            }
        }
        delete []h_tItems;
#endif
    } else if (testType == 1) {
       // concurrent insertion
        setTime(&startTime);

        concurrentKernel<<<blockNum, blockSize, smemSize>>>(d_heap, heapItems, initSize, arrayNum, batchSize);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        setTime(&endTime);
        insertTime = getTime(&startTime, &endTime);
        /*cout << insertTime << " ms\n";*/
        printf("%s,utl,%dM,%dM,%.f\n", 
                argv[0]==std::string("./BGPQ_T") ? "BGPQ_T" : "BGPQ_B",_initSize,_arrayNum,insertTime);

    }
    cudaFree(heapItems); heapItems = NULL;
    cudaFree(d_heap); d_heap = NULL;

    return 0;

}
