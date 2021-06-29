#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <time.h>
#include <algorithm>
#include <cmath>

#include <vector>

#include "heap.cuh"
#include "util.hpp"

using namespace std;
 
template <class T>
__global__ void insertKernel(Heap<T> *heap, 
                             T *items,
                             T *startLoc,
                             T *nItems,
                             uint32_t nOps,
                             uint32_t arraySize,
                             uint32_t batchSize) {
    // insertion
    for (uint32_t i = blockIdx.x; i < nOps; i += gridDim.x) {
        // insert items to buffer
        heap->insertion(items + startLoc[i],
                        nItems[i], 0);
        __syncthreads();
    }
}

template <class T>
__global__ void deleteKernel(Heap<T> *heap, 
                             T *items, 
                             T *startLoc,
                             T *nItems,
                             uint32_t nOps,
                             uint32_t arraySize, 
                             uint32_t batchSize) {
    // deletion
    for (uint32_t i = blockIdx.x; i < nOps; i += gridDim.x) {

        // delete items from heap
        int size = (int)nItems[i];
        if (heap->deleteRoot(items, size) == true) {
            __syncthreads();

            heap->deleteUpdate(0);
        }
        __syncthreads();
    }
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__);  }
inline void gpuAssert(cudaError_t code, const char *file, uint32_t line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


int main(int argc, char *argv[]) {

    if (argc != 5) {
        cout << argv[0] << " [batch size] [array num] [block num] [block size]\n";
        return -1;
    }

    srand(time(NULL));

    uint32_t batchSize = atoi(argv[1]);
    uint32_t arrayNum = pow(2, atoi(argv[2]));
    arrayNum = (arrayNum + batchSize - 1) / batchSize * batchSize;
    uint32_t batchNum = 1;
    while (batchNum * batchSize < arrayNum * 2) {
        batchNum <<= 1;
    }
    uint32_t blockNum = atoi(argv[3]);
    uint32_t blockSize =atoi(argv[4]);
    //uint32_t blockSize = batchSize / 2;

    // generate # of keys operated for each operation
    uint32_t ii = 0;
    vector<uint32_t> _startLoc;
    vector<uint32_t> _numItems;
        
    while (ii + batchSize < arrayNum) {
        _startLoc.push_back(ii);
        uint32_t _nitems = rand() % (batchSize / 8);
        ii += _nitems;
        _numItems.push_back(_nitems);
    }
    _startLoc.push_back(ii);
    _numItems.push_back(arrayNum - ii);
    uint32_t nOps = _startLoc.size();

    uint32_t *startLoc = new uint32_t[nOps]();
    uint32_t *numItems = new uint32_t[nOps]();
    for (uint32_t i = 0; i < nOps; i++) {
        startLoc[i] = _startLoc[i];
        numItems[i] = _numItems[i];
    }

    printf("%s test size: 2^%u/%u heap[%u/%u] kernel[%u/%u] nOps: %u\n", argv[0], atoi(argv[2]), arrayNum, batchSize, batchNum, blockNum, blockSize, nOps);

    struct timeval startTime;
    struct timeval endTime;

    uint32_t *oriItems = new uint32_t[arrayNum]();
    uint32_t *oriItems_ = new uint32_t[arrayNum]();
    //for (uint32_t i = 0; i < arrayNum / 2; ++i) {
        //oriItems_[i] = oriItems[i] = rand() % (INT_MAX / 2);
    //}
    for (uint32_t i = 0; i < arrayNum; ++i) {
        oriItems_[i] = oriItems[i] = INT_MAX / 2 + rand() % (INT_MAX / 2);
    }

    //vector<uint32_t> npOriItems_(nOps * batchSize, UINT32_MAX);
    //for (int i = 0; i < nOps; i++) {
        //for (int j = 0; j < numItems[i]; j++) {
            //npOriItems_[i * batchSize + j] = oriItems[startLoc[i] + j];
        //}
    //}
    //uint32_t *npOriItems = new uint32_t[nOps * batchSize]();
    //uint32_t *npStartLoc = new uint32_t[nOps]();
    //uint32_t *npNumItems = new uint32_t[nOps]();
    //for (int i = 0; i < nOps * batchSize; i++) {
        //npOriItems[i] = npOriItems_[i];
    //}
    //for (int i = 0; i < nOps; i++) {
        //npStartLoc[i] = i * batchSize;
        //npNumItems[i] = batchSize;
    //}

    // sort original solutions
    std::sort(oriItems_, oriItems_ + arrayNum);

    {
    // bitonic heap sort
    Heap<uint32_t> h_heap(batchNum, batchSize, UINT32_MAX);

    uint32_t *heapItems;
    Heap<uint32_t> *d_heap;

    cudaMalloc((void **)&heapItems, sizeof(uint32_t) * arrayNum);
    cudaMemcpy(heapItems, oriItems, sizeof(uint32_t) * arrayNum, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_heap, sizeof(Heap<uint32_t>));
    cudaMemcpy(d_heap, &h_heap, sizeof(Heap<uint32_t>), cudaMemcpyHostToDevice);

    uint32_t *d_startLoc, *d_numItems;
    cudaMalloc((void **)&d_startLoc, sizeof(uint32_t) * nOps);
    cudaMemcpy(d_startLoc, startLoc, sizeof(uint32_t) * nOps, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_numItems, sizeof(uint32_t) * nOps);
    cudaMemcpy(d_numItems, numItems, sizeof(uint32_t) * nOps, cudaMemcpyHostToDevice);

    uint32_t smemSize = batchSize * 3 * sizeof(uint32_t);
    smemSize += (blockSize + 1) * sizeof(uint32_t) + 2 * batchSize * sizeof(uint32_t);

    // concurrent insertion
    setTime(&startTime);

    insertKernel<uint32_t><<<blockNum, blockSize, smemSize>>>(d_heap, heapItems, d_startLoc, d_numItems, nOps, arrayNum, batchSize);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    setTime(&endTime);
    double insertTime = getTime(&startTime, &endTime);

    // concurrent deletion
    setTime(&startTime);

    deleteKernel<uint32_t><<<blockNum, blockSize, smemSize>>>(d_heap, heapItems, d_startLoc, d_numItems, nOps, arrayNum, batchSize);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    setTime(&endTime);
    double deleteTime = getTime(&startTime, &endTime);

    printf("Success %.f %.f %.f\n", insertTime, deleteTime, insertTime + deleteTime);

    cudaMemcpy(oriItems, heapItems, sizeof(uint32_t) * arrayNum, cudaMemcpyDeviceToHost);
    for (int i = 0; i < arrayNum; i++) {
        if (oriItems[i] != oriItems_[i]) {
            printf("heap: %u stl_sort %u\n", oriItems[i], oriItems_[i]);
            return -1;
        }
    }
    cudaFree(d_heap); d_heap = nullptr;
    cudaFree(heapItems); heapItems = nullptr;
    cudaFree(d_startLoc); d_startLoc = nullptr;
    cudaFree(d_numItems); d_numItems = nullptr;
    }

    return 0;
}
