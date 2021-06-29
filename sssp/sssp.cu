// Insert 32K batches into an empty heap with tbsize = 512, K = 512 and tbnum = 32 to 32K
#include <cstdlib>
#include <cstdio>
#include <time.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>

#include <heap.cuh>
#include "util.hpp"

using namespace std;

__global__ void insertKernel(Heap<int> *heap, 
                             int *items, 
                             int arraySize, 
                             int batchSize) {
    // insertion
    int batchNeed = arraySize / batchSize;
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

    if (argc != 2) {
        cout << "./" << argv[0] << " [filename]\n";
        return -1;
    }


    ifstream fin(argv[1]);

    int blockNum = 32;
    int batchNum = 512 * 1024;
    int batchSize = 1024;
    int blockSize = 512;

    int arrayNum;
    fin >> arrayNum;
    arrayNum = arrayNum / batchSize * batchSize;

    struct timeval startTime;
    struct timeval endTime;

    srand(time(NULL));

    int *oriItems = new int[arrayNum];
#ifdef DEBUG
    int *testItems = new int[arrayNum];
#endif
    for (int i = 0; i < arrayNum; ++i) {
        fin >> oriItems[i];
        oriItems[i] = rand() % arrayNum;
#ifdef DEBUG
        testItems[i] = oriItems[i];
#endif
    }

    fin.close();

#ifdef DEBUG
    std::sort(testItems, testItems + arrayNum);
#endif

    double insertTime, deleteTime;

    // bitonic heap sort
    Heap<int> h_heap(batchNum * 2, batchSize, INT_MAX);

    int *heapItems;
    Heap<int> *d_heap;

    cudaMalloc((void **)&heapItems, sizeof(int) * arrayNum);
    cudaMemcpy(heapItems, oriItems, sizeof(int) * arrayNum, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_heap, sizeof(Heap<int>));
    cudaMemcpy(d_heap, &h_heap, sizeof(Heap<int>), cudaMemcpyHostToDevice);

    int smemSize = batchSize * 3 * sizeof(int);
    smemSize += (blockSize + 1) * sizeof(int) + 2 * batchSize * sizeof(int);


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

    printf("%s,sssp,%s,%d,%.f,%.f,%.f\n",
            argv[0] == std::string("./ssspT") ? "BGPQ_T" : "BGPQ_B",argv[1], arrayNum,
            insertTime,deleteTime,insertTime+deleteTime);

#ifdef DEBUG
    cudaMemcpy(oriItems, heapItems, sizeof(int) * arrayNum, cudaMemcpyDeviceToHost);
    for (int i = 0; i < arrayNum; i++) {
        /*printf("%d | ", oriItems[i]);*/
        if (oriItems[i] != testItems[i]) {
            printf("error %d: BGPQ: %d SEQ: %d\n", i, oriItems[i], testItems[i]);
            break;
        }
    }
    delete []testItems; testItems = NULL;
#endif

    delete []oriItems; oriItems = NULL;
    cudaFree(heapItems); heapItems = NULL;

    /*for (int i = 0; i < arrayNum; i++) {*/
        /*cout << oriItems[i] << " ";*/
    /*}*/
//    h_heap.printHeap();
/*
    //    cout << "heap insert time: " << insertTime << "ms" << endl;

    cudaMemcpy(&h_heap, d_heap, sizeof(Heap<int>), cudaMemcpyDeviceToHost);
    if (h_heap.checkInsertHeap()) cout << "Insert Result Correct\n";
    else cout << "Insert Result Wrong\n";
*/
    return 0;

}
