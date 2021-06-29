#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <time.h>

#include "buffer.cuh"
#include "util.cuh"

using namespace std;

// TODO insert random number, delete a batch until finish
// verify, same input (sorted) same output (sorted)
__global__ void concurrentKernel(Buffer<int> *buffer, int *items, int arraySize, int blockNum, int blockSize) {

    if (blockIdx.x < blockNum / 2) {
        int batchNeed = (arraySize + blockSize - 1) / blockSize;
        for (int i = blockIdx.x; i < batchNeed; i += gridDim.x / 2) {
            int size = blockSize < (arraySize - blockSize * i) ? blockSize : (arraySize - blockSize * i);
//            buffer->insertToBuffer(items + arraySize + i * blockSize, size, 0);
            int tmp_size = size / 3;
            buffer->insertToBuffer(items + arraySize + i * blockSize, tmp_size, 0);
            __syncthreads();
            buffer->insertToBuffer(items + arraySize + i * blockSize + tmp_size, (size - tmp_size), 0);
            __syncthreads();
        }
    }
    else {
        int batchNeed = (arraySize + blockSize - 1) / blockSize;
        for (int i = blockIdx.x - gridDim.x / 2; i < batchNeed; i += gridDim.x / 2) {
            int size = 0;
            while (size == 0) {
                buffer->deleteFromBuffer(items + i * blockSize, size, 0);
                __syncthreads();
            }
        }
    }
}

__global__ void insertKernel(Buffer<int> *buffer, int *items, int arraySize, int blockSize) {
    int batchNeed = (arraySize + blockSize - 1) / blockSize;
    for (int i = blockIdx.x; i < batchNeed; i += gridDim.x) {
        int size = blockSize < (arraySize - blockSize * i) ? blockSize : (arraySize - blockSize * i);
        buffer->insertToBuffer(items + i * blockSize, size, 0);
        __syncthreads();
    }
}

__global__ void deleteKernel(Buffer<int> *buffer, int *items, int arraySize, int blockSize) {
    int batchNeed = (arraySize + blockSize - 1) / blockSize;
    for (int i = blockIdx.x; i < batchNeed; i += gridDim.x) {
        int size = 0;
        buffer->deleteFromBuffer(items + arraySize + i * blockSize, size, 0);
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
        cout << "./bufferTest [arrayNum] [bufferSize] [blockNum] [blockSize]\n";
        return -1;
    }

    srand(time(NULL));

    int arrayNum = atoi(argv[1]);
    int bufferSize = atoi(argv[2]);
    int blockNum = atoi(argv[3]);
    int blockSize = atoi(argv[4]);
#ifdef PRINT_DEBUG
    struct timeval startTime;
    struct timeval endTime;
#endif
    // generate <keys, vals> sequence
    int *oriItems = new int[2 * arrayNum];
    int *resItems = new int[2 * arrayNum];
    for (int i = 0; i < 2 * arrayNum; ++i) {
        oriItems[i] = i;
    }

    Buffer<int> h_buffer(bufferSize);
    Buffer<int> *d_buffer;
    cudaMalloc((void **)&d_buffer, sizeof(Buffer<int>));
    cudaMemcpy(d_buffer, &h_buffer, sizeof(Buffer<int>), cudaMemcpyHostToDevice);

    int *bufferItems;
    cudaMalloc((void **)&bufferItems, sizeof(int) * 2 * arrayNum);
    cudaMemcpy(bufferItems, oriItems, sizeof(int) * 2 * arrayNum, cudaMemcpyHostToDevice);

//    int smemSize = sizeof(int) + 2048 * sizeof(int);
    int smemSize = 2 * sizeof(int);
#ifdef PRINT_DEBUG
    cout << "start:\n";
    setTime(&startTime);
#endif
    insertKernel<<<blockNum, blockSize, smemSize>>>(d_buffer, bufferItems, arrayNum, blockSize);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
#ifdef PRINT_DEBUG
    setTime(&endTime);
    double insertTime = getTime(&startTime, &endTime);
    cout << "buffer insert time: " << insertTime << "ms" << endl;

//    cudaMemcpy(&h_buffer, d_buffer, sizeof(Buffer<int>), cudaMemcpyDeviceToHost);
    h_buffer.printBufferPtr();
//    h_buffer.printBuffer();
    printf("-----------\n");
    setTime(&startTime);
#endif
    concurrentKernel<<<blockNum, blockSize, smemSize>>>(d_buffer, bufferItems, arrayNum, blockNum, blockSize);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
#ifdef PRINT_DEBUG
    setTime(&endTime);
    double concurrentTime = getTime(&startTime, &endTime);
    cout << "buffer concurrent ins/del time: " << concurrentTime << "ms" << endl;
//    cudaMemcpy(&h_buffer, d_buffer, sizeof(Buffer<int>), cudaMemcpyDeviceToHost);
    h_buffer.printBufferPtr();
//    h_buffer.printBuffer();
    printf("-----------\n");
#endif
    cudaMemcpy(resItems, bufferItems, sizeof(int) * arrayNum, cudaMemcpyDeviceToHost);
    sort(resItems, resItems + arrayNum);
    for (int i = 0; i < arrayNum; ++i) {
        if (resItems[i] != oriItems[i]) {
            printf("Wrong Answer! %d buffer: %d ori: %d\n", i, resItems[i], oriItems[i]);
            return -1;
        }
    }
#ifdef PRINT_DEBUG
    setTime(&startTime);
#endif
    deleteKernel<<<blockNum, blockSize, smemSize>>>(d_buffer, bufferItems, arrayNum, blockSize);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
#ifdef PRINT_DEBUG
    setTime(&endTime);
    double deleteTime = getTime(&startTime, &endTime);
    cout << "buffer delete time: " << deleteTime << "ms" << endl;
//    cudaMemcpy(&h_buffer, d_buffer, sizeof(Buffer<int>), cudaMemcpyDeviceToHost);
    h_buffer.printBufferPtr();
//    h_buffer.printBuffer();
    printf("-----------\n");
#endif
    // sort resItems
    cudaMemcpy(resItems, bufferItems, sizeof(int) * 2 * arrayNum, cudaMemcpyDeviceToHost);
    sort(resItems, resItems + arrayNum * 2);


    /*for (int i = 0; i < 2 * arrayNum; i++) {*/
        /*printf("%d ", resItems[i]);*/
    /*}*/
    /*printf("\n");*/

    for (int i = 0; i < 2 * arrayNum; ++i) {
        if (resItems[i] != oriItems[i]) {
            printf("Wrong Answer! %d buffer: %d ori: %d\n", i, resItems[i], oriItems[i]);
            return -1;
        }
    }
#ifdef PRINT_DEBUG
    printf("Correct!\n");
#endif
    return 0;

}
