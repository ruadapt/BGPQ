#ifndef BUFFER_CUH
#define BUFFER_CUH
// TODO we may need to reuse the space and let it be a buffer
using namespace std;

template <typename K=int>
class Buffer {
public:

    K *bufferItems;

/*
beginPos------------readPos------writePos------------endPos
       reading data                      writing data
*/
    unsigned long long int capacity;
    unsigned long long int *begPos;
    unsigned long long int *endPos;
    unsigned long long int *readPos;
    unsigned long long int *writePos;
    int *bufferLock;

#ifdef PBS_MODEL
    int *globalBenefit;
    /*int *tbstate;*/
    /*int *terminate;*/
#endif

    Buffer(unsigned long long int _capacity) : capacity(_capacity) {
        cudaMalloc((void **)&bufferItems, sizeof(K) * capacity);
        cudaMalloc((void **)&begPos, sizeof(unsigned long long int));
        cudaMemset(begPos, 0, sizeof(unsigned long long int));
        cudaMalloc((void **)&readPos, sizeof(unsigned long long int));
        cudaMemset(readPos, 0, sizeof(unsigned long long int));
        cudaMalloc((void **)&writePos, sizeof(unsigned long long int));
        cudaMemset(writePos, 0, sizeof(unsigned long long int));
        cudaMalloc((void **)&endPos, sizeof(unsigned long long int));
        cudaMemset(endPos, 0, sizeof(unsigned long long int));
        cudaMalloc((void **)&bufferLock, sizeof(int));
        cudaMemset(bufferLock, 0, sizeof(int));
#ifdef PBS_MODEL
        cudaMalloc((void **)&globalBenefit, sizeof(int));
        cudaMemset(globalBenefit, 0, sizeof(int));
        /*cudaMalloc((void **)&tbstate, 1024 * sizeof(int));*/
        /*cudaMemset(tbstate, 0, 1024 * sizeof(int));*/
        /*int tmp1 = 1;*/
        /*cudaMemcpy(tbstate, &tmp1, sizeof(int), cudaMemcpyHostToDevice);*/
        /*cudaMalloc((void **)&terminate, sizeof(int));*/
        /*cudaMemset(terminate, 0, sizeof(int));*/
#endif
    }

    ~Buffer() {
        cudaFree(bufferItems);
        bufferItems = NULL;
        cudaFree(begPos);
        begPos = NULL;
        cudaFree(readPos);
        readPos = NULL;
        cudaFree(writePos);
        writePos = NULL;
        cudaFree(endPos);
        endPos = NULL;
        cudaFree(bufferLock);
        bufferLock = NULL;
#ifdef PBS_MODEL
        cudaFree(globalBenefit);
        globalBenefit = NULL;
        /*cudaFree(tbstate);*/
        /*tbstate = NULL;*/
        /*cudaFree(terminate);*/
        /*terminate = NULL;*/
#endif
    }

#ifdef PBS_MODEL
    /*__device__ int ifTerminate() {*/
        /*return *terminate;*/
    /*}*/
#endif

    int getBufferSize() {
        unsigned long long int h_begin, h_end;
        cudaMemcpy(&h_begin, begPos, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_end, endPos, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
        return (int)(h_end - h_begin);
    }

    void printBufferPtr() {
        unsigned long long int h_read, h_write, h_begin, h_end;
        cudaMemcpy(&h_read, readPos, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_write, writePos, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_begin, begPos, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_end, endPos, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);

        cout << h_begin << " " << h_read << " " << h_write << " " << h_end << endl;
    }
    void printBuffer() {
        K *h_items = new K[capacity];
        cudaMemcpy(h_items, bufferItems, capacity * sizeof(K), cudaMemcpyDeviceToHost);
        unsigned long long int h_read, h_write, h_begin, h_end;
        cudaMemcpy(&h_read, readPos, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_write, writePos, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_begin, begPos, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_end, endPos, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);

        int counter = 0;
        cout << "item: ";
        for (unsigned long long int i = 0; i < capacity; ++i) {
            cout << h_items[i] << " ";
            counter++;
            if (counter == 32) {
            counter = 0;
            cout << endl;
            }
        }
        cout << endl;
        cout << h_begin << " " << h_read << " " << h_write << " " << h_end << endl;
    }

    __device__ bool isEmpty() {
        return *endPos == *readPos;
    }

    __device__ unsigned long long int getSize() {
        return *endPos - *readPos;
    }

    // current insert/delete implementation can provide maximum sizeof(int) items
    __device__ void insertToBuffer(K *items,
                                   int size,
                                   int smemOffset) {
        extern __shared__ int s[];
        unsigned long long int *insertStartPos = (unsigned long long int *)&s[smemOffset];
        int *numItemsForRound = (int *)&insertStartPos[1];

        if (!threadIdx.x) {
            // Get the begin position in the buffer for this insertion
            *insertStartPos = atomicAdd(endPos, size);
//            printf("insert %d starts at %d\n", items[0], *insertStartPos);
        }
        __syncthreads();

        int offset = 0;

        // Loop until all items are added to the buffer
        while (offset < size) {
            if (!threadIdx.x) {
                // Wait until there is some available space in the buffer
                while (*insertStartPos + offset - *begPos >= capacity) {}
                // Determine the number of items for this round
                unsigned long long int remain = capacity - (*insertStartPos + offset - *begPos);
                *numItemsForRound = (size - offset) < remain ? (size - offset) : remain;
                /**numItemsForRound = thrust::min(size, capacity - (*insertStartPos - *begPos));*/
            }
            __syncthreads();

            for (int i = threadIdx.x; i < *numItemsForRound; i += blockDim.x) {
                bufferItems[(*insertStartPos + offset + i) % capacity] = items[offset + i];
            }
            __syncthreads();

            offset += *numItemsForRound;
            __syncthreads();
        }
        if (!threadIdx.x && size) {
            while (atomicCAS(writePos, *insertStartPos, *insertStartPos + size) != *insertStartPos) {}
        }
        __syncthreads();
    }

    __device__ bool deleteFromBuffer(K *items,
                                     int &size,
                                     int smemOffset) {
        extern __shared__ int s[];
        unsigned long long int *deleteStartPos = (unsigned long long int *)&s[smemOffset];
        int *deleteSize = (int *)&deleteStartPos[1];

        if (!threadIdx.x) {
            *deleteSize = 0;
            while (1) {
                unsigned long long int tmpStart = *readPos;
                unsigned long long int tmpEnd = *writePos;
                unsigned long long int tmpSize = tmpEnd - tmpStart;
                *deleteSize = tmpSize < blockDim.x ? tmpSize : blockDim.x;
                if (*deleteSize == 0) {
#ifdef PBS_MODEL
                    break;
#else
                    break;
#endif
                }
                if (tmpStart == atomicCAS(readPos, tmpStart, tmpStart + *deleteSize)) {
                    *deleteStartPos = tmpStart;
                    break;
                }
            }
        }
        __syncthreads();

        size = *deleteSize;
        __syncthreads();

        if (size == 0) return false;

        for (int i = threadIdx.x; i < size; i += blockDim.x) {
            items[i] = bufferItems[(*deleteStartPos + i) % capacity];
        }
        __syncthreads();

        if (!threadIdx.x) {
            while (atomicCAS(begPos, *deleteStartPos, *deleteStartPos + size) != *deleteStartPos) {}
        }
        __syncthreads();

        return true;
    }
};

#endif
