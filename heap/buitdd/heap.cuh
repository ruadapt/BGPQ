#ifndef HEAP_CUH
#define HEAP_CUH

#include "../heaputil.cuh"

#define AVAIL 0
#define INSHOLD 1
#define DELMOD 2
#define INUSE 3

using namespace std;

template <typename K>
class Heap {
public:
        K init_limits;

        int batchNum;
        int batchSize;

        int *batchCount;
        int *partialBufferSize;
#ifdef HEAP_SORT
        int *deleteCount;
#endif
#ifdef PBS_MODEL
        int *globalBenefit;
        uint32_t *tbstate;
        uint32_t *terminate;
#endif
        K *heapItems;
        int *status;

        Heap(int _batchNum,
            int _batchSize,
            K _init_limits = 0) : init_limits(_init_limits), batchNum(_batchNum), batchSize(_batchSize) {
            // prepare device heap
            cudaMalloc((void **)&heapItems, sizeof(K) * batchSize * (batchNum + 1));
            // initialize heap items with max value
            K *tmp = new K[batchSize * (batchNum + 1)];
            for (int i = 0; i < (batchNum + 1) * batchSize; i++) {
                tmp[i] = init_limits;
            }
            cudaMemcpy(heapItems, tmp, sizeof(K) * batchSize * (batchNum + 1), cudaMemcpyHostToDevice);
            delete []tmp; tmp = NULL;

            cudaMalloc((void **)&status, sizeof(int) * (batchNum + 1));
            cudaMemset(status, AVAIL, sizeof(int) * (batchNum + 1));

            cudaMalloc((void **)&batchCount, sizeof(int));
            cudaMemset(batchCount, 0, sizeof(int));
            cudaMalloc((void **)&partialBufferSize, sizeof(int));
            cudaMemset(partialBufferSize, 0, sizeof(int));
#ifdef HEAP_SORT
            cudaMalloc((void **)&deleteCount, sizeof(int));
            cudaMemset(deleteCount, 0, sizeof(int));
#endif
#ifdef PBS_MODEL
            cudaMalloc((void **)&globalBenefit, sizeof(int));
            cudaMemset(globalBenefit, 0, sizeof(int));
            cudaMalloc((void **)&tbstate, 1024 * sizeof(uint32_t));
            cudaMemset(tbstate, 0, 1024 * sizeof(uint32_t));
            uint32_t tmp1 = 1;
            cudaMemcpy(tbstate, &tmp1, sizeof(uint32_t), cudaMemcpyHostToDevice);
            cudaMalloc((void **)&terminate, sizeof(uint32_t));
            cudaMemset(terminate, 0, sizeof(uint32_t));
#endif
        }

        void reset() {
			K *tmp = new K[batchSize * (batchNum + 1)];
            for (int i = 0; i < (batchNum + 1) * batchSize; i++) {
                tmp[i] = init_limits;
            }
            cudaMemcpy(heapItems, tmp, sizeof(K) * batchSize * (batchNum + 1), cudaMemcpyHostToDevice);
            delete []tmp; tmp = NULL;

            cudaMemset(status, AVAIL, sizeof(int) * (batchNum + 1));

            cudaMemset(batchCount, 0, sizeof(int));
            cudaMemset(partialBufferSize, 0, sizeof(int));
#ifdef HEAP_SORT
            cudaMemset(deleteCount, 0, sizeof(int));
#endif
#ifdef PBS_MODEL
            cudaMemset(globalBenefit, 0, sizeof(int));
            cudaMemset(tbstate, 0, 1024 * sizeof(uint32_t));
            uint32_t tmp1 = 1;
            cudaMemcpy(tbstate, &tmp1, sizeof(uint32_t), cudaMemcpyHostToDevice);
            cudaMemset(terminate, 0, sizeof(uint32_t));
#endif
		}

        ~Heap() {
            cudaFree(heapItems);
            heapItems = NULL;
            cudaFree(status);
            status = NULL;
            cudaFree(batchCount);
            batchCount = NULL;
            cudaFree(partialBufferSize);
            partialBufferSize = NULL;
#ifdef HEAP_SORT
            cudaFree(deleteCount);
            deleteCount = NULL;
#endif
#ifdef PBS_MODEL
            cudaMemset(globalBenefit, 0, sizeof(int));
            cudaMemset(tbstate, 0, 1024 * sizeof(uint32_t));
            uint32_t tmp1 = 1;
            cudaMemcpy(tbstate, &tmp1, sizeof(uint32_t), cudaMemcpyHostToDevice);
            cudaMemset(terminate, 0, sizeof(uint32_t));
#endif
            batchNum = 0;
            batchSize = 0;
        }

        bool checkInsertHeap() {
            int h_batchCount;
            int h_partialBufferSize;
            cudaMemcpy(&h_batchCount, batchCount, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&h_partialBufferSize, partialBufferSize, sizeof(int), cudaMemcpyDeviceToHost);

            int *h_status = new int[h_batchCount + 1];
            K *h_items = new K[batchSize * (h_batchCount + 1)];
            cudaMemcpy(h_items, heapItems, sizeof(K) * batchSize * (h_batchCount + 1), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_status, status, sizeof(int) * (h_batchCount + 1), cudaMemcpyDeviceToHost);

            // check partial batch
            if (h_status[0] != AVAIL) {
                printf("Partial Batch State Error: state should be AVAIL = 0 while current is %d\n", h_status[0]);
                return false;
            }
            if (h_batchCount != 0 && h_partialBufferSize != 0) {
                if (h_items[batchSize * 2 - 1] > h_items[0]) {
                    printf("Partial Buffer Error: partial batch should be larger than root batch.\n");
                    return false;
                }
                for (int i = 1; i < h_partialBufferSize; i++) {
                    if (h_items[i] < h_items[i - 1]) {
                        printf("Partial Buffer Error: partialBuffer[%d] is smaller than partialBuffer[%d-1]\n", i, i); 
                        return false;
                    }
                }
            }

            for (int i = 1; i <= h_batchCount; ++i) {
                if (h_status[i] != AVAIL) {
                    printf("State Error @ batch %d, state should be AVAIL = 0 while current is %d\n", i, h_status[i]);
                    return false;
                }
                int p = hostGetReversedIdx(hostGetReversedIdx(i) >> 1);
                if (i > 1) {
                    if (h_items[i * batchSize] < h_items[p * batchSize + batchSize - 1]){
                        printf("Batch Keys Error @ batch %d's first item is smaller than batch %d's last item\n", i, p);
                        return false;
                    }
                }
                for (int j = 1; j < batchSize; ++j) {
                    if (h_items[i * batchSize + j] < h_items[i * batchSize + j - 1]) {
                        printf("Batch Keys Error @ batch %d item[%d] is smaller than item[%d]\n", i, j, j - 1);
                        return false;
                    }
                }
            }

            delete []h_items;
            delete []h_status;

            return true;

        }


        void printHeap() {
            
            // TODO if you need this, print each item of the K

            int h_batchCount;
            int h_partialBufferSize;
            cudaMemcpy(&h_batchCount, batchCount, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&h_partialBufferSize, partialBufferSize, sizeof(int), cudaMemcpyDeviceToHost);

            int *h_status = new int[h_batchCount + 1];
            K *h_items = new K[batchSize * (h_batchCount + 1)];
            cudaMemcpy(h_items, heapItems, sizeof(K) * batchSize * (h_batchCount + 1), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_status, status, sizeof(int) * (h_batchCount + 1), cudaMemcpyDeviceToHost);

            printf("batch partial %d_%d:", h_partialBufferSize, h_status[0]);

            for (int i = 0; i < h_partialBufferSize; ++i) {
                printf(" %d", h_items[i]);
            }
            printf("\n");

            for (int i = 1; i <= h_batchCount; ++i) {
                printf("batch %d_%d:", i, h_status[i]);
                for (int j = 0; j < batchSize; ++j) {
                    printf(" %d", h_items[i * batchSize + j]);
                }
                printf("\n");
            }

        }

       __device__ int getItemCount() {
            changeStatus(&status[0], AVAIL, INUSE);
            int itemCount = *partialBufferSize + *batchCount * batchSize;
            changeStatus(&status[0], INUSE, AVAIL);
            return itemCount;
        }

        int nodeCount() {
            int bcount;
            cudaMemcpy(&bcount, batchCount, sizeof(int), cudaMemcpyDeviceToHost);
            return bcount;
        }

        int itemCount() {
            int psize, bcount;
            cudaMemcpy(&bcount, batchCount, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&psize, partialBufferSize, sizeof(int), cudaMemcpyDeviceToHost);
            return psize + bcount * batchSize;
        }

        __host__ bool isEmpty() {
            int psize, bsize;
            cudaMemcpy(&psize, partialBufferSize, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&bsize, batchCount, sizeof(int), cudaMemcpyDeviceToHost);
            return !psize && !bsize;
        }

        __inline__ __device__ uint32_t getReversedIdx(uint32_t oriIdx) {
            int l = __clz(oriIdx) + 1;
            return (__brev(oriIdx) >> l) | (1 << (32-l));
        }

        uint32_t hostGetReversedIdx(uint32_t oriIdx) {
            if (oriIdx == 1) return 1;
            uint32_t i = oriIdx;
            int l = 0;
            while (i > 0) {
                l++;
                i>>= 1;
            }
            l = 32 - (l - 1);
            uint32_t res = 0;
            for (int i = 0; i < 32; i++) {
                int n = oriIdx % 2;
                oriIdx >>= 1;
                res <<= 1;
                res += n;
            }
            return (res >> l) | (1 << (32 - l));
        }

    // changeStatus must make sure that original status = ori and new status = new
    __device__ bool changeStatus(int *status, int oriS, int newS) {
        if ((oriS == AVAIL   && newS == INUSE  ) ||
            (oriS == INUSE   && newS == AVAIL  ) ||
            (oriS == INUSE   && newS == INSHOLD) ||
            (oriS == INSHOLD && newS == INUSE  ) ||
            (oriS == INSHOLD && newS == DELMOD ) ||
            (oriS == DELMOD  && newS == INUSE  ) ||
            (oriS == INUSE   && newS == DELMOD )) {
                while (atomicCAS(status, oriS, newS) != oriS){
            }
            return true;
        }
        else {
            printf("LOCK ERROR ori %d new %d\n", oriS, newS);
            return false;
        }
    }

    // determine the next batch when insert operation updating the heap
    // given the current batch index and the target batch index
    // return the next batch index to the target batch
    __device__ int getNextIdxToTarget(int currentIdx, int targetIdx) {
        return targetIdx >> (__clz(currentIdx) - __clz(targetIdx) - 1);
    }

     __device__ bool deleteRoot(K *items, int &size) {

        if (threadIdx.x == 0) {
            changeStatus(&status[0], AVAIL, INUSE);
        }
        __syncthreads();

#ifdef HEAP_SORT
        int deleteOffset = *deleteCount;
#else
        int deleteOffset = 0;
#endif

        if (*batchCount == 0 && *partialBufferSize == 0) {
            if (threadIdx.x == 0) {
                changeStatus(&status[0], INUSE, AVAIL);
            }
            size = 0;
            __syncthreads();
            return false;
        }

        if (*batchCount == 0 && *partialBufferSize != 0) {
            // only partial batch has items
            // output the partial batch
            size = *partialBufferSize;
            batchCopy(items + deleteOffset, heapItems, size, true, init_limits);

            if (threadIdx.x == 0) {
#ifdef HEAP_SORT
                *deleteCount += *partialBufferSize;
#endif
                *partialBufferSize = 0;
                changeStatus(&status[0], INUSE, AVAIL);
            }
            __syncthreads();
            return false;
        }

        if (threadIdx.x == 0) {
            changeStatus(&status[1], AVAIL, INUSE);
#ifdef HEAP_SORT
            *deleteCount += batchSize;
#endif
        }
        __syncthreads();

        size = batchSize;
        batchCopy(items + deleteOffset, heapItems + batchSize, size);
        /*
           if (threadIdx.x == 0) {
           printf("delete index: %d\n", *deleteIdx);
           for (int i = 0; i < batchSize; ++i) {
           printf("%d ", keys[*deleteIdx * batchSize + i]);
           }
           printf("\n");
           }
           __syncthreads();
         */
        return true;
    }

    
    // deleteUpdate is used to update the heap
    // it will fill the empty root batch(may be full)
    __device__ void deleteUpdate(int smemOffset) {
        
        extern __shared__ int s[];
        K *sMergedItems = (K *)&s[smemOffset];
        int *tmpIdx = (int *)&s[smemOffset];
        smemOffset += sizeof(K) * 3 * batchSize / sizeof(int);
//        int *tmpType = (int *)&s[smemOffset - 1];

        if (threadIdx.x == 0) {
            *tmpIdx = atomicSub(batchCount, 1);
            if (*tmpIdx == 1) {
                changeStatus(&status[1], INUSE, AVAIL);
                changeStatus(&status[0], INUSE, AVAIL);
            }
        }
        __syncthreads();

        // no full batch exist just stop delete worker
        if (*tmpIdx == 1) return;
        __syncthreads();

        int lastIdx = *tmpIdx;
        __syncthreads();

        if (threadIdx.x == 0) {
            int lstatus = INUSE;
            while (lstatus == INUSE) {
                lstatus = atomicMax(&status[lastIdx], INUSE);
            }
        }
        __syncthreads();

        batchCopy(sMergedItems, 
                  heapItems + lastIdx * batchSize, 
                  batchSize, true, init_limits);

        if (threadIdx.x == 0) {
            changeStatus(&status[lastIdx], INUSE, AVAIL);
        }
        __syncthreads();

        /* start handling partial batch */
        batchCopy(sMergedItems + batchSize, heapItems, batchSize);

        imergePath(sMergedItems, sMergedItems + batchSize,
                   sMergedItems, heapItems,
                   batchSize, smemOffset);
        __syncthreads();

        if (threadIdx.x == 0) {
            changeStatus(&status[0], INUSE, AVAIL);
        }
        __syncthreads();
        /* end handling partial batch */

        int currentIdx = 1;
        int curPrevStatus = AVAIL;
        while (1) {
            int leftIdx = getReversedIdx(getReversedIdx(currentIdx) << 1);
            int rightIdx = getReversedIdx(getReversedIdx(leftIdx) + 1);
            int leftPrevStatus = INUSE, rightPrevStatus = INUSE;
            if (threadIdx.x == 0) {
                while (leftPrevStatus == INUSE) {
                    leftPrevStatus = atomicMax(&status[leftIdx], INUSE);
                }
                while (rightPrevStatus == INUSE) {
                    rightPrevStatus = atomicMax(&status[rightIdx], INUSE);
                }
                if (leftPrevStatus == INSHOLD) leftPrevStatus = DELMOD;
                if (rightPrevStatus == INSHOLD) rightPrevStatus = DELMOD;
            }
            __syncthreads();

            // move leftIdx and rightIdx to shared memory
            batchCopy(sMergedItems + batchSize, 
                      heapItems + leftIdx * batchSize,
                      batchSize);
            batchCopy(sMergedItems + 2 * batchSize,
                      heapItems + rightIdx * batchSize,
                      batchSize);

            int targetIdx = sMergedItems[2 * batchSize - 1] < sMergedItems[3 * batchSize - 1] ? rightIdx : leftIdx;
            int targetPrevStatus = targetIdx == rightIdx ? rightPrevStatus : leftPrevStatus;
            int newIdx = targetIdx == rightIdx ? leftIdx : rightIdx;
            int newPrevStatus = targetIdx == rightIdx ? leftPrevStatus : rightPrevStatus;
            __syncthreads();

            imergePath<K>(sMergedItems + batchSize, sMergedItems + 2 * batchSize,
                          sMergedItems + batchSize, heapItems + targetIdx * batchSize,
                          batchSize, smemOffset);
            __syncthreads();
            
            if (threadIdx.x == 0) {
                changeStatus(&status[targetIdx], INUSE, targetPrevStatus);
            }
            __syncthreads();

            if (sMergedItems[0] >= sMergedItems[2 * batchSize - 1]) {
                __syncthreads();
                batchCopy(heapItems + currentIdx * batchSize, 
                          sMergedItems + batchSize, 
                          batchSize);
            }
            else if (sMergedItems[batchSize - 1] < sMergedItems[batchSize]) {
                __syncthreads();
                batchCopy(heapItems + currentIdx * batchSize,
                          sMergedItems,
                          batchSize);
                batchCopy(heapItems + newIdx * batchSize,
                          sMergedItems + batchSize,
                          batchSize);
                if (threadIdx.x == 0) {
                    changeStatus(&status[currentIdx], INUSE, curPrevStatus);
                    changeStatus(&status[newIdx], INUSE, newPrevStatus);
                }
                __syncthreads();
                return;
            }
            else {
                __syncthreads();
                imergePath<K>(sMergedItems, sMergedItems + batchSize,
                              heapItems + currentIdx * batchSize, sMergedItems,
                              batchSize, smemOffset);
            }

            if (threadIdx.x == 0) {
                changeStatus(&status[currentIdx], INUSE, curPrevStatus);
            }
            __syncthreads();

            currentIdx = newIdx;
            curPrevStatus = newPrevStatus;
        }
       
    }

    __device__ void insertion(K *items, int size, int smemOffset) {

#ifdef INSERT_SMEM // insert items is already in smem
            extern __shared__ int s[];
            K *sMergedItems1 = (K *)&items[0];
            K *sMergedItems2 = (K *)&s[smemOffset];
            smemOffset += sizeof(K) * batchSize / sizeof(int);
            int *tmpIdx = (int *)&s[smemOffset - 1];
#else
            // allocate shared memory space
            extern __shared__ int s[];
            K *sMergedItems = (K *)&s[smemOffset];
            smemOffset += sizeof(K) * 2 * batchSize / sizeof(int);
            int *tmpIdx = (int *)&s[smemOffset - 1];


            // move insert batch to shared memory
            // may be a partial batch, fill rest part with INT_MAX
            // TODO in this way, we can use bitonic sorting
            // but the performance may not be good when size is small
            for (int i = threadIdx.x; i < batchSize; i += blockDim.x) {
                if (i < size) {
                    sMergedItems[i] = items[i];
                }
                else {
                    sMergedItems[i] = init_limits;
                }
            }
            __syncthreads();
#endif
            ibitonicSort(sMergedItems, batchSize);
            __syncthreads();

            if (threadIdx.x == 0) {
                changeStatus(&status[0], AVAIL, INUSE);
            }
            __syncthreads();

            /* start handling partial batch */
            // Case 1: the heap has no full batch
            // TODO current not support size > batchSize, app should handle this
            if (*batchCount == 0 && size < batchSize) {
                // Case 1.1: partial batch is empty
                if (*partialBufferSize == 0) {
                    batchCopy(heapItems, sMergedItems, batchSize);
                    if (threadIdx.x == 0) {
                        *partialBufferSize = size;
                        changeStatus(&status[0], INUSE, AVAIL);
                    }
                    __syncthreads();
                    return;
                }
                // Case 1.2: no full batch is generated
                else if (size + *partialBufferSize < batchSize) {
                    batchCopy(sMergedItems + batchSize, heapItems, batchSize);
                    imergePath(sMergedItems, sMergedItems + batchSize,
                               heapItems, sMergedItems,
                               batchSize, smemOffset);
                    __syncthreads();
                    if (threadIdx.x == 0) {
                        *partialBufferSize += size;
                        changeStatus(&status[0], INUSE, AVAIL);
                    }
                    __syncthreads();
                    return;
                }
                // Case 1.3: a full batch is generated
                else if (size + *partialBufferSize >= batchSize) {
                    batchCopy(sMergedItems + batchSize, heapItems, batchSize);
                    if (threadIdx.x == 0) {
                        // increase batchCount and change root batch to INUSE
                        atomicAdd(batchCount, 1);
                        changeStatus(&status[1], AVAIL, INUSE);
                    }
                    __syncthreads();
                    imergePath(sMergedItems, sMergedItems + batchSize,
                               heapItems + batchSize, heapItems,
                               batchSize, smemOffset);
                    __syncthreads();
                    if (threadIdx.x == 0) {
                        *partialBufferSize += (size - batchSize);
                        changeStatus(&status[0], INUSE, AVAIL);
                        changeStatus(&status[1], INUSE, AVAIL);
                    }
                    __syncthreads();
                    return;
                }
            }
            // Case 2: the heap is non empty
            else {
                // Case 2.1: no full batch is generated
                if (size + *partialBufferSize < batchSize) {
                    batchCopy(sMergedItems + batchSize, heapItems, batchSize);
                    // Merge insert batch with partial batch
                    imergePath(sMergedItems, sMergedItems + batchSize,
                               sMergedItems, sMergedItems + batchSize,
                               batchSize, smemOffset);
                    __syncthreads();
                    if (threadIdx.x == 0) {
                        changeStatus(&status[1], AVAIL, INUSE);
                    }
                    __syncthreads();
                    batchCopy(sMergedItems + batchSize, heapItems + batchSize, batchSize);
                    imergePath(sMergedItems, sMergedItems + batchSize,
                               heapItems + batchSize, heapItems,
                               batchSize, smemOffset);
                    __syncthreads();
                    if (threadIdx.x == 0) {
                        *partialBufferSize += size;
                        changeStatus(&status[0], INUSE, AVAIL);
                        changeStatus(&status[1], INUSE, AVAIL);
                    }
                    __syncthreads();
                    return;
                }
                // Case 2.2: a full batch is generated and needed to be propogated
                else if (size + *partialBufferSize >= batchSize) {
                    batchCopy(sMergedItems + batchSize, heapItems, batchSize);
                    // Merge insert batch with partial batch, leave larger half in the partial batch
                    imergePath(sMergedItems, sMergedItems + batchSize,
                               sMergedItems, heapItems,
                               batchSize, smemOffset);
                    __syncthreads();
                    if (threadIdx.x == 0) {
                        // update partial batch size 
                        *partialBufferSize += (size - batchSize);
                    }
                    __syncthreads();
                }
            }
            /* end handling partial batch */

         if (threadIdx.x == 0) {
            *tmpIdx = atomicAdd(batchCount, 1) + 1;
//            printf("block %d insert target %d\n", blockIdx.x, *tmpIdx);
            changeStatus(&status[*tmpIdx], AVAIL, INUSE);
            changeStatus(&status[0], INUSE, AVAIL);
        }
        __syncthreads();

        int currentIdx = *tmpIdx;
        __syncthreads();

        batchCopy(heapItems + currentIdx * batchSize,
                  sMergedItems,
                  batchSize);

        if (threadIdx.x == 0) {
            changeStatus(&status[currentIdx], INUSE, INSHOLD);
        }
        __syncthreads();

        while (currentIdx != 1) {
            int parentIdx = getReversedIdx(getReversedIdx(currentIdx) >> 1);
            int cstatus = INUSE;
            if (threadIdx.x == 0) {
                   changeStatus(&status[parentIdx], AVAIL, INUSE);
                while (cstatus == INUSE) {
                    cstatus = atomicMax(&status[currentIdx], INUSE);
                }
               }
            __syncthreads();
            
            if (heapItems[parentIdx * batchSize] >= heapItems[currentIdx * batchSize + batchSize - 1]) {
                __syncthreads();
                batchCopy(sMergedItems, 
                          heapItems + parentIdx * batchSize, 
                          batchSize);
                batchCopy(sMergedItems + batchSize,
                          heapItems + currentIdx * batchSize,
                          batchSize);
                batchCopy(heapItems + parentIdx * batchSize,
                          sMergedItems + batchSize,
                          batchSize);
                batchCopy(heapItems + currentIdx * batchSize,
                          sMergedItems,
                          batchSize);
            }
            else {
                __syncthreads();
                batchCopy(sMergedItems, 
                          heapItems + currentIdx * batchSize,
                          batchSize);
                batchCopy(sMergedItems + batchSize,
                          heapItems + parentIdx * batchSize,
                          batchSize);

                imergePath<K>(sMergedItems, sMergedItems + batchSize,
                              heapItems + parentIdx * batchSize, heapItems + currentIdx * batchSize,
                              batchSize, smemOffset);
                __syncthreads();
            }

            if (threadIdx.x == 0) {
                changeStatus(&status[parentIdx], INUSE, INSHOLD);
                changeStatus(&status[currentIdx], INUSE, AVAIL);
            }
            currentIdx = parentIdx;
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            changeStatus(&status[currentIdx], INSHOLD, INUSE);
            changeStatus(&status[currentIdx], INUSE, AVAIL);
        }
        __syncthreads();
    }
};

#endif
