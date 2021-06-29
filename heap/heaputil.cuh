#ifndef HEAPUTIL_CUH
#define HEAPUTIL_CUH

template<typename K>
__inline__ __device__ void batchCopy(K *dest, K *source, int size, bool reset = false, K init_limits = 0)
{
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        dest[i] = source[i];
        if (reset) source[i] = init_limits;
    }
    __syncthreads();
}


template<typename T>
__inline__ __device__ void _swap(T &a, T &b) {
    T tmp = a;
    a = b;
    b = tmp;
}


template<typename K>
__inline__ __device__ void ibitonicSort(K *items, int size) {

    for (int k = 2; k <= size; k <<= 1) {
        for (int j = k / 2; j > 0; j >>= 1) {
            for (int i =  threadIdx.x; i < size; i += blockDim.x) {
                int ixj = i ^ j;
                if (ixj > i) {
                    if ((i & k) == 0) {
                        if (items[i] > items[ixj]) {
                            _swap<K>(items[i], items[ixj]);
                        }
                    }
                    else {
                        if (items[i] < items[ixj]) {
                            _swap<K>(items[i], items[ixj]);
                        }
                    }
                }
                __syncthreads();
            }
        }
    }

}

template<typename K>
__inline__ __device__ void dbitonicSort(K *items, int size) {

    for (int k = 2; k <= size; k <<= 1) {
        for (int j = k / 2; j > 0; j >>= 1) {
            for (int i =  threadIdx.x; i < size; i += blockDim.x) {
                int ixj = i ^ j;
                if (ixj > i) {
                    if ((i & k) == 0) {
                        if (items[i] < items[ixj]) {
                            _swap<K>(items[i], items[ixj]);
                        }
                    }
                    else {
                        if (items[i] > items[ixj]) {
                            _swap<K>(items[i], items[ixj]);
                        }
                    }
                }
                __syncthreads();
            }
        }
    }

}

template<typename K>
__inline__ __device__ void dbitonicMerge(K *items, int size) {
    for (int j = size / 2; j > 0; j /= 2) {
        for (int i =  threadIdx.x; i < size; i += blockDim.x) {
            int ixj = i ^ j;
            if ((ixj > i) && (items[i] < items[ixj]))
                _swap<K>(items[i], items[ixj]);
            __syncthreads();
        }
    }
}

template<typename K>
__inline__ __device__ void ibitonicMerge(K *items, int size) {
    for (int j = size / 2; j > 0; j /= 2) {
        for (int i =  threadIdx.x; i < size; i += blockDim.x) {
            int ixj = i ^ j;
            if ((ixj > i) && (items[i] > items[ixj]))
                _swap<K>(items[i], items[ixj]);
            __syncthreads();
        }
    }
}

template<typename K>
__inline__ __device__ void imergePath(K *aItems, K *bItems,
                                      K *smallItems, K *largeItems,
                                      int size, int smemOffset) {

    extern __shared__ int s[];
    K *tmpItems = (K *)&s[smemOffset];

    int lengthPerThread = size * 2 / blockDim.x;

    int index= threadIdx.x * lengthPerThread;
    int aTop = (index > size) ? size : index;
    int bTop = (index > size) ? index - size : 0;
    int aBottom = bTop;
    
    int offset, aI, bI;
    
    // binary search for diagonal intersections
    while (1) {
        offset = (aTop - aBottom) / 2;
        aI = aTop - offset;
        bI = bTop + offset;

        if (aTop == aBottom || (bI < size && (aI == size || aItems[aI] > bItems[bI]))) {
            if (aTop == aBottom || aItems[aI - 1] <= bItems[bI]) {
                break;
            }
            else {
                aTop = aI - 1;
                bTop = bI + 1;
            }
        }
        else {
            aBottom = aI;
        }
     }

     // start from [aI, bI], found a path with lengthPerThread
    for (int i = lengthPerThread * threadIdx.x; i < lengthPerThread * threadIdx.x + lengthPerThread; ++i) {
        if (bI == size || (aI < size && aItems[aI] <= bItems[bI])) {
            tmpItems[i] = aItems[aI];
            aI++;
        }
        else if (aI == size || (bI < size && aItems[aI] > bItems[bI])) {
            tmpItems[i] = bItems[bI];
            bI++;
        }
    }
    __syncthreads();

    batchCopy(smallItems, tmpItems, size);
    batchCopy(largeItems, tmpItems + size, size);
}

#endif
