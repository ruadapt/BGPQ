#ifndef SORT_CUH
#define SORT_CUH

#include <cstdio>
#include <cstdint>

namespace astar {
namespace sort {
template <class T>
__device__ __forceinline__ void _swap(T &a, T &b) {
    T tmp = a;
    a = b;
    b = tmp;
}

/// Sort the inserted items before activate heap insert update.
template <class T>
__device__ __forceinline__ void ibitonicSort(T *items, uint32_t size) {

    for (uint32_t k = 2; k <= size; k *= 2) {
        for (uint32_t j = k / 2; j > 0; j /= 2) {
            for (uint32_t i =  threadIdx.x; i < size; i += blockDim.x) {
                uint32_t ixj = i ^ j;
                if (ixj > i) {
                    if ((i & k) == 0) {
                        if (items[i] > items[ixj]) {
                            _swap<T>(items[i], items[ixj]);
                        }
                    }
                    else {
                        if (items[i] < items[ixj]) {
                            _swap<T>(items[i], items[ixj]);
                        }
                    }
                }
                __syncthreads();
            }
        }
    }
}

template <class T>
__device__ __forceinline__ void BinarySearch(T *input1, T *input2,
                                             uint32_t size, uint32_t smemOffset,
                                             uint32_t &ai, uint32_t &bi,
                                             uint32_t &ai_next, uint32_t &bi_next) {
    extern __shared__ int s[];
    uint32_t *aI = (uint32_t *)&s[smemOffset];
    uint32_t *bI = (uint32_t *)&aI[blockDim.x + 1];

    int lengthPerThread = size * 2 / blockDim.x;

    int index= threadIdx.x * lengthPerThread;
    int aTop = (index > size) ? size : index;
    int bTop = (index > size) ? index - size : 0;
    int aBottom = bTop;
    
    int offset;
    
    // binary search for diagonal intersections
    if (threadIdx.x == 0) {
        aI[blockDim.x] = size;
        bI[blockDim.x] = size;
    }
    __syncthreads();

    while (1) {
        offset = (aTop - aBottom) / 2;
        ai = aTop - offset;
        bi = bTop + offset;
        
        if (aTop == aBottom || (bi < size && (ai == size || input1[ai] > input2[bi]))) {
            if (aTop == aBottom || input1[ai - 1] <= input2[bi]) {
                aI[threadIdx.x] = ai;
                bI[threadIdx.x] = bi;
                break;
            } else {
                aTop = ai - 1;
                bTop = bi + 1;
            }
        } else {
            aBottom = ai;
        }
     }
    __syncthreads();

    ai_next = aI[threadIdx.x + 1];
    bi_next = bI[threadIdx.x + 1];
    __syncthreads();
}

template <class T>
__device__ __forceinline__ void imergePath(T *input1, T *input2, T *output1, T *output2,
                                           uint32_t size, uint32_t smemOffset) {

    extern __shared__ int s[];
    T *s_items = (T *)&s[smemOffset];

    uint32_t ai, bi, ai_next, bi_next;
    BinarySearch(input1, input2, size, smemOffset, 
        ai, bi, ai_next, bi_next);
    int lengthPerThread = size * 2 / blockDim.x;

    // start from [ai, bi], found a path with lengthPerThread
    for (int i = lengthPerThread * threadIdx.x; i < lengthPerThread * threadIdx.x + lengthPerThread; ++i) {
        if (bi == bi_next || (ai < ai_next && input1[ai] <= input2[bi])) {
            s_items[i] = input1[ai];
            ai++;
        }
        else if (ai == ai_next || (bi < bi_next && input1[ai] > input2[bi])) {
            s_items[i] = input2[bi];
            bi++;
        }
    }
    __syncthreads();

    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        output1[i] = s_items[i];
    }
    __syncthreads();

    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        output2[i] = s_items[i + size];
    }
    __syncthreads();


}

} // namespace sort
} // namespace astar

#endif
