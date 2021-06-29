#ifndef KNAPSACK_KERNEL_CUH
#define KNAPSACK_KERNEL_CUH

__device__ int ComputeBound(int newBenefit, int newWeight, int index, int inputSize, int capacity, int *weight, int *benefit)
{	
    // if weight overcomes the knapsack capacity, return
    // 0 as expected bound
    if (newWeight >= capacity)
        return 0;

    // initialize bound on profit by current profit
    int profit_bound = 0;

    // start including items from index 1 more to current
    // item index
    int j = index + 1;
    double totweight = newWeight;
	
    // checking index condition and knapsack capacity
    // condition
    while ((j < inputSize) && (totweight + ((double) weight[j]) <= capacity))
    {
        totweight    += (double) weight[j];
        profit_bound += benefit[j];
        j++;
    }
    // If k is not n, include last item partially for
    // upper bound on profit
    if (j < inputSize)
        profit_bound += (capacity - totweight) * benefit[j] / ((double)weight[j]);

    return profit_bound;
}

__device__ void appKernel(int *weight, int *benefit, float *benefitPerWeight,
                          int *max_benefit, int inputSize, int capacity,
                          KnapsackItem *delItem, int *delSize,
                          KnapsackItem *insItem, int *insSize) 
{
    for(int i = threadIdx.x; i < *delSize; i += blockDim.x){
        KnapsackItem item = delItem[i];
        int oldBenefit = -item.first;
        int oldWeight = item.second;
        int oldIndex = item.third;
        int oldBound = -item.fourth;

//        int _bound = ComputeBound(oldBenefit, oldWeight, oldIndex, inputSize, capacity, weight, benefit);
//        if (oldBenefit + _bound < *max_benefit) continue;
        if (oldBound < *max_benefit) continue;

        int index = oldIndex + 1;

        if (index == inputSize) continue;

        // check for 1: accept item at current level
        int newBenefit = oldBenefit + benefit[index];
        int newWeight = oldWeight + weight[index];
        int newBound = ComputeBound(newBenefit, newWeight, index, inputSize, capacity, weight, benefit);
        // int newBound = bound[index + 1];

        if(newWeight <= capacity){
            int oldMax = atomicMax(max_benefit, newBenefit);
        }
        
        // printf("bid: %d, processing: %d %u %d, %llu\n", blockIdx.x, -oldBenefit, oldWeight, index, oldSeq);
        if(newWeight <= capacity && newBenefit + newBound > *max_benefit){
            int insIndex = atomicAdd(insSize, 1);
            // printf("choose 1: %d %u %llu\n", -oldBenefit, oldWeight, oldSeq, -newBenefit, newWeight, ((oldSeq << 1) + 1));
            insItem[insIndex].first = -newBenefit;
            insItem[insIndex].second = newWeight;
            insItem[insIndex].third = index;
//            insItem[insIndex].fourth = ((oldSeq << 1) + 1);
            insItem[insIndex].fourth = -(newBenefit + newBound);
        }
        int newBound1 = ComputeBound(oldBenefit, oldWeight, index, inputSize, capacity, weight, benefit);
        // newBound = bound[index + 1];
//        printf("%d-%d i: %d 0: %d %d 1: %d %d\n", blockIdx.x, threadIdx.x, index,
//				oldWeight <= capacity, oldBenefit + newBound1 > *max_benefit, 
//				newWeight <= capacity, newBenefit + newBound > *max_benefit);
        // check for 0: reject current item
		if(oldWeight <= capacity && oldBenefit + newBound1 > *max_benefit){
            int insIndex = atomicAdd(insSize, 1);
            // printf("old: %d %u %llu, choose 0: %d %u %llu\n", -oldBenefit, oldWeight, oldSeq, -oldBenefit, oldWeight, oldSeq << 1);
            insItem[insIndex].first = -oldBenefit;
            insItem[insIndex].second = oldWeight;
            insItem[insIndex].third = index;
//            insItem[insIndex].fourth = oldSeq << 1;
            insItem[insIndex].fourth = -(oldBenefit + newBound1);
        }
    }
}

// addKernel wrapper which allows items being expanded more than once in smem
__device__ void appKernelWrapper(int *weight, int *benefit, float *benefitPerWeight,
                              int *max_benefit, int inputSize, int capacity,
                              int *explored_nodes,
                              KnapsackItem *delItems, int *delSize,
                              KnapsackItem *insItems, int *insSize)
{
    // store original *delSize for termination check
    int tmpDelSize = 0;
    if (threadIdx.x == 0) {
        tmpDelSize = *delSize;
    }
    __syncthreads();
    while (1) {
        appKernel(weight, benefit, benefitPerWeight,
                  max_benefit, inputSize, capacity,
                  delItems, delSize,
                  insItems, insSize);
        __syncthreads();
        if (*insSize >= blockDim.x || *insSize == 0) {
            if (threadIdx.x == 0) {
                *delSize = tmpDelSize;
            }
            __syncthreads();
            break;
        }
        __syncthreads();
        // copy items in insItems to delItems
        for (int i = threadIdx.x; i < *insSize; i += blockDim.x) {
            delItems[i] = insItems[i];
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            atomicAdd(explored_nodes, *insSize);
            *delSize = *insSize;
            *insSize = 0;
        }
        __syncthreads();
    }

}

#endif
