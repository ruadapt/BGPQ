#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>

#include "models.cuh"
#include "models_pq_fifo.cuh"
#include "models_pq.cuh"
#include "models_fifo.cuh"
#include "util.cuh"
#include "datastructure.hpp"

using namespace std;

bool cmp(KnapsackItem a, KnapsackItem b)
{
    double r1 = (double)a.first / (double)a.second;
    double r2 = (double)b.first / (double)b.second;
    return r1 > r2;
}

int main(int argc, char *argv[])
{
    if (argc != 2) {
        cout << "./knapsack [dataset]\n";
        /*cout << "./knapsack [dataset] [batchnum] [batchsize] [blocknum] [blocksize] [gc_threshold] [model] [has_maxbenefit] [switch_counter]\n";*/
        return 1;
    }
    ifstream inputFile;

    /*int batchNum = atoi(argv[2]);*/
    /*int batchSize = atoi(argv[3]);*/
    /*int blockNum = atoi(argv[4]);*/
    /*int blockSize = atoi(argv[5]);*/
    /*int gc_threshold = atoi(argv[6]);*/
    /*int model = atoi(argv[7]);*/
    /*int has_maxbenefit = atoi(argv[8]);*/
    /*int switch_counter = atoi(argv[9]);*/

    int batchNum = 102400;
    int batchSize = 256;
    int blockNum = 4;
    int blockSize = 256;
    int gc_threshold = 150;
    int model = 0;
    int has_maxbenefit = 0;
    int switch_counter = 20;
    /*int endingBlockNum = atoi(argv[10]);*/

    inputFile.open(argv[1]);

    int capacity, inputSize;
    inputFile >> inputSize >> capacity;

    int *weight = new int[inputSize];
    int *benefit = new int[inputSize];
    float *benefitPerWeight = new float[inputSize];
    int max_benefit = 0;

    for (int i = 0; i < inputSize; i++) {
        inputFile >> benefit[i] >> weight[i];
        benefitPerWeight[i] = (float)benefit[i] / (float)weight[i];
    }

    inputFile.close();

    if (has_maxbenefit == 1) {
        inputFile.open(strcat(argv[1], ".res"));
        inputFile >> max_benefit;
        inputFile.close();
        cout << max_benefit << " ";
    }
	// Sort items by ppw
	KnapsackItem *items = new KnapsackItem[inputSize];
	for (int i = 0; i < inputSize; i++){
		items[i] = KnapsackItem(benefit[i], weight[i], 0, 0);
	}
	sort(items, items + inputSize, cmp);
	for (int i = 0; i < inputSize; i++){
		benefit[i] = items[i].first;
		weight[i] = items[i].second;
		benefitPerWeight[i] = (float)(benefit[i]) / (float)(weight[i]);

	}
	delete[]items;

	int *d_weight, *d_benefit;
    float *d_benefitPerWeight;
    cudaMalloc((void **)&d_weight, sizeof(int) * inputSize);
    cudaMalloc((void **)&d_benefit, sizeof(int) * inputSize); 
    cudaMalloc((void **)&d_benefitPerWeight, sizeof(float) * inputSize);
    cudaMemcpy(d_weight, weight, sizeof(int) * inputSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_benefit, benefit, sizeof(int) * inputSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_benefitPerWeight, benefitPerWeight, sizeof(float) * inputSize, cudaMemcpyHostToDevice);

    int *d_max_benefit;
    cudaMalloc((void **)&d_max_benefit, sizeof(int));
    cudaMemset(d_max_benefit, 0, sizeof(int));

    printf("%s,knapsack,%s,",argv[0] == std::string("./knapsackT") ? "BGPQ_T" : "BGPQ_B",argv[1]);

    if (model == 0) /* heap */ {
        oneheap(d_weight, d_benefit, d_benefitPerWeight,
                d_max_benefit, capacity, inputSize,
                batchNum, batchSize, blockNum, blockSize,
                gc_threshold);
    } else if (model == 1) /* fifo queue */ {
        onebuffer(d_weight, d_benefit, d_benefitPerWeight,
                d_max_benefit, capacity, inputSize,
                batchNum, batchSize, blockNum, blockSize,
                gc_threshold);
    } else if (model == 2) /* heap + fifo queue */ {
        if (has_maxbenefit == 0) {
            oneheap(d_weight, d_benefit, d_benefitPerWeight,
                    d_max_benefit, capacity, inputSize,
                    batchNum, batchSize, blockNum, blockSize,
                    gc_threshold);
            cudaMemcpy(&max_benefit, d_max_benefit, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemset(d_max_benefit, 0, sizeof(int));
        }

        oneheapearlystop(d_weight, d_benefit, d_benefitPerWeight,
                d_max_benefit, capacity, inputSize,
                batchNum, batchSize, blockNum, blockSize,
                gc_threshold, max_benefit);
    } else if (model == 3) /* heap + fifo switch */ {
        oneheapfifoswitch(d_weight, d_benefit, d_benefitPerWeight,
                d_max_benefit, capacity, inputSize,
                batchNum, batchSize, blockNum, blockSize,
                gc_threshold, switch_counter, max_benefit);
    }
    cudaMemcpy(&max_benefit, d_max_benefit, sizeof(int), cudaMemcpyDeviceToHost);
    /*cout << max_benefit << " ";*/
    delete[] weight; weight = NULL;
    delete[] benefit; benefit = NULL;
    delete[] benefitPerWeight; benefitPerWeight = NULL;
    cudaFree(d_weight);
    cudaFree(d_benefit);
    cudaFree(d_benefitPerWeight);
    cudaFree(d_max_benefit);

    return 0;

}
