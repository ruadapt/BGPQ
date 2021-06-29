#ifndef PQ_MODEL_CUH
#define PQ_MODEL_CUH

#include "heap.cuh"
#include "astar_map.cuh"

#define EXPAND_NUMBER 8

#ifdef DEBUG
__device__ uint32_t del_explored_node_number;
__device__ uint32_t explored_node_number;
__device__ uint32_t opt_flag;
#endif

namespace astar {

__device__ __forceinline__ uint32_t ManhattanDistance(uint32_t x1, uint32_t y1, uint32_t x2, uint32_t y2) {
	return x2 - x1 + y2 - y1;
}


// This class contains all application specific data and functions.
class AppItem {
public:
	AppItem(AstarMap *map) {
	    cudaMalloc((void **)&d_map, sizeof(AstarMap));
	    cudaMemcpy(d_map, map, sizeof(AstarMap), cudaMemcpyHostToDevice);
        uint32_t node_num = map->h_ * map->w_;
	    uint32_t *h_dist = new uint32_t[node_num]();
        for (int i = 1; i < node_num; i++) {
            h_dist[i] = UINT_MAX;
        }
        cudaMalloc((void **)&d_dist, node_num * sizeof(uint32_t));
	    cudaMemcpy(d_dist, h_dist, node_num * sizeof(uint32_t), cudaMemcpyHostToDevice);
        delete []h_dist;
    }
    
    ~AppItem() {
        cudaFree(d_map); d_map = nullptr;
        cudaFree(d_dist); d_dist = nullptr;
    }

    __device__ __forceinline__ uint32_t H(uint32_t i) {
        return ManhattanDistance(d_map->GetX(i), d_map->GetY(i), d_map->w_ - 1, d_map->h_ - 1);
    }

	AstarMap *d_map;
	uint32_t *d_dist;
};	

template <class HeapItem>
__forceinline__ __device__ void AppKernel1(AppItem *app_item, uint32_t w, uint32_t h,
				HeapItem *del_items, uint32_t *del_size, 
				HeapItem *ins_items, uint32_t *ins_size) {
	uint32_t *dist = app_item->d_dist;
    int warpIdx = threadIdx.x / 32;
    int laneIdx = threadIdx.x % 32;
    for (int i = laneIdx; i < *del_size; i += 32 /*warp_size*/) {
		HeapItem item = del_items[i];
		uint32_t item_node = item.node_;
        uint32_t item_f = item.f_;
		uint32_t item_g = item_f - app_item->H(item_node);
		if (item_g > dist[item_node]) continue;

        // 1 2 3
		// 4 * 5
		// 6 7 8

        // This only work for 256 block size
        if (warpIdx == 0) {
            uint32_t node2 = item_node < w ? UINT_MAX : item_node - w;
            if (app_item->d_map->GetMap(node2) && atomicMin(&dist[node2], item_g + 2) > item_g + 2) {
                ins_items[atomicAdd(ins_size, 1)] = HeapItem(item_g + 2 + app_item->H(node2), node2);
            }
        } else if (warpIdx == 1) {
            uint32_t node1 = (item_node < w || item_node % w == 0) ? UINT_MAX : item_node - w - 1;
            if (app_item->d_map->GetMap(node1) && atomicMin(&dist[node1], item_g + 3) > item_g + 3) {
                ins_items[atomicAdd(ins_size, 1)] = HeapItem(item_g + 3 + app_item->H(node1), node1);
            }
        } else if (warpIdx == 2) {
            uint32_t node3 = (item_node < w || item_node % w == (w - 1)) ? UINT_MAX : item_node - w + 1;
            if (app_item->d_map->GetMap(node3) && atomicMin(&dist[node3], item_g + 2) > item_g + 2) {
                ins_items[atomicAdd(ins_size, 1)] = HeapItem(item_g + 2 + app_item->H(node3), node3);
            }
        } else if (warpIdx == 3) {
            uint32_t node4 = item_node % w == 0 ? UINT_MAX : item_node - 1;
            if (app_item->d_map->GetMap(node4) && atomicMin(&dist[node4], item_g + 2) > item_g + 2) {
                ins_items[atomicAdd(ins_size, 1)] = HeapItem(item_g + 2 + app_item->H(node4), node4);
            }
        } else if (warpIdx == 4) {
            uint32_t node5 = item_node % w == (w - 1) ? UINT_MAX : item_node + 1;
            if (app_item->d_map->GetMap(node5) && atomicMin(&dist[node5], item_g + 1) > item_g + 1) {
                ins_items[atomicAdd(ins_size, 1)] = HeapItem(item_g + 1 + app_item->H(node5), node5);
            }
        } else if (warpIdx == 5) {
            uint32_t node7 = item_node / w == (h - 1) ? UINT_MAX : item_node + w;
            if (app_item->d_map->GetMap(node7) && atomicMin(&dist[node7], item_g + 1) > item_g + 1) {
                ins_items[atomicAdd(ins_size, 1)] = HeapItem(item_g + 1 + app_item->H(node7), node7);
            }
        } else if (warpIdx == 6) {
            uint32_t node6 = (item_node / w  == (h - 1) || item_node % w == 0) ? UINT_MAX : item_node +w - 1;
            if (app_item->d_map->GetMap(node6) && atomicMin(&dist[node6], item_g + 3) > item_g + 3) {
                ins_items[atomicAdd(ins_size, 1)] = HeapItem(item_g + 3 + app_item->H(node6), node6);
            }
        } else if (warpIdx == 7) {
            uint32_t node8 = (item_node / w == (h - 1) ||  item_node % w == w - 1) ? UINT_MAX : item_node + w + 1;
            if (app_item->d_map->GetMap(node8) && atomicMin(&dist[node8], item_g + 2) > item_g + 2) {
                ins_items[atomicAdd(ins_size, 1)] = HeapItem(item_g + 2 + app_item->H(node8), node8);
            }
        }
	}
}

template <class HeapItem>
__forceinline__ __device__ void AppKernel(AppItem *app_item, uint32_t w, uint32_t h,
				HeapItem *del_items, uint32_t *del_size, 
				HeapItem *ins_items, uint32_t *ins_size) {
	uint32_t *dist = app_item->d_dist;

	for (int i = threadIdx.x; i < *del_size; i += blockDim.x) {
		HeapItem item = del_items[i];
		uint32_t item_node = item.node_;
        uint32_t item_f = item.f_;
		uint32_t item_g = item_f - app_item->H(item_node);
		if (item_g > dist[item_node]) continue;
        int wmap[8] = {3, 2, 2, 2, 1, 3, 1, 2};
		/*uint8_t *wmap = &app_item->d_map->d_wmap[8 * item_node];*/
        
		// 1 2 3
		// 4 * 5
		// 6 7 8
/*
        uint32_t node2 = item_node < w ? UINT_MAX : item_node - w;
        uint32_t node1 = (node2 == UINT_MAX || node2 % w == 0) ? UINT_MAX : node2 - 1;
        uint32_t node3 = (node2 == UINT_MAX || node2 % w == w - 1) ? UINT_MAX : node2 + 1;

        uint32_t node4 = item_node % w == 0 ? UINT_MAX : item_node - 1;
        uint32_t node5 = item_node % w == w - 1 ? UINT_MAX : item_node + 1;

        uint32_t node7 = item_node / w == h - 1 ? UINT_MAX : item_node + w;
        uint32_t node6 = (node7 == UINT_MAX || node7 % w == 0) ? UINT_MAX : node7 - 1;
        uint32_t node8 = (node7 == UINT_MAX || node7 % w == w - 1) ? UINT_MAX : node7 + 1;
*/
		uint32_t node1 = app_item->d_map->Get1(item_node);
		if (app_item->d_map->GetMap(node1) 
        && atomicMin(&dist[node1], item_g + wmap[0]) > 
        item_g + wmap[0]) {
			ins_items[atomicAdd(ins_size, 1)] = 
            HeapItem(item_g + wmap[0] + 
            app_item->H(node1), node1);
		}
		uint32_t node2 = app_item->d_map->Get2(item_node);
		if (app_item->d_map->GetMap(node2) && atomicMin(&dist[node2], item_g + wmap[1]) > item_g + wmap[1]) {
			ins_items[atomicAdd(ins_size, 1)] = HeapItem(item_g + wmap[1] + app_item->H(node2), node2);
		}
		uint32_t node3 = app_item->d_map->Get3(item_node);
		if (app_item->d_map->GetMap(node3) && atomicMin(&dist[node3], item_g + wmap[2]) > item_g + wmap[2]) {
			ins_items[atomicAdd(ins_size, 1)] = HeapItem(item_g + wmap[2] + app_item->H(node3), node3);
		}
		uint32_t node4 = app_item->d_map->Get4(item_node);
		if (app_item->d_map->GetMap(node4) && atomicMin(&dist[node4], item_g + wmap[3]) > item_g + wmap[3]) {
			ins_items[atomicAdd(ins_size, 1)] = HeapItem(item_g + wmap[3] + app_item->H(node4), node4);
		}
		uint32_t node5 = app_item->d_map->Get5(item_node);
		if (app_item->d_map->GetMap(node5) && atomicMin(&dist[node5], item_g + wmap[4]) > item_g + wmap[4]) {
			ins_items[atomicAdd(ins_size, 1)] = HeapItem(item_g + wmap[4] + app_item->H(node5), node5);
//            if (node5 == 999999) printf("node 5 %u %u %u %u %u\n", item_node, item_g, item_f, item_g + 1 + app_item->H(node5), dist[999999]);
		}
		uint32_t node6 = app_item->d_map->Get6(item_node);
		if (app_item->d_map->GetMap(node6) && atomicMin(&dist[node6], item_g + wmap[5]) > item_g + wmap[5]) {
			ins_items[atomicAdd(ins_size, 1)] = HeapItem(item_g + wmap[5] + app_item->H(node6), node6);
		}
		uint32_t node7 = app_item->d_map->Get7(item_node);
		if (app_item->d_map->GetMap(node7) && atomicMin(&dist[node7], item_g + wmap[6]) > item_g + wmap[6]) {
			ins_items[atomicAdd(ins_size, 1)] = HeapItem(item_g + wmap[6] + app_item->H(node7), node7);
//            if (node7 == 999999) printf("node 7 %u %u %u %u %u\n", item_node, item_g, item_f, item_g + 1 + app_item->H(node7), dist[999999]);
		}
		uint32_t node8 = app_item->d_map->Get8(item_node);
		if (app_item->d_map->GetMap(node8) && atomicMin(&dist[node8], item_g + wmap[7]) > item_g + wmap[7]) {
			ins_items[atomicAdd(ins_size, 1)] = HeapItem(item_g + wmap[7] + app_item->H(node8), node8);
//	        if (node8 == 999999) printf("node 8 %u %u %u %u %u\n", item_node, item_g, item_f, item_g + 2 + app_item->H(node8), dist[999999]);
	    }
	}
}

template <class HeapItem>
class PQModel {
public:
	PQModel(uint32_t block_num, uint32_t block_size, 
		uint32_t batch_num, uint32_t batch_size,
        uint32_t gc_threshold,
		Heap<HeapItem> *heap, AppItem *app_item) :
	block_num_(block_num), block_size_(block_size),
	batch_num_(batch_num), batch_size_(batch_size),
    gc_threshold_(gc_threshold) {
		cudaMalloc((void **)&d_heap_, sizeof(Heap<HeapItem>));
		cudaMemcpy(d_heap_, heap, sizeof(Heap<HeapItem>), cudaMemcpyHostToDevice);

		cudaMalloc((void **)&d_app_item_, sizeof(AppItem));
		cudaMemcpy(d_app_item_, app_item, sizeof(AppItem), cudaMemcpyHostToDevice);

		cudaMalloc((void **)&ins_items, EXPAND_NUMBER * batch_size * block_num * sizeof(HeapItem));
        cudaMalloc((void **)&gc_flag_, sizeof(uint32_t));
        cudaMemset(gc_flag_, 0, sizeof(uint32_t));
	}
	~PQModel() {
		cudaFree(d_heap_); d_heap_ = nullptr;
		cudaFree(d_app_item_); d_app_item_ = nullptr;

		cudaFree(ins_items); ins_items = nullptr;
        cudaFree(gc_flag_); gc_flag_ = nullptr;
	}

    __device__ uint32_t GetGCFlag() {
        return *gc_flag_;
    }

	Heap<HeapItem> *d_heap_;
	AppItem *d_app_item_;

	uint32_t block_num_;
	uint32_t block_size_;
	uint32_t batch_num_;
	uint32_t batch_size_;

    uint32_t seq_path_weight_;

    HeapItem *ins_items;

    // for garbage collection
    uint32_t *gc_flag_;
    uint32_t gc_threshold_;
};

template <class HeapItem>
__global__ void Init(PQModel<HeapItem> *model) {

    if (blockIdx.x == 0) {
		if (threadIdx.x == 0) {
			// This is a hacky, our heap is min heap.
			model->ins_items[0] = HeapItem(model->d_app_item_->H(0), 0);
		}
		__syncthreads();
		model->d_heap_->insertion(model->ins_items, 1, 0);
		__syncthreads();
	}
#ifdef DEBUG
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        del_explored_node_number = 0;
        explored_node_number = 0;
        opt_flag = 0;
    }
#endif
}

template <class HeapItem>
__global__ void Run(PQModel<HeapItem> *model) {

	uint32_t batch_size = model->batch_size_;
	Heap<HeapItem> *d_heap = model->d_heap_;
	// HeapItem *ins_items is not on smem due to undetermined large size
//	HeapItem *ins_items = model->ins_items + blockIdx.x * batch_size * EXPAND_NUMBER;

	extern __shared__ int smem[];
    HeapItem *ins_items = (HeapItem *)&smem[0];
    HeapItem *del_items = (HeapItem *)&ins_items[(EXPAND_NUMBER + 1) * batch_size];
//	HeapItem *del_items = (HeapItem *)&smem[0];
	uint32_t *del_size = (uint32_t *)&del_items[2 * batch_size];
	// HeapItem *ins_items is not on smem due to undetermined large size
	uint32_t *ins_size = (uint32_t *)&del_size[1];
    uint32_t *partial_ins_size = (uint32_t *)&ins_size[1];

    if (threadIdx.x == 0) {
        *partial_ins_size = 0;
    }
    __syncthreads();

	/*uint32_t smemOffset = (sizeof(HeapItem) * batch_size*/
							/*+ sizeof(uint32_t) * 2) / sizeof(int);*/
    uint32_t smemOffset = (sizeof(HeapItem) * batch_size * (EXPAND_NUMBER + 1 + 2)
                            + sizeof(uint32_t)
                            + sizeof(uint32_t) * 2) / sizeof(int);

	while(!d_heap->ifTerminate() && !model->GetGCFlag()) {
        __syncthreads();

        if (threadIdx.x == 0) {
            *ins_size = *partial_ins_size;
            // del_size is temporarily used to store target node distance
    		*del_size = model->d_app_item_->d_dist[model->d_app_item_->d_map->target_];
#ifdef DEBUG
            printf("result: %u\n", *del_size);
#endif
        }
        __syncthreads();
        uint32_t app_t_val = *del_size;
        __syncthreads();

		if(d_heap->deleteRoot(del_items, *del_size, app_t_val) == true){
			__syncthreads();
			d_heap->deleteUpdate(smemOffset);
			__syncthreads();
		}
		__syncthreads();
		if(threadIdx.x == 0){
			*ins_size = *partial_ins_size;
		}
		__syncthreads();

		if(*del_size > 0){
            __syncthreads();
#ifdef DEBUG
            if(threadIdx.x == 0){
                atomicAdd(&del_explored_node_number, *del_size);
                printf("thread %d del %d remain ins %d\n", 
                    blockIdx.x, *del_size, *partial_ins_size);
/*
                printf("del_item\n");
                for (int i = 0; i < *del_size; i++) {
                    printf("%d %d | ", del_items[i].node_, del_items[i].f_);
                }
                printf("\nremain_ins_item\n");
                for (int i = 0; i < *ins_size; i++) {
                    printf("%d %d | ", ins_items[i].node_, ins_items[i].f_);
                }
                printf("\n");
*/
            }
            __syncthreads();
#endif
	         while (1) {
                __syncthreads();
                AppKernel(model->d_app_item_, model->d_app_item_->d_map->w_, model->d_app_item_->d_map->h_,
                        del_items, del_size, ins_items, ins_size);
                __syncthreads();
#ifdef DEBUG
                if (threadIdx.x == 0) {
                    printf("thread %d del %d ins %d p %d\n", 
                    blockIdx.x, *del_size, *ins_size, *partial_ins_size);
                }
                __syncthreads();
#endif
                if (*ins_size >= batch_size || *ins_size == 0) break;
                __syncthreads();
                if (threadIdx.x == 0) {
                    *del_size = *ins_size;
                    *ins_size = 0;
                }
                __syncthreads();
                for (int i = threadIdx.x; i < *del_size; i += blockDim.x) {
                    del_items[i] = ins_items[i];
                }
                __syncthreads();
            }
#ifdef RESERVED
			AppKernel(model->d_app_item_, model->d_app_item_->d_map->w_, model->d_app_item_->d_map->h_,
                    del_items, del_size, ins_items, ins_size);
			__syncthreads();
#endif
		}
        __syncthreads();
		if(*ins_size > 0) {
            __syncthreads();
            uint32_t first_ins_size = min(batch_size, *ins_size);
            __syncthreads();
            d_heap->insertion(ins_items, first_ins_size, smemOffset);
            __syncthreads();
            uint32_t real_ins_size = first_ins_size;
            for (int i = 1; i < *ins_size / batch_size; i++) {
                real_ins_size += batch_size;
                d_heap->insertion(ins_items + i * batch_size, batch_size, smemOffset);
                __syncthreads();
            }
            // update partial_ins_size
            if (threadIdx.x == 0) {
                *partial_ins_size = *ins_size - real_ins_size;
#ifdef DEBUG
                printf("thread %d ins size %d / %d\n", blockIdx.x, real_ins_size, *ins_size);
//                atomicAdd(&explored_node_number, *ins_size);
                atomicAdd(&explored_node_number, real_ins_size);
#endif
            }
            __syncthreads();
            // move partial items to its place
            for (int i = threadIdx.x; i < *partial_ins_size; i += blockDim.x) {
                ins_items[i] = ins_items[real_ins_size + i];
            }
            __syncthreads();
/*
			for(uint32_t offset = 0; offset < *ins_size; offset += batch_size){
				uint32_t size = min(batch_size, *ins_size - offset);
				__syncthreads();
				d_heap->insertion(ins_items + offset, size, smemOffset);
				__syncthreads();
			}
            __syncthreads();
*/
		}
        // check gc
        if (threadIdx.x == 0) {
//                printf("bid %d ins size %u\n", blockIdx.x, *ins_size);
//                if (blockIdx.x == 0) printf("%u\n", *d_heap->batchCount);
            if (*d_heap->batchCount > model->gc_threshold_) {
//                    atomicCAS(d_heap->terminate, 0, 1);
                atomicCAS(model->gc_flag_, 0, 1);
            }
        }
        __syncthreads();
#ifdef DEBUG
        if (threadIdx.x == 0) {
            printf("thread %d gc_flag %d\n", blockIdx.x, *model->gc_flag_);
        }
        __syncthreads();
#endif
	}
    if (*partial_ins_size != 0) {
        __syncthreads();
        d_heap->insertion(ins_items, *partial_ins_size, smemOffset);
        __syncthreads();
    }
#ifdef DEBUG
    __syncthreads();
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("explored_nodes_number: %u / %u\n", explored_node_number, del_explored_node_number);
    }
#endif
}

template <class HeapItem>
__global__ void RunRemain(PQModel<HeapItem> *model) {

	uint32_t batch_size = model->batch_size_;
	Heap<HeapItem> *d_heap = model->d_heap_;
	// HeapItem *ins_items is not on smem due to undetermined large size
	HeapItem *ins_items = model->ins_items + blockIdx.x * batch_size * EXPAND_NUMBER;

	extern __shared__ int smem[];
	HeapItem *del_items = (HeapItem *)&smem[0];
	uint32_t *del_size = (uint32_t *)&del_items[batch_size];
	// HeapItem *ins_items is not on smem due to undetermined large size
	uint32_t *ins_size = (uint32_t *)&del_size[1];

	uint32_t smemOffset = (sizeof(HeapItem) * batch_size
							+ sizeof(uint32_t) * 2) / sizeof(int);
	
	while(1) {
		if(d_heap->deleteRoot(del_items, *del_size) == true){
			__syncthreads();
			d_heap->deleteUpdate(smemOffset);
			__syncthreads();
		}
		__syncthreads();
		if(threadIdx.x == 0){
			*ins_size = 0;
		}
		__syncthreads();
		
		if(*del_size > 0){
			AppKernel(model->d_app_item_, del_items, del_size, ins_items, ins_size);
			__syncthreads();
        } else {
            break;
        }
#ifdef DEBUG
        if (threadIdx.x == 0) {
            atomicAdd(&explored_node_number, *ins_size);
        }
        __syncthreads();
#endif
		if(*ins_size > 0){
			for(uint32_t offset = 0; offset < *ins_size; offset += batch_size){
				uint32_t size = min(batch_size, *ins_size - offset);
				__syncthreads();
				d_heap->insertion(ins_items + offset, size, smemOffset);
				__syncthreads();
			}
		}
	}	
#ifdef DEBUG
    __syncthreads();
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("explored_nodes_number: %u\n", explored_node_number);
    }
#endif
}

} // astar

#endif
