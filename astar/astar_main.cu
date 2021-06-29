#include "pq_model.cuh"
#include "gc.cuh"
#include "astar_map.cuh"
#include "heap.cuh"
#include "util.hpp"
#include "seq_astar.hpp"

#include <cstdio>
#include <cstdint>
#include <assert.h>

int main(int argc, char *argv[]) {
/*
	if (argc != 8) {
		fprintf(stderr, "Usage: ./astar [mapH] [mapW] [block_rate] [batchNum] [batchSize] [blockNum] [blockSize]\n");
		return 1;
	}

	uint32_t map_h = atoi(argv[1]);
	uint32_t map_w = atoi(argv[2]);
	uint32_t block_rate = atoi(argv[3]);
	assert(block_rate < 100);
	uint32_t batch_num = atoi(argv[4]);
	uint32_t batch_size = atoi(argv[5]);
	uint32_t block_num = atoi(argv[6]);
	uint32_t block_size = atoi(argv[7]);
	astar::AstarMap map(map_h, map_w, block_rate);
*/
	if (argc != 2) {
        fprintf(stderr, "Usage: ./astar [map_file]\n");
        /*fprintf(stderr, "Usage: ./astar [map_file] [batchNum] [batchSize] [blockNum] [blockSize] [seq_flag] [mode: 0/pq] [gc_threshold]\n");*/
        return 1;
    }

    /*uint32_t batch_num = atoi(argv[2]);*/
    /*uint32_t batch_size = atoi(argv[3]);*/
    /*uint32_t block_num = atoi(argv[4]);*/
    /*uint32_t block_size = atoi(argv[5]);*/
    /*bool seq_flag = atoi(argv[6]) == 1 ? true : false;*/
    /*astar::AstarMap map(argv[1]);*/
    /*uint32_t mode = atoi(argv[7]);*/
    /*uint32_t gc_threshold = atoi(argv[8]);*/

    uint32_t batch_num = 2560000;
    uint32_t batch_size = 256;
    uint32_t block_num = 22;
    uint32_t block_size = 256;
    bool seq_flag = false;
    astar::AstarMap map(argv[1]);
    uint32_t mode = 0;
    uint32_t gc_threshold = 256000;
    uint32_t map_h = map.h_;
    uint32_t map_w = map.w_;
    
    struct timeval start_time, end_time;
    struct timeval heap_start_time, heap_end_time;
    struct timeval gc_start_time, gc_end_time;
    double heap_time = 0, gc_time = 0;

	Heap<astar::AstarItem> heap(batch_num, batch_size);
#ifdef DEBUG_PRINT
    map.PrintMap();
#else
    if (map_h <= 20) map.PrintMap();
#endif
    astar::AppItem app_item(&map);

	astar::PQModel<astar::AstarItem> h_model(block_num, block_size, 
											batch_num, batch_size,
                                            gc_threshold,
											&heap, &app_item);
	astar::PQModel<astar::AstarItem> *d_model;
	cudaMalloc((void **)&d_model, sizeof(astar::PQModel<astar::AstarItem>));

    vector<uint32_t> g_list(map_h * map_w, UINT_MAX);
    g_list[0] = 0;
    if (seq_flag == true) {
        // Start handling sequential astar.
        uint32_t *seq_map = new uint32_t[map.size_]();
        cudaMemcpy(seq_map, map.d_map, map.size_ * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        /*uint8_t *seq_wmap = new uint8_t[map_h * map_w * 8];*/
        /*cudaMemcpy(seq_wmap, map.d_wmap, 8 * map_h * map_w * sizeof(uint8_t), cudaMemcpyDeviceToHost);*/

        setTime(&start_time);

        seq_astar::SeqAstarSearch1(seq_map, map_h, map_w, g_list);
        uint32_t seq_path_weight = g_list[map_h * map_w - 1];
        if (seq_path_weight == UINT_MAX) {
            h_model.seq_path_weight_ = UINT_MAX;
            printf("No Available Path\n");
        } else {
            h_model.seq_path_weight_ = seq_path_weight;
            printf("Sequential Result: %u\n", seq_path_weight);
        }
        setTime(&end_time);
        printf("Sequential Time: %.4f ms\n", getTime(&start_time, &end_time));
        uint32_t seq_explored_node_count = 0;
        for (int i =0; i < map_h * map_w; i++) {
            if (g_list[i] != UINT_MAX) seq_explored_node_count++;
        }
        printf("Sequential Explored Count: %u\n", seq_explored_node_count);
        delete []seq_map;
        /*delete []seq_wmap;*/
        // End handling sequential astar.
    }

    cudaMemcpy(d_model, &h_model, sizeof(astar::PQModel<astar::AstarItem>), cudaMemcpyHostToDevice);
	size_t smemSize = 2 * batch_size * sizeof(astar::AstarItem) // del_items
                    + 8 * batch_size * sizeof(astar::AstarItem)
                    + batch_size * sizeof(astar::AstarItem) // for reducing partial buffer insertion
					+ 2 * sizeof(uint32_t) // ins_size, del_size
					+ 5 * batch_size * sizeof(astar::AstarItem); // ins/del operations

    setTime(&start_time);

    astar::Init<astar::AstarItem><<<1, 1, smemSize>>>(d_model);
    cudaDeviceSynchronize();

    setTime(&end_time);
#ifdef DEBUG
    printf("Init Time: %.4f ms\n", getTime(&start_time, &end_time));
#endif
    // should be in debug
    uint32_t gc_count = 0;
    setTime(&start_time);

    if (mode == /* pure pq mode */ 0) {
        while (1) {
            setTime(&heap_start_time);
            astar::Run<astar::AstarItem><<<block_num, block_size, smemSize>>>(d_model);
            cudaDeviceSynchronize();
            setTime(&heap_end_time);
            heap_time += getTime(&heap_start_time, &heap_end_time);
#ifdef DEBUG
            printf("heap time: %.4f\n", getTime(&heap_start_time, &heap_end_time));
#endif
            cudaMemcpy(&heap, h_model.d_heap_, sizeof(Heap<astar::AstarItem>), cudaMemcpyDeviceToHost);
            uint32_t h_terminate = 0;
            cudaMemcpy(&h_terminate, heap.terminate, sizeof(uint32_t), cudaMemcpyDeviceToHost);
            if (h_terminate == 1 || heap.itemCount() == 0) break;
            uint32_t old_heap_item_count = heap.itemCount();
#ifdef DEBUG
            printf("heap nodes before gc: %u / %u\nstart gc...", 
            old_heap_item_count, old_heap_item_count / batch_size);
#endif
            /*uint32_t h_gc_flag = 0;*/
            /*cudaMemcpy(&h_gc_flag, h_model.gc_flag_, sizeof(uint32_t), cudaMemcpyDeviceToHost);*/
            /*if (h_gc_flag == 0) break;*/

            setTime(&gc_start_time);
            astar::invalidFilter<astar::AstarItem>(h_model, heap);
            setTime(&gc_end_time);
            gc_time += getTime(&gc_start_time, &gc_end_time);
            // reset gc_flag
            // TODO should be in pqmodel
            cudaMemset(h_model.gc_flag_, 0, sizeof(uint32_t));
            cudaMemcpy(d_model, &h_model, sizeof(astar::PQModel<astar::AstarItem>), cudaMemcpyHostToDevice);
            uint32_t new_heap_item_count = heap.itemCount();
            if (new_heap_item_count > old_heap_item_count * 2 / 3) {
//                gc_threshold *= 2;
                h_model.gc_threshold_ = gc_threshold;
                cudaMemcpy(d_model, &h_model, sizeof(astar::PQModel<astar::AstarItem>), cudaMemcpyHostToDevice);
            }
            // should be in debug
            gc_count++;
#ifdef DEBUG
            printf("gc time: %.4f\n", getTime(&gc_start_time, &gc_end_time));
            {
            printf("finish\nheap nodes after gc: %u / %u\n", 
            new_heap_item_count, new_heap_item_count / batch_size);
            uint32_t gpu_path_weight = 0;
            cudaMemcpy(&h_model, d_model, sizeof(astar::PQModel<astar::AstarItem>), cudaMemcpyDeviceToHost);
            cudaMemcpy(&app_item, h_model.d_app_item_, sizeof(astar::AppItem), cudaMemcpyDeviceToHost);
            cudaMemcpy(&gpu_path_weight, &app_item.d_dist[map_h * map_w - 1], sizeof(uint32_t), cudaMemcpyDeviceToHost);
            printf("current dest dist: %u\n", gpu_path_weight);
            }
#endif
        }
    }
    setTime(&end_time);
    /*printf("heap time: %.4f gc time: %.4f/%u Run Time: %.4f ms\n", heap_time, gc_time, gc_count, getTime(&start_time, &end_time));*/

    cudaMemcpy(&h_model, d_model, sizeof(astar::PQModel<astar::AstarItem>), cudaMemcpyDeviceToHost);
    cudaMemcpy(&app_item, h_model.d_app_item_, sizeof(astar::AppItem), cudaMemcpyDeviceToHost);
    uint32_t *h_dist = new uint32_t[map_h * map_w];
    cudaMemcpy(h_dist, app_item.d_dist, map_h * map_w * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    uint32_t visited_nodes_count = 0;
    for (int i = 0; i < map_h * map_w; ++i) {
        if (h_dist[i] != UINT_MAX) visited_nodes_count++;
    }
    /*printf("visited_nodes_count: %u\n", visited_nodes_count);*/
    printf("%s,astar,%d*%d,%d,%d,%.f\n",
            argv[0] == std::string("./astarT") ? "BGPQ_T" : "BGPQ_B",
            map_h,map_w,h_dist[map_h*map_w - 1],visited_nodes_count,heap_time+gc_time);
#ifdef DEBUG
    uint32_t tmp_seq_counter = 0;
    uint32_t tmp_gpu_counter = 0;
    for (int i = 0; i < map_h * map_w; ++i) {
        if (h_dist[i] == UINT_MAX && g_list[i] != UINT_MAX) {
            tmp_seq_counter++;
            /*if (g_list[i] + (map_h + map_w - 2 - i % map_h - i / map_w) >= h_model.seq_path_weight_) {*/
                /*tmp_seq_counter++;*/
            /*}*/
        }
        if (h_dist[i] != UINT_MAX && g_list[i] == UINT_MAX) {
            tmp_gpu_counter++;
        }
    }
    printf("only seq visited %u only gpu visited %u\n", tmp_seq_counter, tmp_gpu_counter);

    uint32_t gpu_path_weight = 0;
    cudaMemcpy(&gpu_path_weight, &app_item.d_dist[map_h * map_w - 1], sizeof(uint32_t), cudaMemcpyDeviceToHost);
    if (gpu_path_weight == UINT_MAX) {
        printf("No Available Path\n");
    } else {
        printf("Gpu Result: %u\n", gpu_path_weight);
    }
    if (h_model.seq_path_weight_ != gpu_path_weight) {
        printf("Error: Sequential Result (%u) is not equal to GPU Result (%u).\n", 
                h_model.seq_path_weight_, gpu_path_weight);
    }
#endif
    cudaFree(d_model); d_model = NULL;
/*
    astar::RunRemain<astar::AstarItem><<<block_num, block_size, smemSize>>>(d_model);
    cudaDeviceSynchronize();
    cudaMemcpy(&gpu_path_weight, &app_item.d_dist[map_h * map_w - 1], sizeof(uint32_t), cudaMemcpyDeviceToHost);
    if (gpu_path_weight == UINT_MAX) {
        printf("No Available Path\n");
    } else {
        printf("Remain Gpu Result: %u\n", gpu_path_weight);
    }
*/
	return 0;
}
