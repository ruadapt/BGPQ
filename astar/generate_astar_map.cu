#include "astar_map.cuh"
#include <iostream>

int main(int argc, char *argv[]) {
    if (argc != 5) {
        std::cout << "Usage: ./generate_astar_map [h] [w] [obstacle_rate] [outfilename]\n";
        return -1;
    }
    uint32_t h = atoi(argv[1]);
    uint32_t w = atoi(argv[2]);
    uint32_t rate = atoi(argv[3]);
	astar::AstarMap map(h, w, rate);
    map.WriteMap(argv[4]);

    return 0;
}
