#ifndef SEQ_ASTAR_HPP
#define SEQ_ASTAR_HPP

#include <cstdio>
#include <cstdint>
#include <queue>
#include <climits>
#ifdef DEBUG
#include <cstdio>
#endif

namespace seq_astar {

uint32_t ManhattanDistance(uint32_t x1, uint32_t y1, uint32_t x2, uint32_t y2) {
	return x2 - x1 + y2 - y1;
}

uint32_t GetX(uint32_t i, uint32_t w) {
    return i % w;
}

uint32_t GetY(uint32_t i, uint32_t w) {
    return i / w;
}

uint32_t H(uint32_t i, uint32_t h, uint32_t w) {
    return ManhattanDistance(GetX(i, w), GetY(i, w), w - 1, h - 1);
}

uint32_t GetNode(uint64_t item) {
    return (uint32_t)item;
}

uint32_t GetF(uint64_t item) {
    return (uint32_t)(item >> 32);
}

uint64_t GenerateItem(uint32_t f, uint32_t node) {
    return (((uint64_t)f) << 32) + (uint64_t)node;
}

uint32_t GetNodeFromXY(int x, int y, uint32_t h, uint32_t w) {
    if (x < 0 || y < 0 || x >= w || y >= h) return UINT_MAX;
    else return w * y + x;
}
uint32_t Get1(uint32_t i, uint32_t h, uint32_t w) {
    return GetNodeFromXY((int)GetX(i, w) - 1, (int)GetY(i, w) - 1, h, w);
}
uint32_t Get2(uint32_t i, uint32_t h, uint32_t w) {
    return GetNodeFromXY((int)GetX(i, w), (int)GetY(i, w) - 1, h, w);
}
uint32_t Get3(uint32_t i, uint32_t h, uint32_t w) {
    return GetNodeFromXY((int)GetX(i, w) + 1, (int)GetY(i, w) - 1, h, w);
}
uint32_t Get4(uint32_t i, uint32_t h, uint32_t w) {
    return GetNodeFromXY((int)GetX(i, w) - 1, (int)GetY(i, w), h, w);
}
uint32_t Get5(uint32_t i, uint32_t h, uint32_t w) {
    return GetNodeFromXY((int)GetX(i, w) + 1, (int)GetY(i, w), h, w);
}
uint32_t Get6(uint32_t i, uint32_t h, uint32_t w) {
    return GetNodeFromXY((int)GetX(i, w) - 1, (int)GetY(i, w) + 1, h, w);
}
uint32_t Get7(uint32_t i, uint32_t h, uint32_t w) {
    return GetNodeFromXY((int)GetX(i, w), (int)GetY(i, w) + 1, h, w);
}
uint32_t Get8(uint32_t i, uint32_t h, uint32_t w) {
    return GetNodeFromXY((int)GetX(i, w) + 1, (int)GetY(i, w) + 1, h, w);
}

uint32_t GetMap(uint32_t i, uint32_t *map) {
    if (i == UINT_MAX) return 0;
    uint32_t index = i / 32;
    uint32_t offset = i % 32;
    return (map[index] >> offset) & 1U;
}

uint32_t GetWMap(uint32_t i, uint32_t n, uint8_t *wmap) {
    return wmap[8 * i + n];
}

bool IsTarget(uint32_t i, uint32_t h, uint32_t w) {
    return i != UINT_MAX && (GetX(i, w) == w - 1) && (GetY(i, w) == h - 1);
}

uint32_t SeqAstarSearch(uint32_t *map, uint32_t h, uint32_t w) {

    priority_queue<uint64_t, vector<uint64_t>, std::greater<uint64_t>> open_list;
    vector<uint32_t> close_list(h * w, UINT_MAX);
    close_list[0] = 0;
    int wmap[8] = {3,2,2,2,1,3,1,2};
#ifdef DEBUG
    uint64_t explored_nodes_number = 0;
#endif
    open_list.push(GenerateItem(H(0, h, w), 0));

    while (!open_list.empty()) {
        uint64_t item = open_list.top();
        open_list.pop();
#ifdef DEBUG
        explored_nodes_number++;
#endif
        uint32_t item_node = GetNode(item);
        uint32_t item_f = GetF(item);
        uint32_t item_g = item_f - H(item_node, h, w);
        if (item_g > close_list[item_node]) continue;
        uint32_t node1 = Get1(item_node, h, w);
        if (IsTarget(node1, h, w)) {
            close_list[node1] = item_g + wmap[0];
            break;
        }
        if (GetMap(node1, map) && close_list[node1] > item_g + wmap[0]) {
            close_list[node1] = item_g + wmap[0];
            open_list.push(GenerateItem(H(node1, h, w) + item_g + wmap[0], node1));
        }
        uint32_t node2 = Get2(item_node, h, w);
        if (IsTarget(node2, h, w)) {
            close_list[node2] = item_g + wmap[1];
            break;
        }
        if (GetMap(node2, map) && close_list[node2] > item_g + wmap[1]) {
            close_list[node2] = item_g + wmap[1];
            open_list.push(GenerateItem(H(node2, h, w) + item_g + wmap[1], node2));
        }
        uint32_t node3 = Get3(item_node, h, w);
        if (IsTarget(node3, h, w)) {
            close_list[node3] = item_g + wmap[2];
            break;
        }
        if (GetMap(node3, map) && close_list[node3] > item_g + wmap[2]) {
            close_list[node3] = item_g + wmap[2];
            open_list.push(GenerateItem(H(node3, h, w) + item_g + wmap[2], node3));
        }
        uint32_t node4 = Get4(item_node, h, w);
        if (IsTarget(node4, h, w)) {
            close_list[node4] = item_g + wmap[3];
            break;
        }
        if (GetMap(node4, map) && close_list[node4] > item_g + wmap[3]) {
            close_list[node4] = item_g + wmap[3];
            open_list.push(GenerateItem(H(node4, h, w) + item_g + wmap[3], node4));
        }
        uint32_t node5 = Get5(item_node, h, w);
        if (IsTarget(node5, h, w)) {
            close_list[node5] = item_g + wmap[4];
            break;
        }
        if (GetMap(node5, map) && close_list[node5] > item_g + wmap[4]) {
            close_list[node5] = item_g + wmap[4];
            open_list.push(GenerateItem(H(node5, h, w) + item_g + wmap[4], node5));
        }
        uint32_t node6 = Get6(item_node, h, w);
        if (IsTarget(node6, h, w)) {
            close_list[node6] = item_g + wmap[5];
            break;
        }
        if (GetMap(node6, map) && close_list[node6] > item_g + wmap[5]) {
            close_list[node6] = item_g + wmap[5];
            open_list.push(GenerateItem(H(node6, h, w) + item_g + wmap[5], node6));
        }
        uint32_t node7 = Get7(item_node, h, w);
        if (IsTarget(node7, h, w)) {
            close_list[node7] = item_g + wmap[6];
            break;
        }
        if (GetMap(node7, map) && close_list[node7] > item_g + wmap[6]) {
            close_list[node7] = item_g + wmap[6];
            open_list.push(GenerateItem(H(node7, h, w) + item_g + wmap[6], node7));
        }
        uint32_t node8 = Get8(item_node, h, w);
        if (IsTarget(node8, h, w)) {
            close_list[node8] = item_g + wmap[7];
            break;
        }
        if (GetMap(node8, map) && close_list[node8] > item_g + wmap[7]) {
            close_list[node8] = item_g + wmap[7];
            open_list.push(GenerateItem(H(node8, h, w) + item_g + wmap[7], node8));
        }
    }
#ifdef DEBUG
    printf("explored_nodes_number: %u\n", explored_nodes_number);
#endif
    return close_list[h * w - 1];
}

void SeqAstarSearch1(uint32_t *map, uint32_t h, uint32_t w, vector<uint32_t> &g_list) {

    priority_queue<uint64_t, vector<uint64_t>, std::greater<uint64_t>> open_list;
    vector<uint32_t> close_list(h * w, 0);
    open_list.push(GenerateItem(H(0, h, w), 0));
    int wmap[8] = {3,2,2,2,1,3,1,2};
#ifdef DEBUG
    uint32_t explored_nodes_number = 0;
    uint32_t max_open_list_size = 0;
#endif
    while (!open_list.empty()) {
        uint64_t item = open_list.top();
        open_list.pop();
        uint32_t item_node = GetNode(item);
        if (IsTarget(item_node, h, w)) break;
        if (close_list[item_node] == 1) continue;
        uint32_t item_f = GetF(item);
        uint32_t item_g = item_f - H(item_node, h, w);
        close_list[item_node] = 1;
#ifdef DEBUG
        uint32_t origin_size = open_list.size();
#endif
        uint32_t node1 = Get1(item_node, h, w);
        if (GetMap(node1, map) && close_list[node1] == 0) {
            if (item_g + wmap[0] < g_list[node1]) {
                g_list[node1] = item_g + wmap[0];
                open_list.push(GenerateItem(H(node1, h, w) + item_g + wmap[0], node1));
            }
        }
        uint32_t node2 = Get2(item_node, h, w);
        if (GetMap(node2, map) && close_list[node2] == 0) {
            if (item_g + wmap[1] < g_list[node2]) {
                g_list[node2] = item_g + wmap[1];
                open_list.push(GenerateItem(H(node2, h, w) + item_g + wmap[1], node2));
            }
        }
        uint32_t node3 = Get3(item_node, h, w);
        if (GetMap(node3, map) && close_list[node3] == 0) {
            if (item_g + wmap[2] < g_list[node3]) {
                g_list[node3] = item_g + wmap[2];
                open_list.push(GenerateItem(H(node3, h, w) + item_g + wmap[2], node3));
            }
        }
        uint32_t node4 = Get4(item_node, h, w);
        if (GetMap(node4, map) && close_list[node4] == 0) {
            if (item_g + wmap[3] < g_list[node4]) {
                g_list[node4] = item_g + wmap[3];
                open_list.push(GenerateItem(H(node4, h, w) + item_g + wmap[3], node4));
            }
        }
        uint32_t node5 = Get5(item_node, h, w);
        if (GetMap(node5, map) && close_list[node5] == 0) {
            if (item_g + wmap[4] < g_list[node5]) {
                g_list[node5] = item_g + wmap[4];
                open_list.push(GenerateItem(H(node5, h, w) + item_g + wmap[4], node5));
            }
        }
        uint32_t node6 = Get6(item_node, h, w);
        if (GetMap(node6, map) && close_list[node6] == 0) {
            if (item_g + wmap[5] < g_list[node6]) {
                g_list[node6] = item_g + wmap[5];
                open_list.push(GenerateItem(H(node6, h, w) + item_g + wmap[5], node6));
            }
        }
        uint32_t node7 = Get7(item_node, h, w);
        if (GetMap(node7, map) && close_list[node7] == 0) {
            if (item_g + wmap[6] < g_list[node7]) {
                g_list[node7] = item_g + wmap[6];
                open_list.push(GenerateItem(H(node7, h, w) + item_g + wmap[6], node7));
            }
        }
        uint32_t node8 = Get8(item_node, h, w);
        if (GetMap(node8, map) && close_list[node8] == 0) {
            if (item_g + wmap[7] < g_list[node8]) {
                g_list[node8] = item_g + wmap[7];
                open_list.push(GenerateItem(H(node8, h, w) + item_g + wmap[7], node8));
            }
        }
#ifdef DEBUG
        explored_nodes_number += (open_list.size() - origin_size);
        max_open_list_size = max_open_list_size < open_list.size() ? open_list.size() : max_open_list_size;
#endif
    }
#ifdef DEBUG
    printf("explored_nodes_number: %u\n", explored_nodes_number);
    uint32_t visted_nodes_number = 0;
    for (auto n : close_list) {
        if (n == 1) visted_nodes_number++;
    }
    printf("visted_nodes_number: %u\n", visted_nodes_number);
    printf("max open list size: %u\n", max_open_list_size);
#endif
    return;
}

} // namespace seq_astar

#endif
