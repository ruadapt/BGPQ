#ifndef ASTAR_ITEM_CUH
#define ASTAR_ITEM_CUH

#include <climits>
#include <iostream>
#include <cstdint>
#include <cstdlib>
#include <cstdio>

namespace astar {

class AstarItem {
public:
	// uint32_t g;
	// uintre_t prev;
    uint32_t f_;
    uint32_t node_;

    __host__ __device__ AstarItem() : f_(UINT_MAX), node_(0) {}
    __host__ __device__ AstarItem(uint32_t f, uint32_t node)
        : f_(f), node_(node) {}

    __host__ __device__ AstarItem& operator=(const AstarItem &rhs) {
        node_ = rhs.node_;
        f_ = rhs.f_;
        return *this;
    }

    // rewrite this if you need more accurate comparison
    __host__ __device__ bool operator<(const AstarItem &rhs) const {
        return f_ < rhs.f_;
    }
    __host__ __device__ bool operator<=(const AstarItem &rhs) const {
        return f_ <= rhs.f_;
    }
    __host__ __device__ bool operator>(const AstarItem &rhs) const {
        return f_ > rhs.f_;
    }
    __host__ __device__ bool operator>=(const AstarItem &rhs) const {
        return f_ >= rhs.f_;
    }
    __host__ __device__ bool operator==(const AstarItem &rhs) const {
        return f_ == rhs.f_;
    }
    __host__ __device__ bool operator!=(const AstarItem &rhs) const {
        return f_ != rhs.f_ || node_ != rhs.node_;
    }

    friend std::ostream& operator<< (std::ostream &os, const AstarItem &rhs) {
        return os << rhs.f_;
    }

};

} // namespace astar

#endif
