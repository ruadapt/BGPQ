#ifndef DATA_STRUCTURE_HPP
#define DATA_STRUCTURE_HPP

#include <bitset>

typedef unsigned short int uint16;
typedef unsigned int uint32;
typedef unsigned long long uint64;

#define INIT_LIMITS INT_MAX

struct KnapsackItem {
    int first; // benefit
    int second; // weight
    int third;
    int fourth;

    __host__ __device__ KnapsackItem(int a = 0, int b = 0, short c = 0, short d = 0)
        : first(a), second(b), third(c), fourth(d) {
    }

    __host__ __device__ KnapsackItem& operator=(const KnapsackItem &rhs) {
        first = rhs.first;
        second = rhs.second;
        third = rhs.third;
        fourth = rhs.fourth;
        return *this;
    }

    // rewrite this if you need more accurate comparison
    __host__ __device__ bool operator<(const KnapsackItem &rhs) const {
        return (first < rhs.first);
    }
    __host__ __device__ bool operator<=(const KnapsackItem &rhs) const {
        return (first <= rhs.first);
    }
    __host__ __device__ bool operator>(const KnapsackItem &rhs) const {
        return (first > rhs.first);
    }
    __host__ __device__ bool operator>=(const KnapsackItem &rhs) const {
        return (first >= rhs.first);
    }
    __host__ __device__ bool operator==(const KnapsackItem &rhs) const {
        return (first == rhs.first);
    }
    __host__ __device__ bool operator!=(const KnapsackItem &rhs) const {
        return (first != rhs.first || second != rhs.second ||
                third != rhs.third)|| fourth != rhs.fourth;
    }

    // rewrite this if you need more accurate comparison
    //__host__ __device__ bool operator<(const KnapsackItem &rhs) const {
        //return (fourth < rhs.fourth);
    //}
    //__host__ __device__ bool operator<=(const KnapsackItem &rhs) const {
        //return (fourth <= rhs.fourth);
    //}
    //__host__ __device__ bool operator>(const KnapsackItem &rhs) const {
        //return (fourth > rhs.fourth);
    //}
    //__host__ __device__ bool operator>=(const KnapsackItem &rhs) const {
        //return (fourth >= rhs.fourth);
    //}
    //__host__ __device__ bool operator==(const KnapsackItem &rhs) const {
        //return (fourth == rhs.fourth);
    //}
    //__host__ __device__ bool operator!=(const KnapsackItem &rhs) const {
        //return (first != rhs.first || second != rhs.second ||
                //third != rhs.third)|| fourth != rhs.fourth;
    //}


};

inline std::ostream& operator << (std::ostream& o, const KnapsackItem& a)
{
    //o << "Benefit: " << -a.first << " Weight: " << a.second << " Index: " << a.third << " Sequence: " << std::bitset<64>(a.fourth);
    o << "Benefit: " << -a.first << " Weight: " << a.second << " Index: " << (int)a.third;
    return o;
}

inline __host__ __device__ void bin(uint64 n)
{
    if (n > 1)
    bin(n>>1);

    printf("%d", n & 1);
}
#endif
