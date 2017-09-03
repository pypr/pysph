#ifndef Z_ORDER_H
#define Z_ORDER_H
#include <iostream>
#include <algorithm>
#include <cmath>

#ifdef _WIN32
    typedef unsigned int uint32_t;
    typedef unsigned long long uint64_t;
#else
    #include <stdint.h>
#endif

using namespace std;

inline void find_cell_id(double x, double y, double z, double h,
        int &c_x, int &c_y, int &c_z)
{
    c_x = floor(x/h);
    c_y = floor(y/h);
    c_z = floor(z/h);
}

inline uint64_t get_key(uint64_t i, uint64_t j, uint64_t k)
{

    i = (i | (i << 32)) & 0x1f00000000ffff;
    i = (i | (i << 16)) & 0x1f0000ff0000ff;
    i = (i | (i <<  8)) & 0x100f00f00f00f00f;
    i = (i | (i <<  4)) & 0x10c30c30c30c30c3;
    i = (i | (i <<  2)) & 0x1249249249249249;

    j = (j | (j << 32)) & 0x1f00000000ffff;
    j = (j | (j << 16)) & 0x1f0000ff0000ff;
    j = (j | (j <<  8)) & 0x100f00f00f00f00f;
    j = (j | (j <<  4)) & 0x10c30c30c30c30c3;
    j = (j | (j <<  2)) & 0x1249249249249249;

    k = (k | (k << 32)) & 0x1f00000000ffff;
    k = (k | (k << 16)) & 0x1f0000ff0000ff;
    k = (k | (k <<  8)) & 0x100f00f00f00f00f;
    k = (k | (k <<  4)) & 0x10c30c30c30c30c3;
    k = (k | (k <<  2)) & 0x1249249249249249;

    return (i | (j << 1) | (k << 2));
}

class CompareSortWrapper
{
private:
    uint32_t* current_pids;
    uint64_t* current_keys;
    int length;
public:
    CompareSortWrapper()
    {
        this->current_pids = NULL;
        this->current_keys = NULL;
        this->length = 0;
    }

    CompareSortWrapper(uint32_t* current_pids, uint64_t* current_keys,
            int length)
    {
        this->current_pids = current_pids;
        this->current_keys = current_keys;
        this->length = length;
    }

    struct CompareFunctionWrapper
    {
        CompareSortWrapper* data;

        CompareFunctionWrapper(CompareSortWrapper* data)
        {
            this->data = data;
        }

        inline bool operator()(const int &a, const int &b)
        {
            return this->data->current_keys[a] < this->data->current_keys[b];
        }
    };

    inline void compare_sort()
    {
        sort(this->current_pids, this->current_pids + this->length,
                CompareFunctionWrapper(this));

        sort(this->current_keys, this->current_keys + this->length);
    }
};

#endif

