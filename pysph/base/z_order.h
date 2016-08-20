#ifndef Z_ORDER_H
#define Z_ORDER_H
#include <iostream>
#include <algorithm>
#include <cmath>

using namespace std;

typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;

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
    double* x_ptr;
    double* y_ptr;
    double* z_ptr;

    double* xmin;
    double cell_size;

    uint32_t* current_pids;
    int length;
public:
    CompareSortWrapper()
    {
        this->x_ptr = NULL;
        this->y_ptr = NULL;
        this->z_ptr = NULL;

        this->xmin = NULL;
        this->cell_size = 0; 

        this->current_pids = NULL;
        this->length = 0;
    }

    CompareSortWrapper(double* x_ptr, double* y_ptr, double* z_ptr,
            double* xmin, double cell_size, uint32_t* current_pids,
            int length)
    {
        this->x_ptr = x_ptr;
        this->y_ptr = y_ptr;
        this->z_ptr = z_ptr;

        this->xmin = xmin;
        this->cell_size = cell_size; 

        this->current_pids = current_pids;
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
            int c_x, c_y, c_z;
            int id_x, id_y, id_z;

            find_cell_id(
                    this->data->x_ptr[a] - this->data->xmin[0],
                    this->data->y_ptr[a] - this->data->xmin[1],
                    this->data->z_ptr[a] - this->data->xmin[2],
                    this->data->cell_size,
                    c_x, c_y, c_z
                    );

            find_cell_id(
                    this->data->x_ptr[b] - this->data->xmin[0],
                    this->data->y_ptr[b] - this->data->xmin[1],
                    this->data->z_ptr[b] - this->data->xmin[2],
                    this->data->cell_size,
                    id_x, id_y, id_z
                    );

            return get_key(c_x, c_y, c_z) < get_key(id_x, id_y, id_z);
        }
    };

    inline void compare_sort()
    {
        sort(this->current_pids, this->current_pids + this->length,
                CompareFunctionWrapper(this));
    }
};

#endif

