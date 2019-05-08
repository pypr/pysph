from compyle.api import annotate, Elementwise
from compyle.parallel import Scan
from math import floor
from pytools import memoize


@memoize(key=lambda *args: tuple(args))
def get_elwise(f, backend):
    return Elementwise(f, backend=backend)


@memoize(key=lambda *args: tuple(args))
def get_scan(inp_f, out_f, dtype, backend):
    return Scan(input=inp_f, output=out_f, dtype=dtype,
                backend=backend)


@annotate
def exclusive_input(i, ary):
    return ary[i]


@annotate
def exclusive_output(i, prev_item, ary):
    ary[i] = prev_item


@annotate
def norm2(x, y, z):
    return x * x + y * y + z * z


@annotate
def find_cell_id(x, y, z, h, c):
    c[0] = floor((x) / h)
    c[1] = floor((y) / h)
    c[2] = floor((z) / h)


@annotate(p='ulong', return_='ulong')
def interleave1(p):
    return p


@annotate(ulong='p, q', return_='ulong')
def interleave2(p, q):
    p = p & 0xffffffff
    p = (p | (p << 16)) & 0x0000ffff0000ffff
    p = (p | (p << 8)) & 0x00ff00ff00ff00ff
    p = (p | (p << 4)) & 0x0f0f0f0f0f0f0f0f
    p = (p | (p << 2)) & 0x3333333333333333
    p = (p | (p << 1)) & 0x5555555555555555

    q = q & 0xffffffff
    q = (q | (q << 16)) & 0x0000ffff0000ffff
    q = (q | (q << 8)) & 0x00ff00ff00ff00ff
    q = (q | (q << 4)) & 0x0f0f0f0f0f0f0f0f
    q = (q | (q << 2)) & 0x3333333333333333
    q = (q | (q << 1)) & 0x5555555555555555

    return (p | (q << 1))


@annotate(ulong='p, q, r', return_='ulong')
def interleave3(p, q, r):
    p = (p | (p << 32)) & 0x1f00000000ffff
    p = (p | (p << 16)) & 0x1f0000ff0000ff
    p = (p | (p << 8)) & 0x100f00f00f00f00f
    p = (p | (p << 4)) & 0x10c30c30c30c30c3
    p = (p | (p << 2)) & 0x1249249249249249

    q = (q | (q << 32)) & 0x1f00000000ffff
    q = (q | (q << 16)) & 0x1f0000ff0000ff
    q = (q | (q << 8)) & 0x100f00f00f00f00f
    q = (q | (q << 4)) & 0x10c30c30c30c30c3
    q = (q | (q << 2)) & 0x1249249249249249

    r = (r | (r << 32)) & 0x1f00000000ffff
    r = (r | (r << 16)) & 0x1f0000ff0000ff
    r = (r | (r << 8)) & 0x100f00f00f00f00f
    r = (r | (r << 4)) & 0x10c30c30c30c30c3
    r = (r | (r << 2)) & 0x1249249249249249

    return (p | (q << 1) | (r << 2))


@annotate
def find_idx(keys, num_particles, key):
    first = 0
    last = num_particles - 1
    middle = (first + last) / 2

    while first <= last:
        if keys[middle] < key:
            first = middle + 1
        elif keys[middle] > key:
            last = middle - 1
        elif keys[middle] == key:
            if middle == 0:
                return 0
            if keys[middle - 1] != key:
                return middle
            else:
                last = middle - 1
        middle = (first + last) / 2

    return -1


@annotate
def neighbor_boxes(c_x, c_y, c_z, nbr_boxes):
    nbr_boxes_length = 1

    nbr_boxes[0] = interleave3(c_x, c_y, c_z)

    key = declare('ulong')
    for j in range(-1, 2):
        for k in range(-1, 2):
            for m in range(-1, 2):
                if (j != 0 or k != 0 or m != 0) and c_x + m >= 0 and c_y + k >= 0 and c_z + j >= 0:
                    key = interleave3(c_x + m, c_y + k, c_z + j)
                    nbr_boxes[nbr_boxes_length] = key
                    nbr_boxes_length += 1

    return nbr_boxes_length
