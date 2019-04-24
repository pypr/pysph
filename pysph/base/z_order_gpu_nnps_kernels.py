from gpu_helper_kernels import *
from compyle.api import declare


################# fill pids ###################3

def fill_pids(i, x, y, z, xmin, ymin, zmin, keys, pids):
    find_cell_id(
        x[i] - xmin,
        y[i] - ymin,
        z[i] - zmin,
        cell_size, c_x, c_y, c_z
        )
    key = interleave3(c_x, c_y, c_z)
    keys[i] = key
    pids[i] = i


################# fill unique cids ###################3

def inp_fill_unique_cids(i, keys, cids):
    return 1 if i != 0 and keys[i] != keys[i-1] else 0


def out_fill_unique_cids(item, cids):
    cids[i] = item


################# map cid to idx ###################3

def map_cid_to_idx(x, y, z, num_particles, cell_size, xmin, ymin, zmin,
                   pids, keys, cids, cid_to_idx):
    cid = cids[i]

    if i != 0 and cid == 0:
        return

    c_x, c_y, c_z = declare('double', 3)

    pid = pids[i]

    find_cell_id(
        x[pid] - xmin,
        y[pid] - ymin,
        z[pid] - zmin,
        cell_size, c_x, c_y, c_z
        )

    nbr_boxes = declare('matrix(27, "ulong")')

    nbr_boxes_length = neighbor_boxes(c.x, c.y, c.z, nbr_boxes)

    for j in range(nbr_boxes_length):
        key = nbr_boxes[j]
        idx = find_idx(keys, num_particles, key)
        cid_to_idx[27*cid + j] = idx


################# map dst to src ###################3

def map_dst_to_src(dst_to_src, cids_dst, cid_to_idx_dst,
        keys_dst, keys_src, cids_src, num_particles_src, max_cid_src):
    idx = cid_to_idx_dst[27*i]
    key = keys_dst[idx]
    idx_src = find_idx(keys_src, num_particles_src, key)
    dst_to_src[i] = atomic_inc(max_cid_src[0]) if (idx_src == -1) else cids_src[idx_src]


def fill_overflow_map:
    cid = dst_to_src[i]
    # i is the cid in dst

    if cid < max_cid_src
        return

    idx = cid_to_idx_dst[27*i]

    pid = pids_dst[idx]

    find_cell_id(
        x[pid] - min.x,
        y[pid] - min.y,
        z[pid] - min.z,
        cell_size, c.x, c.y, c.z
        );

    unsigned long* nbr_boxes[27];

    nbr_boxes_length = neighbor_boxes(c.x, c.y, c.z, nbr_boxes);

    unsigned int start_idx = cid - max_cid_src;

    #pragma unroll
    for(j=0; j<nbr_boxes_length; j++)
    {
        key = nbr_boxes[j];
        idx = find_idx(keys_src, num_particles_src, key);
        overflow_cid_to_idx[27*start_idx + j] = idx;
    }

