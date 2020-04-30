from pysph.base.gpu_helper_kernels import *
from compyle.api import declare
from compyle.template import Template
from compyle.low_level import atomic_inc


@annotate
def fill_pids(i, x, y, z, cell_size, xmin, ymin, zmin, keys, pids):
    c = declare('matrix(3, "int")')
    find_cell_id(
        x[i] - xmin,
        y[i] - ymin,
        z[i] - zmin,
        cell_size, c
    )
    key = declare('ulong')
    key = interleave3(c[0], c[1], c[2])
    keys[i] = key
    pids[i] = i


@annotate
def inp_fill_unique_cids(i, keys, cids):
    return 1 if i != 0 and keys[i] != keys[i - 1] else 0


@annotate
def out_fill_unique_cids(i, item, cids):
    cids[i] = item


@annotate
def map_cid_to_idx(i, x, y, z, num_particles, cell_size, xmin, ymin, zmin,
                   pids, keys, cids, cid_to_idx):
    cid = cids[i]

    if i != 0 and cids[i] == cids[i - 1]:
        return

    c = declare('matrix(3, "int")')

    pid = pids[i]

    find_cell_id(
        x[pid] - xmin,
        y[pid] - ymin,
        z[pid] - zmin,
        cell_size, c
    )

    nbr_boxes = declare('matrix(27, "ulong")')

    nbr_boxes_length = neighbor_boxes(c[0], c[1], c[2], nbr_boxes)

    for j in range(nbr_boxes_length):
        key = nbr_boxes[j]
        idx = find_idx(keys, num_particles, key)
        cid_to_idx[27 * cid + j] = idx


@annotate
def map_dst_to_src(i, dst_to_src, cids_dst, cid_to_idx_dst,
                   keys_dst, keys_src, cids_src, num_particles_src, max_cid_src):
    idx = cid_to_idx_dst[27 * i]
    key = keys_dst[idx]
    idx_src = find_idx(keys_src, num_particles_src, key)
    dst_to_src[i] = atomic_inc(
        max_cid_src[0]) if (
        idx_src == -
        1) else cids_src[idx_src]


@annotate
def fill_overflow_map(i, dst_to_src, cid_to_idx_dst, x, y, z,
                      num_particles_src, cell_size, xmin, ymin, zmin, keys_src,
                      pids_dst, overflow_cid_to_idx, max_cid_src):
    cid = dst_to_src[i]
    # i is the cid in dst

    if cid < max_cid_src:
        return

    idx = cid_to_idx_dst[27 * i]

    pid = pids_dst[idx]

    c = declare('matrix(3, "int")')

    find_cell_id(
        x[pid] - xmin,
        y[pid] - ymin,
        z[pid] - zmin,
        cell_size, c
    )

    nbr_boxes = declare('matrix(27, "ulong")')

    nbr_boxes_length = neighbor_boxes(c[0], c[1], c[2], nbr_boxes)

    start_idx = cid - max_cid_src

    for j in range(nbr_boxes_length):
        key = nbr_boxes[j]
        idx = find_idx(keys_src, num_particles_src, key)
        overflow_cid_to_idx[27 * start_idx + j] = idx


class ZOrderNNPSKernel(Template):
    def __init__(self, name, dst_src=False, z_order_length=False,
                 z_order_nbrs=False):
        super(ZOrderNNPSKernel, self).__init__(name=name)
        self.z_order_length = z_order_length
        self.z_order_nbrs = z_order_nbrs
        assert self.z_order_nbrs != self.z_order_length
        self.dst_src = dst_src

    def template(self, i, d_x, d_y, d_z, d_h, s_x, s_y, s_z, s_h, xmin, ymin,
                 zmin, num_particles, keys, pids_dst, pids_src, max_cid_src,
                 cids, cid_to_idx, overflow_cid_to_idx, dst_to_src,
                 radius_scale2, cell_size):
        '''
        q = declare('matrix(4)')

        qid = pids_dst[i]

        q[0] = d_x[qid]
        q[1] = d_y[qid]
        q[2] = d_z[qid]
        q[3] = d_h[qid]

        cid = cids[i]
        nbr_boxes = declare('GLOBAL_MEM int*')
        nbr_boxes = cid_to_idx
        h_i = radius_scale2*q[3]*q[3]

        % if obj.dst_src:
        cid = dst_to_src[cid]
        start_id_nbr_boxes = 27*cid
        if cid >= max_cid_src:
            start_id_nbr_boxes = 27*(cid - max_cid_src)
            nbr_boxes = overflow_cid_to_idx
        % else:
        start_id_nbr_boxes = 27*cid
        % endif

        % if obj.z_order_length:
        length = 0
        % elif obj.z_order_nbrs:
        start_idx = start_indicies[qid]
        curr_idx = 0
        % endif

        s = declare('matrix(4)')
        j = declare('int')

        for j in range(27):
            idx = nbr_boxes[start_id_nbr_boxes + j]
            if idx == -1:
                continue
            key = keys[idx]

            while (idx < num_particles and keys[idx] == key):
                pid = pids_src[idx]
                s[0] = s_x[pid]
                s[1] = s_y[pid]
                s[2] = s_z[pid]
                s[3] = s_h[pid]

                h_j = radius_scale2 * s[3] * s[3]

                % if obj.z_order_nbrs:
                dist = norm2(q[0] - s[0], q[1] - s[1], q[2] - s[2])
                if dist < h_i or dist < h_j:
                    nbrs[start_idx + curr_idx] = pid
                    curr_idx += 1
                % else:
                dist = norm2(q[0] - s[0], q[1] - s[1], q[2] - s[2])
                if dist < h_i or dist < h_j:
                    length += 1
                %endif
                idx += 1

        % if obj.z_order_length:
        nbr_lengths[qid] = length
        % endif
        '''


class ZOrderLengthKernel(ZOrderNNPSKernel):
    def __init__(self, name, dst_src):
        super(ZOrderLengthKernel, self).__init__(
            name, dst_src, z_order_length=True,
        )

    def extra_args(self):
        return ['nbr_lengths'], {}


class ZOrderNbrsKernel(ZOrderNNPSKernel):
    def __init__(self, name, dst_src):
        super(ZOrderNbrsKernel, self).__init__(
            name, dst_src, z_order_nbrs=True
        )

    def extra_args(self):
        return ['start_indicies', 'nbrs'], {}
