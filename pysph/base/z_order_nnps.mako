//CL//

<%def name="preamble()" cached="True">
    #define ULLONG_MAX (1 << 62)

    #define NORM2(X, Y, Z) ((X)*(X) + (Y)*(Y) + (Z)*(Z))

    #define FIND_CELL_ID(x, y, z, h, c_x, c_y, c_z) \
    c_x = floor((x)/h); c_y = floor((y)/h); c_z = floor((z)/h)

    inline unsigned long interleave(unsigned long p, \
            unsigned long q, unsigned long r);

    inline int neighbor_boxes(int c_x, int c_y, int c_z, \
            unsigned long* nbr_boxes);

    inline unsigned long interleave(unsigned long p, \
            unsigned long q, unsigned long r)
    {
        p = (p | (p << 32)) & 0x1f00000000ffff;
        p = (p | (p << 16)) & 0x1f0000ff0000ff;
        p = (p | (p <<  8)) & 0x100f00f00f00f00f;
        p = (p | (p <<  4)) & 0x10c30c30c30c30c3;
        p = (p | (p <<  2)) & 0x1249249249249249;

        q = (q | (q << 32)) & 0x1f00000000ffff;
        q = (q | (q << 16)) & 0x1f0000ff0000ff;
        q = (q | (q <<  8)) & 0x100f00f00f00f00f;
        q = (q | (q <<  4)) & 0x10c30c30c30c30c3;
        q = (q | (q <<  2)) & 0x1249249249249249;

        r = (r | (r << 32)) & 0x1f00000000ffff;
        r = (r | (r << 16)) & 0x1f0000ff0000ff;
        r = (r | (r <<  8)) & 0x100f00f00f00f00f;
        r = (r | (r <<  4)) & 0x10c30c30c30c30c3;
        r = (r | (r <<  2)) & 0x1249249249249249;

        return (p | (q << 1) | (r << 2));
    }

    inline int find_idx(__global unsigned long* keys, \
            int num_particles, unsigned long key)
    {
        int first = 0;
        int last = num_particles - 1;
        int middle = (first + last) / 2;

        while(first <= last)
        {
            if(keys[middle] < key)
                first = middle + 1;
            else if(keys[middle] > key)
                last = middle - 1;
            else if(keys[middle] == key)
            {
                if(middle == 0)
                    return 0;
                if(keys[middle - 1] != key)
                    return middle;
                else
                    last = middle - 1;
            }
            middle = (first + last) / 2;
        }

        return -1;
    }

    inline int neighbor_boxes(int c_x, int c_y, int c_z, \
        unsigned long* nbr_boxes)
    {
        int nbr_boxes_length = 1;
        int j, k, m;
        unsigned long key;
        nbr_boxes[0] = interleave(c_x, c_y, c_z);

        #pragma unroll
        for(j=-1; j<2; j++)
        {
            #pragma unroll
            for(k=-1; k<2; k++)
            {
                #pragma unroll
                for(m=-1; m<2; m++)
                {
                    if((j != 0 || k != 0 || m != 0) && c_x+m >= 0 && c_y+k >= 0 && c_z+j >= 0)
                    {
                        key = interleave(c_x+m, c_y+k, c_z+j);
                        nbr_boxes[nbr_boxes_length] = key;
                        nbr_boxes_length++;
                    }
                }
            }
        }

        return nbr_boxes_length;
    }

</%def>


<%def name="fill_pids_args(data_t)" cached="True">
    ${data_t}* x, ${data_t}* y, ${data_t}* z, ${data_t} cell_size,
    ${data_t} xmin, ${data_t} ymin, ${data_t} zmin,
    unsigned long* keys, unsigned int* pids
</%def>

<%def name="fill_pids_src(data_t)" cached="True">
    unsigned long c_x, c_y, c_z;
    FIND_CELL_ID(
        x[i] - xmin,
        y[i] - ymin,
        z[i] - zmin,
        cell_size, c_x, c_y, c_z
        );
    unsigned long key;
    key = interleave(c_x, c_y, c_z);
    keys[i] = key;
    pids[i] = i;
</%def>


<%def name="fill_unique_cids_args(data_t)" cached="True">
    unsigned long* keys, unsigned int* cids, unsigned int* curr_cid
</%def>

<%def name="fill_unique_cids_src(data_t)" cached="True">
    cids[i] = (i != 0 && keys[i] != keys[i-1]) ? atomic_inc(&curr_cid[0]) : 0;
</%def>


<%def name="map_cid_to_idx_args(data_t)" cached="True">
    ${data_t}* x, ${data_t}* y, ${data_t}* z, int num_particles,
    ${data_t} cell_size, ${data_t}3 min, unsigned int* pids,
    unsigned long* keys, unsigned int* cids, int* cid_to_idx
</%def>

<%def name="map_cid_to_idx_src(data_t)" cached="True">
    unsigned int cid = cids[i];

    if(i != 0 && cid == 0)
        PYOPENCL_ELWISE_CONTINUE;

    unsigned int j;
    int idx;
    unsigned long key;
    int nbr_boxes_length;
    ${data_t}3 c;

    unsigned int pid = pids[i];

    FIND_CELL_ID(
        x[pid] - min.x,
        y[pid] - min.y,
        z[pid] - min.z,
        cell_size, c.x, c.y, c.z
        );

    unsigned long* nbr_boxes[27];

    nbr_boxes_length = neighbor_boxes(c.x, c.y, c.z, nbr_boxes);

    #pragma unroll
    for(j=0; j<nbr_boxes_length; j++)
    {
        key = nbr_boxes[j];
        idx = find_idx(keys, num_particles, key);
        cid_to_idx[27*cid + j] = idx;
    }
</%def>


<%def name="fill_cids_args(data_t)" cached="True">
    unsigned long* keys, unsigned int* cids, unsigned int num_particles
</%def>

<%def name="fill_cids_src(data_t)" cached="True">
    unsigned int cid = cids[i];
    if(cid == 0)
        PYOPENCL_ELWISE_CONTINUE;
    unsigned int j = i + 1;
    while(j < num_particles && cids[j] == 0)
    {
        cids[j] = cid;
        j++;
    }
</%def>

<%def name="map_dst_to_src_args(data_t)" cached="True">
    unsigned int* dst_to_src, unsigned int* cids_dst, int* cid_to_idx_dst, unsigned long* keys_dst,
    unsigned long* keys_src, unsigned int* cids_src, unsigned int num_particles_src,
    int* max_cid_src
</%def>

<%def name="map_dst_to_src_src(data_t)" cached="True">
    // ENH: number of work items can be set using dst_to_src
    int idx = cid_to_idx_dst[27*i];
    //unsigned int cid_dst = i;
    unsigned long key = keys_dst[idx];
    int idx_src = find_idx(keys_src, num_particles_src, key);
    dst_to_src[i] = (idx_src == -1) ? atomic_inc(&max_cid_src[0]) : cids_src[idx_src];
</%def>

<%def name="fill_overflow_map_args(data_t)" cached="True">
    unsigned int* dst_to_src, int* cid_to_idx_dst, ${data_t}* x,
    ${data_t}* y, ${data_t}* z, int num_particles_src,
    ${data_t} cell_size, ${data_t}3 min, unsigned long* keys_src,
    unsigned int* pids_dst, int* overflow_cid_to_idx,
    unsigned int max_cid_src
</%def>

<%def name="fill_overflow_map_src(data_t)" cached="True">
    unsigned int cid = dst_to_src[i];
    // i is the cid in dst

    if(cid < max_cid_src)
        PYOPENCL_ELWISE_CONTINUE;

    int idx = cid_to_idx_dst[27*i];

    unsigned int j;
    unsigned long key;
    int nbr_boxes_length;
    ${data_t}3 c;

    unsigned int pid = pids_dst[idx];

    FIND_CELL_ID(
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
</%def>

// IMPORTANT: Check if refactoring of the following kernels is needed for caching
// the result

<%def name="z_order_nbrs_prep(data_t, sorted, dst_src)", cached="True">
     unsigned int qid;

    // IMPORTANT: Make sure the 'cid' approach works for sorted particle arrays
    % if sorted:
        qid = i;
    % else:
        qid = pids_dst[i];
    % endif

    ${data_t}4 q = (${data_t}4)(d_x[qid], d_y[qid], d_z[qid], d_h[qid]);

    int3 c;

    FIND_CELL_ID(
        q.x - min.x,
        q.y - min.y,
        q.z - min.z,
        cell_size, c.x, c.y, c.z
        );

    int idx;
    unsigned int j;
    ${data_t} dist;
    ${data_t} h_i = radius_scale2*q.w*q.w;
    ${data_t} h_j;

    unsigned long key;
    unsigned int pid;


    unsigned int cid = cids[i];
    __global int* nbr_boxes = cid_to_idx;
    unsigned int start_id_nbr_boxes;

    % if dst_src:
        cid = dst_to_src[cid];
        start_id_nbr_boxes = 27*cid;
        if(cid >= max_cid_src)
        {
            start_id_nbr_boxes = 27*(cid - max_cid_src);
            nbr_boxes = overflow_cid_to_idx;
        }
    % else:
        start_id_nbr_boxes = 27*cid;
    % endif

</%def>

<%def name="z_order_nbr_lengths_args(data_t)" cached="True">
    ${data_t}* d_x, ${data_t}* d_y, ${data_t}* d_z,
    ${data_t}* d_h, ${data_t}* s_x, ${data_t}* s_y,
    ${data_t}* s_z, ${data_t}* s_h,
    ${data_t}3 min, unsigned int num_particles, unsigned long* keys,
    unsigned int* pids_dst, unsigned int* pids_src, unsigned int max_cid_src,
    unsigned int* cids, int* cid_to_idx, int* overflow_cid_to_idx,
    unsigned int* dst_to_src, unsigned int* nbr_lengths, ${data_t} radius_scale2,
    ${data_t} cell_size
</%def>

<%def name="z_order_nbr_lengths_src(data_t, sorted, dst_src)" cached="True">
    ${z_order_nbrs_prep(data_t, sorted, dst_src)}

    #pragma unroll
    for(j=0; j<27; j++)
    {
        idx = nbr_boxes[start_id_nbr_boxes + j];
        if(idx == -1)
            continue;
        key = keys[idx];

        while(idx < num_particles && keys[idx] == key)
        {
            pid = pids_src[idx];
            h_j = radius_scale2*s_h[pid]*s_h[pid];
            dist = NORM2(q.x - s_x[pid], q.y - s_y[pid], \
                    q.z - s_z[pid]);
            if(dist < h_i || dist < h_j)
                nbr_lengths[qid] += 1;
            idx++;
        }
    }

</%def>


<%def name="z_order_nbrs_args(data_t)" cached="True">
    ${data_t}* d_x, ${data_t}* d_y, ${data_t}* d_z,
    ${data_t}* d_h, ${data_t}* s_x, ${data_t}* s_y,
    ${data_t}* s_z, ${data_t}* s_h,
    ${data_t}3 min, unsigned int num_particles, unsigned long* keys,
    unsigned int* pids_dst, unsigned int* pids_src, unsigned int max_cid_src,
    unsigned int* cids, int* cid_to_idx, int* overflow_cid_to_idx,
    unsigned int* dst_to_src, unsigned int* start_indices, unsigned int* nbrs,
    ${data_t} radius_scale2, ${data_t} cell_size
</%def>

<%def name="z_order_nbrs_src(data_t, sorted, dst_src)" cached="True">
    ${z_order_nbrs_prep(data_t, sorted, dst_src)}

    unsigned long start_idx = (unsigned long) start_indices[qid];
    unsigned long curr_idx = 0;

    #pragma unroll
    for(j=0; j<27; j++)
    {
        idx = nbr_boxes[start_id_nbr_boxes + j];
        if(idx == -1)
            continue;
        key = keys[idx];

        while(idx < num_particles && keys[idx] == key)
        {
            pid = pids_src[idx];
            h_j = radius_scale2*s_h[pid]*s_h[pid];
            dist = NORM2(q.x - s_x[pid], q.y - s_y[pid], \
                    q.z - s_z[pid]);
            if(dist < h_i || dist < h_j)
            {
                nbrs[start_idx + curr_idx] = pid;
                curr_idx++;
            }
            idx++;
        }
    }

</%def>


