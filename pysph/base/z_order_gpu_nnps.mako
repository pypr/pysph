//CL//

// IMPORTANT NOTE: pyopencl uses the length of the first argument
// to determine the global work size

<%def name="preamble()" cached="True">
</%def>


<%def name="fill_pids_args(data_t)" cached="True">
    ${data_t}* x, ${data_t}* y, ${data_t}* z, ${data_t} cell_size,
    ${data_t}3 min, unsigned long* keys, unsigned int* pids
</%def>

<%def name="fill_pids_src(data_t)" cached="True">
    unsigned long c_x, c_y, c_z;
    FIND_CELL_ID(
        x[i] - min.x,
        y[i] - min.y,
        z[i] - min.z,
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
    unsigned int* dst_to_src, unsigned int* cids_dst, int* cid_to_idx_dst,
    unsigned long* keys_dst, unsigned long* keys_src, unsigned int* cids_src,
    unsigned int num_particles_src, int* max_cid_src
</%def>

<%def name="map_dst_to_src_src(data_t)" cached="True">
    int idx = cid_to_idx_dst[27*i];
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

<%def name="z_order_nbrs_prep(data_t, sorted, dst_src)", cached="False">
     unsigned int qid;

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

<%def name="z_order_nbr_lengths_src(data_t, sorted, dst_src)" cached="False">
    ${z_order_nbrs_prep(data_t, sorted, dst_src)}

    unsigned int length = 0;

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
                length++;
            idx++;
        }
    }

    nbr_lengths[qid] = length;

</%def>


<%def name="z_order_nbrs_args(data_t)" cached="False">
    ${data_t}* d_x, ${data_t}* d_y, ${data_t}* d_z,
    ${data_t}* d_h, ${data_t}* s_x, ${data_t}* s_y,
    ${data_t}* s_z, ${data_t}* s_h,
    ${data_t}3 min, unsigned int num_particles, unsigned long* keys,
    unsigned int* pids_dst, unsigned int* pids_src, unsigned int max_cid_src,
    unsigned int* cids, int* cid_to_idx, int* overflow_cid_to_idx,
    unsigned int* dst_to_src, unsigned int* start_indices, unsigned int* nbrs,
    ${data_t} radius_scale2, ${data_t} cell_size
</%def>

<%def name="z_order_nbrs_src(data_t, sorted, dst_src)" cached="False">
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


