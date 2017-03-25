// Template file for OpenCL kernels

<%def name="preamble()" cached="True">
    #define NORM2(X, Y, Z) ((X)*(X) + (Y)*(Y) + (Z)*(Z))

    #define FIND_CELL_ID(x, y, z, h, c_x, c_y, c_z) \
    c_x = floor((x)/h); c_y = floor((y)/h); c_z = floor((z)/h)

    inline unsigned long interleave(unsigned long p, \
            unsigned long q, unsigned long r);

    inline int neighbor_boxes(int c_x, int c_y, int c_z, \
            unsigned int num_particles, unsigned long* nbr_boxes);

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
            unsigned int num_particles, unsigned long* nbr_boxes)
    {
        int nbr_boxes_length = 0;
        int j, k, m;
        unsigned long key;
        #pragma unroll
        for(j=-1; j<2; j++)
        {
            #pragma unroll
            for(k=-1; k<2; k++)
            {
                #pragma unroll
                for(m=-1; m<2; m++)
                {
                    if(c_x+m >= 0 && c_y+k >= 0 && c_z+j >= 0)
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


<%def name="fill_pids_arguments(data_t)" cached="True">
    ${data_t}* x, ${data_t}* y, ${data_t}* z, ${data_t} cell_size,
    ${data_t} xmin, ${data_t} ymin, ${data_t} zmin,
    unsigned long* keys, unsigned int* pids
</%def>

<%def name="fill_pids_source(data_t)">
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


<%def name="fill_unique_cids_arguments(data_t)" cached="True">
    unsigned long* keys, unsigned int* cids, unsigned int* curr_cid
</%def>

<%def name="fill_unique_cids_source(data_t)" cached="True">
    cids[i] = (i != 0 && keys[i] != keys[i-1]) ? atomic_inc(&curr_cid[0]) : 0;
</%def>


<%def name="map_cid_to_idx_arguments(data_t)" cached="True">
    ${data_t}* x, ${data_t}* y, ${data_t}* z, int num_particles,
    ${data_t} cell_size, ${data_t}3 min, unsigned int* pids,
    unsigned long* keys, unsigned int* cids, int* cid_to_idx
</%def>

<%def name="map_cid_to_idx_source(data_t)" cached="True">
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

    nbr_boxes_length = neighbor_boxes(c.x, c.y, c.z,
        num_particles, nbr_boxes);

    #pragma unroll
    for(j=0; j<nbr_boxes_length; j++)
    {
        key = nbr_boxes[j];
        idx = find_idx(keys, num_particles, key);
        cid_to_idx[27*cid + j] = idx;
    }
</%def>


<%def name="fill_cids_arguments(data_t)" cached="True">
    unsigned long* keys, unsigned int* cids, unsigned int num_particles
</%def>

<%def name="fill_cids_source(data_t)" cached="True">
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


<%def name="find_neighbor_lengths_arguments(data_t)" cached="True">
    const ${data_t}* d_x, const ${data_t}* d_y, const ${data_t}* d_z,
    const ${data_t}* d_h, const ${data_t}* s_x, const ${data_t}* s_y,
    const ${data_t}* s_z, const ${data_t}* s_h,
    ${data_t}3 min, unsigned int num_particles, const unsigned long* keys,
    const unsigned int* pids, unsigned int* cids, int* cid_to_idx,
    unsigned int* nbr_lengths, ${data_t} radius_scale2, ${data_t} cell_size
</%def>

<%def name="find_neighbor_lengths_source(data_t, sorted)" cached="True">
    unsigned int qid;

    % if sorted:
        qid = i;
    % else:
        qid = pids[i];
    % endif

    ${data_t}4 q = (${data_t}4)(d_x[qid], d_y[qid], d_z[qid], d_h[qid]);

    int3 c;
    unsigned int cid = cids[i];

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

    #pragma unroll
    for(j=0; j<27; j++)
    {
        idx = cid_to_idx[27*cid + j];
        if(idx == -1)
            continue;
        key = keys[idx];

        while(idx < num_particles && keys[idx] == key)
        {
            pid = pids[idx];
            h_j = radius_scale2*s_h[pid]*s_h[pid];
            dist = NORM2(q.x - s_x[pid], q.y - s_y[pid], \
                    q.z - s_z[pid]);
            if(dist < h_i || dist < h_j)
                nbr_lengths[qid] += 1;
            idx++;
        }
    }

</%def>


<%def name="find_neighbors_arguments(data_t)" cached="True">
    const ${data_t}* d_x, const ${data_t}* d_y, const ${data_t}* d_z,
    const ${data_t}* d_h, const ${data_t}* s_x, const ${data_t}* s_y,
    const ${data_t}* s_z, const ${data_t}* s_h,
    ${data_t}3 min, unsigned int num_particles, const unsigned long* keys,
    const unsigned int* pids, unsigned int* cids, int* cid_to_idx,
    const unsigned int* start_indices, unsigned int* nbrs,
    ${data_t} radius_scale2, ${data_t} cell_size
</%def>

<%def name="find_neighbors_source(data_t, sorted)" cached="True">
    unsigned int qid;

    % if sorted:
        qid = i;
    % else:
        qid = pids[i];
    % endif

    ${data_t}4 q = (${data_t}4)(d_x[qid], d_y[qid], d_z[qid], d_h[qid]);

    int3 c;
    unsigned int cid = cids[i];

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

    unsigned long start_idx = (unsigned long) start_indices[qid];
    unsigned long curr_idx = 0;

    #pragma unroll
    for(j=0; j<27; j++)
    {
        idx = cid_to_idx[27*cid + j];
        if(idx == -1)
            continue;
        key = keys[idx];

        while(idx < num_particles && keys[idx] == key)
        {
            pid = pids[idx];
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


