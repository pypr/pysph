//CL//

<%def name="preamble(data_t)" cached="True">
    inline int get_level(${data_t} radius_scale, ${data_t} hmin, \
            ${data_t} interval_size, ${data_t} h)
    {
        return (int) floor((radius_scale*h - hmin)/interval_size);
    }

    inline int extract_level(ulong key, int max_num_bits)
    {
        return key >> max_num_bits;
    }

    inline ${data_t} get_cell_size(int level, ${data_t} hmin,
            ${data_t} interval_size)
    {
        return hmin + (level + 1)*interval_size;
    }

</%def>


<%def name="fill_pids_args(data_t)" cached="True">
    ${data_t}* x, ${data_t}* y, ${data_t}* z, ${data_t}* h, ${data_t} interval_size,
    ${data_t} xmin, ${data_t} ymin, ${data_t} zmin, ${data_t} hmin,
    ulong* keys, uint* pids, ${data_t} radius_scale,
    int max_num_bits
</%def>

<%def name="fill_pids_src(data_t)" cached="True">
    ulong c_x, c_y, c_z;
    ulong key;
    
    int level = get_level(radius_scale, hmin, interval_size, h[i]);
    pids[i] = i;

    FIND_CELL_ID(
        x[i] - xmin,
        y[i] - ymin,
        z[i] - zmin,
        get_cell_size(level, hmin, interval_size), c_x, c_y, c_z
        );

    key = interleave(c_x, c_y, c_z);
    ulong mask = level << max_num_bits;
    keys[i] = key + mask;

</%def>


<%def name="fill_start_indices_args(data_t)" cached="True">
    ulong* keys, uint* start_idx_levels, int max_num_bits, uint* num_particles_levels
</%def>

<%def name="fill_start_indices_src(data_t)" cached="True">
    int curr_level = extract_level(keys[i], max_num_bits);
    if(i != 0 && curr_level != extract_level(keys[i-1], max_num_bits))
        atomic_min(&start_idx_levels[curr_level], i);
    else
        atomic_inc(&num_particles_levels[curr_level])
</%def>


<%def name="find_nbrs_prep(data_t, sorted)", cached="True">
    uint qid = i;

    ${data_t}4 q = (${data_t}4)(d_x[qid], d_y[qid], d_z[qid], d_h[qid]);
    ${data_t} radius_scale2 = radius_scale*radius_scale;

    int3 c;

    int idx, j, k, m;
    ${data_t} dist;
    ${data_t} h_i = radius_scale2*q.w*q.w;
    ${data_t} h_j;

    ulong key;
    uint pid;

    ${data_t} h_max, cell_size_level;
    int H, level;

    ulong mask;
</%def>

<%def name="find_nbr_lengths_args(data_t)" cached="True">
    ${data_t}* d_x, ${data_t}* d_y, ${data_t}* d_z,
    ${data_t}* d_h, ${data_t}* s_x, ${data_t}* s_y,
    ${data_t}* s_z, ${data_t}* s_h,
    ${data_t}3 min, uint num_particles, ulong* keys,
    uint* pids_dst, uint* pids_src,
    uint* nbr_lengths, ${data_t} radius_scale,
    ${data_t} hmin, ${data_t} interval_size, uint* start_idx_levels,
    int max_num_bits, int num_levels, uint* num_particles_levels
</%def>

<%def name="find_nbr_lengths_src(data_t, sorted)" cached="True">
    ${find_nbrs_prep(data_t, sorted)}

    unsigned int length = 0;

    #pragma unroll
    for(level=0; level<num_levels; level++)
    {
        if(start_idx_levels[level] == num_particles)
            continue;
        cell_size_level = get_cell_size(level, hmin, interval_size);
        FIND_CELL_ID(
            q.x - min.x,
            q.y - min.y,
            q.z - min.z,
            cell_size_level, c.x, c.y, c.z
            );

        h_max = fmax(radius_scale*q.w, cell_size_level);
        H = ceil(h_max/cell_size_level);

        mask = level << max_num_bits;

        #pragma unroll
        for(j=-H; j<H+1; j++)
        {
            #pragma unroll
            for(k=-H; k<H+1; k++)
            {
                #pragma unroll
                for(m=-H; m<H+1; m++)
                {
                    if(c.x+m < 0 || c.y+k < 0 || c.z+j < 0)
                        continue;
                    key = mask + interleave(c.x+m, c.y+k, c.z+j);
                    idx = find_idx(keys + start_idx_levels[level],
                        num_particles_levels[level], key);
                    if(idx == -1)
                        continue;
                    idx += start_idx_levels[level];

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
            }
        }
    }

    nbr_lengths[qid] = length;

</%def>


<%def name="find_nbrs_args(data_t)" cached="True">
    ${data_t}* d_x, ${data_t}* d_y, ${data_t}* d_z,
    ${data_t}* d_h, ${data_t}* s_x, ${data_t}* s_y,
    ${data_t}* s_z, ${data_t}* s_h,
    ${data_t}3 min, uint num_particles, ulong* keys,
    uint* pids_dst, uint* pids_src, 
    uint* start_indices, uint* nbrs,
    ${data_t} radius_scale,
    ${data_t} hmin, ${data_t} interval_size, uint* start_idx_levels,
    int max_num_bits, int num_levels, uint* num_particles_levels
</%def>

<%def name="find_nbrs_src(data_t, sorted)" cached="True">
    ${find_nbrs_prep(data_t, sorted)}

    ulong start_idx = (ulong) start_indices[qid];
    ulong curr_idx = 0;

    #pragma unroll
    for(level=0; level<num_levels; level++)
    {
        if(start_idx_levels[level] == num_particles)
            continue;
        cell_size_level = get_cell_size(level, hmin, interval_size);
        FIND_CELL_ID(
            q.x - min.x,
            q.y - min.y,
            q.z - min.z,
            cell_size_level, c.x, c.y, c.z
            );

        h_max = fmax(radius_scale*q.w, cell_size_level);
        H = ceil(h_max/cell_size_level);

        mask = level << max_num_bits;

        #pragma unroll
        for(j=-H; j<H+1; j++)
        {
            #pragma unroll
            for(k=-H; k<H+1; k++)
            {
                #pragma unroll
                for(m=-H; m<H+1; m++)
                {
                    if(c.x+m < 0 || c.y+k < 0 || c.z+j < 0)
                        continue;
                    key = mask + interleave(c.x+m, c.y+k, c.z+j);
                    idx = find_idx(keys + start_idx_levels[level],
                        num_particles_levels[level], key);
                    if(idx == -1)
                        continue;
                    idx += start_idx_levels[level];

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
            }
        }
    }

</%def>


