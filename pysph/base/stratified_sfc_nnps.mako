//CL//

<%def name="preamble(data_t)" cached="True">
    #define NORM2(X, Y, Z) ((X)*(X) + (Y)*(Y) + (Z)*(Z))

    #define FIND_CELL_ID(x, y, z, h, c_x, c_y, c_z) \
    c_x = floor((x)/h); c_y = floor((y)/h); c_z = floor((z)/h)

    #define MIN(a,b) (((a)<(b))?(a):(b))
    #define MAX(a,b) (((a)>(b))?(a):(b))

    inline ulong interleave(ulong p, \
            ulong q, ulong r);

    inline int neighbor_boxes(int c_x, int c_y, int c_z, \
            ulong* nbr_boxes, int H);

    inline ulong interleave(ulong p, \
            ulong q, ulong r)
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

    inline int find_idx(__global ulong* keys, \
            int num_particles, ulong key)
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
            ulong* nbr_boxes, int H)
    {
        int nbr_boxes_length = 0;
        int j, k, m;
        ulong key;

        #pragma unroll
        for(j=-H; j<H+1; j++)
        {
            #pragma unroll
            for(k=-H; k<H+1; k++)
            {
                #pragma unroll
                for(m=-H; m<H+1; m++)
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


// IMPORTANT: Check if refactoring of the following kernels is needed for caching
// the result

<%def name="find_nbrs_prep(data_t, sorted)", cached="True">
    uint qid = i;

    // IMPORTANT: Make sure the 'cid' approach works for sorted particle arrays
    //% if sorted:
    //    qid = i;
    //% else:
    //    qid = pids_dst[i];
    //% endif

    ${data_t}4 q = (${data_t}4)(d_x[qid], d_y[qid], d_z[qid], d_h[qid]);
    ${data_t} radius_scale2 = radius_scale*radius_scale;

    int3 c;

    int idx;
    uint j;
    ${data_t} dist;
    ${data_t} h_i = radius_scale2*q.w*q.w;
    ${data_t} h_j;

    ulong key;
    uint pid;

    ${data_t} h_max, cell_size_level;
    int H, level;

    ulong mask;
    int mask_len, nbr_boxes_length;

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
        mask_len = (2*H+1)*(2*H+1)*(2*H+1);

        ulong nbr_boxes[mask_len];

        nbr_boxes_length = neighbor_boxes(c.x, c.y, c.z, nbr_boxes, H);

        #pragma unroll
        for(j=0; j<nbr_boxes_length; j++)
        {
            key = mask + nbr_boxes[j];
            idx = find_idx(keys + start_idx_levels[level],
                num_particles_levels[level], key);
            if(idx == -1)
                continue;
            idx += start_idx_levels[level];
            //key = keys[idx];

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
    }

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
        mask_len = (2*H+1)*(2*H+1)*(2*H+1);

        ulong nbr_boxes[mask_len];

        nbr_boxes_length = neighbor_boxes(c.x, c.y, c.z, nbr_boxes, H);

        #pragma unroll
        for(j=0; j<nbr_boxes_length; j++)
        {
            key = mask + nbr_boxes[j];
            idx = find_idx(keys + start_idx_levels[level],
                num_particles_levels[level], key);
            if(idx == -1)
                continue;
            idx += start_idx_levels[level];
            //key = keys[idx];

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


</%def>


