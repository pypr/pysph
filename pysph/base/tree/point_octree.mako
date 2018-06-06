//CL//
<%def name="preamble(data_t)" cached="True">
    #define IN_BOUNDS(X, MIN, MAX) ((X >= MIN) && (X < MAX))
    #define NORM2(X, Y, Z) ((X)*(X) + (Y)*(Y) + (Z)*(Z))
    #define MIN(X, Y) ((X) < (Y) ? (X) : (Y))
    #define MAX(X, Y) ((X) > (Y) ? (X) : (Y))
    #define AVG(X, Y) (((X) + (Y)) / 2)
    #define ABS(X) ((X) > 0 ? (X) : -(X))
    #define EPS 1e-6
    #define INF 1e6
    #define SQR(X) ((X) * (X))

    char eye_index(ulong sfc, ulong mask, char rshift) {
        return ((sfc & mask) >> rshift);
    }
</%def>

<%def name="fill_particle_data_args(data_t)" cached="False">
    ${data_t}* x, ${data_t}* y, ${data_t}* z,
    ${data_t} cell_size, ${data_t}3 min,
    unsigned long* keys, unsigned int* pids
</%def>

<%def name="fill_particle_data_src(data_t)" cached="False">
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

<%def name="reorder_particles_args()" cached="False">
    int *pids,
    ulong *sfc, int *cids,
    char *seg_flag,
    uint2 *pbounds,
    int *offsets,
    uint8 *octant_vector,
    int *pids_next, ulong *sfc_next,
    int *cids_next,
    ulong mask, char rshift,
    uint csum_nodes_prev
</%def>

<%def name="reorder_particles_src()" cached="False">
    int curr_cid = cids[i] - csum_nodes_prev;
    if (curr_cid < 0 || offsets[curr_cid] == -1) {
        sfc_next[i] = sfc[i];
        cids_next[i] = cids[i];
        pids_next[i] = pids[i];
    } else {
        uint2 pbound_here = pbounds[curr_cid];
        char octant = eye_index(sfc[i], mask, rshift);

        global uint *octv = (global uint *)(octant_vector + i);
        int sum = octv[octant];
        sum -= (octant == 0) ? 0 : octv[octant - 1];
        octv = (global uint *)(octant_vector + pbound_here.s1 - 1);
        sum += (octant == 0) ? 0 : octv[octant - 1];

        uint new_index = pbound_here.s0 + sum - 1;
        sfc_next[new_index] = sfc[i];
        pids_next[new_index] = pids[i];
        cids_next[new_index] = offsets[curr_cid] + octant;
    }
</%def>

<%def name="append_layer_args()" cached="False">
    int *offsets_next, uint2 *pbounds_next,
	int *offsets, uint2 *pbounds,
    int curr_offset, char is_last_level
</%def>

<%def name="append_layer_src()" cached="False">
	pbounds[curr_offset + i] = pbounds_next[i];
    offsets[curr_offset + i] = is_last_level ? -1 : offsets_next[i];
</%def>

<%def name="set_node_data_args()", cached="False">
    int *offsets_prev, uint2 *pbounds_prev,
    int *offsets, uint2 *pbounds,
    char *seg_flag, uint8 *octant_vector,
    uint csum_nodes, uint N
</%def>

<%def name="set_node_data_src()", cached="False">
    uint2 pbound_here = pbounds_prev[i];
    int child_offset = offsets_prev[i];
    if (child_offset == -1) {
        PYOPENCL_ELWISE_CONTINUE;
    }
    child_offset -= csum_nodes;

    uint8 octv = octant_vector[pbound_here.s1 - 1];

    % for i in range(8):
        % if i == 0:
            pbounds[child_offset] = (uint2)(pbound_here.s0, pbound_here.s0 + octv.s0);
        % else:
            pbounds[child_offset + ${i}] = (uint2)(pbound_here.s0 + octv.s${i - 1},
                                                   pbound_here.s0 + octv.s${i});
			if (pbound_here.s0 + octv.s${i - 1} < N)
               seg_flag[pbound_here.s0 + octv.s${i - 1}] = 1;
        % endif
    % endfor

</%def>


<%def name="find_neighbors_template(data_t, sorted, wgs)" cached="False">
    /*
     * Property owners
     * octree_dst: cids, unique_cids
     * octree_src: neighbor_cid_offset, neighbor_cids
     * self evident: xsrc, ysrc, zsrc, hsrc,
                     xdst, ydst, zdst, hdst,
                     pbounds_src, pbounds_dst,
     */
    int idx = i / ${wgs};

    // Fetch dst particles
    ${data_t} xd, yd, zd, hd;

    int cid_dst = unique_cids[idx];
    uint2 pbound_here = pbounds_dst[cid_dst];
    char svalid = (pbound_here.s0 + lid < pbound_here.s1);
    int pid_dst;

    if (svalid) {
        % if sorted:
            pid_dst = pbound_here.s0 + lid;
        % else:
            pid_dst = pids_dst[pbound_here.s0 + lid];
        % endif

        xd = xdst[pid_dst];
        yd = ydst[pid_dst];
        zd = zdst[pid_dst];
        hd = hdst[pid_dst];
    }

    // Set loop parameters
    int cid_src, pid_src;
    int offset_src = neighbor_cid_offset[idx];
    int offset_lim = neighbor_cid_offset[idx + 1];
    uint2 pbound_here2;
    int m;
    local ${data_t} xs[${wgs}];
    local ${data_t} ys[${wgs}];
    local ${data_t} zs[${wgs}];
    local ${data_t} hs[${wgs}];
    ${data_t} r2;


    ${caller.pre_loop()}
    while (offset_src < offset_lim) {
        cid_src = neighbor_cids[offset_src];
        pbound_here2 = pbounds_src[cid_src];

        offset_src++;

        while (pbound_here2.s0 < pbound_here2.s1) {
            // Copy src data
            if (pbound_here2.s0 + lid < pbound_here2.s1) {
                %if sorted:
                    pid_src = pbound_here2.s0 + lid;
                % else:
                    pid_src = pids_src[pbound_here2.s0 + lid];
                %endif
                xs[lid] = xsrc[pid_src];
                ys[lid] = ysrc[pid_src];
                zs[lid] = zsrc[pid_src];
                hs[lid] = hsrc[pid_src];
            }
            m = min(pbound_here2.s1, pbound_here2.s0 + ${wgs}) - pbound_here2.s0;

            barrier(CLK_LOCAL_MEM_FENCE);

            // Everything this point forward is done independently
            // by each thread.
            if (svalid) {
                for (int j=0; j < m; j++) {
                    % if sorted:
                        pid_src= pbound_here2.s0 + j;
                    % else:
                        pid_src = pids_src[pbound_here2.s0 + j];
                    % endif
                    ${data_t} dist2 = NORM2(xs[j] - xd,
                                            ys[j] - yd,
                                            zs[j] - zd);

                    r2 = MAX(hs[j], hd) * radius_scale;
                    r2 *= r2;
                    if (dist2 < r2) {
                        ${caller.query()}
                    }
                }
            }
            pbound_here2.s0 += ${wgs};
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
    ${caller.post_loop()}
</%def>

<%def name="find_neighbor_counts_args(data_t, sorted, wgs)" cached="False">
    int *unique_cids, int *pids_src, int *pids_dst, int *cids,
    uint2 *pbounds_src, uint2 *pbounds_dst,
    ${data_t} *xsrc, ${data_t} *ysrc, ${data_t} *zsrc, ${data_t} *hsrc,
    ${data_t} *xdst, ${data_t} *ydst, ${data_t} *zdst, ${data_t} *hdst,
    ${data_t} radius_scale,
    int *neighbor_cid_offset, int *neighbor_cids,
    int *neighbor_counts
</%def>
<%def name="find_neighbor_counts_src(data_t, sorted, wgs)" cached="False">
     <%self:find_neighbors_template data_t="${data_t}" sorted="${sorted}" wgs="${wgs}">
        <%def name="pre_loop()">
            int count = 0;
        </%def>
        <%def name="query()">
            count++;
        </%def>
        <%def name="post_loop()">
            if(svalid)
                neighbor_counts[pid_dst] = count;
        </%def>
    </%self:find_neighbors_template>
</%def>

<%def name="find_neighbors_args(data_t, sorted, wgs)" cached="False">
    int *unique_cids, int *pids_src, int *pids_dst, int *cids,
    uint2 *pbounds_src, uint2 *pbounds_dst,
    ${data_t} *xsrc, ${data_t} *ysrc, ${data_t} *zsrc, ${data_t} *hsrc,
    ${data_t} *xdst, ${data_t} *ydst, ${data_t} *zdst, ${data_t} *hdst,
    ${data_t} radius_scale,
    int *neighbor_cid_offset, int *neighbor_cids,
    int *neighbor_counts, int *neighbors
</%def>
<%def name="find_neighbors_src(data_t, sorted, wgs)" cached="False">
     <%self:find_neighbors_template data_t="${data_t}" sorted="${sorted}" wgs="${wgs}">
        <%def name="pre_loop()">
            int offset;
            if (svalid)
                offset = neighbor_counts[pid_dst];
        </%def>
        <%def name="query()">
            if (svalid)
                neighbors[offset++] = pid_src;
        </%def>
        <%def name="post_loop()">
        </%def>
    </%self:find_neighbors_template>
</%def>

<%def name="find_neighbors_elementwise_template(data_t, sorted)" cached="False">
    /*
     * Property owners
     * octree_dst: cids, unique_cid_idx
     * octree_src: neighbor_cid_offset, neighbor_cids
     * self evident: xsrc, ysrc, zsrc, hsrc,
                     xdst, ydst, zdst, hdst,
                     pbounds_src, pbounds_dst,
     */
    int idx = unique_cids_map[i];

    // Fetch dst particles
    ${data_t} xd, yd, zd, hd;

    int pid_dst;

    % if sorted:
        pid_dst = i;
    % else:
        pid_dst = pids_dst[i];
    % endif

    xd = xdst[pid_dst];
    yd = ydst[pid_dst];
    zd = zdst[pid_dst];
    hd = hdst[pid_dst];


    // Set loop parameters
    int cid_src, pid_src;
    int offset_src = neighbor_cid_offset[idx];
    int offset_lim = neighbor_cid_offset[idx + 1];
    uint2 pbound_here2;
    ${data_t} r2;


    ${caller.pre_loop()}
    while (offset_src < offset_lim) {
        cid_src = neighbor_cids[offset_src];
        pbound_here2 = pbounds_src[cid_src];
        offset_src++;

        for (int j=pbound_here2.s0; j < pbound_here2.s1; j++) {
            % if sorted:
                pid_src= j;
            % else:
                pid_src = pids_src[j];
            % endif
            ${data_t} dist2 = NORM2(xsrc[pid_src] - xd,
                                    ysrc[pid_src] - yd,
                                    zsrc[pid_src] - zd);

            r2 = MAX(hsrc[pid_src], hd) * radius_scale;
            r2 *= r2;
            if (dist2 < r2) {
                ${caller.query()}
            }

        }
    }
    ${caller.post_loop()}
</%def>

<%def name="find_neighbor_counts_elementwise_args(data_t, sorted)" cached="False">
    int *unique_cids_map, int *pids_src, int *pids_dst, int *cids,
    uint2 *pbounds_src, uint2 *pbounds_dst,
    ${data_t} *xsrc, ${data_t} *ysrc, ${data_t} *zsrc, ${data_t} *hsrc,
    ${data_t} *xdst, ${data_t} *ydst, ${data_t} *zdst, ${data_t} *hdst,
    ${data_t} radius_scale,
    int *neighbor_cid_offset, int *neighbor_cids,
    int *neighbor_counts
</%def>
<%def name="find_neighbor_counts_elementwise_src(data_t, sorted)" cached="False">
     <%self:find_neighbors_elementwise_template data_t="${data_t}" sorted="${sorted}">
        <%def name="pre_loop()">
            int count = 0;
        </%def>
        <%def name="query()">
            count++;
        </%def>
        <%def name="post_loop()">
            neighbor_counts[pid_dst] = count;
        </%def>
    </%self:find_neighbors_elementwise_template>
</%def>

<%def name="find_neighbors_elementwise_args(data_t, sorted)" cached="False">
    int *unique_cids_map, int *pids_src, int *pids_dst, int *cids,
    uint2 *pbounds_src, uint2 *pbounds_dst,
    ${data_t} *xsrc, ${data_t} *ysrc, ${data_t} *zsrc, ${data_t} *hsrc,
    ${data_t} *xdst, ${data_t} *ydst, ${data_t} *zdst, ${data_t} *hdst,
    ${data_t} radius_scale,
    int *neighbor_cid_offset, int *neighbor_cids,
    int *neighbor_counts, int *neighbors
</%def>
<%def name="find_neighbors_elementwise_src(data_t, sorted)" cached="False">
     <%self:find_neighbors_elementwise_template data_t="${data_t}" sorted="${sorted}">
        <%def name="pre_loop()">
            int offset = neighbor_counts[pid_dst];
        </%def>
        <%def name="query()">
            neighbors[offset++] = pid_src;
        </%def>
        <%def name="post_loop()">
        </%def>
    </%self:find_neighbors_elementwise_template>
</%def>