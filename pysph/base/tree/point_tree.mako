//CL//
<%def name="preamble(data_t)" cached="True">
    #define IN_BOUNDS(X, MIN, MAX) ((X >= MIN) && (X < MAX))
    #define NORM2(X, Y, Z) ((X)*(X) + (Y)*(Y) + (Z)*(Z))
    #define MIN(X, Y) ((X) < (Y) ? (X) : (Y))
    #define MAX(X, Y) ((X) > (Y) ? (X) : (Y))
    #define SQR(X) ((X) * (X))

    typedef struct {
        union {
            float s0;
            float x;
        };
    } float1;

    typedef struct {
        union {
            double s0;
            double x;
        };
    } double1;
</%def>

<%def name="fill_particle_data_args(data_t, xvars, dim)" cached="False">
    % for v in xvars:
    ${data_t}* ${v},
    % endfor
    ${data_t} cell_size, ${data_t}${dim} min,
    unsigned long* keys, unsigned int* pids
</%def>


<%def name="fill_particle_data_src(data_t, xvars, dim)" cached="False">
    % for v in xvars:
    unsigned long c_${v} = floor((${v}[i] - min.${v}) / cell_size);
    % endfor
    unsigned long key;
    ## For 3D: key = interleave3(c_x, c_y, c_z);
    key = interleave${dim}(${', '.join('c_' + v for v in xvars)});
    keys[i] = key;
    pids[i] = i;
</%def>


<%def name="find_neighbors_template(data_t, sorted, wgs)" cached="False">
    /*
     * Property owners
     * tree_dst: cids, unique_cids
     * tree_src: neighbor_cid_offset, neighbor_cids
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
     * tree_dst: cids, unique_cid_idx
     * tree_src: neighbor_cid_offset, neighbor_cids
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