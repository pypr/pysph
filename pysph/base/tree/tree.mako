//CL//
<%def name="preamble(data_t)" cached="False">
    char eye_index(ulong sfc, ulong mask, char rshift) {
        return ((sfc & mask) >> rshift);
    }
</%def>

<%def name="reorder_particles_args(k, data_vars, data_var_ctypes, const_vars,
const_var_ctypes, index_code)" cached="False">
    int *pids, int *cids, char *seg_flag,
    uint2 *pbounds, int *offsets, uint${k} *octant_vector,
    int *pids_next, int *cids_next,
    % for var, ctype in zip(data_vars, data_var_ctypes):
        ${ctype} *${var},
    % endfor
    % for var, ctype in zip(data_vars, data_var_ctypes):
        ${ctype} *${var}_next,
    % endfor
    % for var, ctype in zip(const_vars, const_var_ctypes):
        ${ctype} ${var},
    % endfor
    uint csum_nodes_prev
</%def>

<%def name="reorder_particles_src(k, data_vars, data_var_ctypes, const_vars,
const_var_ctypes, index_code)" cached="False">
    int curr_cid = cids[i] - csum_nodes_prev;
    if (curr_cid < 0 || offsets[curr_cid] == -1) {
        cids_next[i] = cids[i];
        pids_next[i] = pids[i];

        % for var in data_vars:
            ${var}_next[i] = ${var}[i];
        % endfor
    } else {
        uint2 pbound_here = pbounds[curr_cid];
        char octant = (${index_code});

        global uint *octv = (global uint *)(octant_vector + i);
        int sum = octv[octant];
        sum -= (octant == 0) ? 0 : octv[octant - 1];
        octv = (global uint *)(octant_vector + pbound_here.s1 - 1);
        sum += (octant == 0) ? 0 : octv[octant - 1];

        uint new_index = pbound_here.s0 + sum - 1;

        pids_next[new_index] = pids[i];
        cids_next[new_index] = offsets[curr_cid] + octant;

        % for var in data_vars:
            ${var}_next[new_index] = ${var}[i];
        % endfor
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

<%def name="set_node_data_args(k)", cached="False">
    int *offsets_prev, uint2 *pbounds_prev,
    int *offsets, uint2 *pbounds,
    char *seg_flag, uint${k} *octant_vector,
    uint csum_nodes, uint N
</%def>

<%def name="set_node_data_src(k)", cached="False">
    uint2 pbound_here = pbounds_prev[i];
    int child_offset = offsets_prev[i];
    if (child_offset == -1) {
        PYOPENCL_ELWISE_CONTINUE;
    }
    child_offset -= csum_nodes;

    uint${k} octv = octant_vector[pbound_here.s1 - 1];

    % for i in range(k):
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