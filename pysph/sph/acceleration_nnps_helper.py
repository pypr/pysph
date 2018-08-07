import sys
from mako.template import Template

disable_unicode = False if sys.version_info.major > 2 else True

NNPS_TEMPLATE = r"""

    /*
     * Property owners
     * octree_dst: cids, unique_cids
     * octree_src: neighbor_cid_offset, neighbor_cids
     * self evident: xsrc, ysrc, zsrc, hsrc,
                     xdst, ydst, zdst, hdst,
                     pbounds_src, pbounds_dst,
     */
    long i = get_global_id(0);
    int lid = get_local_id(0);
    int _idx = get_group_id(0);

    // Fetch dst particles
    ${data_t} _xd, _yd, _zd, _hd;

    int _cid_dst = _unique_cids[_idx];
    uint2 _pbound_here = _pbounds_dst[_cid_dst];
    char _svalid = (_pbound_here.s0 + lid < _pbound_here.s1);
    unsigned int d_idx;

    if (_svalid) {
        % if sorted:
        d_idx = _pbound_here.s0 + lid;
        % else:
        d_idx = _pids_dst[_pbound_here.s0 + lid];
        % endif

        _xd = d_x[d_idx];
        _yd = d_y[d_idx];
        _zd = d_z[d_idx];
        _hd = d_h[d_idx];
    }

    // Set loop parameters
    int _cid_src, _pid_src;
    int _offset_src = _neighbor_cid_offset[_idx];
    int _offset_lim = _neighbor_cid_offset[_idx + 1];
    uint2 _pbound_here2;

    % for var, type in zip(vars, types):
    local ${type} ${var}[${lmem_size}];
    % endfor

    char _nbrs[${lmem_size}];
    int _nbr_cnt, _m, _nbr_saved;
    ${setup}
    while (_offset_src < _offset_lim) {
        _nbr_saved = 0;
        while (_offset_src < _offset_lim) {
            _cid_src = _neighbor_cids[_offset_src];
            _pbound_here2 = _pbounds_src[_cid_src];
            _m = min(_pbound_here2.s1,
                    _pbound_here2.s0 + ${wgs}) - _pbound_here2.s0;
            if (_m + _nbr_saved > ${lmem_size})
                break;

            // Copy src data
            if (lid < _m) {

                %if sorted:
                _pid_src = _pbound_here2.s0 + lid;
                % else:
                _pid_src = _pids_src[_pbound_here2.s0 + lid];
                %endif


                % for var in vars:
                ${var}[_nbr_saved + lid] = ${var}_global[_pid_src];
                % endfor
            }
            _nbr_saved += _m;
            _offset_src++;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Everything this point forward is done independently
        // by each thread.
        if (_svalid) {
            _nbr_cnt = 0;
            for (int _j=0; _j < _nbr_saved; _j++) {
                ${data_t} _dist2 = NORM2(s_x[_j] - _xd,
                            s_y[_j] - _yd,
                            s_z[_j] - _zd);

                ${data_t} _r2 = MAX(s_h[_j], _hd) * _radius_scale;
                _r2 *= _r2;

                if (_dist2 < _r2) {
                    _nbrs[_nbr_cnt++] = _j;
                }
            }

            int _j = 0;
            while (_j < _nbr_cnt) {
                int s_idx = _nbrs[_j];
                ${loop_code}
                _j++;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

"""

NNPS_ARGS_TEMPLATE = """
    __global int *_unique_cids, __global int *_pids_src, __global int
    *_pids_dst,
    __global int *_cids,
    __global uint2 *_pbounds_src, __global uint2 *_pbounds_dst,
    %(data_t)s _radius_scale,
    __global int *_neighbor_cid_offset, __global int *_neighbor_cids
    """


def _generate_nnps_code(sorted, wgs, setup, loop, vars, types,
                        data_t='float'):
    # Note: Properties like the data type and sortedness
    # need to be fixed throughout the simulation since
    # currently this function is only called at the start of
    # the simulation.
    chunksize = 3
    return Template(NNPS_TEMPLATE, disable_unicode=disable_unicode).render(
        data_t=data_t, sorted=sorted, wgs=wgs, setup=setup, loop_code=loop,
        vars=vars, types=types, chunksize=chunksize, lmem_size=wgs*chunksize
    )


def generate_body(setup, loop, vars, types, wgs, c_type):
    return _generate_nnps_code(True, wgs, setup, loop, vars, types,
                               c_type)


def get_kernel_args_list(c_type='float'):
    args = NNPS_ARGS_TEMPLATE % {'data_t': c_type}
    return args.split(",")
