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

        _xd = xd[d_idx];
        _yd = yd[d_idx];
        _zd = zd[d_idx];
        _hd = hd[d_idx];
    }

    // Set loop parameters
    int _cid_src, _pid_src;
    int _offset_src = _neighbor_cid_offset[_idx];
    int _offset_lim = _neighbor_cid_offset[_idx + 1];
    uint2 _pbound_here2;

    LOCAL_MEM ${data_t} _xs[${wgs}];
    LOCAL_MEM ${data_t} _ys[${wgs}];
    LOCAL_MEM ${data_t} _zs[${wgs}];
    LOCAL_MEM ${data_t} _hs[${wgs}];

    % for var, type in zip(vars, types):
    LOCAL_MEM ${type} ${var}[${wgs}];
    % endfor

    char _nbrs[${wgs}];
    int _nbr_cnt, _m;

    ${setup}

    while (_offset_src < _offset_lim) {
        _cid_src = _neighbor_cids[_offset_src];
        _pbound_here2 = _pbounds_src[_cid_src];

        while (_pbound_here2.s0 < _pbound_here2.s1) {
            _m = min(_pbound_here2.s1,
                _pbound_here2.s0 + ${wgs}) - _pbound_here2.s0;

            // Copy src data
            if (lid < _m) {

                % if sorted:
                _pid_src = _pbound_here2.s0 + lid;
                % else:
                _pid_src = _pids_src[_pbound_here2.s0 + lid];
                % endif

                _xs[lid] = xs[_pid_src];
                _ys[lid] = ys[_pid_src];
                _zs[lid] = zs[_pid_src];
                _hs[lid] = hs[_pid_src];

                % for var in vars:
                ${var}[lid] = ${var}_global[_pid_src];
                % endfor
            }
            local_barrier();

            // Everything this point forward is done independently
            // by each thread.
            if (_svalid) {
                _nbr_cnt = 0;
                for (int _j=0; _j < _m; _j++) {
                    ${data_t} _dist2 = NORM2(_xs[_j] - _xd,
                                _ys[_j] - _yd,
                                _zs[_j] - _zd);

                    ${data_t} _r2 = MAX(_hs[_j], _hd) * _radius_scale;
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
            local_barrier();
            _pbound_here2.s0 += ${wgs};
        }

        // Process next neighboring cell
        _offset_src++;
    }

"""

NNPS_ARGS_TEMPLATE = """
    GLOBAL_MEM int *_unique_cids, GLOBAL_MEM int *_pids_src, GLOBAL_MEM int
    *_pids_dst,
    GLOBAL_MEM int *_cids,
    GLOBAL_MEM uint2 *_pbounds_src, GLOBAL_MEM uint2 *_pbounds_dst,
    %(data_t)s _radius_scale,
    GLOBAL_MEM int *_neighbor_cid_offset, GLOBAL_MEM int *_neighbor_cids,
    GLOBAL_MEM %(data_t)s *xd, GLOBAL_MEM %(data_t)s *yd,
    GLOBAL_MEM %(data_t)s *zd, GLOBAL_MEM %(data_t)s *hd,
    GLOBAL_MEM %(data_t)s *xs, GLOBAL_MEM %(data_t)s *ys,
    GLOBAL_MEM %(data_t)s *zs, GLOBAL_MEM %(data_t)s *hs
    """


def _generate_nnps_code(sorted, wgs, setup, loop, vars, types,
                        data_t='float'):
    # Note: Properties like the data type and sortedness
    # need to be fixed throughout the simulation since
    # currently this function is only called at the start of
    # the simulation.
    return Template(NNPS_TEMPLATE, disable_unicode=disable_unicode).render(
        data_t=data_t, sorted=sorted, wgs=wgs, setup=setup, loop_code=loop,
        vars=vars, types=types
    )


def generate_body(setup, loop, vars, types, wgs, c_type='float'):
    return _generate_nnps_code(True, wgs, setup, loop, vars, types,
                               c_type)


def get_kernel_args_list(c_type='float'):
    args = NNPS_ARGS_TEMPLATE % {'data_t': c_type}
    return args.split(",")
