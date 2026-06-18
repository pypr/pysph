"""Dynamic Warp equation-group code generation (ADR-0003).

Compose SPH physics as small ``WarpEquation`` building blocks and generate a
single fused Warp kernel per ``WarpGroup``, mirroring PySPH's equation/group
transpilation. Fusion is a *property of grouping*: the generator unions the
array signature of all equations in a group, computes the shared per-pair
geometry exactly once, and inlines each equation's per-pair ``loop`` body into a
single neighbor traversal.

The model follows PySPH's group semantics: for each destination particle the
generated kernel loops its neighbors once and, for each pair ``(i, j)``, runs
every equation's ``loop`` snippet in order, each accumulating into a shared
per-output register ``_acc_<out>``. The accumulators are written back to the
destination arrays once after the loop.

A generated kernel is materialized by templating source from the equation
snippets, registering it with :mod:`linecache` so Warp can introspect it, and
wrapping it with ``wp.Kernel(func=..., source=...)``. Compiled kernels are
cached by structural signature ``(ordered equation signatures, dtype)`` so each
unique group compiles only once.
"""

import linecache

import numpy as np

try:
    import warp as wp
except ImportError:  # pragma: no cover
    wp = None


# Shared per-pair quantities the generator can compute once and expose to the
# ``loop`` snippets of every equation in a group.
SHARED_QUANTITIES = (
    'dx', 'dy', 'dz', 'rij2', 'rij', 'hij', 'grad', 'wij',
    'vijx', 'vijy', 'vijz',
)

# Position/smoothing arrays implied by the geometric shared quantities, and the
# velocity arrays implied by the relative-velocity shared quantities. These are
# auto-added to a group's array signature so equation blocks only declare the
# *extra* arrays they read.
_GEOM_POS = ('x', 'y', 'z')
_GEOM_H = ('h',)
_GEOM_VEL = ('u', 'v', 'w')


class WarpEquation(object):
    """Base class for a composable Warp SPH equation block.

    Subclasses declare the particle arrays they read/write and the shared
    per-pair quantities they need, then contribute ``initialize`` / ``loop`` /
    ``post_loop`` source snippets. Snippets are plain Warp-Python source lines
    (already indented relative to the generated body) that may reference:

    - ``i`` (destination particle index) and ``j`` (neighbor index);
    - any shared quantity listed in :attr:`requires` (``dx``, ``rij``, ``hij``,
      ``grad``, ``wij``, ``vijx`` ...);
    - source arrays as ``s_<name>`` and destination arrays as ``d_<name>``;
    - shared output accumulators as ``_acc_<name>`` for each name in
      :attr:`out_arrays` (initialized to zero by the generator, written back
      after the loop);
    - declared scalars by name (e.g. ``alpha``);
    - the ``TYPE`` token, replaced by ``wp.float32`` / ``wp.float64``.

    Block-local temporaries should use a trailing underscore (e.g. ``tmpi_``)
    to avoid colliding with generator-owned names.
    """

    #: extra source (neighbor ``j``) arrays read, beyond geometry/velocity
    src_arrays = ()
    #: extra destination (particle ``i``) arrays read, beyond geometry/velocity
    dst_arrays = ()
    #: destination arrays written (accumulated via ``_acc_<name>``)
    out_arrays = ()
    #: scalar kernel parameters consumed by the snippets, in order
    scalars = ()
    #: subset of :data:`SHARED_QUANTITIES` this block needs in the loop
    requires = ()

    def initialize(self):
        """Per-destination-particle setup emitted before the neighbor loop."""
        return ""

    def loop(self):
        """Per-pair body emitted inside the neighbor loop."""
        return ""

    def post_loop(self):
        """Per-destination-particle code emitted after the neighbor loop."""
        return ""

    def signature(self):
        """Structural cache key fragment. Override if behavior depends on flags
        that change the generated source (not on runtime scalar *values*)."""
        return (type(self).__name__,)


class GroupKernel(object):
    """A compiled fused kernel plus the metadata needed to launch it.

    The ordered name lists fully determine both the generated signature and the
    launch-input order, so a caller binds device arrays/scalars by walking these
    lists in order.
    """

    def __init__(self, kernel, src_names, dst_names, scalar_names, out_names,
                 dtype, source, neighbor_mode='flat'):
        self.kernel = kernel
        self.src_names = src_names      # s_<name> arrays, in signature order
        self.dst_names = dst_names      # d_<name> arrays read, in order
        self.scalar_names = scalar_names
        self.out_names = out_names      # d_<name> arrays written, in order
        self.dtype = dtype
        self.source = source
        self.neighbor_mode = neighbor_mode


def _dtype_tokens(dtype):
    """Return ``(type_token, func_suffix, np_dtype)`` for a requested dtype."""
    if dtype in (np.float32, 'float32', 'f32'):
        return 'wp.float32', 'f32', np.float32
    if dtype in (np.float64, 'float64', 'f64'):
        return 'wp.float64', 'f64', np.float64
    raise ValueError("dtype must be float32 or float64, got %r" % (dtype,))


def _collect(equations, neighbor_mode='flat'):
    """Union the array/scalar signature across a group, preserving order and
    auto-adding the arrays implied by the requested shared quantities.

    In ``grid`` mode the destination/source position and smoothing arrays are
    forced into the signature even if no equation requested them: the
    grid-direct loop needs ``x,y,z`` to locate the destination cell and ``h``
    (with ``radius_scale``) to apply the support cutoff that reproduces the
    flat neighbor list's membership.
    """
    requires = set()
    for eq in equations:
        requires.update(eq.requires)
    unknown = requires - set(SHARED_QUANTITIES)
    if unknown:
        raise ValueError("unknown shared quantities: %s" % sorted(unknown))
    if neighbor_mode == 'grid':
        # The cell walk + support cutoff always needs positions and h.
        requires |= {'dx', 'dy', 'dz', 'rij2'}

    src = []
    dst = []
    scalars = []
    out = []

    def add(seq, name):
        if name not in seq:
            seq.append(name)

    # Geometry-implied arrays first so the signature is stable and readable.
    needs_pos = requires & {'dx', 'dy', 'dz', 'rij2', 'rij', 'grad', 'wij'}
    needs_h = requires & {'hij', 'grad', 'wij'}
    needs_vel = requires & {'vijx', 'vijy', 'vijz'}
    if neighbor_mode == 'grid':
        # The support cutoff reads radius_scale * h on both i and j.
        needs_h = needs_h | {'h'}
    if needs_pos:
        for n in _GEOM_POS:
            add(src, n)
            add(dst, n)
    if needs_h:
        for n in _GEOM_H:
            add(src, n)
            add(dst, n)
    if needs_vel:
        for n in _GEOM_VEL:
            add(src, n)
            add(dst, n)

    for eq in equations:
        for n in eq.src_arrays:
            add(src, n)
        for n in eq.dst_arrays:
            add(dst, n)
        for n in eq.scalars:
            add(scalars, n)
        for n in eq.out_arrays:
            add(out, n)
    return src, dst, scalars, out, requires


def _reindent(text, extra):
    """Prefix every non-blank line of ``text`` with ``extra`` spaces, keeping
    the snippet's internal relative indentation intact."""
    if extra <= 0:
        return text
    pad = " " * extra
    return "\n".join(
        (pad + line) if line.strip() else line
        for line in text.split("\n")
    )


def _emit_geometry(requires, type_token, func_suffix, phase='all'):
    """Emit the shared per-pair geometry lines requested by the group.

    ``phase`` controls which lines are emitted, so the grid-direct loop can
    compute ``dx,dy,dz,rij2`` (``pre``) before the support cutoff and the rest
    (``post``) inside it. ``all`` emits the full sequence in the original order
    (used by the flat path, byte-identical to before this split). All lines are
    emitted at the flat 8-space loop indent; the grid path reindents them.
    """
    emit_pre = phase in ('all', 'pre')
    emit_post = phase in ('all', 'post')
    lines = []
    L = lines.append
    needs_pos = requires & {'dx', 'dy', 'dz', 'rij2', 'rij', 'grad', 'wij'}
    if emit_pre and needs_pos:
        L("        dx = d_x[i] - s_x[j]")
        L("        dy = %s(0.0)" % type_token)
        L("        dz = %s(0.0)" % type_token)
        L("        if dim > wp.int32(1):")
        L("            dy = d_y[i] - s_y[j]")
        L("        if dim > wp.int32(2):")
        L("            dz = d_z[i] - s_z[j]")
    if emit_pre and (requires & {'rij2', 'rij', 'grad', 'wij'}):
        L("        rij2 = dx*dx + dy*dy + dz*dz")
    if emit_post and (requires & {'rij', 'grad', 'wij'}):
        L("        rij = wp.sqrt(rij2)")
    if emit_post and (requires & {'hij', 'grad', 'wij'}):
        L("        hij = %s(0.5) * (d_h[i] + s_h[j])" % type_token)
    if emit_post and 'grad' in requires:
        L("        grad = %s(0.0)" % type_token)
        L("        if rij > %s(1.0e-12):" % type_token)
        L("            grad = _kernel_dwdq_%s(rij, hij, dim, kernel_id)"
          " / (hij * rij)" % func_suffix)
    if emit_post and 'wij' in requires:
        L("        wij = _kernel_value_%s(rij, hij, dim, kernel_id)"
          % func_suffix)
    if emit_post and (requires & {'vijx', 'vijy', 'vijz'}):
        L("        vijx = d_u[i] - s_u[j]")
        L("        vijy = %s(0.0)" % type_token)
        L("        vijz = %s(0.0)" % type_token)
        L("        if dim > wp.int32(1):")
        L("            vijy = d_v[i] - s_v[j]")
        L("        if dim > wp.int32(2):")
        L("            vijz = d_w[i] - s_w[j]")
    return lines


def generate_group_source(equations, dtype, func_name='_warp_group_kernel',
                           neighbor_mode='flat'):
    """Generate the Warp kernel source for a group of equations.

    ``neighbor_mode`` selects the neighbor source: ``flat`` reads a prebuilt CSR
    list (``starts/lengths/neighbors``); ``grid`` walks a uniform-grid cell list
    directly (``cell_starts/cell_counts/cell_particles`` + bounds), applying the
    support cutoff inline so it visits exactly the flat list's neighbor set
    (ADR-0004). The per-pair geometry and every equation snippet are identical
    across modes -- only the loop that produces ``j`` differs.

    Returns ``(source, src_names, dst_names, scalar_names, out_names)``.
    """
    if neighbor_mode not in ('flat', 'grid'):
        raise ValueError("neighbor_mode must be 'flat' or 'grid'")
    type_token, func_suffix, _ = _dtype_tokens(dtype)
    src_names, dst_names, scalar_names, out_names, requires = _collect(
        equations, neighbor_mode=neighbor_mode
    )

    def subst(snippet):
        return snippet.replace('TYPE', type_token)

    lines = []
    L = lines.append

    # --- signature ---
    L("def %s(" % func_name)
    for n in src_names:
        L("        s_%s: wp.array(dtype=%s)," % (n, type_token))
    for n in dst_names:
        L("        d_%s: wp.array(dtype=%s)," % (n, type_token))
    if neighbor_mode == 'flat':
        L("        starts: wp.array(dtype=wp.int32),")
        L("        lengths: wp.array(dtype=wp.int32),")
        L("        neighbors: wp.array(dtype=wp.uint32),")
    else:
        L("        cell_starts: wp.array(dtype=wp.int32),")
        L("        cell_counts: wp.array(dtype=wp.int32),")
        L("        cell_particles: wp.array(dtype=wp.uint32),")
        L("        xmin: %s," % type_token)
        L("        ymin: %s," % type_token)
        L("        zmin: %s," % type_token)
        L("        cell_size: %s," % type_token)
        L("        nx: wp.int32,")
        L("        ny: wp.int32,")
        L("        nz: wp.int32,")
        L("        ncells: wp.int32,")
        L("        radius_scale: %s," % type_token)
    L("        dim: wp.int32,")
    L("        kernel_id: wp.int32,")
    for n in scalar_names:
        L("        %s: %s," % (n, type_token))
    for n in out_names:
        L("        d_%s: wp.array(dtype=%s)," % (n, type_token))
    L("):")

    # --- per-particle preamble ---
    L("    i = wp.tid()")
    for n in out_names:
        L("    _acc_%s = %s(0.0)" % (n, type_token))
    for eq in equations:
        snippet = eq.initialize()
        if snippet:
            L(subst(snippet))

    if neighbor_mode == 'flat':
        # --- single CSR neighbor loop ---
        L("    start = starts[i]")
        L("    stop = start + lengths[i]")
        L("    for pos in range(start, stop):")
        L("        j = wp.int32(neighbors[pos])")
        for line in _emit_geometry(requires, type_token, func_suffix):
            L(line)
        for eq in equations:
            snippet = eq.loop()
            if snippet:
                L(subst(snippet))
    else:
        # --- direct uniform-grid cell-list walk (ADR-0004) ---
        # Geometry/snippet lines are authored at the flat 8-space loop indent;
        # reindent them to sit inside the cell walk (pre-cutoff at +20 -> col
        # 28, post-cutoff body at +24 -> col 32).
        L("    ix0 = wp.int32(wp.floor((d_x[i] - xmin) / cell_size))")
        L("    iy0 = wp.int32(0)")
        L("    iz0 = wp.int32(0)")
        L("    if dim > wp.int32(1):")
        L("        iy0 = wp.int32(wp.floor((d_y[i] - ymin) / cell_size))")
        L("    if dim > wp.int32(2):")
        L("        iz0 = wp.int32(wp.floor((d_z[i] - zmin) / cell_size))")
        L("    for dzc in range(-1, 2):")
        L("        for dyc in range(-1, 2):")
        L("            for dxc in range(-1, 2):")
        L("                ix = ix0 + wp.int32(dxc)")
        L("                iy = iy0 + wp.int32(dyc)")
        L("                iz = iz0 + wp.int32(dzc)")
        L("                if ix >= 0 and ix < nx and iy >= 0 and iy < ny"
          " and iz >= 0 and iz < nz:")
        L("                    cid = ix + iy * nx + iz * nx * ny")
        L("                    if cid >= 0 and cid < ncells:")
        L("                        c_start_ = cell_starts[cid]")
        L("                        c_stop_ = c_start_ + cell_counts[cid]")
        L("                        for pos in range(c_start_, c_stop_):")
        L("                            j = wp.int32(cell_particles[pos])")
        pre = _emit_geometry(requires, type_token, func_suffix, phase='pre')
        for line in pre:
            L(_reindent(line, 20))
        L("                            hi_ = radius_scale * d_h[i]")
        L("                            hj_ = radius_scale * s_h[j]")
        L("                            if rij2 < hi_*hi_ or rij2 < hj_*hj_:")
        post = _emit_geometry(requires, type_token, func_suffix, phase='post')
        for line in post:
            L(_reindent(line, 24))
        for eq in equations:
            snippet = eq.loop()
            if snippet:
                L(_reindent(subst(snippet), 24))

    # --- write back / post-loop ---
    for n in out_names:
        L("    d_%s[i] = _acc_%s" % (n, n))
    for eq in equations:
        snippet = eq.post_loop()
        if snippet:
            L(subst(snippet))

    source = "\n".join(lines) + "\n"
    return source, src_names, dst_names, scalar_names, out_names


# Cache of compiled group kernels, keyed by structural signature.
_KERNEL_CACHE = {}


def _cache_key(equations, dtype, neighbor_mode='flat'):
    type_token, _, _ = _dtype_tokens(dtype)
    return (
        tuple(eq.signature() for eq in equations), type_token, neighbor_mode
    )


def build_group_kernel(equations, dtype, device_funcs, key=None,
                       neighbor_mode='flat'):
    """Generate (or fetch from cache) the fused kernel for ``equations``.

    ``device_funcs`` maps the device ``wp.func`` names referenced by the
    generated geometry (``_kernel_dwdq_f32`` etc.) to their ``wp.Function``
    objects; these seed the namespace the generated function executes in so
    Warp can resolve them. ``neighbor_mode`` (``flat``/``grid``) selects the
    neighbor source and is part of the structural cache key, so the two
    variants of a group compile independently (ADR-0004).

    Returns a :class:`GroupKernel`.
    """
    if wp is None:  # pragma: no cover
        raise ImportError("warp is required for build_group_kernel")

    cache_key = _cache_key(equations, dtype, neighbor_mode)
    cached = _KERNEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if key is None:
        type_token, func_suffix, _ = _dtype_tokens(dtype)
        key = "warp_group_%s_%s_%d" % (
            func_suffix, neighbor_mode, len(_KERNEL_CACHE)
        )
    func_name = key

    source, src_names, dst_names, scalar_names, out_names = (
        generate_group_source(
            equations, dtype, func_name=func_name, neighbor_mode=neighbor_mode
        )
    )

    # Register the source with linecache so Warp can introspect the exec'd
    # function, then compile it into a namespace seeded with wp + device funcs.
    fname = "<warp_codegen:%s>" % key
    linecache.cache[fname] = (
        len(source), None, source.splitlines(True), fname
    )
    namespace = {'wp': wp}
    namespace.update(device_funcs)
    exec(compile(source, fname, 'exec'), namespace)
    fn = namespace[func_name]
    kernel = wp.Kernel(func=fn, key=key, source=source)

    group_kernel = GroupKernel(
        kernel=kernel, src_names=src_names, dst_names=dst_names,
        scalar_names=scalar_names, out_names=out_names, dtype=dtype,
        source=source, neighbor_mode=neighbor_mode,
    )
    _KERNEL_CACHE[cache_key] = group_kernel
    return group_kernel


def clear_kernel_cache():
    """Drop the compiled-group cache (used by tests)."""
    _KERNEL_CACHE.clear()
