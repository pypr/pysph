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
                 dtype, source):
        self.kernel = kernel
        self.src_names = src_names      # s_<name> arrays, in signature order
        self.dst_names = dst_names      # d_<name> arrays read, in order
        self.scalar_names = scalar_names
        self.out_names = out_names      # d_<name> arrays written, in order
        self.dtype = dtype
        self.source = source


def _dtype_tokens(dtype):
    """Return ``(type_token, func_suffix, np_dtype)`` for a requested dtype."""
    if dtype in (np.float32, 'float32', 'f32'):
        return 'wp.float32', 'f32', np.float32
    if dtype in (np.float64, 'float64', 'f64'):
        return 'wp.float64', 'f64', np.float64
    raise ValueError("dtype must be float32 or float64, got %r" % (dtype,))


def _collect(equations):
    """Union the array/scalar signature across a group, preserving order and
    auto-adding the arrays implied by the requested shared quantities."""
    requires = set()
    for eq in equations:
        requires.update(eq.requires)
    unknown = requires - set(SHARED_QUANTITIES)
    if unknown:
        raise ValueError("unknown shared quantities: %s" % sorted(unknown))

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


def _emit_geometry(requires, type_token, func_suffix):
    """Emit the shared per-pair geometry lines requested by the group."""
    lines = []
    L = lines.append
    needs_pos = requires & {'dx', 'dy', 'dz', 'rij2', 'rij', 'grad', 'wij'}
    if needs_pos:
        L("        dx = d_x[i] - s_x[j]")
        L("        dy = %s(0.0)" % type_token)
        L("        dz = %s(0.0)" % type_token)
        L("        if dim > wp.int32(1):")
        L("            dy = d_y[i] - s_y[j]")
        L("        if dim > wp.int32(2):")
        L("            dz = d_z[i] - s_z[j]")
    if requires & {'rij2', 'rij', 'grad', 'wij'}:
        L("        rij2 = dx*dx + dy*dy + dz*dz")
    if requires & {'rij', 'grad', 'wij'}:
        L("        rij = wp.sqrt(rij2)")
    if requires & {'hij', 'grad', 'wij'}:
        L("        hij = %s(0.5) * (d_h[i] + s_h[j])" % type_token)
    if 'grad' in requires:
        L("        grad = %s(0.0)" % type_token)
        L("        if rij > %s(1.0e-12):" % type_token)
        L("            grad = _kernel_dwdq_%s(rij, hij, dim, kernel_id)"
          " / (hij * rij)" % func_suffix)
    if 'wij' in requires:
        L("        wij = _kernel_value_%s(rij, hij, dim, kernel_id)"
          % func_suffix)
    if requires & {'vijx', 'vijy', 'vijz'}:
        L("        vijx = d_u[i] - s_u[j]")
        L("        vijy = %s(0.0)" % type_token)
        L("        vijz = %s(0.0)" % type_token)
        L("        if dim > wp.int32(1):")
        L("            vijy = d_v[i] - s_v[j]")
        L("        if dim > wp.int32(2):")
        L("            vijz = d_w[i] - s_w[j]")
    return lines


def generate_group_source(equations, dtype, func_name='_warp_group_kernel'):
    """Generate the Warp kernel source for a group of equations.

    Returns ``(source, src_names, dst_names, scalar_names, out_names)``.
    """
    type_token, func_suffix, _ = _dtype_tokens(dtype)
    src_names, dst_names, scalar_names, out_names, requires = _collect(
        equations
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
    L("        starts: wp.array(dtype=wp.int32),")
    L("        lengths: wp.array(dtype=wp.int32),")
    L("        neighbors: wp.array(dtype=wp.uint32),")
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

    # --- single neighbor loop ---
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


def _cache_key(equations, dtype):
    type_token, _, _ = _dtype_tokens(dtype)
    return (tuple(eq.signature() for eq in equations), type_token)


def build_group_kernel(equations, dtype, device_funcs, key=None):
    """Generate (or fetch from cache) the fused kernel for ``equations``.

    ``device_funcs`` maps the device ``wp.func`` names referenced by the
    generated geometry (``_kernel_dwdq_f32`` etc.) to their ``wp.Function``
    objects; these seed the namespace the generated function executes in so
    Warp can resolve them.

    Returns a :class:`GroupKernel`.
    """
    if wp is None:  # pragma: no cover
        raise ImportError("warp is required for build_group_kernel")

    cache_key = _cache_key(equations, dtype)
    cached = _KERNEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if key is None:
        type_token, func_suffix, _ = _dtype_tokens(dtype)
        key = "warp_group_%s_%d" % (func_suffix, len(_KERNEL_CACHE))
    func_name = key

    source, src_names, dst_names, scalar_names, out_names = (
        generate_group_source(equations, dtype, func_name=func_name)
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
        source=source,
    )
    _KERNEL_CACHE[cache_key] = group_kernel
    return group_kernel


def clear_kernel_cache():
    """Drop the compiled-group cache (used by tests)."""
    _KERNEL_CACHE.clear()
