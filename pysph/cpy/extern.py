class Extern(object):
    """A simple way to support external functions and symbols.
    """
    def link(self, backend):
        """Return a list of extra link args."""
        return []

    def code(self, backend):
        """Return suitable code as a string.

        This code is injected at the top of the generated code.
        """
        raise NotImplementedError()

    def __call__(self, *args, **kw):
        """Implement for a pure Python implementation if needed.
        """
        raise NotImplementedError()


class _printf(Extern):
    def code(self, backend):
        # Always available so no need but in Cython we explicitly define it as
        # an example.

        if backend == 'cython':
            return 'from libc.studio cimport printf'
        return ''

    def __call__(self, *args):
        print(args[0] % args[1:])


# Now make it available publicly.
printf = _printf()

# More examples are available in the low_level.py module.


def get_extern_code(externs, backend):
    links = []
    code = []
    for ex in externs:
        link = ex.link(backend)
        if link:
            links.extend(link)
        c = ex.code(backend)
        if c:
            code.append(c)

    return links, code
