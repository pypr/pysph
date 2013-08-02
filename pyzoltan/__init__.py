def get_include():
    """Return the directory in the package that contains header files."""
    from os.path import dirname, abspath, join
    return abspath( join(dirname(__file__), '../') )

