"""List and run PySPH examples.

One can optionally supply the name of the example and any additional arguments.

"""

from __future__ import print_function

import argparse
import ast
import os
import sys

HERE = os.path.dirname(__file__)


def _exec_file(filename):
    ns = {'__name__': '__main__', '__file__': filename}
    if sys.version_info[0] > 2:
        co = compile(open(filename, 'rb').read(), filename, 'exec')
        exec(co, ns)
    else:
        execfile(filename, ns)


def _extract_full_doc(filename):
    p = ast.parse(open(filename, 'rb').read())
    return ast.get_docstring(p)


def _extract_short_doc(dirname, fname):
    return open(os.path.join(dirname, fname)).readline()[3:].strip()


def _get_module(fname):
    start = fname
    parts = ['pysph.examples']
    while os.path.dirname(start) != '':
        dirname, start = os.path.split(start)
        parts.append(dirname)
    return '.'.join(parts + [start[:-3]])


def example_info(module, filename):
    print("Information for example: %s" % module)
    print(_extract_full_doc(filename))


def get_all_examples():
    basedir = HERE
    examples = []
    _ignore = [['run.py'], ['ghia_cavity_data.py'], ['db_exp_data.py'],
               ['tests', 'test_examples.py'],
               ['tests', 'test_riemann_solver.py'],
               ['gas_dynamics', 'shocktube_setup.py'],
               ['gas_dynamics', 'riemann_2d_config.py'],
               ['sphysics', 'beach_geometry.py'],
               ['sphysics', 'periodic_rigidbody.py']]
    ignore = [os.path.abspath(os.path.join(basedir, *pth))
              for pth in _ignore]
    for dirpath, dirs, files in os.walk(basedir):
        rel_dir = os.path.relpath(dirpath, basedir)
        if rel_dir == '.':
            rel_dir = ''
        py_files = [x for x in files
                    if x.endswith('.py') and not x.startswith('_')]
        data = []
        for f in py_files:
            path = os.path.join(rel_dir, f)
            full_path = os.path.join(basedir, path)
            if os.path.abspath(full_path) in ignore:
                continue
            module = _get_module(path)
            doc = _extract_short_doc(dirpath, f)
            data.append((module, doc))
        examples.extend(data)
    return examples


def get_input(prompt):
    if sys.version_info[0] > 2:
        return input(prompt)
    else:
        return raw_input(prompt)


def get_path(module):
    """Return the path to the module filename given the module.
    """
    x = module[len('pysph.examples.'):].split('.')
    x[-1] = x[-1] + '.py'
    return os.path.join(HERE, *x)


def guess_correct_module(example):
    """Given some form of the example name guess and return a reasonable
    module.

    Examples
    --------

    >>> guess_correct_module('elliptical_drop')
    'pysph.examples.elliptical_drop'
    >>> guess_correct_module('pysph.examples.elliptical_drop')
    'pysph.examples.elliptical_drop'
    >>> guess_correct_module('solid_mech.rings')
    'pysph.examples.solid_mech.rings'
    >>> guess_correct_module('solid_mech/rings.py')
    'pysph.examples.solid_mech.rings'
    >>> guess_correct_module('solid_mech/rings')
    'pysph.examples.solid_mech.rings'
    """
    if example.endswith('.py'):
        example = example[:-3]
    example = example.replace('/', '.')
    if not example.startswith('pysph.examples.'):
        module = 'pysph.examples.' + example
    else:
        module = example
    return module


def cat_example(module):
    filename = get_path(module)
    print("# File: %s" % filename)
    print(open(filename).read())


def list_examples(examples):
    for idx, (module, doc) in enumerate(examples):
        print("%d. %s" % (idx + 1, module[len('pysph.examples.'):]))
        print("   %s" % doc)


def run_command(module, args):
    print("Running example %s.\n" % module)
    filename = get_path(module)
    if '-h' not in args and '--help' not in args:
        example_info(module, filename)

    # FIXME: This is ugly but we want the user to be able to run
    #   mpirun -np 4 pysph run elliptical_drop
    # This necessitates that we do not use subprocess.  The cleaner alternative
    # is to expect each user to write a main function which accepts args that
    # we can call.  For now we just clobber sys.argv.

    sys.argv = [filename] + args
    _exec_file(filename)


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    examples = get_all_examples()
    parser = argparse.ArgumentParser(
        prog="run", description=__doc__, add_help=False
    )
    parser.add_argument(
        "-h", "--help", action="store_true", default=False, dest="help",
        help="show this help message and exit"
    )
    parser.add_argument(
        "-l", "--list", action="store_true", default=False, dest="list",
        help="List examples"
    )
    parser.add_argument(
        "--cat", action="store_true", default=False, dest="cat",
        help="Show/cat the example code on stdout"
    )
    parser.add_argument(
        "args", type=str, nargs="?",
        help='''optional example name (for example both cavity or
        pysph.examples.cavity will work) and arguments to the example.'''
    )

    if len(argv) > 0 and argv[0] in ['-h', '--help']:
        parser.print_help()
        sys.exit()

    options, extra = parser.parse_known_args(argv)
    if options.list:
        return list_examples(examples)
    if options.cat:
        module = guess_correct_module(options.args)
        return cat_example(module)
    if len(argv) > 0:
        module = guess_correct_module(argv[0])
        run_command(module, argv[1:])
    else:
        list_examples(examples)
        try:
            ans = int(get_input("Enter example number you wish to run: "))
        except ValueError:
            ans = 0
        if ans < 1 or ans > len(examples):
            print("Invalid example number, exiting!")
            sys.exit()

        args = str(get_input(
            "Enter additional arguments (leave blank to skip): "
        ))
        module, doc = examples[ans - 1]
        print("-" * 80)
        run_command(module, args.split())

if __name__ == '__main__':
    main()
