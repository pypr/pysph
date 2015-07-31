"""List and run PySPH examples.

One can optionally supply the name of the example and any additional arguments.

"""

from __future__ import print_function

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(__file__)

def _extract_full_doc(module):
    m = module.rsplit('.', 1)[1]
    mod = __import__(module, fromlist=[m])
    return mod.__doc__

def _extract_short_doc(dirname, fname):
    return open(os.path.join(dirname, fname)).readline()[3:].strip()

def _get_module(fname):
    start = fname
    parts = ['pysph.examples']
    while os.path.dirname(start) != '':
        dirname, start = os.path.split(start)
        parts.append(dirname)
    return '.'.join(parts + [start[:-3]])


def get_all_examples():
    basedir = HERE
    examples = []
    ignore = [os.path.abspath(os.path.join(basedir, "run.py"))]
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

def example_info(module):
    print("Information for example: %s"%module)
    print(_extract_full_doc(module))

def list_examples(examples):
    for idx, (module, doc) in enumerate(examples):
        print("%d. %s"%(idx+1, module))
        print("   %s"%doc)

def get_input(prompt):
    if sys.version_info.major > 2:
        return input(prompt)
    else:
        return raw_input(prompt)

def run_command(module, args):
    print("Running example %s.\n"%module)
    example_info(module)
    cmd = [sys.executable, "-m", module] + list(args)
    subprocess.call(cmd)

def main():
    examples = get_all_examples()
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument(
        "-h", "--help", action="store_true", default=False, dest="help",
        help="show this help message and exit"
    )
    parser.add_argument(
        "-l", "--list", action="store_true", default=False, dest="list",
        help="List examples"
    )
    parser.add_argument(
        "args", type=str, nargs="?",
        help="optional example name and arguments to the example."
    )

    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help']:
        parser.print_help()
        sys.exit()

    options, extra = parser.parse_known_args()
    if options.list:
        return list_examples(examples)
    if len(sys.argv) > 1:
        module = 'pysph.examples.' + sys.argv[1]
        run_command(module, sys.argv[2:])
    else:
        list_examples(examples)
        ans = int(get_input("Enter example number you wish to run: "))
        args = str(get_input("Enter additional arguments (lave blank to skip): "))
        module, doc = examples[ans-1]
        print("-"*80)
        run_command(module, args.split())

if __name__ == '__main__':
    main()
