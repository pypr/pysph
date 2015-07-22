"""List and run PySPH examples.

"""
from __future__ import print_function

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
    for dirpath, dirs, files in os.walk(basedir):
        rel_dir = os.path.relpath(dirpath, basedir)
        if rel_dir == '.':
            rel_dir = ''
        py_files = [x for x in files
                    if x.endswith('.py') and not x.startswith('_')]
        data = []
        for f in py_files:
            module = _get_module(os.path.join(rel_dir, f))
            doc = _extract_short_doc(dirpath, f)
            data.append((module, doc))
        examples.extend(data)
    return examples

def main():
    examples = get_all_examples()
    for idx, (module, doc) in enumerate(examples):
        print("%d. %s"%(idx+1, module))
        print("   %s"%doc)

    ans = input("Enter example number you wish to run -> ")
    module, doc = examples[ans-1]
    print("-"*80)
    print("Running example %s.\n"%module)
    print(_extract_full_doc(module))
    cmd = [sys.executable, "-m", module]
    subprocess.call(cmd)

if __name__ == '__main__':
    main()
