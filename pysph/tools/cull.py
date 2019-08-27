"""Culls files in a given directory, one in every 'c' files is spared.

The specified directory can contain other directories that house the output
 files; files in all these directories will be culled, sparing one in every
 'c' files. Note that DELETION IS PERMANENT.
"""

from pysph.tools.binder import find_sim_dirs, find_dir_size
from pysph.solver.utils import get_files
import os
import sys
import argparse


def cull(src_path, c):

    src_path = os.path.abspath(src_path)
    sim_paths_list = find_sim_dirs(src_path)

    initial_size = find_dir_size(src_path)

    for path in sim_paths_list:
        files = get_files(path)
        l = len(files)
        del_files = [
            files[i] for i in set(range(l)) - set(range(0, l, c))
        ]
        if len(del_files) != 0:
            for f in del_files:
                os.remove(f)

    final_size = find_dir_size(src_path)

    print("Initial size of the directory was: "+str(initial_size)+" bytes")
    print("Final size of the directory is: "+str(final_size)+" bytes")
    return


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        prog='cull',
        description=__doc__,
        add_help=False
    )

    parser.add_argument(
        "-h",
        "--help",
        action="store_true",
        default=False,
        dest="help",
        help="show this help message and exit"
    )

    parser.add_argument(
        "src_path",
        metavar='src_path',
        type=str,
        nargs=1,
        help="the directory containing the directories/files to be culled"
    )

    parser.add_argument(
        "-c", "--cull-factor",
        metavar="cull_factor",
        type=int,
        default=2,
        help="one in every 'c' files is spared, all remaining output " +
             "files are deleted [default=2]"
    )

    if len(argv) > 0 and argv[0] in ['-h', '--help']:
        parser.print_help()
        sys.exit()

    options, extra = parser.parse_known_args(argv)

    src_path = options.src_path[0]
    x = options.cull_factor

    cull(src_path, x)


if __name__ == '__main__':
    main()
