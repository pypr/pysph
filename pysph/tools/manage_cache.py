"""Manage the PySPH and Compyle cache directories.

These directories contain the generated sources and extension modules and can
get quite big. The command allows you to see the path and size of these cache
directories and also clear them out if they are too big.

"""
import argparse
from pathlib import Path
import shutil
import sys


def _get_cache_dirs():
    home = Path('~').expanduser()
    cc = home / '.compyle' / 'source'
    pc = home / '.pysph' / 'source'
    return (cc, pc)


def _find_size(pth):
    return sum(f.stat().st_size for f in pth.glob('**/*') if f.is_file())


def show_cache():
    cc, pc = _get_cache_dirs()
    print("PySPH cache directories are at:")
    GB = 2**30
    print("{}  {:<.3g} GB".format(str(cc), _find_size(cc)/GB))
    print("{}    {:<.3g} GB".format(str(pc), _find_size(pc)/GB))


def clear_cache():
    cc, pc = _get_cache_dirs()
    print("Clearing cache in\n", cc, "\n", pc)
    confirm = input('Are you sure? (y/N) ')
    if confirm in ['y', 'Y']:
        shutil.rmtree(cc)
        shutil.rmtree(pc)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog='cache', description=__doc__, add_help=False
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
        "-c", "--clear",
        action="store_true",
        default=False,
        help="Delete all the files in the cache directory."
    )

    if argv is not None and len(argv) > 0 and argv[0] in ['-h', '--help']:
        parser.print_help()
        sys.exit()

    options, extra = parser.parse_known_args(argv)
    if options.clear:
        clear_cache()
    else:
        show_cache()


if __name__ == '__main__':
    main()
