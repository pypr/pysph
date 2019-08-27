"""Make a directory compatible with mybinder.org and ready for
upload to a Github repo.
"""

from pysph.solver.utils import get_files, mkdir
import os
from shutil import copy
import nbformat as nbf
import argparse
import sys
import re
import glob


def find_viewer_type(path):
    '''
    Finds the type of viewer to use in the jupyter notebook.
    Parses the log file at the file path and searches for
    'dim=d', where 'dD' is taken to be the viewer type.

    example: if 'dim=2', then what is returned is '2D'
    '''

    log_file_path = os.path.abspath(path) + '/*.log'
    regex = r'dim=(\d)'
    log_files = glob.glob(log_file_path)
    if not log_files:
        return '2D'

    match_list = []
    with open(log_files[0], 'r') as file:
        for line in file:
            for match in re.finditer(regex, line, re.S):
                match_text = match.group()
                match_list.append(match_text)
                if len(match_list) > 0:
                    break
            if len(match_list) > 0:
                break
    return match_list[0][-1] + 'D'


def make_notebook(path, sim_name, config_dict={}):
    '''
    Makes a jupyter notebook to view simulation results stored in
    a given directory

    path: the directory conatining the output files

    sim_name: name of the simulation
              ex. 'cavity_output'

    config_dict: configuration dictionary for the notbeook viewer [dict]
                 ex. {'fluid': {'frame': 20}}
    '''

    viewer_type = find_viewer_type(path)
    cell1_src = [
        "import os\n",
        "from pysph.tools.ipy_viewer import Viewer" + viewer_type
    ]
    cell2_src = [
        "cwd = os.getcwd()"
    ]
    cell3_src = [
        "viewer = Viewer" + viewer_type + "(cwd)\n",
        "viewer.interactive_plot(" + str(config_dict) + ")"
    ]

    nb = nbf.v4.new_notebook()

    nb.cells = [
        nbf.v4.new_code_cell(source=cell1_src),
        nbf.v4.new_code_cell(source=cell2_src),
        nbf.v4.new_code_cell(source=cell3_src)
    ]

    nbf.write(
        nb,
        os.path.join(
            path,
            sim_name+'.ipynb'
        )
    )
    return


def find_sim_dirs(path, sim_paths_list=[]):
    '''
    Finds all the directories in a given directory that
    contain pysph output files.
    '''
    path = os.path.abspath(path)
    sim_files = get_files(path)
    if len(sim_files) != 0:
        sim_paths_list.append(path)
    elif len(sim_files) == 0:
        files = os.listdir(path)
        files = [f for f in files if os.path.isdir(f)]
        files = [os.path.abspath(f) for f in files if not f.startswith('.')]
        for f in files:
            sim_paths_list = find_sim_dirs(f, sim_paths_list)

    return sim_paths_list


def find_dir_size(path):
    '''
    Finds the size of a given directory.
    '''
    total_size = 0
    for dir_path, dir_names, file_names in os.walk(path):
        for f in file_names:
            fp = os.path.join(dir_path, f)
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

    return total_size


def make_binder(path):

    src_path = os.path.abspath(path)

    sim_paths_list = find_sim_dirs(src_path)

    for path in sim_paths_list:
        sim_name = os.path.split(path)[1]
        make_notebook(path, sim_name)
        if len(sim_paths_list) == 1 and sim_paths_list[0] == src_path:
            files = os.listdir(src_path)
            files = [os.path.join(src_path, f) for f in files]
            mkdir(os.path.join(src_path, sim_name))
            could_not_copy = []
            for f in files:
                try:
                    copy(f, os.path.join(src_path, sim_name))
                except BaseException as exc:
                    could_not_copy.append([f, exc])
                    continue
                os.remove(f)
            if len(could_not_copy) != 0:
                print("Could not copy the following files:\n")
                for f in could_not_copy:
                    print('file: ', f[0])
                    print('error: ', f[1], '\n')

    with open(os.path.join(src_path, 'requirements.txt'), 'w') as file:
        file.write(
            "ipympl\n" +
            "matplotlib\n" +
            "ipyvolume\n" +
            "numpy\n" +
            "pytools\n" +
            "-e git+https://github.com/pypr/compyle#egg=compyle\n" +
            "-e git+https://github.com/pypr/pysph#egg=pysph"
        )

    with open(os.path.join(src_path, 'README.md'), 'w') as file:
        file.write(
            "# Title\n" +
            "[![Binder](https://mybinder.org/badge_logo.svg)]" +
            "(https://mybinder.org/v2/gh/user_name/repo_name/branch_name)" +
            "\n" +
            "[comment]: # (The above link is for repositories hosted " +
            "on GitHub. For links corresponding to other hosting services, " +
            "please visit https://mybinder.readthedocs.io)"
        )


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        prog='binder',
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
        type=str,
        nargs=1,
        help="the directory containing the directories/files to be prepared"
    )

    if len(argv) > 0 and argv[0] in ['-h', '--help']:
        parser.print_help()
        sys.exit()

    options, extra = parser.parse_known_args(argv)

    src_path = options.src_path[0]

    make_binder(src_path)


if __name__ == '__main__':
    main()
