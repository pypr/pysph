""" Generate XDMF file(s) referencing the heavy data stored using HDF5 by
PySPH.

Separate xdmf file will be generated for each hdf5 file input. If directory is
input, a single xdmf file will be generated assuming all the hdf5 files inside
the directory as timeseris data.
"""

import argparse
import sys
from pathlib import Path

import h5py
from mako.template import Template

from pysph.solver.utils import get_files


def main(argv=None):
    """ Main function to generate XDMF file(s) referencing the heavy data
    stored using HDF5 by PySPH.
    """

    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        prog='generate_xdmf', description=__doc__, add_help=False
    )

    parser.add_argument(
        "-h", "--help", action="store_true", default=False, dest="help",
        help="show this help message and exit"
    )

    parser.add_argument(
        "inputfile", type=str, nargs='+',
        help="list of input hdf5 file(s) or/and director(y/ies) with hdf5"
             "file(s)."
    )

    parser.add_argument(
        "-d", "--outdir", metavar="path/to/outdir", type=str,
        default=Path(),
        help="directory to output xdmf file(s), defaults to current working "
             "directory"
    )

    parser.add_argument(
        "--relative-path", action="store_true",
        help="use relative path(s) to reference heavy data in the generated"
             " xdmf file"
    )

    parser.add_argument(
        "--no-vectorize-velocity", action="store_false",
        dest='vectorize_velocity',
        help="reference u,v and w such that they are read as separate scalar "
             "quantities through xdmf"
    )

    if len(argv) > 0 and argv[0] in ['-h', '--help']:
        parser.print_help()
        sys.exit()

    options, extra = parser.parse_known_args(argv)
    run(options)


def run(options):
    for ifile in options.inputfile:
        if Path(ifile).is_dir():
            files = get_files(ifile, endswith='hdf5')
            outfile = Path(ifile).name + '.xdmf'
        else:
            files = [Path(ifile).absolute()]
            outfile = Path(ifile).stem + '.xdmf'

        files2xdmf(files, Path(options.outdir).joinpath(outfile),
                   options.relative_path, options.vectorize_velocity)


def files2xdmf(absolute_files, outfilename, refer_relative_path,
               vectorize_velocity):
    # Assuming output_props and strides does not change for the files in a
    # folder, just obtain those from the first file.
    particles_info = {}
    n_particles = {}
    output_props = []
    stride = {}
    fname = absolute_files[0]
    with h5py.File(fname, 'r') as data:  # will fail here if not .hdf5 file
        for pname in data['particles'].keys():
            n_particles[pname] = []
            for arrname, arr in data[f'particles/{pname}/arrays/'].items():
                if arr.attrs['stored']:
                    output_props.append(arrname)
                stride[arrname] = arr.attrs.get('stride')
    attr_type = {}
    for var_name in output_props:
        if stride[var_name] == 1:
            typ = 'Scalar'
        elif stride[var_name] == 3:
            typ = 'Vector'
        elif stride[var_name] == 9:
            typ = 'Tensor'
        else:
            typ = 'Matrix'
        attr_type[var_name] = typ

        particles_info[pname] = {'output_props': output_props,
                                 'stride': stride,
                                 'attr_type': attr_type}

    # time (and number of particles may) change for different files in a
    # folder, so obtain that from each file.
    times = []
    for fname in absolute_files:
        with h5py.File(fname, 'r') as data:  # will fail here if not .hdf5 file
            times.append(data['solver_data'].attrs.get('t'))
            for pname in data['particles'].keys():
                for arrname, arr in data[f'particles/{pname}/arrays/'].items():
                    if arr.attrs['stored']:
                        n = int(arr.shape[0]/arr.attrs.get('stride'))
                        n_particles[pname].append(n)
                        break

    template_file = Path(__file__).parent.absolute().joinpath(
        'xdmf_template.mako')
    xdmf_template = Template(filename=str(template_file))

    if refer_relative_path:
        outdir = Path(outfilename).parent
        files = [Path(f).relative_to(outdir) for f in absolute_files]
    else:
        files = absolute_files

    with open(outfilename, 'w') as xdmf_file:
        print(xdmf_template.render(times=times, files=files,
                                   particles_info=particles_info,
                                   n_particles=n_particles,
                                   vectorize_velocity=vectorize_velocity),
              file=xdmf_file)


if __name__ == '__main__':
    main()
