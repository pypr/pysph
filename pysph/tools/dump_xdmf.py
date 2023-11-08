""" Generate XDMF file(s) referencing the heavy data stored using HDF5 by
PySPH.

Separate xdmf file will be generated for each hdf5 file input. If directory is
input, a single xdmf file will be generated assuming all the hdf5 files inside
the directory as timeseris data.
"""
import argparse
import shutil
import sys
from pathlib import Path

import h5py
from mako.template import Template

from pysph.solver.utils import get_files, ProgressBar


def main(argv=None):
    """ Main function to generate XDMF file(s) referencing the heavy data
    stored using HDF5 by PySPH.
    """
    cols, _ = shutil.get_terminal_size()
    print("Generating XDMF".center(cols, '-'))

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
        default=None,
        help="directory to output xdmf file(s), defaults to input path"
    )

    parser.add_argument(
        "--refer-absolute-path", action="store_false", dest='relative_path',
        help="use absolute path(s) to reference heavy data in the generated"
             " xdmf file"
    )

    parser.add_argument(
        "--vectorize-velocity", action=argparse.BooleanOptionalAction,
        default=True,
        dest='vectorize_velocity',
        help="reference u,v and w such that they are read as separate scalar "
             "quantities through xdmf"
    )

    parser.add_argument(
        "--combine-particle-arrays", action=argparse.BooleanOptionalAction,
        default=False,
        dest='combine_particle_arrays',
        help="Combine all particle arrays into a single xdmf file. If False, "
             "separate xdmf files will be generated for each particle array."
    )

    if len(argv) > 0 and argv[0] in ['-h', '--help']:
        parser.print_help()
        sys.exit()

    options, extra = parser.parse_known_args(argv)
    run(options)
    print("Done Generating XDMF".center(cols, '-'))


def run(options):
    if options.outdir is not None:
        Path(options.outdir).mkdir(parents=True, exist_ok=True)

    for ifile in options.inputfile:
        # there can be two cases:
        # 1. input is a hdf5 file, refer it in a single xdmf file
        # if --combine-particle-arrays is True else separate xdmf files for
        # each particle array. The name of the xdmf file will be inferred from
        # the name of the hdf5 file. if the input is a list of hdf5 files, then
        # each hdf5 file will be dealt with separately.
        # 2. input is a directory, refer all the hdf5 files inside it in a
        # single xdmf file if --combine-particle-arrays is True else separate
        # xdmf files for each particle array. The name of the xdmf file will
        # be inferred from the name of the directory.
        # Note: pa name will be appended to outfilename later
        # if combine_particle_arrays is false.

        if Path(ifile).is_dir():
            idir = Path(ifile).absolute()
            files = get_files(f'{idir}', endswith='hdf5')

            if options.outdir is not None:
                outdir = Path(options.outdir).absolute()
            else:
                outdir = idir

            if outdir != idir:
                outfilename = Path(idir).name + '.xdmf'
            elif options.combine_particle_arrays:
                outfilename = 'all_pa.xdmf'
            else:
                outfilename = 'pa.xdmf'
        else:
            files = [Path(ifile).absolute()]

            if options.outdir is not None:
                outdir = Path(options.outdir).absolute()
            else:
                outdir = Path(ifile).parent

            outfilename = Path(ifile).stem + '.xdmf'

        outfile = outdir.joinpath(outfilename).absolute()
        files2xdmf(files, outfile,
                   options.relative_path, options.vectorize_velocity,
                   options.combine_particle_arrays)


def files2xdmf(absolute_files, outfilename, refer_relative_path,
               vectorize_velocity, combine_particle_arrays):
    # Assuming output_props and strides does not change for the files in a
    # folder, just obtain those from the first file.
    particles_info = {}
    n_particles = {}
    stride = {}
    fname = absolute_files[0]

    print(f'Reading properties and strides from {fname}...', end='')
    with h5py.File(fname, 'r') as data:  # will fail here if not .hdf5 file
        for pname in data['particles'].keys():
            n_particles[pname] = []
            output_props = []
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
    print('done')

    print(f'Reading number of particles and time')
    # When there is only one file, len(absolute_files) - 1 will be 0.
    # But for ProgressBar, tf should be > 0.
    # So, we set tf to max(1, len(absolute_files) - 1) but do not show the
    # progress bar in this case.
    progress_max = max(len(absolute_files) - 1, 1)
    progress_bar = ProgressBar(ti=0, tf=progress_max,
                               show=True if progress_max > 1 else False)
    # time (and number of particles may) change for different files in a
    # folder, so obtain that from each file.
    times = []
    for i, fname in enumerate(absolute_files):
        progress_bar.update(i)
        with h5py.File(fname, 'r') as data:  # will fail here if not .hdf5 file
            times.append(data['solver_data'].attrs.get('t'))
            for pname in data['particles'].keys():
                for arrname, arr in data[f'particles/{pname}/arrays/'].items():
                    if arr.attrs['stored']:
                        n = int(arr.shape[0] / arr.attrs.get('stride'))
                        n_particles[pname].append(n)
                        break
    progress_bar.finish()
    template_file = Path(__file__).parent.absolute().joinpath(
        'xdmf_template.mako')
    xdmf_template = Template(filename=str(template_file))

    if refer_relative_path:
        outdir = Path(outfilename).parent
        files = [Path(f).relative_to(outdir) for f in absolute_files]
    else:
        files = absolute_files

    if combine_particle_arrays:
        with open(outfilename, 'w') as xdmf_file:
            print(xdmf_template.render(times=times, files=files,
                                       particles_info=particles_info,
                                       n_particles=n_particles,
                                       vectorize_velocity=vectorize_velocity),
                  file=xdmf_file)
    else:
        for pname in particles_info.keys():
            outfilename_pname = outfilename.parent.joinpath(
                f'{outfilename.stem}_{pname}{outfilename.suffix}')
            print(f'Writing {outfilename_pname}...', end='')
            with open(outfilename_pname, 'w') as xdmf_file:
                print(xdmf_template.render(
                    times=times, files=files,
                    particles_info={pname: particles_info[pname]},
                    n_particles={pname: n_particles[pname]},
                    vectorize_velocity=vectorize_velocity),
                    file=xdmf_file)
            print('done')


if __name__ == '__main__':
    main()
