''' convert pysph .npz output to vtk file format '''
from __future__ import print_function
import os
import re

from enthought.tvtk.api import tvtk, write_data
from numpy import array, c_, ravel, load, zeros_like


def write_vtk(data, filename, scalars=None, vectors={'V':('u','v','w')}, tensors={},
              coords=('x','y','z'), dims=None, **kwargs):
    ''' write data in to vtk file

    Parameters
    ----------
    data : dict
        mapping of variable name to their numpy array
    filename : str
        the file to write to (can be any recognized vtk extension)
        if extension is missing .vts extension is appended
    scalars : list
        list of arrays to write as scalars (defaults to data.keys())
    vectors : dict
        mapping of vector name to vector component names to take from data
    tensors : dict
        mapping of tensor name to tensor component names to take from data
    coords : list
        the name of coordinate data arrays (default=('x','y','z'))
    dims : 3 tuple
        the size along the dimensions for (None means x.shape)
    **kwargs : extra arguments for the file writer
        example file_type=binary/ascii

    '''
    x = data[coords[0]]
    y = data.get(coords[1], zeros_like(x))
    z = data.get(coords[2], zeros_like(x))

    if dims is None:
        dims = array([1,1,1])
        dims[:x.ndim] = x.shape
    else:
        dims = array(dims)

    sg = tvtk.StructuredGrid(points=c_[x.flat,y.flat,z.flat],dimensions=array(dims))
    pd = tvtk.PointData()

    if scalars is None:
        scalars = [i for i in data.keys() if i not in coords]
    for v in scalars:
        pd.scalars = ravel(data[v])
        pd.scalars.name = v
        sg.point_data.add_array(pd.scalars)

    for vec,vec_vars in vectors.items():
        u,v,w = [data[i] for i in vec_vars]
        pd.vectors = c_[ravel(u),ravel(v),ravel(w)]
        pd.vectors.name = vec
        sg.point_data.add_array(pd.vectors)

    for ten,ten_vars in tensors.items():
        vars = [data[i] for i in ten_vars]
        tensors = c_[[ravel(i) for i in vars]].T
        pd.tensors = tensors
        pd.tensors.name = ten
        sg.point_data.add_array(pd.tensors)

    write_data(sg, filename, **kwargs)


def detect_vectors_tensors(keys):
    ''' detect the vectors and tensors from given array names

    Vectors are identified as the arrays with common prefix followed by
    0,1 and 2 in their names
    Tensors are identified as the arrays with common prefix followed by
    two character codes representing ij indices
        (00,01,02,11,12,22) for a symmetric tensor
        (00,01,02,10,11,12,20,21,22) for a tensor
    Arrays not belonging to vectors or tensors are returned as scalars

    Returns scalars,vectors,tensors in a format suitable to be used as arguments
    for :py:func:`write_vtk`

    '''
    d = {}
    for k in keys:
        d[len(k)] = d.get(len(k), [])
        d[len(k)].append(k)

    scalars = []
    vectors = {}
    tensors = {}

    for n,l in d.items():
        if n<2:
            continue
        l.sort()

        idx = -1
        while idx<len(l)-1:
            idx += 1
            k = l[idx]

            # check if last char is 0
            if k[-1] == '0':

                # check for tensor
                if k[-2] == '0':

                    # check for 9 tensor
                    ten = []
                    for i in range(3):
                        for j in range(3):
                            ten.append(k[:-2]+str(j)+str(i))
                    ten.sort()

                    if l[idx:idx+9] == ten:
                        tensors[k[:-2]] = ten
                        idx += 8
                        continue

                    # check for symm 6 tensor
                    ten2 = []
                    for i in range(3):
                        for j in range(i+1):
                            ten2.append(k[:-2]+str(j)+str(i))
                    ten2.sort()

                    if l[idx:idx+6] == ten2:
                        ten = []
                        for i in range(3):
                            for j in range(3):
                                ten.append(k[:-2]+str(min(i,j))+str(max(i,j)))
                        tensors[k[:-2]] = ten
                        idx += 5
                        continue

                # check for vector
                vec = []
                for i in range(3):
                    vec.append(k[:-1] + str(i))
                if l[idx:idx+3] == vec:
                    vectors[k[:-1]] = vec
                    idx += 2
                    continue

            scalars.append(k)

    return scalars, vectors, tensors



def get_output_details(path):
    solvers = {}
    if not os.path.isdir(path):
        path = os.path.dirname(path)
    files = os.listdir(path)
    files.sort()

    pat = re.compile(r'(?P<solver>.+)_(?P<rank>\d+)_(?P<entity>.+)_(?P<time>[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?).npz')
    matches = [(f,pat.match(f)) for f in files]

    files = []
    for filename,match in matches:
        if match is None:
            continue
        files.append(filename)
        groups = match.groupdict()
        solvername = groups['solver']
        solver = solvers.get(solvername)
        if solver is None:
            solver = [set([]),set([]),set([])]
            solvers[solvername] = solver
        solver[0].add(groups['rank'])
        solver[1].add(groups['entity'])
        solver[2].add(groups['time'])
    # {solver:(entities,procs,times)}
    return solvers


def pysph_to_vtk(path, merge_procs=False, skip_existing=True, binary=True):
    ''' convert pysph output .npz files into vtk format

    Parameters
    ----------
    path : str
        directory where .npz files are located
    merge_procs : bool
        whether to merge the data from different procs into a single file
        (not yet implemented)
    skip_existing : bool
        skip files where corresponding vtk already exist
        this is useful if you've converted vtk files while a solver is running
        only want to convert the newly added files
    binary : bool
        whether to use binary format in vtk file
    The output vtk files are stored in a directory `solver_name` _vtk within
    the `path` directory

    '''
    if binary:
        data_mode = 'binary'
    else:
        data_mode = 'ascii'

    if merge_procs is True:
        # FIXME: implement
        raise NotImplementedError('merge_procs=True not implemented yet')

    solvers = get_output_details(path)
    for solver, (procs, entities, times) in solvers.items():
        print('converting solver:', solver)
        dir = os.path.join(path,solver+'_vtk')
        if not os.path.exists(dir):
            os.mkdir(dir)
        procs = sorted(procs)
        entities = sorted(entities)
        times = sorted(times, key=float)
        times_file = open(os.path.join(dir,'times'), 'w')

        for entity in entities:
            print('    entity:', entity)
            for proc in procs:
                print('        proc:', proc)
                print('        timesteps:', len(times))
                f = '%s_%s_%s_'%(solver,proc,entity)
                of = os.path.join(dir,f)

                for i, time in enumerate(times):
                    print('\r',i,)
                    if skip_existing and os.path.exists(f+str(i)):
                        continue
                    d = load(os.path.join(path, f+time+'.npz'))
                    arrs = {}
                    for nam,val in d.items():
                        if val.ndim > 0:
                            arrs[nam] = val
                    d.close()

                    scalars, vectors, tensors = detect_vectors_tensors(arrs)
                    vectors['V'] = ['u','v','w']
                    z = zeros_like(arrs['x'])
                    if 'v' not in arrs:
                        arrs['v'] = z
                    if 'w' not in arrs:
                        arrs['w'] = z
                    write_vtk(arrs, of+str(i),
                              scalars=scalars, vectors=vectors, tensors=tensors,
                              data_mode=data_mode)
                    times_file.write('%d\t%s\n'%(i,time))

        times_file.close()

def extract_text(path, particle_idx, props=['x','y','u','v','p','rho','sigma00','sigma01','sigma11'], ent=None, solvers=None):
    if solvers:
        raise NotImplementedError
    else:
        solvers = get_output_details(path)

    for solver, (procs, entities, times) in solvers.items():
        print('converting solver:', solver)
        dir = os.path.join(path,solver+'_vtk')
        if not os.path.exists(dir):
            os.mkdir(dir)
        procs = sorted(procs)
        entities = sorted(entities)
        times = sorted(times, key=float)
        times_file = open(os.path.join(dir,'times'), 'w')
        e = ent
        if ent is None:
            e = entities
        for entity in entities:
            if entity not in e:
                continue
            print('    entity:', entity)
            for proc in procs:
                print('        proc:', proc)
                print('        timesteps:', len(times))
                f = '%s_%s_%s_'%(solver,proc,entity)
                of = os.path.join(dir,f)
                files = [open(os.path.join(path,f+'%d.dat'%particle_id), 'w') for particle_id in particle_idx]
                print(files)
                for file in files:
                    file.write('i\tt\t'+'\t'.join(props))
                for i, time in enumerate(times):
                    print('\r',i,)
                    d = load(os.path.join(path, f+time+'.npz'))
                    s = '\n%d\t%s'%(i,time)
                    for j,file in enumerate(files):
                        file.write(s)
                        for prop in props:
                            file.write('\t')
                            file.write(str(d[prop][particle_idx[j]]))

                    d.close()

                for file in files:
                    file.close()

def test():
    l = ['x'+str(i) for i in range(3)]
    l.append('a0')
    l.append('a1')

    for i in range(3):
        for j in range(3):
            if i == j:
                l.append('XX%d'%i)
            if i <= j:
                l.append('S%d%d'%(i,j))
            l.append('T%d%d'%(i,j))

    scalars, vectors, tensors = detect_vectors_tensors(l)
    assert set(scalars) == set(['a0','a1'])
    assert set(vectors) == set(['x','XX'])
    assert set(tensors) == set(['S','T'])


if __name__ == '__main__':
    import sys
    pysph_to_vtk(path=sys.argv[1])

