from textwrap import dedent


###############################################################################
# `Locator` class.
###############################################################################
class Locator(object):
    def cython_code(self):
        raise NotImplementedError
        
        
###############################################################################
# `AllPairLocator` class.
###############################################################################
class AllPairLocator(Locator):
    def cython_code(self):
        code = dedent('''\
        cdef class AllPairLocator:
            cdef ParticleArrayWrapper src, dest
            cdef long N
            cdef UIntArray nbrs
            def __init__(self, ParticleArrayWrapper src, 
                         ParticleArrayWrapper dest):
                self.src = src
                self.dest = dest
                self.N = src.size()
                self.nbrs = UIntArray(self.N)
                cdef long i
                for i in range(self.N):
                    self.nbrs[i] = i
                
            def get_neighbors(self, long d_idx, UIntArray nbr_array):
                nbr_array.resize(self.N)
                nbr_array.set_data(self.nbrs.get_npy_array())
        '''
        )
        return dict(helper='', code=code)
        
