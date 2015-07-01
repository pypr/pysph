#cython: embedsignature=True

"""A handy set of classes/functions for 3D points.
"""

from cpython cimport *
# numpy imports
cimport numpy
import numpy

# IntPoint's maximum absolute value must be less than `IntPoint_maxint`
# this is due to the hash implementation
cdef int IntPoint_maxint = 2**20

DTYPE = numpy.float
ctypedef numpy.float_t DTYPE_t


###############################################################################
# `Point` class.
###############################################################################
cdef class Point:
    """
    This class represents a Point in 3D space.
    """

    # Declared in the .pxd file.
    #cdef public double x, y, z

    property x:
        def __get__(Point self):
            return self.data.x
        def __set__(Point self, double x):
            self.data.x = x

    property y:
        def __get__(Point self):
            return self.data.y
        def __set__(Point self, double y):
            self.data.y = y

    property z:
        def __get__(Point self):
            return self.data.z
        def __set__(Point self, double z):
            self.data.z = z

    ######################################################################
    # `object` interface.
    ######################################################################
    def __init__(Point self, double x=0.0, double y=0.0, double z=0.0):
        """Constructor for a Point."""
        self.data.x = x
        self.data.y = y
        self.data.z = z

    def __reduce__(Point self):
        """
        Implemented to facilitate pickling of the Point extension type.
        """
        d = {}
        d['x'] = self.data.x
        d['y'] = self.data.y
        d['z'] = self.data.z

        return (Point, (), d)

    def __setstate__(Point self, d):
        self.data.x = d['x']
        self.data.y = d['y']
        self.data.z = d['z']

    def __str__(Point self):
        return '(%f, %f, %f)'%(self.data.x, self.data.y, self.data.z)

    def __repr__(Point self):
        return 'Point(%g, %g, %g)'%(self.data.x, self.data.y, self.data.z)

    def __add__(Point self, Point p):
        return Point_new(self.data.x + p.data.x, self.data.y + p.data.y,
                         self.data.z + p.data.z)

    def __sub__(Point self, Point p):
        return Point_new(self.data.x - p.data.x, self.data.y - p.data.y,
                         self.data.z - p.data.z)

    def __mul__(Point self, double m):
        return Point_new(self.data.x*m, self.data.y*m, self.data.z*m)

    def __div__(Point self, double m):
        return Point_new(self.data.x/m, self.data.y/m, self.data.z/m)

    def __abs__(Point self):
        return cPoint_length(self.data)

    def __neg__(Point self):
        return Point_new(-self.data.x, -self.data.y, -self.data.z)

    def __richcmp__(Point self, Point p, int oper):
        if oper == 2: # ==
            if self.data.x == p.data.x and self.data.y == p.data.y and self.data.z == p.data.z:
                return True
            return False
        elif oper == 3: # !=
            if self.data.x == p.data.x and self.data.y == p.data.y and self.data.z == p.data.z:
                return False
            return True
        else:
            raise TypeError('No ordering is possible for Points.')

    def __iadd__(Point self, Point p):
        self.data.x += p.data.x
        self.data.y += p.data.y
        self.data.z += p.data.z
        return self

    def __isub__(Point self, Point p):
        self.data.x -= p.data.x
        self.data.y -= p.data.y
        self.data.z -= p.data.z
        return self

    def __imul__(Point self, double m):
        self.data.x *= m
        self.data.y *= m
        self.data.z *= m
        return self

    def __idiv__(Point self, double m):
        self.data.x /= m
        self.data.y /= m
        self.data.z /= m
        return self

    ######################################################################
    # `Point` interface.
    ######################################################################
    cpdef set(Point self, double x, double y, double z):
        """Set the position from a given array.
        """
        self.data.x = x
        self.data.y = y
        self.data.z = z

    cdef set_from_cPoint(Point self, cPoint value):
        self.data.x = value.x
        self.data.y = value.y
        self.data.z = value.z

    cpdef numpy.ndarray asarray(Point self):
        """Return a numpy array with the coordinates."""
        cdef numpy.ndarray[DTYPE_t, ndim=1] r = numpy.empty(3)
        r[0] = self.data.x
        r[1] = self.data.y
        r[2] = self.data.z
        return r

    cpdef double norm(Point self):
        """Return the square of the Euclidean distance to this point."""
        return cPoint_norm(self.data)

    cpdef double length(Point self):
        """Return the Euclidean distance to this point."""
        return cPoint_length(self.data)

    cpdef double dot(Point self, Point p):
        """Return the dot product of this point with another."""
        return cPoint_dot(self.data, p.data)

    cpdef Point cross(Point self, Point p):
        """Return the cross product of this point with another, i.e.
        `self` cross `p`."""
        return Point_new(self.data.y*p.data.z - self.data.z*p.data.y,
                         self.data.z*p.data.x - self.data.x*p.data.z,
                         self.data.x*p.data.y - self.data.y*p.data.x)

    cpdef double distance(Point self, Point p):
        """Return the distance between this point and p"""
        return cPoint_distance(self.data, p.data)

    cdef cPoint to_cPoint(Point self):
        return self.data

    def normalize(self):
        """ Normalize the point """
        cdef double norm = cPoint_length(self.data)

        self.data.x /= norm
        self.data.y /= norm
        self.data.z /= norm

cdef class IntPoint:

    property x:
        def __get__(self):
            return self.data.x

    property y:
        def __get__(self):
            return self.data.y

    property z:
        def __get__(self):
            return self.data.z

    def __init__(self, int x=0, int y=0, int z=0):
        self.data.x = x
        self.data.y = y
        self.data.z = z

    def __reduce__(self):
        """
        Implemented to facilitate pickling of the IntPoint extension type.
        """
        d = {}
        d['x'] = self.data.x
        d['y'] = self.data.y
        d['z'] = self.data.z

        return (IntPoint, (), d)

    def __setstate__(self, d):
        self.data.x = d['x']
        self.data.y = d['y']
        self.data.z = d['z']

    def __str__(self):
        return '(%d,%d,%d)'%(self.data.x, self.data.y, self.data.z)

    def __repr__(self):
        return 'IntPoint(%d,%d,%d)'%(self.data.x, self.data.y, self.data.z)

    cdef IntPoint copy(self):
        return IntPoint_new(self.data.x, self.data.y, self.data.z)

    cpdef numpy.ndarray asarray(self):
        cdef numpy.ndarray[ndim=1,dtype=numpy.int_t] arr = numpy.empty(3,
                                                            dtype=numpy.int)
        arr[0] = self.data.x
        arr[1] = self.data.y
        arr[2] = self.data.z

        return arr

    cdef bint is_equal(self, IntPoint p):
        return cIntPoint_is_equal(self.data, p.data)

    cdef IntPoint diff(self, IntPoint p):
        return IntPoint_new(self.data.x-p.data.x, self.data.y-p.data.y,
                            self.data.z-p.data.z)

    cdef tuple to_tuple(self):
        cdef tuple t = (self.data.x, self.data.y, self.data.z)
        return t

    def __richcmp__(IntPoint self, IntPoint p, int oper):
        if oper == 2: # ==
            return cIntPoint_is_equal(self.data, p.data)
        elif oper == 3: # !=
            return not cIntPoint_is_equal(self.data, p.data)
        else:
            raise TypeError('No ordering is possible for Points.')

    def __hash__(self):
        cdef long ret = self.data.x + IntPoint_maxint
        ret = 2 * IntPoint_maxint * ret + self.data.y + IntPoint_maxint
        return 2 * IntPoint_maxint * ret + self.data.z + IntPoint_maxint

    def py_is_equal(self, IntPoint p):
        return self.is_equal(p)

    def py_diff(self, IntPoint p):
        return self.diff(p)

    def py_copy(self):
        return self.copy()
