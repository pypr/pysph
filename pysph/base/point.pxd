cimport numpy

cdef extern from "math.h":
    double sqrt(double) nogil
    double ceil(double) nogil

cdef extern from 'limits.h':
    cdef int INT_MAX

cdef struct cPoint:
    double x, y, z

cdef inline cPoint cPoint_new(double x, double y, double z):
    cdef cPoint p = cPoint(x,y,z)
    return p

cdef inline cPoint cPoint_sub(cPoint pa, cPoint pb):
    return cPoint_new(pa.x-pb.x, pa.y-pb.y, pa.z-pb.z)

cdef inline cPoint cPoint_add(cPoint pa, cPoint pb):
    return cPoint_new(pa.x+pb.x, pa.y+pb.y, pa.z+pb.z)

cdef inline double cPoint_dot(cPoint pa, cPoint pb):
    return pa.x*pb.x + pa.y*pb.y + pa.z*pb.z

cdef inline double cPoint_norm(cPoint p):
    return p.x*p.x + p.y*p.y + p.z*p.z

cdef inline double cPoint_distance(cPoint pa, cPoint pb):
    return sqrt((pa.x-pb.x)*(pa.x-pb.x) +
                (pa.y-pb.y)*(pa.y-pb.y) +
                (pa.z-pb.z)*(pa.z-pb.z)
                )

cdef inline double cPoint_distance2(cPoint pa, cPoint pb):
    return ((pa.x-pb.x)*(pa.x-pb.x) + (pa.y-pb.y)*(pa.y-pb.y) +
                    (pa.z-pb.z)*(pa.z-pb.z))

cdef inline double cPoint_length(cPoint pa):
    return sqrt(cPoint_norm(pa))

cdef inline cPoint cPoint_scale(cPoint p, double k):
    return cPoint_new(p.x*k, p.y*k, p.z*k)

cdef inline cPoint normalized(cPoint p):
    cdef double norm = cPoint_length(p)
    return cPoint_new(p.x/norm, p.y/norm, p.z/norm)

cdef class Point:
    """ Class to represent point in 3D. """
    cdef cPoint data
    cpdef set(self, double x, double y, double z)
    cdef set_from_cPoint(self, cPoint value)
    cpdef numpy.ndarray asarray(self)
    cpdef double norm(self)
    cpdef double length(self)
    cpdef double dot(self, Point p)
    cpdef Point cross(self, Point p)
    cpdef double distance(self, Point p)
    cdef cPoint to_cPoint(self)

cdef inline Point Point_new(double x, double y, double z):
    cdef Point p = Point.__new__(Point)
    p.x = x
    p.y = y
    p.z = z
    return p

cdef inline Point Point_sub(Point pa, Point pb):
    return Point_new(pa.x-pb.x, pa.y-pb.y, pa.z-pb.z)

cdef inline Point Point_add(Point pa, Point pb):
    return Point_new(pa.x+pb.x, pa.y+pb.y, pa.z+pb.z)

cdef inline double Point_length(Point p):
    return sqrt(p.x*p.x + p.y*p.y + p.z*p.z)

cdef inline double Point_length2(Point p):
    return p.x*p.x + p.y*p.y + p.z*p.z

cdef inline double Point_distance(Point pa, Point pb):
    return sqrt((pa.x-pb.x)*(pa.x-pb.x) +
                (pa.y-pb.y)*(pa.y-pb.y) +
                (pa.z-pb.z)*(pa.z-pb.z)
                )

cdef inline double Point_distance2(Point pa, Point pb):
    return ((pa.x-pb.x)*(pa.x-pb.x) + (pa.y-pb.y)*(pa.y-pb.y) +
                    (pa.z-pb.z)*(pa.z-pb.z))

cdef inline Point Point_from_cPoint(cPoint p):
    return Point_new(p.x, p.y, p.z)




cdef struct cIntPoint:
    int x, y, z

cdef inline cIntPoint cIntPoint_new(int x, int y, int z):
    cdef cIntPoint p = cIntPoint(x,y,z)
    return p

cdef inline cIntPoint cIntPoint_sub(cIntPoint pa, cIntPoint pb):
    return cIntPoint_new(pa.x-pb.x, pa.y-pb.y, pa.z-pb.z)

cdef inline cIntPoint cIntPoint_add(cIntPoint pa, cIntPoint pb):
    return cIntPoint_new(pa.x+pb.x, pa.y+pb.y, pa.z+pb.z)

cdef inline long cIntPoint_dot(cIntPoint pa, cIntPoint pb):
    return pa.x*pb.x + pa.y*pb.y + pa.z*pb.z

cdef inline long cIntPoint_norm(cIntPoint p):
    return p.x*p.x + p.y*p.y + p.z*p.z

cdef inline double cIntPoint_length(cIntPoint pa):
    return sqrt(<double>cIntPoint_norm(pa))

cdef inline long cIntPoint_distance2(cIntPoint pa, cIntPoint pb):
    return ((pa.x-pb.x)*(pa.x-pb.x) +
            (pa.y-pb.y)*(pa.y-pb.y) +
            (pa.z-pb.z)*(pa.z-pb.z))

cdef inline double cIntPoint_distance(cIntPoint pa, cIntPoint pb):
    return sqrt(<double>cIntPoint_distance2(pa, pb))

cdef inline cIntPoint cIntPoint_scale(cIntPoint p, int k):
    return cIntPoint_new(p.x*k, p.y*k, p.z*k)

cdef inline bint cIntPoint_is_equal(cIntPoint pa, cIntPoint pb):
    return (pa.x == pb.x and pa.y == pb.y and pa.z == pb.z)

cdef class IntPoint:
    cdef cIntPoint data

    cpdef numpy.ndarray asarray(self)
    cdef bint is_equal(self, IntPoint)
    cdef IntPoint diff(self, IntPoint)
    cdef tuple to_tuple(self)
    cdef IntPoint copy(self)

cdef inline IntPoint IntPoint_sub(IntPoint pa, IntPoint pb):
    return IntPoint_new(pa.x-pb.x, pa.y-pb.y, pa.z-pb.z)

cdef inline IntPoint IntPoint_add(IntPoint pa, IntPoint pb):
    return IntPoint_new(pa.x+pb.x, pa.y+pb.y, pa.z+pb.z)

cdef inline IntPoint IntPoint_new(int x, int y, int z):
    cdef IntPoint p = IntPoint.__new__(IntPoint)
    p.data.x = x
    p.data.y = y
    p.data.z = z
    return p

cdef inline IntPoint IntPoint_from_cIntPoint(cIntPoint p):
    return IntPoint_new(p.x, p.y, p.z)
