# Copyright 2008, 2009 Neil Martinsen-Burrell
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""Defines a file-derived class to read/write Fortran unformatted files taken from

http://wiki.scipy.org/Cookbook/FortranIO/FortranFile

The assumption is that a Fortran unformatted file is being written by
the Fortran runtime as a sequence of records.  Each record consists of
an integer (of the default size [usually 32 or 64 bits]) giving the
length of the following data in bytes, then the data itself, then the
same integer as before.

Examples
--------

To use the default endian and size settings, one can just do::
    >>> f = FortranFile('filename')
    >>> x = f.readReals()

One can read arrays with varying precisions::
    >>> f = FortranFile('filename')
    >>> x = f.readInts('h')
    >>> y = f.readInts('q')
    >>> z = f.readReals('f')
Where the format codes are those used by Python's struct module.

One can change the default endian-ness and header precision::
    >>> f = FortranFile('filename', endian='>', header_prec='l')
for a file with little-endian data whose record headers are long
integers.

"""
import struct
import numpy

class FortranFile(file):

    """File with methods for dealing with fortran unformatted data files"""

    def _get_header_length(self):
        return struct.calcsize(self._header_prec)
    _header_length = property(fget=_get_header_length)

    def _set_endian(self,c):
        """Set endian to big (c='>') or little (c='<') or native (c='@')

        :Parameters:
          `c` : string
            The endian-ness to use when reading from this file.
        """
        if c in '<>@=':
            self._endian = c
        else:
            raise ValueError('Cannot set endian-ness')
    def _get_endian(self):
        return self._endian
    ENDIAN = property(fset=_set_endian,
                      fget=_get_endian,
                      doc="Possible endian values are '<', '>', '@', '='"
                     )

    def _set_header_prec(self, prec):
        if prec in 'hilq':
            self._header_prec = prec
        else:
            raise ValueError('Cannot set header precision')
    def _get_header_prec(self):
        return self._header_prec
    HEADER_PREC = property(fset=_set_header_prec,
                           fget=_get_header_prec,
                           doc="Possible header precisions are 'h', 'i', 'l', 'q'"
                          )

    def __init__(self, fname, endian='@', header_prec='i', *args, **kwargs):
        """Open a Fortran unformatted file for writing.
        
        Parameters
        ----------
        endian : character, optional
            Specify the endian-ness of the file.  Possible values are
            '>', '<', '@' and '='.  See the documentation of Python's
            struct module for their meanings.  The deafult is '>' (native
            byte order)
        header_prec : character, optional
            Specify the precision used for the record headers.  Possible
            values are 'h', 'i', 'l' and 'q' with their meanings from
            Python's struct module.  The default is 'i' (the system's
            default integer).

        """
        file.__init__(self, fname, *args, **kwargs)
        self.ENDIAN = endian
        self.HEADER_PREC = header_prec

    def _read_exactly(self, num_bytes):
        """Read in exactly num_bytes, raising an error if it can't be done."""
        data = ''
        while True:
            l = len(data)
            if l == num_bytes:
                return data
            else:
                read_data = self.read(num_bytes - l)
            if read_data == '':
                raise IOError('Could not read enough data.'
                              '  Wanted %d bytes, got %d.' % (num_bytes, l))
            data += read_data

    def _read_check(self):
        return struct.unpack(self.ENDIAN+self.HEADER_PREC,
                             self._read_exactly(self._header_length)
                            )[0]

    def _write_check(self, number_of_bytes):
        """Write the header for the given number of bytes"""
        self.write(struct.pack(self.ENDIAN+self.HEADER_PREC,
                               number_of_bytes))

    def readRecord(self):
        """Read a single fortran record"""
        l = self._read_check()
        data_str = self._read_exactly(l)
        check_size = self._read_check()
        if check_size != l:
            raise IOError('Error reading record from data file')
        return data_str

    def writeRecord(self,s):
        """Write a record with the given bytes.

        Parameters
        ----------
        s : the string to write

        """
        length_bytes = len(s)
        self._write_check(length_bytes)
        self.write(s)
        self._write_check(length_bytes)

    def readString(self):
        """Read a string."""
        return self.readRecord()

    def writeString(self,s):
        """Write a string

        Parameters
        ----------
        s : the string to write
        
        """
        self.writeRecord(s)

    _real_precisions = 'df'

    def readReals(self, prec='f'):
        """Read in an array of real numbers.
        
        Parameters
        ----------
        prec : character, optional
            Specify the precision of the array using character codes from
            Python's struct module.  Possible values are 'd' and 'f'.
            
        """
        
        _numpy_precisions = {'d': numpy.float64,
                             'f': numpy.float32
                            }

        if prec not in self._real_precisions:
            raise ValueError('Not an appropriate precision')
            
        data_str = self.readRecord()
        num = len(data_str)/struct.calcsize(prec)
        numbers =struct.unpack(self.ENDIAN+str(num)+prec,data_str) 
        return numpy.array(numbers, dtype=_numpy_precisions[prec])

    def writeReals(self, reals, prec='f'):
        """Write an array of floats in given precision

        Parameters
        ----------
        reals : array
            Data to write
        prec` : string
            Character code for the precision to use in writing
        """
        if prec not in self._real_precisions:
            raise ValueError('Not an appropriate precision')
        
        # Don't use writeRecord to avoid having to form a
        # string as large as the array of numbers
        length_bytes = len(reals)*struct.calcsize(prec)
        self._write_check(length_bytes)
        _fmt = self.ENDIAN + prec
        for r in reals:
            self.write(struct.pack(_fmt,r))
        self._write_check(length_bytes)
    
    _int_precisions = 'hilq'

    def readInts(self, prec='i'):
        """Read an array of integers.
        
        Parameters
        ----------
        prec : character, optional
            Specify the precision of the data to be read using 
            character codes from Python's struct module.  Possible
            values are 'h', 'i', 'l' and 'q'
            
        """
        if prec not in self._int_precisions:
            raise ValueError('Not an appropriate precision')
            
        data_str = self.readRecord()
        num = len(data_str)/struct.calcsize(prec)
        return numpy.array(struct.unpack(self.ENDIAN+str(num)+prec,data_str))

    def writeInts(self, ints, prec='i'):
        """Write an array of integers in given precision

        Parameters
        ----------
        reals : array
            Data to write
        prec : string
            Character code for the precision to use in writing
        """
        if prec not in self._int_precisions:
            raise ValueError('Not an appropriate precision')
        
        # Don't use writeRecord to avoid having to form a
        # string as large as the array of numbers
        length_bytes = len(ints)*struct.calcsize(prec)
        self._write_check(length_bytes)
        _fmt = self.ENDIAN + prec
        for item in ints:
            self.write(struct.pack(_fmt,item))
        self._write_check(length_bytes)
