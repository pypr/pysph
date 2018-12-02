ROOT = $(shell pwd)
MAKEFILE = $(ROOT)/Makefile
SRC = $(ROOT)
PKG1 = $(SRC)/pysph
SUBPKG1 = base sph sph/solid_mech parallel
DIRS := $(foreach dir,$(SUBPKG1),$(PKG1)/$(dir))

# this is used for cython files on recursive call to make
PYX = $(wildcard *.pyx)

MPI4PY_INCL = $(shell python -c "import mpi4py; print mpi4py.get_include()")

# the default target to make
all : build

.PHONY : $(DIRS) bench build

build :
	python setup.py build_ext --inplace

$(DIRS) :
	cd $@;  python $(ROOT)/pyzoltan/core/generator.py
	$(MAKE) -f $(MAKEFILE) -C $@ cythoncpp ROOT=$(ROOT)

%.c : %.pyx
	python `which cython` -I$(SRC) -I$(MPI4PY_INCL) $<

%.cpp : %.pyx
	python `which cython` --cplus -I$(SRC) -I$(MPI4PY_INCL) $<

%.html : %.pyx
	python `which cython` -I$(SRC) -I$(MPI4PY_INCL) -a $<

cython : $(PYX:.pyx=.c)

cythoncpp : $(PYX:.pyx=.cpp)

_annotate : $(PYX:.pyx=.html)

annotate :
	for f in $(DIRS); do $(MAKE) -f $(MAKEFILE) -C $${f} _annotate ROOT=$(ROOT); done

clean :
	python setup.py clean
	-for dir in $(DIRS); do rm -f $$dir/*.c; done
	-for dir in $(DIRS); do rm -f $$dir/*.cpp; done

cleanall : clean
	-for dir in $(DIRS); do rm -f $$dir/*.so; done
#	-rm $(patsubst %.pyx,%.c,$(wildcard $(PKG)/*/*.pyx))

test :
	python `which pytest` -m 'not slow' pysph

testall :
	python `which pytest` pysph

epydoc :
	python cython-epydoc.py --config epydoc.cfg pysph

doc :
	cd docs; make html

develop :
	python setup.py develop

install :
	python setup.py install
