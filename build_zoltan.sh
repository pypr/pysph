#!/bin/sh

ZOLTAN_DIR=$1
BUILD_DIR="/tmp"

URL="http://www.cs.sandia.gov/~kddevin/Zoltan_Distributions/zoltan_distrib_v3.8.tar.gz"

Check()
{
    if [ $# -lt 1 ]; then
        echo "Usage: build_zoltan.sh TARGET_DIR"
        echo "Where TARGET_DIR is where the library and includes are installed."
        exit 1
    fi

    HEADER="$ZOLTAN_DIR/include/zoltan.h"
    if [ -f "$HEADER" ]; then
        echo "$HEADER already exists, skipping build."
        exit 0
    else
        mkdir $ZOLTAN_DIR
    fi
}

Download()
{
    FNAME=`basename $URL`
    if [ -x "/usr/bin/curl" ] ; then
        curl -o $BUILD_DIR/$FNAME $URL
    else
        wget -q -O $BUILD_DIR/$FNAME $URL
    fi
}


Build()
{
    cd $BUILD_DIR
    tar xzf zoltan_distrib*.tar.gz
    cd Zoltan_v3.8
    mkdir build
    cd build
    ../configure --with-cflags=-fPIC --enable-mpi --prefix=$ZOLTAN_DIR
    make install
}

Check "$@"
Download
Build
