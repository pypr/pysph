# Install and update required packages
apt-get -y update
apt-get -y upgrade
apt-get install -y build-essential \
        g++ \
        python \
        python-dev \
        python-setuptools \
        nfs-kernel-server \
        nfs-common \
        rpcbind \
        upstart \
        cython \
        cython3 \
        git \
        ipython \
        ipython3 \
        libgomp1 \
        libopenmpi-dev \
        libtrilinos-zoltan-dev \
        mayavi2 \
        python \
        python-execnet \
        python-h5py \
        python-mako \
        python-matplotlib \
        python-mock \
        python-mpi4py \
        python-nose \
        python-numpy \
        python-pip \
        python-psutil \
        python-qt4 \
        python-unittest2 \
        python3 \
        python3-h5py \
        python3-mako \
        python3-matplotlib \
        python3-mpi4py \
        python3-nose \
        python3-numpy \
        python3-pip \
        python3-psutil \
        sudo \
        tox \
        vim \
        wget \
        virtualenv \
    && rm -rf /var/lib/apt/lists/*

# Starcluster seems to look for /etc/init.d/nfs
# Create a symbolic link to point to the right file
ln -s /etc/init.d/nfs-kernel-server /etc/init.d/nfs
ln -s /lib/systemd/system/nfs-kernel-server.service /lib/systemd/system/nfs.service

# rpcbind is shipped instead of portmap on recent Debian installations
# http://star.mit.edu/cluster/mlarchives/2545.html
echo 'exit 0' > /etc/init.d/portmap
chmod +x /etc/init.d/portmap

# Download sge
curl -L https://github.com/brunogrande/starcluster-ami-config/blob/master/sge.tar.gz?raw=true | sudo tar -xz -C /opt/

# Create users
adduser --disabled-password --gecos '' pysph && \
adduser pysph sudo && \
echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
