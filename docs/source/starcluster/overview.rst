.. _starcluster-docs:

==============================
Using StarCluster with PySPH
==============================

StarCluster is an open source cluster-computing toolkit for Amazon’s Elastic
Compute Cloud (EC2). StarCluster has been designed to simplify the process of
building, configuring, and managing clusters of virtual machines on Amazon’s
EC2 cloud.

Using StarCluster along with PySPH's MPI support, you can run PySPH code on
multiple instances in parallel and complete simulations faster.

.. contents::
    :local:
    :depth: 1

Installing StarCluster
++++++++++++++++++++++

StarCluster can be installed via pip as ::

  $ pip install starcluster

Configuring StarCluster
+++++++++++++++++++++++

Creating Configuration File
```````````````````````````

After StarCluster has been installed, the next step is to update your
StarCluster configuration ::

  $ starcluster help

  StarCluster - (http://star.mit.edu/cluster)
  Software Tools for Academics and Researchers (STAR)
  Please submit bug reports to starcluster@mit.edu
  cli.py:87 - ERROR - config file /home/user/.starcluster/config does not exist
  Options:
  --------
  [1] Show the StarCluster config template
  [2] Write config template to /home/user/.starcluster/config
  [q] Quit
  Please enter your selection:

Select the second option by typing 2 and press enter. This will give you a
template to use to create a configuration file containing your AWS credentials,
cluster settings, etc. The next step is to customize this file using your
favorite text-editor ::

  $ emacs ~/.starcluster/config

Updating AWS Credentials
````````````````````````

This file is commented with example “cluster templates”. A cluster template
defines a set of configuration settings used to start a new cluster. The config
template provides a smallcluster template that is ready to go
out-of-the-box. However, first, you must fill in your AWS credentials and
keypair info ::

  [aws info]
  aws_access_key_id = # your aws access key id here
  aws_secret_access_key = # your secret aws access key here
  aws_user_id = # your 12-digit aws user id here

To find your AWS User ID, see `Finding your Account Canonical User ID
<http://docs.aws.amazon.com/general/latest/gr/acct-identifiers.html#FindingCanonicalId>`_

You can get your root user credentials from the `Security Credentials
<https://console.aws.amazon.com/iam/home?#security_credential>`_ page on AWS
Management Console. However, root credentials allow for full access to all
resources on your account and it is recommended that you create separate IAM
(Identity and Access Management) user credentials for managing access to your
EC2 resources. To create IAM user credentials, see `Creating IAM Users
(Console)
<http://docs.aws.amazon.com/IAM/latest/UserGuide/id_users_create.html#id_users_create_console>`_
For StarCluster, create an IAM user with the ``EC2 Full Access`` permission.


If you don't already have a keypair, you can generate one using StarCluster by
running::

  $ starcluster createkey mykey -o ~/.ssh/mykey.rsa

This will create a keypair called mykey on Amazon EC2 and save the private key
to ~/.ssh/mykey.rsa. Once you have a key the next step is to fill in your
keypair info in the StarCluster config file ::

  [key mykey]
  key_location = ~/.ssh/mykey.rsa

Also, update the following information for the smallcluster configuration::

  [cluster smallcluster]
  ..
  KEYNAME = mykey
  ..

Now that the basic configuration for StarCluster is complete, you can directly
launch instances using StarCluster. However, note that EC2 charges are not pro
rata and you will be charged for an entire hour even if you run an instance for
a few minutes. Before attempting to deploy an instance/cluster you can modify
the following information in your cluster configuration::

  [cluster smallcluster]
  ..
  NODE_INSTANCE_TYPE=t2.micro
  NODE_IMAGE_ID=ami-6b211202
  ..

Now you can launch an EC2 instance using::

  $ starcluster start smallcluster

You can SSH into the master node by running::

  $ starcluster sshmaster smallcluster

You can transfer files to the nodes using the ``get`` and ``put`` commands as::

  $ starcluster put /path/to/local/file/or/dir /remote/path/
  $ starcluster get /path/to/remote/file/or/dir /local/path/

Finally, you can terminate the instance by running::

  $ starcluster terminate smallcluster

Setting up PySPH for StarCluster
++++++++++++++++++++++++++++++++

Most of the public AMIs currently distributed for StarCluster are outdated and
have reached their end of life. To ensure a hassle-free experience while
further extending the AMI and installing packages, you can use the 64 bit
Ubuntu 16.04 AMI with AMI ID ``ami-01fdc27a`` which has most StarCluster
dependencies and PySPH dependencies installed.

Base AMI for PySPH [Optional]
`````````````````````````````

The ``ami.sh`` file which can be found in the ``starcluster`` directory in the
PySPH repository automatically launches a vanilla 64-bit Ubuntu 16.04 instance,
installs any necessary StarCluster and PySPH dependencies and saves an AMI with
this configuration on your AWS account ::

  $ ./ami.sh

The AMI ID of the generated image is stored in ``AMI_ID``. You can also see a
list of the AMIs currently in your AWS account by running ::

  $ starcluster listimages

Cluster configuration for PySPH
```````````````````````````````

Modify your StarCluster configuration file with the following
information. Launching a cluster with the following configuration will start 2
t2.micro instances, install the latest version of PySPH in each and keep track
of the nodes loaded in ``/home/pysph/PYSPH_HOSTS``::

   [cluster pysphcluster]
   KEYNAME = mykey
   CLUSTER_SIZE = 2 # Number of nodes in cluster
   CLUSTER_USER = pysph
   CLUSTER_SHELL = bash
   NODE_IMAGE_ID = ami-01fdc27a # Or AMI ID for base AMI generated previously
   NODE_INSTANCE_TYPE = t2.micro # EC2 Instance type
   PLUGINS = pysph_install

   [plugin pysph_install]
   setup_class = sc_pysph.PySPHInstaller

Also, copy ``sc_pysph.py`` from the ``starcluster`` directory to
``~/.starcluster/plugins/``

Running PySPH scripts on a cluster
++++++++++++++++++++++++++++++++++

You can start the cluster configured previously by running ::

  $ starcluster start -c pysphcluster cluster

Assuming your PySPH file ``cube.py`` is in the local home directory, you can
first transfer this file to the cluster::

  $ starcluster put -u pysph cluster ~/cube.py /home/pysph/cube.py

Then run the PySPH code as::

  $ starcluster sshmaster -u pysph cluster "mpirun -n 2 --hostfile ~/PYSPH_HOSTS python ~/cube.py"

Finally, you can get the output generated by PySPH back by running::

  $ starcluster get -u pysph cluster /home/pysph/cube_output .
