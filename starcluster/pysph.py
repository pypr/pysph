from starcluster.clustersetup import DefaultClusterSetup
from starcluster.logger import log


class PySPHInstallerBase(DefaultClusterSetup):
    # PYSPH_PROFILE = '/etc/profile.d/pysph.sh'
    PYSPH_HOSTS = "/home/pysph/PYSPH_HOSTS"

    def _create_env(self, master):
        master.ssh.execute(
            r"""if [ ! -d '/home/pysph/pysph_env' ]; then
            mkdir '/home/pysph/pysph_env' && virtualenv /home/pysph/pysph_env;
            fi""")

    def _install_pysph(self, master):
        commands = r"""
          source '/home/pysph/pysph_env/bin/activate'
          if ! python -c "import pysph" &> /dev/null; then
          export USE_TRILINOS=1
          export ZOLTAN_INCLUDE=/usr/include/trilinos
          export ZOLTAN_LIBRARY=/usr/lib/x86_64-linux-gnu
          cd /home/pysph &&
          git clone https://github.com/pypr/pysph &&
          cd pysph &&
          pip install -r requirements.txt &&
          pip install mpi4py &&
          python setup.py install
          fi"""
        master.ssh.execute(commands)

    def _configure_python(self, node):
        node.ssh.execute(
            r"""sudo update-alternatives --install
            /usr/bin/python python /home/pysph/pysph_env/bin/python2.7 1""")


class PySPHInstaller(PySPHInstallerBase):
    def run(self, nodes, master, user, user_shell, volumes):
        aliases = [n.alias for n in nodes]

        master.ssh.switch_user("pysph")
        log.info("Creating virtual environment")
        self._create_env(master)

        for node in nodes:
            log.info("Updating python link on %s" % node.alias)
            self.pool.simple_job(self._configure_python, (node,))
        self.pool.wait(len(nodes))

        log.info("Installing PySPH")
        self._install_pysph(master)

        log.info("Adding nodes to PYSPH hosts file")
        pysph_hosts = master.ssh.remote_file(self.PYSPH_HOSTS, 'w')
        pysph_hosts.write('\n'.join(aliases) + '\n')

    def on_add_node(self, new_node, nodes, master, user, user_shell, volumes):
        log.info("Updating python link on %s" % new_node.alias)
        self._configure_python(new_node)

        log.info("Adding %s to PYSPH hosts file" % new_node.alias)
        pysph_hosts = master.ssh.remote_file(self.PYSPH_HOSTS, 'a')
        pysph_hosts.write(new_node.alias + '\n')
        pysph_hosts.close()

    def on_remove_node(self, remove_node, nodes, master,
                       user, user_shell, volumes):
        log.info("Removing %s from PYSPH hosts file" % remove_node.alias)
        master.ssh.remove_lines_from_file(self.PYSPH_HOSTS, remove_node.alias)
