from starcluster.clustersetup import DefaultClusterSetup
from starcluster.logger import log


class PySPHInstallerBase(DefaultClusterSetup):
    PYSPH_PROFILE = "/etc/profile.d/pysph.sh"
    PYSPH_HOSTS = "/home/pysph/PYSPH_HOSTS"
    PYSPH_USER = "pysph"

    def _create_env(self, master):
        master.ssh.execute(
            r"""
            echo $HOME
            if [ ! -d ~/pysph_env ]; then
            mkdir ~/pysph_env &&
            virtualenv --system-site-packages ~/pysph_env;
            fi
            """
        )

    def _install_pysph(self, master):
        commands = r"""
          . ~/pysph_env/bin/activate
          if ! python -c "import pysph" &> /dev/null; then
          export USE_TRILINOS=1
          export ZOLTAN_INCLUDE=/usr/include/trilinos
          export ZOLTAN_LIBRARY=/usr/lib/x86_64-linux-gnu
          cd ~ &&
          git clone https://github.com/pypr/pysph &&
          cd pysph &&
          python setup.py install
          fi
          """
        master.ssh.execute(commands)

    def _configure_profile(self, node):
        pysph_profile = node.ssh.remote_file(self.PYSPH_PROFILE, 'w')
        pysph_profile.write("test -e ~/.bashrc && . ~/.bashrc")
        pysph_profile.close()


class PySPHInstaller(PySPHInstallerBase):
    def run(self, nodes, master, user, user_shell, volumes):
        aliases = [n.alias for n in nodes]

        log.info("Configuring PYSPH Profile")
        for node in nodes:
            self.pool.simple_job(self._configure_profile,
                                 (node,))
        self.pool.wait(len(nodes))

        master.ssh.switch_user(self.PYSPH_USER)
        log.info("Creating virtual environment")
        self._create_env(master)
        master.ssh.execute("echo '. ~/pysph_env/bin/activate' > ~/.bashrc")

        log.info("Installing PySPH")
        self._install_pysph(master)

        log.info("Adding nodes to PYSPH hosts file")
        pysph_hosts = master.ssh.remote_file(self.PYSPH_HOSTS, 'w')
        pysph_hosts.write('\n'.join(aliases) + '\n')

    def on_add_node(self, new_node, nodes, master, user, user_shell, volumes):
        log.info("Configuring PYSPH Profile")
        self._configure_profile(new_node)

        master.ssh.switch_user(self.PYSPH_USER)
        log.info("Adding %s to PYSPH hosts file" % new_node.alias)
        pysph_hosts = master.ssh.remote_file(self.PYSPH_HOSTS, 'a')
        pysph_hosts.write(new_node.alias + '\n')
        pysph_hosts.close()

    def on_remove_node(self, remove_node, nodes, master,
                       user, user_shell, volumes):
        master.switch_user(self.PYSPH_USER)
        log.info("Removing %s from PYSPH hosts file" % remove_node.alias)
        master.ssh.remove_lines_from_file(self.PYSPH_HOSTS, remove_node.alias)
