import tempfile
import shutil
import subprocess
import sys

from nose.plugins.attrib import attr

from pysph.examples import run

def run_example(module):
    """This simply runs the example to make sure that the example executes
    correctly.  It wipes out the generated output directory.
    """
    out_dir = tempfile.mkdtemp()
    cmd = [sys.executable, "-m", module, "--max-steps", "1",
           "--disable-output", "-q", "-d", out_dir]
    try:
        subprocess.check_output(' '.join(cmd), shell=True)
    finally:
        shutil.rmtree(out_dir)

def _has_tvtk():
    try:
        from tvtk.api import tvtk
    except ImportError:
        return False
    else:
        return True

@attr(slow=True)
def test_example_should_run():
    for module, doc in run.get_all_examples():
        if module == 'pysph.examples.rigid_body.dam_break3D_sph' and \
           not _has_tvtk():
                continue
        yield run_example, module
