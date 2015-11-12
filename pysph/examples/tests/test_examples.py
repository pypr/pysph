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
        subprocess.check_output(cmd)
    finally:
        shutil.rmtree(out_dir)


@attr(slow=True)
def test_example_should_run():
    for module, doc in run.get_all_examples():
        yield run_example, module
