import os
import tempfile
import shutil
import subprocess
import sys

from pytest import mark

from pysph.examples import run


def check_output(*args, **kw):
    """Simple hack to support Python 2.6 which does not have
    subprocess.check_output.
    """
    if not hasattr(subprocess, 'check_output'):
        subprocess.call(*args, **kw)
    else:
        subprocess.check_output(*args, **kw)


def print_safe(string_or_bytes):
    if type(string_or_bytes) is bytes:
        print(string_or_bytes.decode('utf-8'))
    else:
        print(string_or_bytes)


_orig_ets_toolkit = None
def setup_module():
    # Set the ETS_TOOLKIT to null to avoid errors when importing TVTK.
    global _orig_ets_toolkit
    var = 'ETS_TOOLKIT'
    _orig_ets_toolkit = os.environ.get(var)
    os.environ[var] = 'null'


def teardown_module():
    var = 'ETS_TOOLKIT'
    if _orig_ets_toolkit is None:
        del os.environ[var]
    else:
        os.environ[var] = _orig_ets_toolkit


def run_example(module):
    """This simply runs the example to make sure that the example executes
    correctly.  It wipes out the generated output directory.
    """
    out_dir = tempfile.mkdtemp()
    cmd = [sys.executable, "-m", module, "--max-steps", "1",
           "--disable-output", "-q", "-d", out_dir]
    env_vars = dict(os.environ)
    env_vars['ETS_TOOLKIT'] = 'null'
    try:
        check_output(cmd, env=env_vars)
    except subprocess.CalledProcessError as e:
        print_safe(e.stdout)
        print_safe(e.stderr)
        raise
    finally:
        shutil.rmtree(out_dir)


def _has_tvtk():
    try:
        from tvtk.api import tvtk
    except (ImportError, SystemExit):
        return False
    else:
        return True


def _find_examples():
    examples = []
    for module, doc in run.get_all_examples():
        if module == 'pysph.examples.rigid_body.dam_break3D_sph' and \
           not _has_tvtk():
                continue
        examples.append(module)
    return examples


@mark.slow
@mark.parametrize("module", _find_examples())
def test_example_should_run(module):
    run_example(module)
