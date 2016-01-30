from contextlib import contextmanager
from distutils.sysconfig import get_config_var
import os
from os.path import join, exists
import shutil
import sys
import tempfile
from textwrap import dedent
from multiprocessing import Pool
from unittest import TestCase, main

try:
    from unittest import mock
except ImportError:
    import mock

from pysph.base.ext_module import get_md5, ExtModule


def _check_write_source(root):
    """Used to create an ExtModule and test if a file was opened.

    It returns the number of times "open" was called.
    """
    m = mock.mock_open()
    with mock.patch('pysph.base.ext_module.open', m, create=True):
        s = ExtModule("print 'hello'", root=root)
    if m.called:
        with open(*m.call_args[0]) as fp:
            fp.write("junk")
    return m.call_count

def _check_compile(root):
    with mock.patch('shutil.copy') as m:
        s = ExtModule("print 'hello'", root=root)
        s.build()
    if m.called:
        # If it was called, do the copy to mimic the action.
        shutil.copy(*m.call_args[0])
    return m.call_count


class TestMiscExtMod(TestCase):
    def test_md5(self):
        data = "hello world"
        # Two calls with same data produce same result
        self.assertEqual(get_md5(data), get_md5(data))
        # Two calls with different data produce different md5sums.
        self.assertNotEqual(get_md5(data), get_md5(data + ' '))


class TestExtModule(TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.data = dedent('''\
        def f():
            return "hello world"
        ''')

    def tearDown(self):
        if sys.platform.startswith('win'):
            from exceptions import WindowsError
            try:
                shutil.rmtree(self.root)
            except WindowsError:
                pass
        else:
            shutil.rmtree(self.root)

    def test_constructor(self):
        data = self.data
        s = ExtModule(data, root=self.root)
        self.assertTrue(exists(join(self.root, 'build')))

        self.assertEqual(s.hash, get_md5(data))
        self.assertEqual(s.code, data)
        expect_name = 'm_%s'%(s.hash)
        self.assertEqual(s.name, expect_name)
        self.assertEqual(s.src_path, join(self.root, expect_name +'.pyx'))
        self.assertEqual(s.ext_path,
                         join(self.root, expect_name + get_config_var('SO')))

        self.assertTrue(exists(s.src_path))
        self.assertEqual(data, open(s.src_path).read())

    def test_default_root(self):
        try:
            data = self.data
            s = ExtModule(data)
            self.assertTrue(exists(join(s.root, 'build')))
            self.assertEqual(s.hash, get_md5(data))
            self.assertEqual(s.code, data)
            self.assertTrue(exists(s.src_path))
            self.assertEqual(data, open(s.src_path).read())
        finally:
            os.unlink(s.src_path)

    def test_load_module(self):
        data = self.data
        s = ExtModule(data, root=self.root)
        mod = s.load()
        self.assertEqual(mod.f(), "hello world")
        self.assertTrue(exists(s.ext_path))

    def _create_dummy_module(self):
        code = "def hello(): return 'hello'"
        modname = 'test_rebuild.py'
        f = join(self.root, modname)
        with open(f, 'w') as fp:
            fp.write(code)
        return f

    @contextmanager
    def _add_root_to_sys_path(self):
        import sys
        if self.root not in sys.path:
            sys.path.insert(0, self.root)
        try:
            yield
        finally:
            sys.path.remove(self.root)

    def test_rebuild_when_dependencies_change(self):
        # Given.
        data = self.data
        depends = ["test_rebuild"]
        s = ExtModule(data, root=self.root, depends=depends)
        fname = self._create_dummy_module()
        f_stat = os.stat(fname)

        with self._add_root_to_sys_path():
            # When
            self.assertTrue(s.should_recompile())
            s.build()

            # Then.
            self.assertFalse(s.should_recompile())

            # Now lets re-create the module and try again.

            # When.
            fname = self._create_dummy_module()
            # Update the timestamp to make it newer, otherwise we need to
            # sleep.
            os.utime(fname, (f_stat.st_atime, f_stat.st_mtime + 10))

            # Then.
            self.assertTrue(s.should_recompile())

    def test_that_multiple_writes_do_not_occur_for_same_source(self):
        # Given
        n_proc = 5
        p = Pool(n_proc)

        # When

        # Note that _create_extension cannot be defined here or even in the
        # class as a nested function or instance method cannot be pickled.

        result = p.map(_check_write_source, [self.root]*n_proc)
        p.close()

        # Then
        # The file should have been opened only once.
        self.assertEqual(sum(result), 1)

    def test_that_multiple_compiles_do_not_occur_for_same_source(self):
        # Given
        n_proc = 5
        p = Pool(n_proc)

        # When

        # Note that _check_compile cannot be defined here or even in the
        # class as a nested function or instance method cannot be pickled.

        result = p.map(_check_compile, [self.root]*n_proc)
        p.close()

        # Then
        # The shutil.copy should have been run only once.
        self.assertEqual(sum(result), 1)


if __name__ == '__main__':
    main()
