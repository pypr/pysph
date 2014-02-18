from distutils.sysconfig import get_config_var
import os
from os.path import join, exists
import shutil
import tempfile
from textwrap import dedent
from unittest import TestCase, main

from pysph.base.ext_module import get_md5, ExtModule

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


if __name__ == '__main__':
    main()
