"""Tests for the configuration.
"""
from unittest import TestCase, main

from ..config import Config, get_config, set_config, use_config


class ConfigTestCase(TestCase):

    def setUp(self):
        # Unset any default configuration.
        set_config(None)
        self.config = Config()

    def tearDown(self):
        # Unset any default configuration.
        set_config(None)

    def test_use_openmp_config_default(self):
        # Given
        config = self.config
        # When
        # Then
        self.assertFalse(config.use_openmp)

    def test_set_get_use_openmp_config(self):
        # Given
        config = self.config
        # When
        config.use_openmp = 10
        # Then
        self.assertEqual(config.use_openmp, 10)

    def test_set_get_omp_schedule_config(self):
        # Given
        config = self.config
        # When
        config.omp_schedule = ("static", 10)
        # Then
        self.assertEqual(config.omp_schedule, ("static", 10))

    def test_set_string_omp_schedule(self):
        # Given
        config = self.config
        # When
        config.set_omp_schedule("dynamic,20")
        # Then
        self.assertEqual(config.omp_schedule, ("dynamic", 20))

    def test_set_omp_schedule_config_exception(self):
        # Given
        config = self.config
        # When
        # Then
        with self.assertRaises(ValueError):
            config.omp_schedule = ("random", 20)

    def test_use_opencl_config_default(self):
        # Given
        config = self.config
        # When
        # Then
        self.assertFalse(config.use_opencl)

    def test_set_get_use_opencl_config(self):
        # Given
        config = self.config
        # When
        config.use_opencl = 10
        # Then
        self.assertEqual(config.use_opencl, 10)

    def test_use_double_config_default(self):
        # Given
        config = self.config
        # When
        # Then
        self.assertFalse(config.use_double)

    def test_set_get_use_double_config(self):
        # Given
        config = self.config
        # When
        config.use_double = 10
        # Then
        self.assertEqual(config.use_double, 10)

    def test_default_global_config_is_really_global(self):
        # Given.
        config = get_config()
        self.assertTrue(isinstance(config, Config))

        # When
        config.use_openmp = 100

        # Then.
        config1 = get_config()
        self.assertEqual(config1.use_openmp, 100)

    def test_set_global(self):
        # Given.
        self.config.use_openmp = 200
        set_config(self.config)

        # When
        config = get_config()

        # Then.
        self.assertEqual(config.use_openmp, 200)

    def test_use_config(self):
        # Given
        self.config.use_openmp = 200
        set_config(self.config)

        # When/Then
        with use_config(use_openmp=300) as cfg:
            config = get_config()
            self.assertEqual(config.use_openmp, 300)
            self.assertEqual(cfg.use_openmp, 300)
            cfg.use_openmp = 100
            cfg.use_double = False
            self.assertEqual(config.use_openmp, 100)
            self.assertEqual(config.use_double, False)

        # Then
        self.assertEqual(get_config().use_openmp, 200)


if __name__ == '__main__':
    main()
