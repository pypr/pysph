"""Tests for the configuration.
"""
import sys
from unittest import TestCase, main

from pysph.base.config import Config, get_config, set_config

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


if __name__ == '__main__':
    main()
