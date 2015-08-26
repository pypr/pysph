import subprocess
import sys
import unittest

from pysph.base.capture_stream import CaptureMultipleStreams, CaptureStream


def write_stderr():
    subprocess.call(
        [sys.executable, "-S", "-s", "-c",
         "import sys;sys.stderr.write('stderr')"]
    )

def write_stdout():
    subprocess.call(
        [sys.executable, "-S", "-s", "-c",
         "import sys;sys.stdout.write('stdout')"]
    )

class TestCaptureStream(unittest.TestCase):
    def test_that_stderr_is_captured_by_default(self):
        # Given
        # When
        with CaptureStream() as stream:
            write_stderr()
        # Then
        self.assertEqual(stream.get_output(), "stderr")

    def test_that_stdout_can_be_captured(self):
        # Given
        # When
        with CaptureStream(sys.stdout) as stream:
            write_stdout()
        # Then
        self.assertEqual(stream.get_output(), "stdout")

    def test_that_output_is_available_in_context_and_outside(self):
        # Given
        # When
        with CaptureStream(sys.stderr) as stream:
            write_stderr()
            # Then
            self.assertEqual(stream.get_output(), "stderr")

        # Then
        self.assertEqual(stream.get_output(), "stderr")

class TestCaptureMultipleStreams(unittest.TestCase):
    def test_that_stdout_stderr_are_captured_by_default(self):
        # Given
        # When
        with CaptureMultipleStreams() as stream:
            write_stderr()
            write_stdout()
        # Then
        outputs = stream.get_output()
        self.assertEqual(outputs[0], "stdout")
        self.assertEqual(outputs[1], "stderr")

    def test_that_order_is_preserved(self):
        # Given
        # When
        with CaptureMultipleStreams((sys.stderr, sys.stdout)) as stream:
            write_stderr()
            write_stdout()
        # Then
        outputs = stream.get_output()
        self.assertEqual(outputs[0], "stderr")
        self.assertEqual(outputs[1], "stdout")


if __name__ == '__main__':
    unittest.main()
