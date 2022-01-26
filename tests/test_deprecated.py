import dddm
from unittest import TestCase
import os


class TestContext(TestCase):
    def test_context_exists(self):
        assert dddm.context.context

    def test_load_context(self):
        dddm.context.get_stbc_context(check=False)

    def test_load_with_check(self):
        with self.assertRaises((FileNotFoundError, ValueError)):
            dddm.context.get_stbc_context(check=True)
        tempdir = os.environ.copy().get('TMPDIR')
        os.environ.update({'TMPDIR': '.'})

        with self.assertRaises((FileNotFoundError, ValueError)):
            dddm.context.get_stbc_context(check=True)
        if tempdir:
            os.environ.update({'TMPDIR': tempdir})
        else:
            del os.environ['TMPDIR']
