import DirectDmTargets as dddm
from unittest import TestCase
import os


class TestAddPid(TestCase):
    def test_add_pid_to_csv_filename(self):
        with self.assertRaises(AssertionError):
            dddm.utils.add_pid_to_csv_filename('bla')

    def test_add_tmp(self):
        dddm.utils.add_pid_to_csv_filename(os.path.join(dddm.context._naive_tmp, 'bla.csv'))

    def test_add_no_dir(self):
        dddm.utils.add_pid_to_csv_filename('bla.csv')

    def test_add_bad_dir(self):
        dddm.utils.add_pid_to_csv_filename('bla.csv')

    def test_host(self):
        dddm.utils.add_host_and_pid_to_csv_filename('bla.csv')


class TestContext(TestCase):
    def test_context_exists(self):
        assert dddm.context.context

    def test_load_context(self):
        dddm.context.get_stbc_context(check=False)

    def test_load_with_check(self):
        with self.assertRaises((FileNotFoundError, ValueError)):
            dddm.context.get_stbc_context(check=True)

        os.environ.update({'TMPDIR': '.'})

        with self.assertRaises((FileNotFoundError, ValueError)):
            dddm.context.get_stbc_context(check=True)
