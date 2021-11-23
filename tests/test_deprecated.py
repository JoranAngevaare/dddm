import DirectDmTargets as dddm
from unittest import TestCase


class TestAddPid(TestCase):
    def test_add_pid_to_csv_filename(self):
        with self.assertRaises(AssertionError):
            dddm.utils.add_pid_to_csv_filename('bla')

    def test_add_tmp(self):
        dddm.utils.add_pid_to_csv_filename('/tmp/bla.csv')

    def test_add_no_dir(self):
        dddm.utils.add_pid_to_csv_filename('bla.csv')

    def test_add_bad_dir(self):
        dddm.utils.add_pid_to_csv_filename('bla.csv')

    def test_host(self):
        dddm.utils.add_host_and_pid_to_csv_filename('bla.csv')
