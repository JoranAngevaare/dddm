import dddm
from unittest import TestCase


class TestContext(TestCase):
    def test_utils(self):
        ct = dddm.test_utils.test_context()
        ct.show_folders()
