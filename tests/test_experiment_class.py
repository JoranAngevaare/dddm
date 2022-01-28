from unittest import TestCase
import dddm


class TestExperimentClass(TestCase):
    """See if we can init the Experiment class as we expect"""

    def test_dummy_init(self):
        class DummyExperiment(dddm.Experiment):
            pass

        dummy_experiment = DummyExperiment()
        # dummy_experiment._check_class()

    def test_incomplete_init(self):
        class IncompleteExperiment(dddm.Experiment):
            pass

        incomplete = IncompleteExperiment()
        with self.assertRaises(NotImplementedError):
            incomplete._check_class()
