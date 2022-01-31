import unittest
from hypothesis import given, settings
from hypothesis import strategies as st
import numpy as np
from dddm.recoil_rates import detector_spectrum


@st.composite
def same_len_lists(draw):
    n = draw(st.integers(min_value=3, max_value=20))
    fixed_length_list = st.lists(
        st.integers(min_value=0, max_value=int(1e6)),
        min_size=n, max_size=n)
    fixed_length_list_floats = st.lists(
        st.floats(min_value=1,
                  max_value=int(1e6)),
        min_size=n,
        max_size=n)
    return (draw(fixed_length_list),
            draw(fixed_length_list_floats))


class TestSmearing(unittest.TestCase):
    _tests_seen = 0

    @settings(deadline=None, max_examples=100)
    @given(same_len_lists(),
           st.floats(min_value=0.01),
           )
    def test_smearing(self, counts_and_bin_widths, resolution):
        raw, widths = np.array(counts_and_bin_widths)
        energies = np.cumsum(widths)
        smeared = np.zeros(len(raw))
        res = np.ones(len(raw))*resolution  # just one is easier for debugging
        detector_spectrum._smear_signal(raw, energies, res, widths, smeared)

        numeric_tolerance = 1.01  # one shouldn't trust floats for this kind of operations
        self.assertLessEqual(np.sum(smeared),
                             np.sum(raw*widths)*numeric_tolerance,
                             f"Somehow got more events? {smeared}")
        if np.sum(raw*widths) > 0:
            self.assertGreaterEqual(np.sum(smeared),
                                    0,
                                    f"Lost all events? {smeared}")


