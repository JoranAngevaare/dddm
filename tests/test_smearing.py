import unittest
from hypothesis import given, settings
from hypothesis import strategies as st
import numpy as np
from dddm.recoil_rates import detector_spectrum


@st.composite
def same_len_list(draw):
    n = draw(st.integers(min_value=4, max_value=10))
    fixed_length_list = st.lists(
        st.integers(min_value=1, max_value=int(1e6)),
        min_size=n, max_size=n)

    n2 = draw(st.integers(min_value=1, max_value=10))
    fixed_value_list = st.lists(
        st.integers(min_value=n2, max_value=n2),
        min_size=n, max_size=n)
    return draw(fixed_length_list), draw(fixed_value_list)


class TestSmearing(unittest.TestCase):
    @settings(deadline=None, max_examples=1000)
    @given(same_len_list(),
           st.floats(min_value=0.01, max_value=10),
           )
    def test_smearing(self, counts_and_bin_widths, resolution):
        raw, widths = np.array(counts_and_bin_widths)
        energies = np.cumsum(widths)
        smeared = np.zeros(len(raw))
        res = np.ones(len(raw)) * resolution  # just one is easier for debugging
        detector_spectrum._smear_signal(raw, energies, res, widths, smeared)

        numeric_tolerance = 1.25  # one shouldn't trust floats for this kind of operations
        if np.sum(smeared) > np.sum(raw) * numeric_tolerance:
            print(np.sum(smeared), np.sum(raw))
            print(raw)
            print(energies)
            print(res)
            print(widths)
            print(smeared)
        self.assertLessEqual(np.sum(smeared),
                             np.sum(raw) * numeric_tolerance,
                             f"Somehow got more events? {smeared}"
                             f"")
        if np.sum(raw * widths) > 0:
            self.assertGreaterEqual(np.sum(smeared),
                                    0,
                                    f"Lost all events? {smeared}")
