# import dddm
# import numpy as np
# import pandas as pd
#
#
# def test_print_versions():
#     dddm.print_versions()
#
#
# def test_to_str_tuple():
#     tests = [
#         'a',
#         ['a', 'b'],
#         ('a', 'b'),
#         np.array(['a', 'b']),
#         pd.Series(['a', 'b'])
#     ]
#     for t in tests:
#         res = dddm.to_str_tuple(t)
#         assert isinstance(res, tuple)
#         assert isinstance(res[0], str)
