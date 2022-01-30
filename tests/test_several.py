import dddm


def test_utils():
    l = [str(i) for i in range(10)]
    assert dddm.utils.is_str_in_list('1', l)
    assert dddm.utils.str_in_list('1', l)
