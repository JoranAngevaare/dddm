import dddm
import os
import typing as ty

export, __all__ = dddm.exporter()


@export
def test_context():
    """just returns the base contexts, might be different one day"""
    return dddm.base_context()


def skip_long_test() -> ty.Tuple[bool, str]:
    """
    Wrapper for checking and mentioning if a test gets skipped because
    we are doing a short test.
    """
    do = os.environ.get('RUN_TEST_EXTENDED', False)
    skip = not do
    why = 'running quick test, set "export RUN_TEST_EXTENDED=1" to activate'
    return skip, why
