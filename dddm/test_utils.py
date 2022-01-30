import dddm
from unittest import skipIf
import os

export, __all__ = dddm.exporter()


@export
def test_context():
    ct = dddm.base_context()
    ct.register(dddm.examples.XenonSimple)
    ct.register(dddm.examples.ArgonSimple)
    ct.register(dddm.examples.GermaniumSimple)

    return ct


def skif_if_quick_test():
    return skipIf(os.environ.get('RUN_TEST_EXTENDED', False), 'running quick test, set "export RUN_TEST_EXTENDED=1" to activate')