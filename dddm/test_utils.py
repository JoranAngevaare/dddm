import dddm
import os

export, __all__ = dddm.exporter()


@export
def test_context():
    ct = dddm.base_context()
    ct.register(dddm.examples.XenonSimple)
    ct.register(dddm.examples.ArgonSimple)
    ct.register(dddm.examples.GermaniumSimple)

    return ct


def skip_long_test():
    do = os.environ.get('RUN_TEST_EXTENDED', False)
    skip = not do
    why = 'running quick test, set "export RUN_TEST_EXTENDED=1" to activate'
    return skip, why
