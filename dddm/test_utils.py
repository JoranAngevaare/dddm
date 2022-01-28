import dddm

export, __all__ = dddm.exporter()


@export
def test_context():
    ct = dddm.base_context()
    ct.register(dddm.examples.XenonSimple)
    ct.register(dddm.examples.ArgonSimple)
    ct.register(dddm.examples.GermaniumSimple)

    return ct
