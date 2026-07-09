from cross_asset_engine.signals import SIGNAL_REGISTRY, list_signals


def test_momentum_and_carry_are_registered():
    assert "momentum" in SIGNAL_REGISTRY
    assert "carry" in SIGNAL_REGISTRY
    assert set(list_signals()) >= {"momentum", "carry"}


def test_registered_entries_are_callable():
    for name, fn in SIGNAL_REGISTRY.items():
        assert callable(fn), f"{name} is not callable"
