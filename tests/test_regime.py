import numpy as np
import pandas as pd

from cross_asset_engine.signals.regime import (
    regime_from_z,
    regime_from_p_high,
    ewma_realized_vol,
    classify_threshold,
    classify_markov,
    carry_scale,
    MarkovRegimeModel,
)


def test_regime_from_z_buckets():
    assert regime_from_z(-1.0) == "LOW"
    assert regime_from_z(0.0) == "NORMAL"
    assert regime_from_z(1.0) == "HIGH"
    assert regime_from_z(2.0) == "CRISIS"


def test_regime_from_p_high_buckets():
    assert regime_from_p_high(0.1) == "LOW"
    assert regime_from_p_high(0.3) == "NORMAL"
    assert regime_from_p_high(0.6) == "HIGH"
    assert regime_from_p_high(0.9) == "CRISIS"


def test_ewma_vol_higher_for_more_volatile_series():
    rng = np.random.default_rng(0)
    calm = pd.Series(rng.normal(0, 0.005, 500))
    wild = pd.Series(rng.normal(0, 0.03, 500))
    assert ewma_realized_vol(wild).iloc[-1] > ewma_realized_vol(calm).iloc[-1]


def test_classify_threshold_flags_a_vol_spike_as_high_or_crisis():
    # 400 calm days then a burst of large moves -> latest regime elevated.
    rng = np.random.default_rng(1)
    calm = rng.normal(0, 0.004, 400)
    spike = rng.normal(0, 0.04, 40)
    returns = pd.Series(np.concatenate([calm, spike]))
    vol = ewma_realized_vol(returns)
    classified = classify_threshold(vol)
    assert classified.iloc[-1]["regime"] in {"HIGH", "CRISIS"}


def test_markov_separates_two_variance_states():
    # Construct data with an obvious calm block and turbulent block.
    rng = np.random.default_rng(2)
    calm = rng.normal(0, 0.004, 300)
    wild = rng.normal(0, 0.03, 300)
    returns = pd.Series(np.concatenate([calm, wild]))
    model = MarkovRegimeModel().fit(returns)
    p_high = model.prob_high_vol()
    # High-vol probability should be much larger in the wild block than calm.
    assert p_high.iloc[-50:].mean() > p_high.iloc[:50].mean() + 0.3


def test_markov_is_deterministic_across_runs():
    rng = np.random.default_rng(3)
    returns = pd.Series(np.concatenate([rng.normal(0, 0.004, 200), rng.normal(0, 0.03, 200)]))
    a = classify_markov(returns)["p_high"].to_numpy()
    b = classify_markov(returns)["p_high"].to_numpy()
    assert np.allclose(a, b)  # reproducibility constraint


def test_carry_scale_decreases_with_stress():
    params = {"signals": {"regime": {"carry_scale": {"LOW": 1.0, "NORMAL": 1.0, "HIGH": 0.5, "CRISIS": 0.25}}}}
    assert carry_scale("LOW", params) == 1.0
    assert carry_scale("HIGH", params) == 0.5
    assert carry_scale("CRISIS", params) == 0.25
