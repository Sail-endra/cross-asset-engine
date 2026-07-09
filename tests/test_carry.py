import numpy as np
import pandas as pd
import pytest

from cross_asset_engine.signals.carry import (
    credit_carry,
    fx_carry,
    rates_carry,
    equity_carry,
    apply_regime_scaling,
)


def _tidy(rows):
    """rows: list of (date, asset_id, value); returns a tidy frame."""
    df = pd.DataFrame(rows, columns=["date", "asset_id", "value"])
    df["date"] = pd.to_datetime(df["date"])
    return df


class FakeMarketData:
    def __init__(self):
        self.instruments = {
            "fx": [
                {"asset_id": "EURUSD", "from_symbol": "EUR", "to_symbol": "USD"},
                {"asset_id": "USDJPY", "from_symbol": "USD", "to_symbol": "JPY"},
            ]
        }
        dates = pd.bdate_range("2026-01-01", periods=2)
        self._credit = _tidy([(dates[-1], "US_IG_OAS", 0.75), (dates[-1], "US_HY_OAS", 2.72)])
        self._fx = _tidy(
            [(d, a, 1.1) for d in dates for a in ("EURUSD", "USDJPY")]
        )
        self._short = _tidy(
            [(d, c, v) for d in dates for c, v in
             {"USD": 3.63, "EUR": 2.25, "JPY": 1.24}.items()]
        )
        self._curve = _tidy([(dates[-1], "2Y", 4.13), (dates[-1], "10Y", 4.48)])
        # long equity history so regime can be computed
        eq_dates = pd.bdate_range("2024-01-01", periods=400)
        rng = np.random.default_rng(0)
        levels = 4000 * np.cumprod(1 + rng.normal(0.0003, 0.008, 400))
        self._equities = _tidy([(d, "SP500", v) for d, v in zip(eq_dates, levels)])

    def get_credit(self):
        return self._credit

    def get_fx(self):
        return self._fx

    def get_short_rates(self):
        return self._short

    def get_curve(self):
        return self._curve

    def get_equities(self):
        return self._equities


def test_credit_carry_is_the_oas_level():
    sigs = credit_carry(FakeMarketData())
    by = {s.asset_id: s for s in sigs}
    assert by["US_IG_OAS"].score == pytest.approx(0.75)
    assert by["US_HY_OAS"].score == pytest.approx(2.72)
    assert by["US_HY_OAS"].direction == 1


def test_fx_carry_is_base_minus_quote_rate_diff():
    sigs = fx_carry(FakeMarketData())
    by = {s.asset_id: s for s in sigs}
    # EURUSD: r_EUR - r_USD = 2.25 - 3.63 = -1.38 (negative carry long EUR)
    assert by["EURUSD"].score == pytest.approx(2.25 - 3.63)
    assert by["EURUSD"].direction == -1
    # USDJPY: r_USD - r_JPY = 3.63 - 1.24 = +2.39 (positive carry long USD)
    assert by["USDJPY"].score == pytest.approx(3.63 - 1.24)
    assert by["USDJPY"].direction == 1


def test_rates_carry_defaults_to_yield_with_rolldown_hook_off():
    sigs = rates_carry(FakeMarketData())
    by = {s.asset_id: s for s in sigs}
    assert by["10Y"].score == pytest.approx(4.48)
    assert by["10Y"].inputs["rolldown_wired"] is False
    # supplying rolldown shifts the score, proving the hook works
    sigs2 = rates_carry(FakeMarketData(), rolldown={"10Y": 0.20})
    by2 = {s.asset_id: s for s in sigs2}
    assert by2["10Y"].score == pytest.approx(4.68)
    assert by2["10Y"].inputs["rolldown_wired"] is True


def test_equity_carry_is_a_hook_without_dividend_data():
    # No dividend yields supplied -> empty, never fabricated.
    assert equity_carry(FakeMarketData()) == []
    # With dividend yields -> div yield minus financing.
    sigs = equity_carry(FakeMarketData(), dividend_yields={"SP500": 1.5}, financing_rate=3.63)
    assert sigs[0].score == pytest.approx(1.5 - 3.63)
    assert sigs[0].direction == -1


def test_apply_regime_scaling_records_scale_and_scaled_score():
    params = {
        "signals": {
            "regime": {
                "risk_asset": "SP500",
                "ewma_lambda": 0.94,
                "carry_scale": {"LOW": 1.0, "NORMAL": 1.0, "HIGH": 0.5, "CRISIS": 0.25},
            }
        }
    }
    md = FakeMarketData()
    sigs = credit_carry(md)
    apply_regime_scaling(sigs, md, params)
    for s in sigs:
        assert "regime" in s.inputs
        assert "regime_scale" in s.inputs
        assert s.inputs["scaled_score"] == pytest.approx(s.score * s.inputs["regime_scale"])
