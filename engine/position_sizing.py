"""Inverse-volatility position sizing.

Allocates portfolio weight inversely proportional to each holding's
realized volatility — lower-vol stocks get larger allocations.  This is
a robust, calibration-free alternative to Kelly sizing that doesn't
require estimating expected returns.
"""

import logging

import numpy as np

import config
from engine.stops import _realized_volatility

logger = logging.getLogger(__name__)

# Default annualized vol when data is unavailable (30%)
DEFAULT_VOL = 0.30


def calculate_inverse_vol_weights(
    holdings: list[dict],
    results: list[dict],
) -> list[dict]:
    """Compute suggested portfolio weights using inverse-volatility.

    Args:
        holdings: Portfolio holdings from portfolio.json
        results: Analysis results from analyse_holding() per holding

    Returns:
        List of dicts per holding with:
            ticker, suggested_weight, current_weight, rebalance_delta, volatility
    """
    if not holdings or not results:
        return []

    max_weight = getattr(config, "MAX_POSITION_WEIGHT", 0.25)

    # Step 1: Gather volatility per ticker
    vols = []
    for r in results:
        ticker = r["ticker"]
        try:
            vol, _ = _realized_volatility(ticker)
            if vol is None or vol <= 0:
                vol = DEFAULT_VOL
        except Exception:
            vol = DEFAULT_VOL
        vols.append(vol)

    # Step 2: Compute current weights from market values
    values = []
    for r, h in zip(results, holdings):
        price = r.get("current_price") or 0
        qty = h.get("quantity", 0)
        currency = h.get("currency", "GBP")
        value = price * qty
        if currency == "GBX":
            value *= 0.01  # GBX → GBP
        values.append(value)

    total_value = sum(values)
    current_weights = [v / total_value if total_value > 0 else 0.0 for v in values]

    # Step 3: Inverse-vol raw weights
    inv_vols = [1.0 / v for v in vols]
    total_inv = sum(inv_vols)
    raw_weights = [iv / total_inv for iv in inv_vols]

    # Step 4: Cap at MAX_POSITION_WEIGHT and redistribute excess
    suggested = list(raw_weights)
    for _ in range(10):  # Iterative capping (converges quickly)
        excess = 0.0
        uncapped_sum = 0.0
        for i in range(len(suggested)):
            if suggested[i] > max_weight:
                excess += suggested[i] - max_weight
                suggested[i] = max_weight
            else:
                uncapped_sum += suggested[i]

        if excess <= 0.001:
            break

        # Redistribute excess proportionally among uncapped positions
        if uncapped_sum > 0:
            for i in range(len(suggested)):
                if suggested[i] < max_weight:
                    suggested[i] += excess * (suggested[i] / uncapped_sum)

    # Final normalize (in case of floating point drift)
    total_s = sum(suggested)
    if total_s > 0:
        suggested = [s / total_s for s in suggested]

    # Step 5: Build output
    output = []
    for i, (r, h) in enumerate(zip(results, holdings)):
        output.append({
            "ticker": r["ticker"],
            "name": r.get("name", r["ticker"]),
            "suggested_weight": round(suggested[i], 4),
            "current_weight": round(current_weights[i], 4),
            "rebalance_delta": round(suggested[i] - current_weights[i], 4),
            "volatility": round(vols[i], 4),
            "current_value": round(values[i], 2),
        })

    return output
