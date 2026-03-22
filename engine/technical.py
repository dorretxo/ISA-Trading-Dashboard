"""Technical analysis: MA, RSI, MACD, ATR, Bollinger Bands, Volume, Stochastic,
ADX (trend strength), Williams %R (momentum).

ADX and Williams %R are computed locally from OHLC data using the `ta` library.
FMP technical indicators are used as a supplement when available (US tickers only
on the Starter plan), but local computation ensures coverage for ALL tickers.
"""

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator, StochRSIIndicator, WilliamsRIndicator
from ta.trend import MACD, SMAIndicator, ADXIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import OnBalanceVolumeIndicator

import config
from utils.data_fetch import get_price_history


def analyse(ticker: str) -> dict:
    """Run full technical analysis. Returns dict of indicators + a score from -1 to 1."""
    df = get_price_history(ticker)
    fmp_data = _get_fmp_technicals(ticker)
    return analyse_from_df(df, fmp_data=fmp_data)


def _get_fmp_technicals(ticker: str) -> dict | None:
    """Fetch ADX and Williams %R from FMP. Returns None if FMP unavailable."""
    try:
        from utils.fmp_client import get_technical_indicator, is_available
        if not is_available():
            return None
        adx_data = get_technical_indicator(ticker, "adx", period=14)
        williams_data = get_technical_indicator(ticker, "williams", period=14)
        if not adx_data and not williams_data:
            return None
        result = {}
        if adx_data and isinstance(adx_data, list) and adx_data:
            try:
                result["adx"] = float(adx_data[0].get("adx", 0))
            except (ValueError, TypeError):
                result["adx"] = None
        if williams_data and isinstance(williams_data, list) and williams_data:
            try:
                result["williams_r"] = float(williams_data[0].get("williams", 0))
            except (ValueError, TypeError):
                result["williams_r"] = None
        return result if result else None
    except Exception:
        return None


def analyse_from_df(df: pd.DataFrame, fmp_data: dict | None = None) -> dict:
    """Run technical analysis on a provided DataFrame. Used by analyse() and backtesting."""
    if df.empty or len(df) < 200:
        return _empty_result("Insufficient price data")

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"] if "Volume" in df.columns else None

    # Moving averages
    sma_50 = SMAIndicator(close, window=50).sma_indicator().iloc[-1]
    sma_200 = SMAIndicator(close, window=200).sma_indicator().iloc[-1]
    current_price = float(close.iloc[-1])

    # RSI
    rsi = RSIIndicator(close, window=14).rsi().iloc[-1]

    # MACD
    macd_ind = MACD(close)
    macd_line = macd_ind.macd().iloc[-1]
    signal_line = macd_ind.macd_signal().iloc[-1]
    macd_histogram = macd_ind.macd_diff().iloc[-1]

    # ATR (for stop-loss calc, exported for stops.py)
    atr = AverageTrueRange(high, low, close, window=14).average_true_range().iloc[-1]

    # Bollinger Bands (20-day, 2 std)
    bb = BollingerBands(close, window=20, window_dev=2)
    bb_upper = float(bb.bollinger_hband().iloc[-1])
    bb_lower = float(bb.bollinger_lband().iloc[-1])
    bb_mid = float(bb.bollinger_mavg().iloc[-1])
    bb_width = (bb_upper - bb_lower) / bb_mid if bb_mid else 0
    bb_pct = float(bb.bollinger_pband().iloc[-1])  # 0-1 position within bands

    # Stochastic RSI
    stoch_rsi_ind = StochRSIIndicator(close, window=14, smooth1=3, smooth2=3)
    stoch_k = float(stoch_rsi_ind.stochrsi_k().iloc[-1])
    stoch_d = float(stoch_rsi_ind.stochrsi_d().iloc[-1])

    # On-Balance Volume (OBV) trend
    obv_trend = None
    obv_divergence = None
    if volume is not None and not volume.isna().all():
        obv_series = OnBalanceVolumeIndicator(close, volume).on_balance_volume()
        if len(obv_series) >= 20:
            obv_sma = obv_series.rolling(20).mean()
            obv_trend = "rising" if float(obv_series.iloc[-1]) > float(obv_sma.iloc[-1]) else "falling"

            # Check for divergence: price up but OBV down (bearish) or price down but OBV up (bullish)
            price_change_20d = (close.iloc[-1] - close.iloc[-20]) / close.iloc[-20]
            obv_change_20d = obv_series.iloc[-1] - obv_series.iloc[-20]
            if price_change_20d > 0.02 and obv_change_20d < 0:
                obv_divergence = "bearish"
            elif price_change_20d < -0.02 and obv_change_20d > 0:
                obv_divergence = "bullish"

    # --- Scoring ---
    score = 0.0
    reasons = []

    # SMA trend: price vs 50-day and 200-day
    if current_price < sma_200:
        score -= 0.4
        reasons.append("below 200-day MA")
    elif current_price < sma_50:
        score -= 0.2
        reasons.append("below 50-day MA")
    else:
        score += 0.2

    # Death cross / golden cross
    if sma_50 < sma_200:
        score -= 0.2
        reasons.append("death cross")
    else:
        score += 0.1

    # RSI
    if rsi > config.RSI_OVERBOUGHT:
        score -= 0.3
        reasons.append("RSI overbought")
    elif rsi < config.RSI_OVERSOLD:
        score += 0.2
        reasons.append("RSI oversold (potential bounce)")
    else:
        score += 0.1

    # MACD
    if macd_line < signal_line:
        score -= 0.2
        reasons.append("MACD bearish crossover")
    else:
        score += 0.2

    # Bollinger Bands: price position within bands
    if bb_pct > 1.0:
        score -= 0.15
        reasons.append("price above upper Bollinger Band")
    elif bb_pct < 0.0:
        score += 0.15
        reasons.append("price below lower Bollinger Band")
    elif bb_pct > 0.8:
        score -= 0.05
    elif bb_pct < 0.2:
        score += 0.05

    # Stochastic RSI confirmation
    if stoch_k > 0.8 and stoch_d > 0.8:
        score -= 0.1
        reasons.append("Stochastic RSI overbought")
    elif stoch_k < 0.2 and stoch_d < 0.2:
        score += 0.1
        reasons.append("Stochastic RSI oversold")

    # Volume confirmation via OBV
    if obv_divergence == "bearish":
        score -= 0.15
        reasons.append("bearish OBV divergence (price up, volume down)")
    elif obv_divergence == "bullish":
        score += 0.15
        reasons.append("bullish OBV divergence (price down, volume up)")
    elif obv_trend == "rising" and current_price > sma_50:
        score += 0.05  # Volume confirms uptrend
    elif obv_trend == "falling" and current_price < sma_50:
        score -= 0.05  # Volume confirms downtrend

    # ADX — trend strength (computed locally; FMP overrides if available)
    adx = None
    try:
        adx_ind = ADXIndicator(high, low, close, window=14)
        adx_val = adx_ind.adx().iloc[-1]
        if not np.isnan(adx_val):
            adx = float(adx_val)
    except Exception:
        pass
    # FMP override for US tickers (may have smoother data)
    if fmp_data and fmp_data.get("adx") is not None:
        adx = fmp_data["adx"]

    if adx is not None:
        if adx > config.ADX_VERY_STRONG_TREND:
            if current_price > sma_50:
                score += 0.15
                reasons.append(f"strong uptrend (ADX {adx:.0f})")
            else:
                score -= 0.15
                reasons.append(f"strong downtrend (ADX {adx:.0f})")
        elif adx > config.ADX_STRONG_TREND:
            if current_price > sma_50:
                score += 0.10
            else:
                score -= 0.10

    # Williams %R — momentum (computed locally; FMP overrides if available)
    williams_r = None
    try:
        wr_ind = WilliamsRIndicator(high, low, close, lbp=14)
        wr_val = wr_ind.williams_r().iloc[-1]
        if not np.isnan(wr_val):
            williams_r = float(wr_val)
    except Exception:
        pass
    # FMP override for US tickers
    if fmp_data and fmp_data.get("williams_r") is not None:
        williams_r = fmp_data["williams_r"]

    if williams_r is not None:
        if williams_r > config.WILLIAMS_OVERBOUGHT:
            score -= 0.10
            reasons.append("Williams %R overbought")
        elif williams_r < config.WILLIAMS_OVERSOLD:
            score += 0.10
            reasons.append("Williams %R oversold")

    # Clamp to [-1, 1]
    score = max(-1.0, min(1.0, score))

    return {
        "score": score,
        "reasons": reasons,
        "sma_50": sma_50,
        "sma_200": sma_200,
        "rsi": rsi,
        "macd": macd_line,
        "macd_signal": signal_line,
        "macd_histogram": macd_histogram,
        "atr": atr,
        "current_price": current_price,
        "bb_upper": bb_upper,
        "bb_lower": bb_lower,
        "bb_pct": bb_pct,
        "stoch_k": stoch_k,
        "stoch_d": stoch_d,
        "obv_trend": obv_trend,
        "obv_divergence": obv_divergence,
        "adx": adx,
        "williams_r": williams_r,
    }


def _empty_result(reason: str) -> dict:
    return {
        "score": 0.0,
        "reasons": [reason],
        "sma_50": None,
        "sma_200": None,
        "rsi": None,
        "macd": None,
        "macd_signal": None,
        "macd_histogram": None,
        "atr": None,
        "current_price": None,
        "bb_upper": None,
        "bb_lower": None,
        "bb_pct": None,
        "stoch_k": None,
        "stoch_d": None,
        "obv_trend": None,
        "obv_divergence": None,
        "adx": None,
        "williams_r": None,
    }
