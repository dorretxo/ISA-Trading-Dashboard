"""Structured global equity universe for the Discovery Engine.

Provides a tiered, metadata-rich universe across major global exchanges.
Each ticker has region, tier, index source, and validation metadata.

The original implementation exposed Tier 2 names only on selected weekdays.
Phase 1 replaces that rotation with a daily systematic snapshot so the
discovery engine can screen the same investable universe every day.
"""

from __future__ import annotations

from typing import NamedTuple

import config


class UniverseEntry(NamedTuple):
    """Single ticker in the structured universe."""

    ticker: str
    country: str       # ISO 2-letter
    exchange: str      # e.g. "LSE", "EPA", "BME", "BIT"
    tier: int          # 1 = core, 2 = mid-cap
    index_source: str  # e.g. "FTSE100", "CAC40", "SBF120", "IBEX_MID"
    sector: str        # broad sector (best-effort, empty if unknown)


# Last manual refresh date — bump when updating ticker lists
UNIVERSE_LAST_REFRESHED = "2026-03-22"


def _excluded_country_tokens() -> set[str]:
    return {
        str(value).strip().upper()
        for value in getattr(config, "DISCOVERY_EXCLUDED_COUNTRIES", set())
        if str(value).strip()
    }


def _ticker_aliases() -> dict[str, str]:
    return {
        str(src).upper().strip(): str(dst).upper().strip()
        for src, dst in getattr(config, "DISCOVERY_TICKER_ALIASES", {}).items()
        if str(src).strip() and str(dst).strip()
    }


def resolve_yahoo_ticker(ticker: str) -> str:
    """Return the Yahoo-compatible symbol for a configured universe ticker."""
    symbol = str(ticker or "").upper().strip()
    return _ticker_aliases().get(symbol, symbol)


def _quarantined_tickers() -> set[str]:
    return {
        str(value).upper().strip()
        for value in getattr(config, "DISCOVERY_TICKER_QUARANTINE", set())
        if str(value).strip()
    }


def is_quarantined_ticker(ticker: str) -> bool:
    """Return True when a symbol is known to be unpriceable in Yahoo Finance."""
    symbol = resolve_yahoo_ticker(ticker)
    return symbol in _quarantined_tickers()


def is_excluded_ticker(ticker: str, country: str | None = None, region: str | None = None) -> bool:
    """Return True when a ticker is outside the configured investable universe."""
    raw_symbol = str(ticker or "").upper().strip()
    if not raw_symbol:
        return False
    symbol = resolve_yahoo_ticker(raw_symbol)

    excluded_tickers = {
        str(value).upper().strip()
        for value in getattr(config, "DISCOVERY_EXCLUDED_TICKERS", set())
    }
    if symbol in excluded_tickers:
        return True

    if symbol in _quarantined_tickers():
        return True

    excluded_suffixes = tuple(
        str(value).upper().strip()
        for value in getattr(config, "DISCOVERY_EXCLUDED_TICKER_SUFFIXES", ())
        if str(value).strip()
    )
    if excluded_suffixes and symbol.endswith(excluded_suffixes):
        return True

    country_tokens = _excluded_country_tokens()
    for value in (country, region):
        token = str(value or "").strip().upper()
        if token and token in country_tokens:
            return True
    return False


# ---------------------------------------------------------------------------
# UK — FTSE 100 + FTSE 250 selection
# ---------------------------------------------------------------------------
_UK = [
    # FTSE 100 — Tier 1
    ("SHEL.L", "GB", "LSE", 1, "FTSE100", "Energy"),
    ("AZN.L", "GB", "LSE", 1, "FTSE100", "Healthcare"),
    ("HSBA.L", "GB", "LSE", 1, "FTSE100", "Financials"),
    ("BP.L", "GB", "LSE", 1, "FTSE100", "Energy"),
    ("GSK.L", "GB", "LSE", 1, "FTSE100", "Healthcare"),
    ("RIO.L", "GB", "LSE", 1, "FTSE100", "Materials"),
    ("ULVR.L", "GB", "LSE", 1, "FTSE100", "Consumer Staples"),
    ("DGE.L", "GB", "LSE", 1, "FTSE100", "Consumer Staples"),
    ("REL.L", "GB", "LSE", 1, "FTSE100", "Industrials"),
    ("LSEG.L", "GB", "LSE", 1, "FTSE100", "Financials"),
    ("BA.L", "GB", "LSE", 1, "FTSE100", "Industrials"),
    ("NG.L", "GB", "LSE", 1, "FTSE100", "Utilities"),
    ("GLEN.L", "GB", "LSE", 1, "FTSE100", "Materials"),
    ("AAL.L", "GB", "LSE", 1, "FTSE100", "Materials"),
    ("LLOY.L", "GB", "LSE", 1, "FTSE100", "Financials"),
    ("BARC.L", "GB", "LSE", 1, "FTSE100", "Financials"),
    ("NWG.L", "GB", "LSE", 1, "FTSE100", "Financials"),
    ("PRU.L", "GB", "LSE", 1, "FTSE100", "Financials"),
    ("STAN.L", "GB", "LSE", 1, "FTSE100", "Financials"),
    ("ABF.L", "GB", "LSE", 1, "FTSE100", "Consumer Staples"),
    ("III.L", "GB", "LSE", 1, "FTSE100", "Financials"),
    ("IMB.L", "GB", "LSE", 1, "FTSE100", "Consumer Staples"),
    ("INF.L", "GB", "LSE", 1, "FTSE100", "Industrials"),
    ("ITRK.L", "GB", "LSE", 1, "FTSE100", "Industrials"),
    ("JD.L", "GB", "LSE", 1, "FTSE100", "Consumer Discretionary"),
    ("KGF.L", "GB", "LSE", 1, "FTSE100", "Consumer Discretionary"),
    ("LAND.L", "GB", "LSE", 1, "FTSE100", "Real Estate"),
    ("MNG.L", "GB", "LSE", 1, "FTSE100", "Consumer Discretionary"),
    ("MNDI.L", "GB", "LSE", 1, "FTSE100", "Materials"),
    ("PSON.L", "GB", "LSE", 1, "FTSE100", "Industrials"),
    ("RKT.L", "GB", "LSE", 1, "FTSE100", "Consumer Staples"),
    ("RR.L", "GB", "LSE", 1, "FTSE100", "Industrials"),
    ("SGE.L", "GB", "LSE", 1, "FTSE100", "Industrials"),
    ("SN.L", "GB", "LSE", 1, "FTSE100", "Healthcare"),
    ("SPX.L", "GB", "LSE", 1, "FTSE100", "Industrials"),
    ("SSE.L", "GB", "LSE", 1, "FTSE100", "Utilities"),
    ("SVT.L", "GB", "LSE", 1, "FTSE100", "Utilities"),
    ("TSCO.L", "GB", "LSE", 1, "FTSE100", "Consumer Staples"),
    ("WPP.L", "GB", "LSE", 1, "FTSE100", "Communication"),
    ("WTB.L", "GB", "LSE", 1, "FTSE100", "Financials"),
    ("EXPN.L", "GB", "LSE", 1, "FTSE100", "Industrials"),
    ("CRH.L", "GB", "LSE", 1, "FTSE100", "Materials"),
    ("CPG.L", "GB", "LSE", 1, "FTSE100", "Consumer Staples"),
    ("CRDA.L", "GB", "LSE", 1, "FTSE100", "Materials"),
    ("DCC.L", "GB", "LSE", 1, "FTSE100", "Industrials"),
    ("FRAS.L", "GB", "LSE", 1, "FTSE100", "Consumer Discretionary"),
    ("HLMA.L", "GB", "LSE", 1, "FTSE100", "Technology"),
    ("HLN.L", "GB", "LSE", 1, "FTSE100", "Consumer Staples"),
    ("HWDN.L", "GB", "LSE", 1, "FTSE100", "Financials"),
    ("ICP.L", "GB", "LSE", 1, "FTSE100", "Industrials"),
    ("IHG.L", "GB", "LSE", 1, "FTSE100", "Consumer Discretionary"),
    ("JMAT.L", "GB", "LSE", 1, "FTSE100", "Materials"),
    ("PSN.L", "GB", "LSE", 1, "FTSE100", "Industrials"),
    ("PHNX.L", "GB", "LSE", 1, "FTSE100", "Financials"),
    ("RTO.L", "GB", "LSE", 1, "FTSE100", "Industrials"),
    ("SBRY.L", "GB", "LSE", 1, "FTSE100", "Consumer Staples"),
    ("SDR.L", "GB", "LSE", 1, "FTSE100", "Financials"),
    ("SKG.L", "GB", "LSE", 1, "FTSE100", "Materials"),
    ("SMDS.L", "GB", "LSE", 1, "FTSE100", "Materials"),
    ("SMT.L", "GB", "LSE", 1, "FTSE100", "Financials"),
    ("SMIN.L", "GB", "LSE", 1, "FTSE100", "Industrials"),
    ("WEIR.L", "GB", "LSE", 1, "FTSE100", "Industrials"),
    ("AV.L", "GB", "LSE", 1, "FTSE100", "Financials"),
    ("AUTO.L", "GB", "LSE", 1, "FTSE100", "Consumer Discretionary"),
    ("BDEV.L", "GB", "LSE", 1, "FTSE100", "Consumer Discretionary"),
    ("BKG.L", "GB", "LSE", 1, "FTSE100", "Consumer Discretionary"),
    ("BNZL.L", "GB", "LSE", 1, "FTSE100", "Industrials"),
    ("BRBY.L", "GB", "LSE", 1, "FTSE100", "Consumer Discretionary"),
    ("CCH.L", "GB", "LSE", 1, "FTSE100", "Consumer Staples"),
    ("CNA.L", "GB", "LSE", 1, "FTSE100", "Industrials"),
    ("ENT.L", "GB", "LSE", 1, "FTSE100", "Consumer Discretionary"),
    ("FLTR.L", "GB", "LSE", 1, "FTSE100", "Consumer Discretionary"),
    ("FRES.L", "GB", "LSE", 1, "FTSE100", "Materials"),
    ("HIK.L", "GB", "LSE", 1, "FTSE100", "Technology"),
    ("IAG.L", "GB", "LSE", 1, "FTSE100", "Industrials"),
    ("ICG.L", "GB", "LSE", 1, "FTSE100", "Financials"),
    ("IMI.L", "GB", "LSE", 1, "FTSE100", "Industrials"),
    ("LGEN.L", "GB", "LSE", 1, "FTSE100", "Financials"),
    ("NXT.L", "GB", "LSE", 1, "FTSE100", "Consumer Discretionary"),
    ("VOD.L", "GB", "LSE", 1, "FTSE100", "Communication"),
    ("BT-A.L", "GB", "LSE", 1, "FTSE100", "Communication"),
    ("AHT.L", "GB", "LSE", 1, "FTSE100", "Healthcare"),
    ("ANTO.L", "GB", "LSE", 1, "FTSE100", "Materials"),
    ("ADM.L", "GB", "LSE", 1, "FTSE100", "Financials"),
    ("EDV.L", "GB", "LSE", 1, "FTSE100", "Financials"),
    ("BAE.L", "GB", "LSE", 1, "FTSE100", "Industrials"),
    # FTSE 250 selection — Tier 2
    ("DARK.L", "GB", "LSE", 2, "FTSE250", "Technology"),
    ("TW.L", "GB", "LSE", 2, "FTSE250", "Industrials"),
    ("BGEO.L", "GB", "LSE", 2, "FTSE250", "Financials"),
    ("JET2.L", "GB", "LSE", 2, "FTSE250", "Industrials"),
    ("WIZZ.L", "GB", "LSE", 2, "FTSE250", "Industrials"),
    ("FOUR.L", "GB", "LSE", 2, "FTSE250", "Technology"),
    ("MONY.L", "GB", "LSE", 2, "FTSE250", "Financials"),
    ("BOKU.L", "GB", "LSE", 2, "FTSE250", "Technology"),
    ("ALFA.L", "GB", "LSE", 2, "FTSE250", "Industrials"),
    ("ASC.L", "GB", "LSE", 2, "FTSE250", "Technology"),
    ("BME.L", "GB", "LSE", 2, "FTSE250", "Industrials"),
    ("BNKR.L", "GB", "LSE", 2, "FTSE250", "Financials"),
    ("CINE.L", "GB", "LSE", 2, "FTSE250", "Consumer Discretionary"),
    ("CMC.L", "GB", "LSE", 2, "FTSE250", "Financials"),
    ("CURY.L", "GB", "LSE", 2, "FTSE250", "Consumer Discretionary"),
    ("DPLM.L", "GB", "LSE", 2, "FTSE250", "Industrials"),
    ("FCIT.L", "GB", "LSE", 2, "FTSE250", "Financials"),
    ("GAW.L", "GB", "LSE", 2, "FTSE250", "Consumer Discretionary"),
    ("GNS.L", "GB", "LSE", 2, "FTSE250", "Consumer Discretionary"),
    ("GPAY.L", "GB", "LSE", 2, "FTSE250", "Technology"),
    ("HTWS.L", "GB", "LSE", 2, "FTSE250", "Consumer Discretionary"),
    ("IGG.L", "GB", "LSE", 2, "FTSE250", "Technology"),
    ("IPX.L", "GB", "LSE", 2, "FTSE250", "Industrials"),
    ("ITV.L", "GB", "LSE", 2, "FTSE250", "Communication"),
    ("JLEN.L", "GB", "LSE", 2, "FTSE250", "Utilities"),
    ("LUCE.L", "GB", "LSE", 2, "FTSE250", "Industrials"),
    ("MGAM.L", "GB", "LSE", 2, "FTSE250", "Consumer Discretionary"),
    ("MRO.L", "GB", "LSE", 2, "FTSE250", "Industrials"),
    ("OSB.L", "GB", "LSE", 2, "FTSE250", "Financials"),
    ("OXB.L", "GB", "LSE", 2, "FTSE250", "Healthcare"),
    ("PAGE.L", "GB", "LSE", 2, "FTSE250", "Consumer Discretionary"),
    ("PRTC.L", "GB", "LSE", 2, "FTSE250", "Industrials"),
    ("QQ.L", "GB", "LSE", 2, "FTSE250", "Industrials"),
    ("RWS.L", "GB", "LSE", 2, "FTSE250", "Industrials"),
    ("SEPL.L", "GB", "LSE", 2, "FTSE250", "Financials"),
    ("SGC.L", "GB", "LSE", 2, "FTSE250", "Industrials"),
    ("SGRO.L", "GB", "LSE", 2, "FTSE250", "Real Estate"),
    ("SHI.L", "GB", "LSE", 2, "FTSE250", "Industrials"),
    ("SNDR.L", "GB", "LSE", 2, "FTSE250", "Consumer Discretionary"),
    ("SOI.L", "GB", "LSE", 2, "FTSE250", "Industrials"),
    ("SFOR.L", "GB", "LSE", 2, "FTSE250", "Industrials"),
    ("TRN.L", "GB", "LSE", 2, "FTSE250", "Industrials"),
    ("TRG.L", "GB", "LSE", 2, "FTSE250", "Consumer Discretionary"),
    ("VSVS.L", "GB", "LSE", 2, "FTSE250", "Consumer Discretionary"),
    ("WIN.L", "GB", "LSE", 2, "FTSE250", "Consumer Discretionary"),
    ("YOU.L", "GB", "LSE", 2, "FTSE250", "Consumer Discretionary"),
    ("ZTF.L", "GB", "LSE", 2, "FTSE250", "Industrials"),
    ("BOY.L", "GB", "LSE", 2, "FTSE250", "Consumer Staples"),
    ("CHG.L", "GB", "LSE", 2, "FTSE250", "Consumer Staples"),
    ("FTC.L", "GB", "LSE", 2, "FTSE250", "Consumer Discretionary"),
    ("BVIC.L", "GB", "LSE", 2, "FTSE250", "Consumer Staples"),
    ("MGGT.L", "GB", "LSE", 2, "FTSE250", "Industrials"),
]

# ---------------------------------------------------------------------------
# Germany — DAX 40 + MDAX selection
# ---------------------------------------------------------------------------
_GERMANY = [
    # DAX 40 — Tier 1
    ("SAP.DE", "DE", "XETRA", 1, "DAX40", "Technology"),
    ("SIE.DE", "DE", "XETRA", 1, "DAX40", "Industrials"),
    ("ALV.DE", "DE", "XETRA", 1, "DAX40", "Financials"),
    ("DTE.DE", "DE", "XETRA", 1, "DAX40", "Communication"),
    ("AIR.DE", "DE", "XETRA", 1, "DAX40", "Industrials"),
    ("BAS.DE", "DE", "XETRA", 1, "DAX40", "Materials"),
    ("MBG.DE", "DE", "XETRA", 1, "DAX40", "Consumer Discretionary"),
    ("BMW.DE", "DE", "XETRA", 1, "DAX40", "Consumer Discretionary"),
    ("MUV2.DE", "DE", "XETRA", 1, "DAX40", "Financials"),
    ("DHL.DE", "DE", "XETRA", 1, "DAX40", "Industrials"),
    ("IFX.DE", "DE", "XETRA", 1, "DAX40", "Technology"),
    ("ADS.DE", "DE", "XETRA", 1, "DAX40", "Consumer Discretionary"),
    ("DB1.DE", "DE", "XETRA", 1, "DAX40", "Financials"),
    ("BEI.DE", "DE", "XETRA", 1, "DAX40", "Consumer Staples"),
    ("RWE.DE", "DE", "XETRA", 1, "DAX40", "Utilities"),
    ("HEN3.DE", "DE", "XETRA", 1, "DAX40", "Consumer Staples"),
    ("VNA.DE", "DE", "XETRA", 1, "DAX40", "Real Estate"),
    ("CON.DE", "DE", "XETRA", 1, "DAX40", "Consumer Discretionary"),
    ("FRE.DE", "DE", "XETRA", 1, "DAX40", "Healthcare"),
    ("MTX.DE", "DE", "XETRA", 1, "DAX40", "Technology"),
    ("SRT3.DE", "DE", "XETRA", 1, "DAX40", "Technology"),
    ("ENR.DE", "DE", "XETRA", 1, "DAX40", "Industrials"),
    ("PAH3.DE", "DE", "XETRA", 1, "DAX40", "Consumer Discretionary"),
    ("HEI.DE", "DE", "XETRA", 1, "DAX40", "Technology"),
    ("QIA.DE", "DE", "XETRA", 1, "DAX40", "Technology"),
    ("P911.DE", "DE", "XETRA", 1, "DAX40", "Consumer Discretionary"),
    ("MRK.DE", "DE", "XETRA", 1, "DAX40", "Healthcare"),
    ("DTG.DE", "DE", "XETRA", 1, "DAX40", "Industrials"),
    ("SHL.DE", "DE", "XETRA", 1, "DAX40", "Healthcare"),
    ("FME.DE", "DE", "XETRA", 1, "DAX40", "Healthcare"),
    ("ZAL.DE", "DE", "XETRA", 1, "DAX40", "Consumer Discretionary"),
    ("HNR1.DE", "DE", "XETRA", 1, "DAX40", "Financials"),
    ("DHER.DE", "DE", "XETRA", 1, "DAX40", "Consumer Discretionary"),
    ("PUM.DE", "DE", "XETRA", 1, "DAX40", "Consumer Discretionary"),
    ("1COV.DE", "DE", "XETRA", 1, "DAX40", "Materials"),
    ("SY1.DE", "DE", "XETRA", 1, "DAX40", "Financials"),
    ("TKA.DE", "DE", "XETRA", 1, "DAX40", "Materials"),
    ("LEG.DE", "DE", "XETRA", 1, "DAX40", "Technology"),
    # MDAX selection — Tier 2
    ("HAG.DE", "DE", "XETRA", 2, "MDAX", "Industrials"),
    ("RHM.DE", "DE", "XETRA", 2, "MDAX", "Industrials"),
    ("EVT.DE", "DE", "XETRA", 2, "MDAX", "Healthcare"),
    ("AIXA.DE", "DE", "XETRA", 2, "MDAX", "Technology"),
    ("NDX1.DE", "DE", "XETRA", 2, "MDAX", "Industrials"),
    ("KGX.DE", "DE", "XETRA", 2, "MDAX", "Consumer Discretionary"),
    ("GXI.DE", "DE", "XETRA", 2, "MDAX", "Technology"),
    ("WAF.DE", "DE", "XETRA", 2, "MDAX", "Technology"),
    ("TEG.DE", "DE", "XETRA", 2, "MDAX", "Communication"),
    ("BNR.DE", "DE", "XETRA", 2, "MDAX", "Industrials"),
]

# ---------------------------------------------------------------------------
# France — CAC 40 + SBF 120 + CAC Mid 60 (EXPANDED)
# ---------------------------------------------------------------------------
_FRANCE = [
    # CAC 40 — Tier 1
    ("MC.PA", "FR", "EPA", 1, "CAC40", "Consumer Discretionary"),
    ("OR.PA", "FR", "EPA", 1, "CAC40", "Consumer Staples"),
    ("TTE.PA", "FR", "EPA", 1, "CAC40", "Energy"),
    ("SAN.PA", "FR", "EPA", 1, "CAC40", "Healthcare"),
    ("AI.PA", "FR", "EPA", 1, "CAC40", "Industrials"),
    ("SU.PA", "FR", "EPA", 1, "CAC40", "Technology"),
    ("BN.PA", "FR", "EPA", 1, "CAC40", "Consumer Staples"),
    ("CS.PA", "FR", "EPA", 1, "CAC40", "Technology"),
    ("DG.PA", "FR", "EPA", 1, "CAC40", "Industrials"),
    ("AIR.PA", "FR", "EPA", 1, "CAC40", "Industrials"),
    ("SAF.PA", "FR", "EPA", 1, "CAC40", "Industrials"),
    ("BNP.PA", "FR", "EPA", 1, "CAC40", "Financials"),
    ("GLE.PA", "FR", "EPA", 1, "CAC40", "Financials"),
    ("CA.PA", "FR", "EPA", 1, "CAC40", "Financials"),
    ("RI.PA", "FR", "EPA", 1, "CAC40", "Consumer Staples"),
    ("KER.PA", "FR", "EPA", 1, "CAC40", "Consumer Discretionary"),
    ("CAP.PA", "FR", "EPA", 1, "CAC40", "Technology"),
    ("DSY.PA", "FR", "EPA", 1, "CAC40", "Technology"),
    ("HO.PA", "FR", "EPA", 1, "CAC40", "Industrials"),
    ("SGO.PA", "FR", "EPA", 1, "CAC40", "Materials"),
    ("STLAP.PA", "FR", "EPA", 1, "CAC40", "Consumer Discretionary"),
    ("VIE.PA", "FR", "EPA", 1, "CAC40", "Utilities"),
    ("ACA.PA", "FR", "EPA", 1, "CAC40", "Financials"),
    ("EN.PA", "FR", "EPA", 1, "CAC40", "Utilities"),
    ("ML.PA", "FR", "EPA", 1, "CAC40", "Industrials"),
    ("PUB.PA", "FR", "EPA", 1, "CAC40", "Communication"),
    ("RMS.PA", "FR", "EPA", 1, "CAC40", "Consumer Discretionary"),
    ("VIV.PA", "FR", "EPA", 1, "CAC40", "Communication"),
    ("STM.PA", "FR", "EPA", 1, "CAC40", "Technology"),
    ("WLN.PA", "FR", "EPA", 1, "CAC40", "Communication"),
    ("RNO.PA", "FR", "EPA", 1, "CAC40", "Consumer Discretionary"),
    ("ORA.PA", "FR", "EPA", 1, "CAC40", "Communication"),
    ("LR.PA", "FR", "EPA", 1, "CAC40", "Technology"),
    ("URW.PA", "FR", "EPA", 1, "CAC40", "Real Estate"),
    ("ATO.PA", "FR", "EPA", 1, "CAC40", "Utilities"),
    # SBF 120 large-caps — Tier 1
    ("EL.PA", "FR", "EPA", 1, "SBF120", "Healthcare"),
    ("DIM.PA", "FR", "EPA", 1, "SBF120", "Healthcare"),
    ("AMUN.PA", "FR", "EPA", 1, "SBF120", "Financials"),
    ("BVI.PA", "FR", "EPA", 1, "SBF120", "Industrials"),
    ("ALO.PA", "FR", "EPA", 1, "SBF120", "Industrials"),
    ("BIM.PA", "FR", "EPA", 1, "SBF120", "Healthcare"),
    ("ADP.PA", "FR", "EPA", 1, "SBF120", "Industrials"),
    ("GET.PA", "FR", "EPA", 1, "SBF120", "Industrials"),
    ("LI.PA", "FR", "EPA", 1, "SBF120", "Real Estate"),
    ("RXL.PA", "FR", "EPA", 1, "SBF120", "Industrials"),
    ("SPIE.PA", "FR", "EPA", 1, "SBF120", "Industrials"),
    ("SW.PA", "FR", "EPA", 1, "SBF120", "Consumer Discretionary"),
    ("ELIS.PA", "FR", "EPA", 1, "SBF120", "Industrials"),
    # SBF 120 / CAC Mid 60 — Tier 2
    ("TEP.PA", "FR", "EPA", 2, "SBF120", "Utilities"),
    ("AM.PA", "FR", "EPA", 2, "SBF120", "Industrials"),
    ("SOI.PA", "FR", "EPA", 2, "SBF120", "Materials"),
    ("DBG.PA", "FR", "EPA", 2, "SBF120", "Consumer Staples"),
    ("ERF.PA", "FR", "EPA", 2, "SBF120", "Financials"),
    ("FGR.PA", "FR", "EPA", 2, "SBF120", "Consumer Discretionary"),
    ("GFI.PA", "FR", "EPA", 2, "SBF120", "Consumer Staples"),
    ("GTT.PA", "FR", "EPA", 2, "SBF120", "Technology"),
    ("ILD.PA", "FR", "EPA", 2, "SBF120", "Communication"),
    ("MF.PA", "FR", "EPA", 2, "SBF120", "Consumer Staples"),
    ("NEX.PA", "FR", "EPA", 2, "CAC_MID60", "Industrials"),
    ("EDEN.PA", "FR", "EPA", 2, "CAC_MID60", "Technology"),
    ("AKE.PA", "FR", "EPA", 2, "CAC_MID60", "Materials"),
    ("TKO.PA", "FR", "EPA", 2, "CAC_MID60", "Energy"),
    ("AF.PA", "FR", "EPA", 2, "CAC_MID60", "Industrials"),
    ("SK.PA", "FR", "EPA", 2, "CAC_MID60", "Consumer Discretionary"),
    ("SOP.PA", "FR", "EPA", 2, "CAC_MID60", "Technology"),
    ("COFA.PA", "FR", "EPA", 2, "CAC_MID60", "Financials"),
    ("OPM.PA", "FR", "EPA", 2, "CAC_MID60", "Consumer Discretionary"),
    ("VRLA.PA", "FR", "EPA", 2, "CAC_MID60", "Materials"),
    ("ITP.PA", "FR", "EPA", 2, "CAC_MID60", "Consumer Discretionary"),
    ("RCO.PA", "FR", "EPA", 2, "CAC_MID60", "Consumer Staples"),
    ("NK.PA", "FR", "EPA", 2, "CAC_MID60", "Materials"),
    ("RBT.PA", "FR", "EPA", 2, "CAC_MID60", "Energy"),
    ("TFI.PA", "FR", "EPA", 2, "CAC_MID60", "Communication"),
    ("IPS.PA", "FR", "EPA", 2, "CAC_MID60", "Industrials"),
    ("ERA.PA", "FR", "EPA", 2, "CAC_MID60", "Materials"),
    ("OVH.PA", "FR", "EPA", 2, "CAC_MID60", "Technology"),
    ("FNAC.PA", "FR", "EPA", 2, "CAC_MID60", "Consumer Discretionary"),
    ("VLA.PA", "FR", "EPA", 2, "CAC_MID60", "Healthcare"),
    ("UBI.PA", "FR", "EPA", 2, "CAC_MID60", "Technology"),
    ("RF.PA", "FR", "EPA", 2, "CAC_MID60", "Healthcare"),
    ("TRI.PA", "FR", "EPA", 2, "CAC_MID60", "Consumer Discretionary"),
    ("MMB.PA", "FR", "EPA", 2, "CAC_MID60", "Communication"),
]

# ---------------------------------------------------------------------------
# Spain — IBEX 35 + IBEX Medium Cap (EXPANDED)
# ---------------------------------------------------------------------------
_SPAIN = [
    # IBEX 35 — Tier 1
    ("SAN.MC", "ES", "BME", 1, "IBEX35", "Financials"),
    ("BBVA.MC", "ES", "BME", 1, "IBEX35", "Financals"),
    ("ITX.MC", "ES", "BME", 1, "IBEX35", "Consumer Discretionary"),
    ("IBE.MC", "ES", "BME", 1, "IBEX35", "Utilities"),
    ("TEF.MC", "ES", "BME", 1, "IBEX35", "Communication"),
    ("REP.MC", "ES", "BME", 1, "IBEX35", "Energy"),
    ("FER.MC", "ES", "BME", 1, "IBEX35", "Industrials"),
    ("AMS.MC", "ES", "BME", 1, "IBEX35", "Healthcare"),
    ("GRF.MC", "ES", "BME", 1, "IBEX35", "Healthcare"),
    ("ENG.MC", "ES", "BME", 1, "IBEX35", "Utilities"),
    ("MAP.MC", "ES", "BME", 1, "IBEX35", "Financials"),
    ("ACS.MC", "ES", "BME", 1, "IBEX35", "Industrials"),
    ("CABK.MC", "ES", "BME", 1, "IBEX35", "Financials"),
    ("CLNX.MC", "ES", "BME", 1, "IBEX35", "Communication"),
    ("FDR.MC", "ES", "BME", 1, "IBEX35", "Consumer Discretionary"),
    ("MRL.MC", "ES", "BME", 1, "IBEX35", "Industrials"),
    ("SAB.MC", "ES", "BME", 1, "IBEX35", "Financials"),
    ("BKT.MC", "ES", "BME", 1, "IBEX35", "Financials"),
    ("SCYR.MC", "ES", "BME", 1, "IBEX35", "Industrials"),
    ("IAG.MC", "ES", "BME", 1, "IBEX35", "Industrials"),
    ("ACX.MC", "ES", "BME", 1, "IBEX35", "Materials"),
    ("CIE.MC", "ES", "BME", 1, "IBEX35", "Consumer Discretionary"),
    ("COL.MC", "ES", "BME", 1, "IBEX35", "Industrials"),
    ("MTS.MC", "ES", "BME", 1, "IBEX35", "Materials"),
    ("RED.MC", "ES", "BME", 1, "IBEX35", "Communication"),
    ("AENA.MC", "ES", "BME", 1, "IBEX35", "Industrials"),
    ("LOG.MC", "ES", "BME", 1, "IBEX35", "Industrials"),
    ("PHM.MC", "ES", "BME", 1, "IBEX35", "Healthcare"),
    ("ROVI.MC", "ES", "BME", 1, "IBEX35", "Healthcare"),
    ("ELE.MC", "ES", "BME", 1, "IBEX35", "Utilities"),
    ("NTGY.MC", "ES", "BME", 1, "IBEX35", "Utilities"),
    ("ANA.MC", "ES", "BME", 1, "IBEX35", "Industrials"),
    ("UNI.MC", "ES", "BME", 1, "IBEX35", "Financials"),
    # IBEX Medium Cap — Tier 2
    ("GRE.MC", "ES", "BME", 2, "IBEX_MID", "Utilities"),
    ("SLR.MC", "ES", "BME", 2, "IBEX_MID", "Utilities"),
    ("EBRO.MC", "ES", "BME", 2, "IBEX_MID", "Consumer Staples"),
    ("VIS.MC", "ES", "BME", 2, "IBEX_MID", "Materials"),
    ("VID.MC", "ES", "BME", 2, "IBEX_MID", "Materials"),
    ("ALM.MC", "ES", "BME", 2, "IBEX_MID", "Healthcare"),
    ("ENO.MC", "ES", "BME", 2, "IBEX_MID", "Industrials"),
    ("APAM.MC", "ES", "BME", 2, "IBEX_MID", "Industrials"),
    ("TRE.MC", "ES", "BME", 2, "IBEX_MID", "Industrials"),
    ("MEL.MC", "ES", "BME", 2, "IBEX_MID", "Consumer Discretionary"),
    ("CAF.MC", "ES", "BME", 2, "IBEX_MID", "Industrials"),
    ("GEST.MC", "ES", "BME", 2, "IBEX_MID", "Consumer Discretionary"),
    ("FAE.MC", "ES", "BME", 2, "IBEX_MID", "Healthcare"),
    ("PSG.MC", "ES", "BME", 2, "IBEX_MID", "Industrials"),
    ("A3M.MC", "ES", "BME", 2, "IBEX_MID", "Communication"),
    ("AEDAS.MC", "ES", "BME", 2, "IBEX_MID", "Real Estate"),
    ("CASH.MC", "ES", "BME", 2, "IBEX_MID", "Industrials"),
    ("ENC.MC", "ES", "BME", 2, "IBEX_MID", "Materials"),
    ("DOM.MC", "ES", "BME", 2, "IBEX_MID", "Technology"),
    ("TLGO.MC", "ES", "BME", 2, "IBEX_MID", "Industrials"),
    ("TUB.MC", "ES", "BME", 2, "IBEX_MID", "Materials"),
]

# ---------------------------------------------------------------------------
# Netherlands — AEX 25 + AMX selection
# ---------------------------------------------------------------------------
_NETHERLANDS = [
    ("ASML.AS", "NL", "AMS", 1, "AEX25", "Technology"),
    ("INGA.AS", "NL", "AMS", 1, "AEX25", "Financials"),
    ("PHIA.AS", "NL", "AMS", 1, "AEX25", "Healthcare"),
    ("AD.AS", "NL", "AMS", 1, "AEX25", "Consumer Staples"),
    ("UNA.AS", "NL", "AMS", 1, "AEX25", "Consumer Staples"),
    ("WKL.AS", "NL", "AMS", 1, "AEX25", "Industrials"),
    ("HEIA.AS", "NL", "AMS", 1, "AEX25", "Consumer Staples"),
    ("REN.AS", "NL", "AMS", 1, "AEX25", "Utilities"),
    ("DSM.AS", "NL", "AMS", 1, "AEX25", "Materials"),
    ("AGN.AS", "NL", "AMS", 1, "AEX25", "Financials"),
    ("KPN.AS", "NL", "AMS", 1, "AEX25", "Communication"),
    ("AKZA.AS", "NL", "AMS", 1, "AEX25", "Materials"),
    ("NN.AS", "NL", "AMS", 1, "AEX25", "Financials"),
    ("ASM.AS", "NL", "AMS", 1, "AEX25", "Technology"),
    ("RAND.AS", "NL", "AMS", 1, "AEX25", "Industrials"),
    ("URW.AS", "NL", "AMS", 1, "AEX25", "Real Estate"),
    ("PRX.AS", "NL", "AMS", 1, "AEX25", "Technology"),
    ("BESI.AS", "NL", "AMS", 1, "AEX25", "Technology"),
    ("LIGHT.AS", "NL", "AMS", 1, "AEX25", "Technology"),
    ("IMCD.AS", "NL", "AMS", 1, "AEX25", "Materials"),
    ("JDEP.AS", "NL", "AMS", 1, "AEX25", "Consumer Staples"),
    ("ABN.AS", "NL", "AMS", 1, "AEX25", "Financials"),
    ("TKWY.AS", "NL", "AMS", 1, "AEX25", "Consumer Discretionary"),
    ("SBMO.AS", "NL", "AMS", 2, "AMX", "Industrials"),
    ("ADYEN.AS", "NL", "AMS", 1, "AEX25", "Technology"),
    ("FLOW.AS", "NL", "AMS", 2, "AMX", "Technology"),
    ("ALFEN.AS", "NL", "AMS", 2, "AMX", "Industrials"),
    ("TOM2.AS", "NL", "AMS", 2, "AMX", "Technology"),
]

# ---------------------------------------------------------------------------
# Italy — FTSE MIB + FTSE Italia Mid Cap (EXPANDED)
# ---------------------------------------------------------------------------
_ITALY = [
    # FTSE MIB — Tier 1
    ("ENI.MI", "IT", "BIT", 1, "FTSE_MIB", "Energy"),
    ("ISP.MI", "IT", "BIT", 1, "FTSE_MIB", "Financials"),
    ("UCG.MI", "IT", "BIT", 1, "FTSE_MIB", "Financials"),
    ("ENEL.MI", "IT", "BIT", 1, "FTSE_MIB", "Utilities"),
    ("STM.MI", "IT", "BIT", 1, "FTSE_MIB", "Technology"),
    ("G.MI", "IT", "BIT", 1, "FTSE_MIB", "Financials"),
    ("TRN.MI", "IT", "BIT", 1, "FTSE_MIB", "Utilities"),
    ("RACE.MI", "IT", "BIT", 1, "FTSE_MIB", "Consumer Discretionary"),
    ("LDO.MI", "IT", "BIT", 1, "FTSE_MIB", "Industrials"),
    ("MONC.MI", "IT", "BIT", 1, "FTSE_MIB", "Consumer Discretionary"),
    ("PRY.MI", "IT", "BIT", 1, "FTSE_MIB", "Industrials"),
    ("TEN.MI", "IT", "BIT", 1, "FTSE_MIB", "Utilities"),
    ("SPM.MI", "IT", "BIT", 1, "FTSE_MIB", "Healthcare"),
    ("SRG.MI", "IT", "BIT", 1, "FTSE_MIB", "Utilities"),
    ("BAMI.MI", "IT", "BIT", 1, "FTSE_MIB", "Financials"),
    ("PST.MI", "IT", "BIT", 1, "FTSE_MIB", "Financials"),
    ("CPR.MI", "IT", "BIT", 1, "FTSE_MIB", "Consumer Discretionary"),
    ("HER.MI", "IT", "BIT", 1, "FTSE_MIB", "Consumer Discretionary"),
    ("MB.MI", "IT", "BIT", 1, "FTSE_MIB", "Financials"),
    ("FBK.MI", "IT", "BIT", 1, "FTSE_MIB", "Financials"),
    ("STLAM.MI", "IT", "BIT", 1, "FTSE_MIB", "Consumer Discretionary"),
    ("PIRC.MI", "IT", "BIT", 1, "FTSE_MIB", "Consumer Discretionary"),
    ("REC.MI", "IT", "BIT", 1, "FTSE_MIB", "Consumer Discretionary"),
    ("BMPS.MI", "IT", "BIT", 1, "FTSE_MIB", "Financials"),
    ("BPE.MI", "IT", "BIT", 1, "FTSE_MIB", "Financials"),
    ("UNI.MI", "IT", "BIT", 1, "FTSE_MIB", "Financials"),
    ("TIT.MI", "IT", "BIT", 1, "FTSE_MIB", "Communication"),
    ("BMED.MI", "IT", "BIT", 1, "FTSE_MIB", "Financials"),
    ("IG.MI", "IT", "BIT", 1, "FTSE_MIB", "Utilities"),
    ("BZU.MI", "IT", "BIT", 1, "FTSE_MIB", "Materials"),
    ("BPSO.MI", "IT", "BIT", 1, "FTSE_MIB", "Financials"),
    ("IVG.MI", "IT", "BIT", 1, "FTSE_MIB", "Industrials"),
    # FTSE Italia Mid Cap — Tier 2
    ("BC.MI", "IT", "BIT", 2, "FTSE_MID", "Consumer Discretionary"),
    ("AZM.MI", "IT", "BIT", 2, "FTSE_MID", "Financials"),
    ("DANR.MI", "IT", "BIT", 2, "FTSE_MID", "Materials"),
    ("TGYM.MI", "IT", "BIT", 2, "FTSE_MID", "Consumer Discretionary"),
    ("NEXI.MI", "IT", "BIT", 2, "FTSE_MID", "Technology"),
    ("DIA.MI", "IT", "BIT", 2, "FTSE_MID", "Healthcare"),
    ("IP.MI", "IT", "BIT", 2, "FTSE_MID", "Industrials"),
    ("ERG.MI", "IT", "BIT", 2, "FTSE_MID", "Utilities"),
    ("REY.MI", "IT", "BIT", 2, "FTSE_MID", "Technology"),
    ("ENAV.MI", "IT", "BIT", 2, "FTSE_MID", "Industrials"),
    ("CRL.MI", "IT", "BIT", 2, "FTSE_MID", "Industrials"),
    ("WBD.MI", "IT", "BIT", 2, "FTSE_MID", "Industrials"),
    ("ANIM.MI", "IT", "BIT", 2, "FTSE_MID", "Financials"),
    ("AMP.MI", "IT", "BIT", 2, "FTSE_MID", "Healthcare"),
    ("AVIO.MI", "IT", "BIT", 2, "FTSE_MID", "Industrials"),
    ("RWAY.MI", "IT", "BIT", 2, "FTSE_MID", "Communication"),
    ("IF.MI", "IT", "BIT", 2, "FTSE_MID", "Financials"),
    ("TIP.MI", "IT", "BIT", 2, "FTSE_MID", "Financials"),
    ("SES.MI", "IT", "BIT", 2, "FTSE_MID", "Technology"),
    ("SFER.MI", "IT", "BIT", 2, "FTSE_MID", "Consumer Discretionary"),
    ("PIA.MI", "IT", "BIT", 2, "FTSE_MID", "Consumer Discretionary"),
    ("BGN.MI", "IT", "BIT", 2, "FTSE_MID", "Consumer Discretionary"),
]

# ---------------------------------------------------------------------------
# Switzerland — SMI + SPI selection
# ---------------------------------------------------------------------------
_SWITZERLAND = [
    ("NESN.SW", "CH", "SWX", 1, "SMI", "Consumer Staples"),
    ("ROG.SW", "CH", "SWX", 1, "SMI", "Healthcare"),
    ("NOVN.SW", "CH", "SWX", 1, "SMI", "Healthcare"),
    ("UBSG.SW", "CH", "SWX", 1, "SMI", "Financials"),
    ("ABBN.SW", "CH", "SWX", 1, "SMI", "Industrials"),
    ("CSGN.SW", "CH", "SWX", 1, "SMI", "Financials"),
    ("SREN.SW", "CH", "SWX", 1, "SMI", "Financials"),
    ("ZURN.SW", "CH", "SWX", 1, "SMI", "Financials"),
    ("GIVN.SW", "CH", "SWX", 1, "SMI", "Materials"),
    ("LONN.SW", "CH", "SWX", 1, "SMI", "Healthcare"),
    ("GEBN.SW", "CH", "SWX", 1, "SMI", "Industrials"),
    ("SGSN.SW", "CH", "SWX", 1, "SMI", "Industrials"),
    ("PGHN.SW", "CH", "SWX", 1, "SMI", "Financials"),
    ("HOLN.SW", "CH", "SWX", 1, "SMI", "Materials"),
    ("SCMN.SW", "CH", "SWX", 1, "SMI", "Consumer Discretionary"),
    ("SIKA.SW", "CH", "SWX", 1, "SMI", "Materials"),
    ("SLHN.SW", "CH", "SWX", 1, "SMI", "Financials"),
    ("TEMN.SW", "CH", "SWX", 1, "SMI", "Technology"),
    ("VACN.SW", "CH", "SWX", 2, "SPI", "Industrials"),
    ("ALC.SW", "CH", "SWX", 1, "SMI", "Healthcare"),
]

# ---------------------------------------------------------------------------
# Nordics — OMXS30, OMXC25, OMXH25, OBX
# ---------------------------------------------------------------------------
_NORDICS = [
    # Sweden
    ("VOLV-B.ST", "SE", "OMX", 1, "OMXS30", "Industrials"),
    ("ERIC-B.ST", "SE", "OMX", 1, "OMXS30", "Technology"),
    ("SAND.ST", "SE", "OMX", 1, "OMXS30", "Industrials"),
    ("ATCO-A.ST", "SE", "OMX", 1, "OMXS30", "Industrials"),
    ("SEB-A.ST", "SE", "OMX", 1, "OMXS30", "Financials"),
    ("SWED-A.ST", "SE", "OMX", 1, "OMXS30", "Financials"),
    ("SHB-A.ST", "SE", "OMX", 1, "OMXS30", "Financials"),
    ("INVE-B.ST", "SE", "OMX", 1, "OMXS30", "Financials"),
    ("HM-B.ST", "SE", "OMX", 1, "OMXS30", "Consumer Discretionary"),
    ("ASSA-B.ST", "SE", "OMX", 1, "OMXS30", "Industrials"),
    ("ALFA.ST", "SE", "OMX", 1, "OMXS30", "Industrials"),
    ("ABB.ST", "SE", "OMX", 1, "OMXS30", "Industrials"),
    ("SKF-B.ST", "SE", "OMX", 1, "OMXS30", "Industrials"),
    ("TELIA.ST", "SE", "OMX", 1, "OMXS30", "Communication"),
    ("ESSITY-B.ST", "SE", "OMX", 1, "OMXS30", "Consumer Staples"),
    ("HEXA-B.ST", "SE", "OMX", 1, "OMXS30", "Technology"),
    ("SAAB-B.ST", "SE", "OMX", 1, "OMXS30", "Industrials"),
    ("ELUX-B.ST", "SE", "OMX", 2, "OMXS_MID", "Consumer Discretionary"),
    # Denmark
    ("NOVO-B.CO", "DK", "OMX", 1, "OMXC25", "Healthcare"),
    ("MAERSK-B.CO", "DK", "OMX", 1, "OMXC25", "Industrials"),
    ("VWS.CO", "DK", "OMX", 1, "OMXC25", "Industrials"),
    ("CARL-B.CO", "DK", "OMX", 1, "OMXC25", "Consumer Staples"),
    ("DSV.CO", "DK", "OMX", 1, "OMXC25", "Industrials"),
    ("ORSTED.CO", "DK", "OMX", 1, "OMXC25", "Utilities"),
    ("PNDORA.CO", "DK", "OMX", 1, "OMXC25", "Consumer Discretionary"),
    ("COLO-B.CO", "DK", "OMX", 1, "OMXC25", "Consumer Staples"),
    ("GMAB.CO", "DK", "OMX", 1, "OMXC25", "Healthcare"),
    # Finland
    ("NOKIA.HE", "FI", "OMX", 1, "OMXH25", "Technology"),
    ("SAMPO.HE", "FI", "OMX", 1, "OMXH25", "Financials"),
    ("KNEBV.HE", "FI", "OMX", 1, "OMXH25", "Industrials"),
    ("NESTE.HE", "FI", "OMX", 1, "OMXH25", "Energy"),
    ("FORTUM.HE", "FI", "OMX", 1, "OMXH25", "Utilities"),
    ("UPM.HE", "FI", "OMX", 1, "OMXH25", "Materials"),
    ("WRT1V.HE", "FI", "OMX", 1, "OMXH25", "Industrials"),
    # Norway
    ("EQNR.OL", "NO", "OSE", 1, "OBX", "Energy"),
    ("DNB.OL", "NO", "OSE", 1, "OBX", "Financials"),
    ("TEL.OL", "NO", "OSE", 1, "OBX", "Communication"),
    ("MOWI.OL", "NO", "OSE", 1, "OBX", "Consumer Staples"),
    ("ORK.OL", "NO", "OSE", 1, "OBX", "Consumer Staples"),
    ("YAR.OL", "NO", "OSE", 1, "OBX", "Materials"),
    ("AKRBP.OL", "NO", "OSE", 1, "OBX", "Energy"),
]

# ---------------------------------------------------------------------------
# Canada — TSX 60 + selection
# ---------------------------------------------------------------------------
_CANADA = [
    ("RY.TO", "CA", "TSX", 1, "TSX60", "Financials"),
    ("TD.TO", "CA", "TSX", 1, "TSX60", "Financials"),
    ("BNS.TO", "CA", "TSX", 1, "TSX60", "Financials"),
    ("BMO.TO", "CA", "TSX", 1, "TSX60", "Financials"),
    ("ENB.TO", "CA", "TSX", 1, "TSX60", "Energy"),
    ("CNR.TO", "CA", "TSX", 1, "TSX60", "Industrials"),
    ("CP.TO", "CA", "TSX", 1, "TSX60", "Industrials"),
    ("TRP.TO", "CA", "TSX", 1, "TSX60", "Energy"),
    ("SU.TO", "CA", "TSX", 1, "TSX60", "Energy"),
    ("CNQ.TO", "CA", "TSX", 1, "TSX60", "Energy"),
    ("MFC.TO", "CA", "TSX", 1, "TSX60", "Financials"),
    ("SLF.TO", "CA", "TSX", 1, "TSX60", "Financials"),
    ("BCE.TO", "CA", "TSX", 1, "TSX60", "Communication"),
    ("T.TO", "CA", "TSX", 1, "TSX60", "Communication"),
    ("ABX.TO", "CA", "TSX", 1, "TSX60", "Materials"),
    ("NTR.TO", "CA", "TSX", 1, "TSX60", "Materials"),
    ("FNV.TO", "CA", "TSX", 1, "TSX60", "Materials"),
    ("WFG.TO", "CA", "TSX", 1, "TSX60", "Materials"),
    ("ATD.TO", "CA", "TSX", 1, "TSX60", "Consumer Staples"),
    ("CSU.TO", "CA", "TSX", 1, "TSX60", "Technology"),
    ("BAM.TO", "CA", "TSX", 1, "TSX60", "Financials"),
    ("BN.TO", "CA", "TSX", 1, "TSX60", "Financials"),
    ("RCI-B.TO", "CA", "TSX", 1, "TSX60", "Communication"),
    ("IFC.TO", "CA", "TSX", 1, "TSX60", "Financials"),
    ("GIB-A.TO", "CA", "TSX", 1, "TSX60", "Technology"),
    ("SHOP.TO", "CA", "TSX", 1, "TSX60", "Technology"),
    ("L.TO", "CA", "TSX", 1, "TSX60", "Consumer Staples"),
    ("MG.TO", "CA", "TSX", 1, "TSX60", "Consumer Discretionary"),
    ("DOL.TO", "CA", "TSX", 1, "TSX60", "Consumer Discretionary"),
    ("CCL-B.TO", "CA", "TSX", 1, "TSX60", "Materials"),
    ("WCN.TO", "CA", "TSX", 1, "TSX60", "Industrials"),
    ("QSR.TO", "CA", "TSX", 1, "TSX60", "Consumer Discretionary"),
    ("FFH.TO", "CA", "TSX", 1, "TSX60", "Financials"),
    ("AEM.TO", "CA", "TSX", 1, "TSX60", "Materials"),
    ("IMO.TO", "CA", "TSX", 1, "TSX60", "Energy"),
    ("CVE.TO", "CA", "TSX", 1, "TSX60", "Energy"),
    ("TOU.TO", "CA", "TSX", 1, "TSX60", "Energy"),
    ("ARX.TO", "CA", "TSX", 1, "TSX60", "Energy"),
    ("WPM.TO", "CA", "TSX", 1, "TSX60", "Materials"),
]

# ---------------------------------------------------------------------------
# Australia — ASX 50 selection
# ---------------------------------------------------------------------------
_AUSTRALIA = [
    ("BHP.AX", "AU", "ASX", 1, "ASX50", "Materials"),
    ("CBA.AX", "AU", "ASX", 1, "ASX50", "Financials"),
    ("CSL.AX", "AU", "ASX", 1, "ASX50", "Healthcare"),
    ("NAB.AX", "AU", "ASX", 1, "ASX50", "Financials"),
    ("WBC.AX", "AU", "ASX", 1, "ASX50", "Financials"),
    ("ANZ.AX", "AU", "ASX", 1, "ASX50", "Financials"),
    ("MQG.AX", "AU", "ASX", 1, "ASX50", "Financials"),
    ("WES.AX", "AU", "ASX", 1, "ASX50", "Consumer Discretionary"),
    ("WOW.AX", "AU", "ASX", 1, "ASX50", "Consumer Staples"),
    ("TLS.AX", "AU", "ASX", 1, "ASX50", "Communication"),
    ("RIO.AX", "AU", "ASX", 1, "ASX50", "Materials"),
    ("FMG.AX", "AU", "ASX", 1, "ASX50", "Materials"),
    ("ALL.AX", "AU", "ASX", 1, "ASX50", "Consumer Discretionary"),
    ("COL.AX", "AU", "ASX", 1, "ASX50", "Consumer Staples"),
    ("GMG.AX", "AU", "ASX", 1, "ASX50", "Real Estate"),
    ("TCL.AX", "AU", "ASX", 1, "ASX50", "Industrials"),
    ("STO.AX", "AU", "ASX", 1, "ASX50", "Energy"),
    ("WDS.AX", "AU", "ASX", 1, "ASX50", "Energy"),
    ("REA.AX", "AU", "ASX", 1, "ASX50", "Technology"),
    ("XRO.AX", "AU", "ASX", 1, "ASX50", "Technology"),
    ("JHX.AX", "AU", "ASX", 1, "ASX50", "Materials"),
    ("S32.AX", "AU", "ASX", 1, "ASX50", "Materials"),
    ("MIN.AX", "AU", "ASX", 1, "ASX50", "Materials"),
    ("ORG.AX", "AU", "ASX", 1, "ASX50", "Energy"),
    ("SHL.AX", "AU", "ASX", 1, "ASX50", "Healthcare"),
    ("QBE.AX", "AU", "ASX", 1, "ASX50", "Financials"),
    ("IAG.AX", "AU", "ASX", 1, "ASX50", "Financials"),
    ("MPL.AX", "AU", "ASX", 1, "ASX50", "Financials"),
    ("NCM.AX", "AU", "ASX", 1, "ASX50", "Materials"),
    ("NST.AX", "AU", "ASX", 1, "ASX50", "Materials"),
]

# ---------------------------------------------------------------------------
# Japan — Nikkei 225 top
# ---------------------------------------------------------------------------
_JAPAN = [
    ("7203.T", "JP", "TSE", 1, "NIKKEI225", "Consumer Discretionary"),
    ("6758.T", "JP", "TSE", 1, "NIKKEI225", "Technology"),
    ("9984.T", "JP", "TSE", 1, "NIKKEI225", "Technology"),
    ("6902.T", "JP", "TSE", 1, "NIKKEI225", "Consumer Discretionary"),
    ("8306.T", "JP", "TSE", 1, "NIKKEI225", "Financials"),
    ("9432.T", "JP", "TSE", 1, "NIKKEI225", "Communication"),
    ("6861.T", "JP", "TSE", 1, "NIKKEI225", "Technology"),
    ("7741.T", "JP", "TSE", 1, "NIKKEI225", "Healthcare"),
    ("6501.T", "JP", "TSE", 1, "NIKKEI225", "Industrials"),
    ("4063.T", "JP", "TSE", 1, "NIKKEI225", "Materials"),
    ("8035.T", "JP", "TSE", 1, "NIKKEI225", "Technology"),
    ("6098.T", "JP", "TSE", 1, "NIKKEI225", "Industrials"),
    ("9433.T", "JP", "TSE", 1, "NIKKEI225", "Communication"),
    ("4502.T", "JP", "TSE", 1, "NIKKEI225", "Healthcare"),
    ("4503.T", "JP", "TSE", 1, "NIKKEI225", "Healthcare"),
    ("6301.T", "JP", "TSE", 1, "NIKKEI225", "Industrials"),
    ("7267.T", "JP", "TSE", 1, "NIKKEI225", "Consumer Discretionary"),
    ("8316.T", "JP", "TSE", 1, "NIKKEI225", "Financials"),
    ("3382.T", "JP", "TSE", 1, "NIKKEI225", "Consumer Staples"),
    ("6954.T", "JP", "TSE", 1, "NIKKEI225", "Technology"),
    ("7974.T", "JP", "TSE", 1, "NIKKEI225", "Consumer Discretionary"),
    ("4519.T", "JP", "TSE", 1, "NIKKEI225", "Healthcare"),
    ("8058.T", "JP", "TSE", 1, "NIKKEI225", "Industrials"),
    ("8031.T", "JP", "TSE", 1, "NIKKEI225", "Industrials"),
    ("6326.T", "JP", "TSE", 1, "NIKKEI225", "Industrials"),
    ("4543.T", "JP", "TSE", 1, "NIKKEI225", "Healthcare"),
    ("6367.T", "JP", "TSE", 1, "NIKKEI225", "Industrials"),
    ("2914.T", "JP", "TSE", 1, "NIKKEI225", "Consumer Staples"),
    ("6594.T", "JP", "TSE", 1, "NIKKEI225", "Industrials"),
    ("7751.T", "JP", "TSE", 1, "NIKKEI225", "Technology"),
]

# ---------------------------------------------------------------------------
# Hong Kong — Hang Seng selection
# ---------------------------------------------------------------------------
_HONG_KONG = [
    ("0700.HK", "HK", "HKEX", 1, "HSI", "Technology"),
    ("9988.HK", "HK", "HKEX", 1, "HSI", "Technology"),
    ("0005.HK", "HK", "HKEX", 1, "HSI", "Financials"),
    ("1299.HK", "HK", "HKEX", 1, "HSI", "Financials"),
    ("0941.HK", "HK", "HKEX", 1, "HSI", "Communication"),
    ("2318.HK", "HK", "HKEX", 1, "HSI", "Financials"),
    ("0388.HK", "HK", "HKEX", 1, "HSI", "Financials"),
    ("0883.HK", "HK", "HKEX", 1, "HSI", "Energy"),
    ("0016.HK", "HK", "HKEX", 1, "HSI", "Real Estate"),
    ("0003.HK", "HK", "HKEX", 1, "HSI", "Utilities"),
    ("1038.HK", "HK", "HKEX", 1, "HSI", "Industrials"),
    ("0011.HK", "HK", "HKEX", 1, "HSI", "Real Estate"),
    ("0066.HK", "HK", "HKEX", 1, "HSI", "Industrials"),
    ("0823.HK", "HK", "HKEX", 1, "HSI", "Real Estate"),
    ("1113.HK", "HK", "HKEX", 1, "HSI", "Real Estate"),
    ("2688.HK", "HK", "HKEX", 1, "HSI", "Industrials"),
    ("0027.HK", "HK", "HKEX", 1, "HSI", "Consumer Discretionary"),
    ("1997.HK", "HK", "HKEX", 1, "HSI", "Real Estate"),
    ("0669.HK", "HK", "HKEX", 1, "HSI", "Technology"),
    ("9618.HK", "HK", "HKEX", 1, "HSI", "Consumer Discretionary"),
]

# ---------------------------------------------------------------------------
# Singapore — STI selection
# ---------------------------------------------------------------------------
_SINGAPORE = [
    ("D05.SI", "SG", "SGX", 1, "STI", "Financials"),
    ("O39.SI", "SG", "SGX", 1, "STI", "Financials"),
    ("U11.SI", "SG", "SGX", 1, "STI", "Financials"),
    ("Z74.SI", "SG", "SGX", 1, "STI", "Communication"),
    ("C6L.SI", "SG", "SGX", 1, "STI", "Industrials"),
    ("A17U.SI", "SG", "SGX", 1, "STI", "Real Estate"),
    ("C38U.SI", "SG", "SGX", 1, "STI", "Real Estate"),
    ("G13.SI", "SG", "SGX", 1, "STI", "Industrials"),
    ("S58.SI", "SG", "SGX", 1, "STI", "Industrials"),
    ("BN4.SI", "SG", "SGX", 1, "STI", "Industrials"),
]

# ---------------------------------------------------------------------------
# Internal: all region lists for iteration
# ---------------------------------------------------------------------------
_ALL_REGIONS: dict[str, list[tuple]] = {
    "UK": _UK,
    "Germany": _GERMANY,
    "France": _FRANCE,
    "Spain": _SPAIN,
    "Netherlands": _NETHERLANDS,
    "Italy": _ITALY,
    "Switzerland": _SWITZERLAND,
    "Nordics": _NORDICS,
    "Canada": _CANADA,
    "Australia": _AUSTRALIA,
    "Japan": _JAPAN,
    "Hong Kong": _HONG_KONG,
    "Singapore": _SINGAPORE,
}


# ---------------------------------------------------------------------------
# Build the full structured universe (once, at import time)
# ---------------------------------------------------------------------------
_UNIVERSE: list[UniverseEntry] = []
for _region_name, _region_list in _ALL_REGIONS.items():
    for _entry in _region_list:
        _UNIVERSE.append(UniverseEntry(*_entry))

# Deduplicate by ticker (keep first occurrence)
_SEEN_TICKERS: set[str] = set()
_UNIVERSE_DEDUPED: list[UniverseEntry] = []
for _e in _UNIVERSE:
    _t = _e.ticker.upper()
    if _t not in _SEEN_TICKERS:
        _SEEN_TICKERS.add(_t)
        _UNIVERSE_DEDUPED.append(_e)
_UNIVERSE = _UNIVERSE_DEDUPED
del _SEEN_TICKERS, _UNIVERSE_DEDUPED

_TICKER_TO_REGION = {
    entry[0].upper(): region_name
    for region_name, region_entries in _ALL_REGIONS.items()
    for entry in region_entries
}
_RESOLVED_TICKER_TO_REGION: dict[str, str] = {}
for _region_name, _region_entries in _ALL_REGIONS.items():
    for _entry in _region_entries:
        _resolved = resolve_yahoo_ticker(_entry[0])
        if _resolved and not is_quarantined_ticker(_resolved):
            _RESOLVED_TICKER_TO_REGION.setdefault(_resolved.upper(), _region_name)
del _region_name, _region_entries, _entry, _resolved


def _iter_filtered_entries(
    exclude_tickers: set[str] | None = None,
    *,
    max_tier: int = 2,
    countries: list[str] | None = None,
    regions: list[str] | None = None,
    include_tier2: bool = True,
):
    """Yield universe entries that satisfy the requested filters."""
    exclude: set[str] = set()
    for t in exclude_tickers or set():
        raw = str(t or "").upper().strip()
        if raw:
            exclude.add(raw)
            exclude.add(resolve_yahoo_ticker(raw))
    allowed_countries = set(countries or [])
    allowed_regions = set(regions or [])
    seen: set[str] = set()

    for entry in _UNIVERSE:
        raw_ticker = entry.ticker.upper()
        resolved_ticker = resolve_yahoo_ticker(raw_ticker)
        if raw_ticker in exclude or resolved_ticker in exclude:
            continue
        if entry.tier > max_tier:
            continue
        if not include_tier2 and entry.tier > 1:
            continue
        if allowed_countries and entry.country not in allowed_countries:
            continue
        region = _TICKER_TO_REGION.get(raw_ticker, "")
        if is_excluded_ticker(raw_ticker, country=entry.country, region=region):
            continue
        if allowed_regions and region not in allowed_regions:
            continue
        if resolved_ticker in seen:
            continue
        seen.add(resolved_ticker)
        yield entry._replace(ticker=resolved_ticker)


def get_region_for_ticker(ticker: str) -> str:
    """Return the configured universe region for a ticker, if known."""
    raw = str(ticker or "").upper().strip()
    resolved = resolve_yahoo_ticker(raw)
    region = _RESOLVED_TICKER_TO_REGION.get(resolved, _TICKER_TO_REGION.get(raw, ""))
    return "" if is_excluded_ticker(resolved, region=region) else region


def build_daily_universe_snapshot(
    exclude_tickers: set[str] | None = None,
    *,
    max_tier: int = 2,
    countries: list[str] | None = None,
    regions: list[str] | None = None,
    include_tier2: bool = True,
    region_caps: dict[str, int] | None = None,
) -> list[UniverseEntry]:
    """Return the daily systematic universe snapshot with metadata.

    The default behavior is intentionally broad: include all configured
    regions every day, optionally capped per region when callers need a
    smaller operational subset.
    """
    per_region_caps = region_caps or {}
    region_counts: dict[str, int] = {}
    selected: list[UniverseEntry] = []

    for entry in _iter_filtered_entries(
        exclude_tickers=exclude_tickers,
        max_tier=max_tier,
        countries=countries,
        regions=regions,
        include_tier2=include_tier2,
    ):
        region = _RESOLVED_TICKER_TO_REGION.get(entry.ticker.upper(), "Unknown")
        cap = per_region_caps.get(region)
        if cap is not None and region_counts.get(region, 0) >= cap:
            continue
        selected.append(entry)
        region_counts[region] = region_counts.get(region, 0) + 1

    return selected


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_full_universe() -> list[UniverseEntry]:
    """Return the active structured universe with metadata."""
    return list(_iter_filtered_entries(max_tier=99, include_tier2=True))


def get_global_universe(
    exclude_tickers: set[str] | None = None,
    *,
    max_tier: int = 2,
    countries: list[str] | None = None,
    include_dynamic: bool = True,
) -> list[str]:
    """Return ticker symbols from the universe, with optional filters.

    Args:
        exclude_tickers: Tickers to exclude (e.g. existing holdings).
        max_tier: Maximum tier to include (1 = core only, 2 = core + mid-cap).
        countries: ISO country codes to include (None = all).
        include_dynamic: Include dynamically validated tickers from reconstitution.

    Returns:
        Deduplicated list of ticker strings.
    """
    static = [
        entry.ticker
        for entry in build_daily_universe_snapshot(
            exclude_tickers=exclude_tickers,
            max_tier=max_tier,
            countries=countries,
        )
    ]
    if not include_dynamic:
        return static

    # Merge dynamic supplement (non-stale, non-excluded, non-duplicate)
    seen = {t.upper() for t in static}
    excluded: set[str] = set()
    for t in exclude_tickers or set():
        raw = str(t or "").upper().strip()
        if raw:
            excluded.add(raw)
            excluded.add(resolve_yahoo_ticker(raw))
    dynamic = get_dynamic_tickers()
    for t in dynamic:
        t_upper = resolve_yahoo_ticker(t).upper()
        if is_excluded_ticker(t_upper):
            continue
        if t_upper not in seen and t_upper not in excluded:
            static.append(t_upper)
            seen.add(t_upper)
    return static


def get_universe_for_rotation(
    day_of_week: int,
    exclude_tickers: set[str] | None = None,
) -> list[str]:
    """Backward-compatible alias for the daily systematic snapshot.

    ``day_of_week`` is retained for API compatibility but no longer affects
    membership. Tier 2 names are now available every day by default.
    """
    _ = day_of_week
    return [
        entry.ticker
        for entry in build_daily_universe_snapshot(
            exclude_tickers=exclude_tickers,
            max_tier=2,
            include_tier2=True,
        )
    ]


def get_universe_by_region(
    max_tier: int = 2,
) -> dict[str, list[str]]:
    """Return the universe organised by region (for progress reporting).

    Args:
        max_tier: Maximum tier to include.
    """
    result: dict[str, list[str]] = {}
    for region_name in _ALL_REGIONS:
        tickers = [
            entry.ticker
            for entry in build_daily_universe_snapshot(
                max_tier=max_tier,
                regions=[region_name],
            )
        ]
        if tickers:
            result[region_name] = tickers
    return result


def get_universe_stats() -> dict:
    """Return summary statistics about the universe."""
    active_universe = get_full_universe()
    total = len(active_universe)
    tier1 = sum(1 for e in active_universe if e.tier == 1)
    tier2 = sum(1 for e in active_universe if e.tier == 2)
    by_country: dict[str, int] = {}
    by_region: dict[str, int] = {}
    for e in active_universe:
        by_country[e.country] = by_country.get(e.country, 0) + 1
        region = _RESOLVED_TICKER_TO_REGION.get(e.ticker.upper(), "Unknown")
        by_region[region] = by_region.get(region, 0) + 1

    # Include dynamic supplement stats if available
    dynamic = load_dynamic_supplement()
    dynamic_count = len(get_dynamic_entries())

    return {
        "total": total,
        "total_with_dynamic": total + dynamic_count,
        "tier1": tier1,
        "tier2": tier2,
        "dynamic": dynamic_count,
        "by_country": by_country,
        "by_region": by_region,
        "last_refreshed": UNIVERSE_LAST_REFRESHED,
        "last_dynamic_refresh": dynamic.get("refreshed_at"),
    }


# ---------------------------------------------------------------------------
# Dynamic Universe Reconstitution (Priority B)
# ---------------------------------------------------------------------------
# Supplements the static universe with validated tickers from yfinance.
# Prunes stale tickers (no data for 3+ consecutive refreshes).
# Runs on-demand or from the orchestrator.

import json
import logging
from datetime import datetime
from pathlib import Path

_logger = logging.getLogger(__name__)
_DYNAMIC_CACHE_PATH = Path(__file__).parent.parent / "feature_cache" / "universe_dynamic.json"
_MIN_AVG_VOLUME = 500_000       # $500K daily volume floor
_MIN_MARKET_CAP = 100_000_000   # $100M market cap floor
_MAX_STALE_RUNS = 3             # Remove after N consecutive no-data refreshes


# yfinance returns country as a free-text label ("Germany", "Japan"). Map the
# common ones to ISO-3166-1-alpha-2 so downstream gates (dollar-volume tiers,
# region filters) can match on a single convention.
_YF_COUNTRY_TO_ISO: dict[str, str] = {
    "united states": "US", "usa": "US", "us": "US",
    "united kingdom": "GB", "uk": "GB", "britain": "GB",
    "germany": "DE", "france": "FR", "spain": "ES", "italy": "IT",
    "netherlands": "NL", "switzerland": "CH", "sweden": "SE",
    "denmark": "DK", "norway": "NO", "finland": "FI", "belgium": "BE",
    "ireland": "IE", "austria": "AT", "portugal": "PT", "luxembourg": "LU",
    "japan": "JP", "china": "CN", "hong kong": "HK", "singapore": "SG",
    "south korea": "KR", "korea": "KR", "taiwan": "TW", "india": "IN",
    "australia": "AU", "new zealand": "NZ",
    "canada": "CA", "mexico": "MX", "brazil": "BR",
    "israel": "IL",
}


def _iso_country(raw: str) -> str:
    if not raw:
        return ""
    s = raw.strip().lower()
    if s in _YF_COUNTRY_TO_ISO:
        return _YF_COUNTRY_TO_ISO[s]
    # Fallback: if already a 2-letter code, pass through uppercased.
    if len(raw) == 2 and raw.isalpha():
        return raw.upper()
    return ""


def load_dynamic_supplement() -> dict:
    """Load the cached dynamic universe supplement."""
    if _DYNAMIC_CACHE_PATH.exists():
        try:
            return json.loads(_DYNAMIC_CACHE_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {"tickers": [], "stale_counts": {}, "refreshed_at": None}


def _save_dynamic_supplement(data: dict) -> None:
    """Persist the dynamic supplement to disk."""
    from utils.atomic_io import atomic_write_json
    _DYNAMIC_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(_DYNAMIC_CACHE_PATH, data, indent=2)


def reconstitute_universe(
    extra_tickers: list[str] | None = None,
    prune_stale: bool = True,
) -> dict:
    """Validate and refresh the dynamic universe supplement.

    1. Validates all existing dynamic tickers still return data from yfinance.
    2. Adds new tickers from `extra_tickers` if they pass liquidity/mcap gates.
    3. Prunes tickers that have returned no data for 3+ consecutive refreshes.
    4. Persists the result to feature_cache/universe_dynamic.json.

    Returns summary dict with counts.
    """
    import yfinance as yf

    existing = load_dynamic_supplement()
    current_tickers = {
        resolve_yahoo_ticker(t.get("ticker", ""))
        for t in existing.get("tickers", [])
        if not is_excluded_ticker(
            t.get("ticker", ""),
            country=t.get("country"),
            region=t.get("country"),
        )
    }
    stale_counts: dict[str, int] = existing.get("stale_counts", {})
    static_tickers = {e.ticker.upper() for e in get_full_universe()}
    for ticker in list(stale_counts):
        if is_excluded_ticker(ticker):
            stale_counts.pop(ticker, None)

    # Candidates to validate: existing dynamic + new extras
    candidates = list(current_tickers)
    if extra_tickers:
        for t in extra_tickers:
            t_upper = resolve_yahoo_ticker(t)
            if not t_upper or is_excluded_ticker(t_upper):
                continue
            if t_upper not in static_tickers and t_upper not in current_tickers:
                candidates.append(t_upper)

    validated: list[dict] = []
    removed = 0

    for ticker in candidates:
        ticker = resolve_yahoo_ticker(ticker)
        if not ticker or is_excluded_ticker(ticker):
            stale_counts.pop(ticker, None)
            removed += 1
            continue
        try:
            info = yf.Ticker(ticker).info or {}
            mcap = info.get("marketCap", 0) or 0
            avg_vol = info.get("averageVolume", 0) or 0
            price = info.get("currentPrice") or info.get("regularMarketPrice") or 0
            country_iso = _iso_country(info.get("country", "")) or "US"
            if is_excluded_ticker(ticker, country=country_iso, region=info.get("country", "")):
                stale_counts.pop(ticker, None)
                removed += 1
                continue

            # Liquidity gate
            avg_dollar_vol = avg_vol * price if price else 0

            if mcap >= _MIN_MARKET_CAP and avg_dollar_vol >= _MIN_AVG_VOLUME:
                stale_counts.pop(ticker, None)
                validated.append({
                    "ticker": ticker,
                    "country": country_iso,
                    "exchange": info.get("exchange", ""),
                    "sector": info.get("sector", ""),
                    "market_cap": mcap,
                    "avg_dollar_volume": round(avg_dollar_vol),
                })
            else:
                # Below threshold — count as stale
                stale_counts[ticker] = stale_counts.get(ticker, 0) + 1
                if prune_stale and stale_counts[ticker] >= _MAX_STALE_RUNS:
                    removed += 1
                    stale_counts.pop(ticker, None)
                    _logger.info("Pruned stale ticker %s (no data for %d runs)", ticker, _MAX_STALE_RUNS)
                else:
                    # Keep it but mark stale
                    validated.append({
                        "ticker": ticker,
                        "country": "??",
                        "exchange": "",
                        "sector": "",
                        "market_cap": 0,
                        "avg_dollar_volume": 0,
                        "stale": True,
                    })
        except Exception:
            stale_counts[ticker] = stale_counts.get(ticker, 0) + 1
            if prune_stale and stale_counts.get(ticker, 0) >= _MAX_STALE_RUNS:
                removed += 1
                stale_counts.pop(ticker, None)

    result = {
        "tickers": validated,
        "stale_counts": stale_counts,
        "refreshed_at": datetime.now().isoformat(),
    }
    _save_dynamic_supplement(result)

    _logger.info(
        "Universe reconstitution: %d validated, %d pruned, %d total dynamic",
        len(validated), removed, len(validated),
    )
    return {
        "validated": len(validated),
        "pruned": removed,
        "total_dynamic": len(validated),
        "total_with_static": len(static_tickers) + len(validated),
    }


def get_dynamic_tickers() -> list[str]:
    """Return validated dynamic ticker symbols (non-stale only)."""
    data = load_dynamic_supplement()
    result: list[str] = []
    seen: set[str] = set()
    for t in data.get("tickers", []):
        ticker = resolve_yahoo_ticker(t.get("ticker", ""))
        if (
            t.get("stale")
            or not ticker
            or ticker in seen
            or is_excluded_ticker(ticker, country=t.get("country"), region=t.get("country"))
        ):
            continue
        result.append(ticker)
        seen.add(ticker)
    return result


def get_dynamic_entries() -> list[dict]:
    """Return full validated dynamic entries with metadata (non-stale only).

    Each entry carries ticker, country (ISO-2), exchange, sector, market_cap
    and avg_dollar_volume. Used by discovery so the dynamic supplement does
    not lose the metadata that reconstitute_universe() already fetched.
    """
    data = load_dynamic_supplement()
    result: list[dict] = []
    seen: set[str] = set()
    for t in data.get("tickers", []):
        ticker = resolve_yahoo_ticker(t.get("ticker", ""))
        if (
            t.get("stale")
            or not ticker
            or ticker in seen
            or is_excluded_ticker(ticker, country=t.get("country"), region=t.get("country"))
        ):
            continue
        entry = dict(t)
        entry["ticker"] = ticker
        result.append(entry)
        seen.add(ticker)
    return result


def get_dynamic_rejected() -> set[str]:
    """Tickers currently counted as stale (failed last reconstitution gate).

    These should be excluded from any ETF/fallback direct-inject path — the
    whole point of routing ETF holdings through reconstitute_universe() is
    to keep names that fail the liquidity/market-cap floors out of discovery.
    """
    data = load_dynamic_supplement()
    rejected: set[str] = set(data.get("stale_counts", {}).keys())
    # A currently-stored entry with stale=True is also effectively rejected.
    for t in data.get("tickers", []):
        if t.get("stale"):
            sym = t.get("ticker")
            if sym:
                rejected.add(sym)
                rejected.add(resolve_yahoo_ticker(sym))
    return rejected


# ---------------------------------------------------------------------------
# ETF Holdings Decomposition (Petajisto 2011)
# ---------------------------------------------------------------------------

_ETF_HOLDINGS_CACHE_PATH = Path("feature_cache/etf_holdings.json")

# Sector ETFs (SPDR Select Sector)
_UNIVERSE_ETFS_SECTOR = [
    "XLK", "XLF", "XLV", "XLE", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE",
]
# Country / Region ETFs (iShares MSCI)
_UNIVERSE_ETFS_COUNTRY = [
    "EWJ", "EWG", "EWU", "EWA", "EWC", "EWH", "EWS",
]
# Size / Style ETFs
_UNIVERSE_ETFS_SIZE = ["IWM", "MDY"]

_ALL_UNIVERSE_ETFS = _UNIVERSE_ETFS_SECTOR + _UNIVERSE_ETFS_COUNTRY + _UNIVERSE_ETFS_SIZE


def decompose_etf_holdings(
    exclude_tickers: set[str] | None = None,
    force_refresh: bool = False,
) -> list[str]:
    """Extract constituent tickers from key sector/country/size ETFs.

    Uses yfinance get_holdings() with a 7-day cache to avoid repeated API calls.
    Returns deduplicated ticker symbols not already in the static universe.

    Petajisto (2011): index reconstitution creates predictable demand patterns.
    Chen, Noronha & Singal (2004): index-add stocks earn abnormal returns.
    """
    import yfinance as yf

    cache_ttl = getattr(config, "ETF_HOLDINGS_CACHE_TTL", 604800)  # 7 days
    excluded: set[str] = set()
    for t in exclude_tickers or set():
        raw = str(t or "").upper().strip()
        if raw:
            excluded.add(raw)
            excluded.add(resolve_yahoo_ticker(raw))
    static_tickers = {e.ticker.upper() for e in get_full_universe()}

    # Check cache freshness
    if not force_refresh and _ETF_HOLDINGS_CACHE_PATH.exists():
        try:
            cached = json.loads(_ETF_HOLDINGS_CACHE_PATH.read_text())
            saved_at = cached.get("saved_at", "")
            if saved_at:
                from datetime import datetime
                saved_dt = datetime.fromisoformat(saved_at)
                age_secs = (datetime.now() - saved_dt).total_seconds()
                if age_secs < cache_ttl:
                    tickers = [
                        resolve_yahoo_ticker(t) for t in cached.get("tickers", [])
                        if resolve_yahoo_ticker(t).upper() not in excluded
                        and not is_excluded_ticker(resolve_yahoo_ticker(t))
                    ]
                    _logger.info("ETF decomposition: %d cached tickers (age %.1fh)",
                                 len(tickers), age_secs / 3600)
                    return tickers
        except (json.JSONDecodeError, OSError, ValueError):
            pass

    # Fetch holdings from each ETF
    all_tickers: set[str] = set()
    etf_source: dict[str, list[str]] = {}

    for etf_symbol in _ALL_UNIVERSE_ETFS:
        try:
            etf = yf.Ticker(etf_symbol)
            # get_holdings() returns a DataFrame with symbol column
            holdings_df = None
            if hasattr(etf, 'get_holdings'):
                holdings_df = etf.get_holdings()
            elif hasattr(etf, 'funds_data'):
                try:
                    holdings_df = etf.funds_data.top_holdings
                except Exception:
                    pass

            if holdings_df is not None and len(holdings_df) > 0:
                # Try different column names for the ticker symbol
                sym_col = None
                for candidate in ("Symbol", "symbol", "Ticker", "ticker"):
                    if candidate in holdings_df.columns:
                        sym_col = candidate
                        break
                if sym_col is None and holdings_df.index.name in ("Symbol", "symbol"):
                    symbols = [str(s) for s in holdings_df.index if isinstance(s, str) and len(s) <= 10]
                else:
                    symbols = (
                        [str(s) for s in holdings_df[sym_col] if isinstance(s, str) and len(s) <= 10]
                        if sym_col else []
                    )

                for sym in symbols:
                    s_upper = resolve_yahoo_ticker(sym)
                    if (
                        s_upper
                        and s_upper not in static_tickers
                        and not is_excluded_ticker(s_upper)
                    ):
                        all_tickers.add(s_upper)
                        etf_source.setdefault(s_upper, []).append(etf_symbol)

                _logger.debug("ETF %s: %d holdings extracted", etf_symbol, len(symbols))
        except Exception as e:
            _logger.debug("ETF %s: holdings fetch failed: %s", etf_symbol, e)
            continue

    # Filter out excluded tickers
    result = sorted(
        t for t in all_tickers
        if t not in excluded and not is_excluded_ticker(t)
    )

    # Cache result
    try:
        from utils.atomic_io import atomic_write_json
        _ETF_HOLDINGS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(_ETF_HOLDINGS_CACHE_PATH, {
            "tickers": result,
            "etf_source": etf_source,
            "n_etfs_queried": len(_ALL_UNIVERSE_ETFS),
            "saved_at": datetime.now().isoformat(),
        }, indent=2)
    except Exception as e:
        _logger.debug("ETF holdings cache write failed: %s", e)

    _logger.info("ETF decomposition: %d unique tickers from %d ETFs",
                 len(result), len(_ALL_UNIVERSE_ETFS))
    return result


# ---------------------------------------------------------------------------
# ADR Cross-Listing Mapping (saves FX round-trip costs)
# ---------------------------------------------------------------------------

ADR_MAPPING: dict[str, str] = {
    # UK → US ADR
    "SHEL.L": "SHEL", "AZN.L": "AZN", "BP.L": "BP", "GSK.L": "GSK",
    "RIO.L": "RIO", "HSBA.L": "HSBC", "ULVR.L": "UL", "DEO.L": "DEO",
    "NGG.L": "NGG", "VOD.L": "VOD", "BTI.L": "BTI", "LIN.L": "LIN",
    "RELI.L": "RELX", "LSEG.L": "LNSTY", "BHP.L": "BHP",
    # Germany → US ADR
    "SAP.DE": "SAP", "SIE.DE": "SIEGY", "DTE.DE": "DTEGY",
    "ALV.DE": "ALIZY", "BAS.DE": "BASFY", "MBG.DE": "MBGAF",
    "BMW.DE": "BMWYY", "MUV2.DE": "MURGY", "ADS.DE": "ADDYY",
    # France → US ADR
    "SAN.PA": "SNY", "OR.PA": "LRLCY", "AI.PA": "AIQUY",
    "MC.PA": "LVMUY", "SU.PA": "SCGLY", "BNP.PA": "BNPQY",
    "DG.PA": "VEOEY", "KER.PA": "PPRUY", "EL.PA": "ESLOY",
    # Spain → US ADR
    "SAN.MC": "SAN", "TEF.MC": "TEF", "BBVA.MC": "BBVA",
    "IBE.MC": "IBDRY",
    # Nordics → US ADR
    "NOVO-B.CO": "NVO", "AZN.ST": "AZN", "VOLV-B.ST": "VLVLY",
    "ERIC-B.ST": "ERIC", "NESTE.HE": "NTOIY",
    # Japan → US ADR
    "7203.T": "TM", "6758.T": "SONY", "9984.T": "SFTBY",
    "8306.T": "MUFG", "6861.T": "KYOCY", "7267.T": "HMC",
    # Australia → US ADR
    "BHP.AX": "BHP", "CSL.AX": "CSLLY", "WBC.AX": "WBCFY",
    # Canada → US ADR
    "RY.TO": "RY", "TD.TO": "TD", "ENB.TO": "ENB",
    "CNR.TO": "CNI", "BMO.TO": "BMO", "BNS.TO": "BNS",
    # Hong Kong → US ADR
    "0700.HK": "TCEHY", "9988.HK": "BABA", "3690.HK": "MPNGY",
}

# Reverse mapping: ADR → local ticker
ADR_REVERSE_MAPPING: dict[str, str] = {v: k for k, v in ADR_MAPPING.items()}
