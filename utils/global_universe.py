"""Structured global equity universe for the Discovery Engine.

Provides a tiered, metadata-rich universe across major global exchanges.
Each ticker has region, tier, index source, and validation metadata.

Tier 1: Core liquid leaders (index constituents, large caps) — screened daily
Tier 2: Mid-cap opportunity set — screened 2x/week (Mon, Thu)
"""

from __future__ import annotations

import datetime as _dt
from typing import NamedTuple


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
# South Korea — KOSPI top
# ---------------------------------------------------------------------------
_KOREA = [
    ("005930.KS", "KR", "KRX", 1, "KOSPI", "Technology"),
    ("000660.KS", "KR", "KRX", 1, "KOSPI", "Technology"),
    ("035420.KS", "KR", "KRX", 1, "KOSPI", "Technology"),
    ("051910.KS", "KR", "KRX", 1, "KOSPI", "Materials"),
    ("006400.KS", "KR", "KRX", 1, "KOSPI", "Consumer Discretionary"),
    ("035720.KS", "KR", "KRX", 1, "KOSPI", "Technology"),
    ("068270.KS", "KR", "KRX", 1, "KOSPI", "Consumer Discretionary"),
    ("028260.KS", "KR", "KRX", 1, "KOSPI", "Consumer Discretionary"),
    ("105560.KS", "KR", "KRX", 1, "KOSPI", "Consumer Staples"),
    ("055550.KS", "KR", "KRX", 1, "KOSPI", "Financials"),
    ("003670.KS", "KR", "KRX", 1, "KOSPI", "Materials"),
    ("034730.KS", "KR", "KRX", 1, "KOSPI", "Technology"),
    ("066570.KS", "KR", "KRX", 1, "KOSPI", "Industrials"),
    ("032830.KS", "KR", "KRX", 1, "KOSPI", "Consumer Discretionary"),
    ("096770.KS", "KR", "KRX", 1, "KOSPI", "Technology"),
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
    "South Korea": _KOREA,
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_full_universe() -> list[UniverseEntry]:
    """Return the complete structured universe with metadata."""
    return list(_UNIVERSE)


def get_global_universe(
    exclude_tickers: set[str] | None = None,
    *,
    max_tier: int = 2,
    countries: list[str] | None = None,
) -> list[str]:
    """Return ticker symbols from the universe, with optional filters.

    Args:
        exclude_tickers: Tickers to exclude (e.g. existing holdings).
        max_tier: Maximum tier to include (1 = core only, 2 = core + mid-cap).
        countries: ISO country codes to include (None = all).

    Returns:
        Deduplicated list of ticker strings.
    """
    exclude = {t.upper() for t in (exclude_tickers or set())}
    result = []
    for entry in _UNIVERSE:
        if entry.tier > max_tier:
            continue
        if countries and entry.country not in countries:
            continue
        if entry.ticker.upper() in exclude:
            continue
        result.append(entry.ticker)
    return result


def get_universe_for_rotation(
    day_of_week: int,
    exclude_tickers: set[str] | None = None,
) -> list[str]:
    """Return tickers for today's rotation schedule.

    Rotation logic:
        - Tier 1 (core): screened every day
        - Tier 2 (mid-cap): screened on Monday (0) and Thursday (3) only

    Args:
        day_of_week: 0=Mon, 1=Tue, ..., 6=Sun (from datetime.weekday())
        exclude_tickers: Tickers to skip.

    Returns:
        List of ticker symbols to screen today.
    """
    tier2_days = {0, 3}  # Monday, Thursday
    include_tier2 = day_of_week in tier2_days
    max_tier = 2 if include_tier2 else 1
    return get_global_universe(exclude_tickers=exclude_tickers, max_tier=max_tier)


def get_universe_by_region(
    max_tier: int = 2,
) -> dict[str, list[str]]:
    """Return the universe organised by region (for progress reporting).

    Args:
        max_tier: Maximum tier to include.
    """
    result: dict[str, list[str]] = {}
    for region_name, region_list in _ALL_REGIONS.items():
        tickers = [
            e[0] for e in region_list
            if e[3] <= max_tier  # e[3] is tier
        ]
        if tickers:
            result[region_name] = tickers
    return result


def get_universe_stats() -> dict:
    """Return summary statistics about the universe."""
    total = len(_UNIVERSE)
    tier1 = sum(1 for e in _UNIVERSE if e.tier == 1)
    tier2 = sum(1 for e in _UNIVERSE if e.tier == 2)
    by_country: dict[str, int] = {}
    for e in _UNIVERSE:
        by_country[e.country] = by_country.get(e.country, 0) + 1
    return {
        "total": total,
        "tier1": tier1,
        "tier2": tier2,
        "by_country": by_country,
        "last_refreshed": UNIVERSE_LAST_REFRESHED,
    }
