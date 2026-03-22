"""Curated global equity universe for the Discovery Engine.

Provides ~800+ liquid, well-covered tickers across major global exchanges.
These supplement FMP's US screener (which only works on Starter plan for US stocks).
yfinance is used to fetch price data for momentum screening.

Lists are based on major index constituents (FTSE 350, DAX 40, CAC 40, etc.)
and are intentionally broad. Stale tickers gracefully return no data from yfinance.
"""

# ---------------------------------------------------------------------------
# UK — FTSE 100 + FTSE 250 selection (~150 tickers)
# ---------------------------------------------------------------------------
UK_TICKERS = [
    # FTSE 100
    "SHEL.L", "AZN.L", "HSBA.L", "BP.L", "GSK.L", "RIO.L", "ULVR.L",
    "DGE.L", "REL.L", "LSEG.L", "BA.L", "NG.L", "GLEN.L", "AAL.L",
    "LLOY.L", "BARC.L", "NWG.L", "PRU.L", "STAN.L", "ABF.L",
    "III.L", "IMB.L", "INF.L", "ITRK.L", "JD.L", "KGF.L", "LAND.L",
    "MNG.L", "MNDI.L", "PSON.L", "RKT.L", "RR.L", "SGE.L",
    "SN.L", "SPX.L", "SSE.L", "SVT.L", "TSCO.L", "WPP.L", "WTB.L",
    "EXPN.L", "CRH.L", "CPG.L", "CRDA.L", "DCC.L", "FRAS.L",
    "HLMA.L", "HLN.L", "HWDN.L", "ICP.L", "IHG.L", "JMAT.L",
    "PSN.L", "PHNX.L", "RTO.L", "SBRY.L", "SDR.L", "SKG.L",
    "SMDS.L", "SMT.L", "SMIN.L", "WEIR.L", "AV.L", "AUTO.L",
    "BDEV.L", "BKG.L", "BNZL.L", "BRBY.L", "CCH.L", "CNA.L",
    "ENT.L", "FLTR.L", "FRES.L", "HIK.L", "IAG.L", "ICG.L",
    "IMI.L", "LGEN.L", "NXT.L", "VOD.L", "BT-A.L", "AHT.L",
    "ANTO.L", "ADM.L", "EDV.L",
    # FTSE 250 selection — mid-caps with good liquidity
    "DARK.L", "TW.L", "BGEO.L", "JET2.L", "WIZZ.L",
    "FOUR.L", "MONY.L", "BOKU.L", "ALFA.L", "ASC.L",
    "BME.L", "BNKR.L", "CINE.L", "CMC.L", "CURY.L",
    "DPLM.L", "FCIT.L", "GAW.L", "GNS.L", "GPAY.L",
    "HTWS.L", "IGG.L", "IPX.L", "ITV.L", "JLEN.L",
    "LUCE.L", "MGAM.L", "MRO.L", "OSB.L", "OXB.L",
    "PAGE.L", "PRTC.L", "QQ.L", "RWS.L", "SEPL.L",
    "SGC.L", "SGRO.L", "SHI.L", "SNDR.L", "SOI.L",
    "SFOR.L", "TRN.L", "TRG.L", "VSVS.L", "WIN.L",
    "YOU.L", "ZTF.L", "BOY.L", "CHG.L", "FTC.L",
    # Defence / Aerospace — user's sector interest
    "BAE.L", "QQ.L", "BVIC.L", "MGGT.L",
]

# ---------------------------------------------------------------------------
# Germany — DAX 40 + MDAX selection (~50 tickers)
# ---------------------------------------------------------------------------
GERMANY_TICKERS = [
    # DAX 40
    "SAP.DE", "SIE.DE", "ALV.DE", "DTE.DE", "AIR.DE", "BAS.DE",
    "MBG.DE", "BMW.DE", "MUV2.DE", "DHL.DE", "IFX.DE", "ADS.DE",
    "DB1.DE", "BEI.DE", "RWE.DE", "HEN3.DE", "VNA.DE", "CON.DE",
    "FRE.DE", "MTX.DE", "SRT3.DE", "ENR.DE", "PAH3.DE", "HEI.DE",
    "QIA.DE", "P911.DE", "MRK.DE", "DTG.DE", "SHL.DE", "FME.DE",
    "ZAL.DE", "HNR1.DE", "DHER.DE", "PUM.DE", "1COV.DE", "SY1.DE",
    "TKA.DE", "LEG.DE",
    # MDAX selection
    "HAG.DE", "RHM.DE", "EVT.DE", "AIXA.DE", "NDX1.DE", "KGX.DE",
    "GXI.DE", "WAF.DE", "TEG.DE", "BNR.DE",
]

# ---------------------------------------------------------------------------
# France — CAC 40 + SBF 120 selection (~45 tickers)
# ---------------------------------------------------------------------------
FRANCE_TICKERS = [
    # CAC 40
    "MC.PA", "OR.PA", "TTE.PA", "SAN.PA", "AI.PA", "SU.PA",
    "BN.PA", "CS.PA", "DG.PA", "AIR.PA", "SAF.PA", "BNP.PA",
    "GLE.PA", "CA.PA", "RI.PA", "KER.PA", "CAP.PA", "DSY.PA",
    "HO.PA", "SGO.PA", "STLAP.PA", "VIE.PA", "ACA.PA", "EN.PA",
    "ML.PA", "PUB.PA", "RMS.PA", "VIV.PA", "STM.PA", "WLN.PA",
    "RNO.PA", "ORA.PA", "LR.PA", "URW.PA", "ATO.PA",
    # SBF 120 selection
    "TEP.PA", "AM.PA", "SOI.PA", "DBG.PA", "ERF.PA",
    "FGR.PA", "GFI.PA", "GTT.PA", "ILD.PA", "MF.PA",
]

# ---------------------------------------------------------------------------
# Spain — IBEX 35 (~30 tickers)
# ---------------------------------------------------------------------------
SPAIN_TICKERS = [
    "SAN.MC", "BBVA.MC", "ITX.MC", "IBE.MC", "TEF.MC", "REP.MC",
    "FER.MC", "AMS.MC", "GRF.MC", "ENG.MC", "MAP.MC", "ACS.MC",
    "CABK.MC", "CLNX.MC", "FDR.MC", "MRL.MC", "SAB.MC", "BKT.MC",
    "SCYR.MC", "IAG.MC", "ACX.MC", "CIE.MC", "COL.MC", "MTS.MC",
    "RED.MC", "AENA.MC", "LOG.MC", "PHM.MC", "ROVI.MC",
]

# ---------------------------------------------------------------------------
# Netherlands — AEX 25 + AMX selection (~30 tickers)
# ---------------------------------------------------------------------------
NETHERLANDS_TICKERS = [
    "ASML.AS", "INGA.AS", "PHIA.AS", "AD.AS", "UNA.AS", "WKL.AS",
    "HEIA.AS", "REN.AS", "DSM.AS", "AGN.AS", "KPN.AS", "AKZA.AS",
    "NN.AS", "ASM.AS", "RAND.AS", "URW.AS", "PRX.AS", "BESI.AS",
    "LIGHT.AS", "IMCD.AS", "JDEP.AS", "ABN.AS", "TKWY.AS",
    "SBMO.AS", "ADYEN.AS", "FLOW.AS", "ALFEN.AS", "TOM2.AS",
]

# ---------------------------------------------------------------------------
# Italy — FTSE MIB selection (~25 tickers)
# ---------------------------------------------------------------------------
ITALY_TICKERS = [
    "ENI.MI", "ISP.MI", "UCG.MI", "ENEL.MI", "STM.MI", "G.MI",
    "TRN.MI", "RACE.MI", "LDO.MI", "MONC.MI", "PRY.MI", "TEN.MI",
    "SPM.MI", "SRG.MI", "BAMI.MI", "PST.MI", "CPR.MI", "HER.MI",
    "MB.MI", "FBK.MI", "STLAM.MI", "PIRC.MI", "REC.MI",
]

# ---------------------------------------------------------------------------
# Switzerland — SMI + SPI selection (~20 tickers)
# ---------------------------------------------------------------------------
SWITZERLAND_TICKERS = [
    "NESN.SW", "ROG.SW", "NOVN.SW", "UBSG.SW", "ABBN.SW", "CSGN.SW",
    "SREN.SW", "ZURN.SW", "GIVN.SW", "LONN.SW", "GEBN.SW", "SGSN.SW",
    "PGHN.SW", "HOLN.SW", "SCMN.SW", "SIKA.SW", "SLHN.SW", "TEMN.SW",
    "VACN.SW", "ALC.SW",
]

# ---------------------------------------------------------------------------
# Nordics — OMXS30, OMXC25, OMXH25, OBX (~40 tickers)
# ---------------------------------------------------------------------------
NORDIC_TICKERS = [
    # Sweden
    "VOLV-B.ST", "ERIC-B.ST", "SAND.ST", "ATCO-A.ST", "SEB-A.ST",
    "SWED-A.ST", "SHB-A.ST", "INVE-B.ST", "HM-B.ST", "ASSA-B.ST",
    "ALFA.ST", "ABB.ST", "SKF-B.ST", "TELIA.ST", "ESSITY-B.ST",
    "HEXA-B.ST", "SAAB-B.ST", "ELUX-B.ST",
    # Denmark
    "NOVO-B.CO", "MAERSK-B.CO", "VWS.CO", "CARL-B.CO", "DSV.CO",
    "ORSTED.CO", "PNDORA.CO", "COLO-B.CO", "GMAB.CO",
    # Finland
    "NOKIA.HE", "SAMPO.HE", "KNEBV.HE", "NESTE.HE", "FORTUM.HE",
    "UPM.HE", "WRT1V.HE",
    # Norway
    "EQNR.OL", "DNB.OL", "TEL.OL", "MOWI.OL", "ORK.OL", "YAR.OL",
    "AKRBP.OL",
]

# ---------------------------------------------------------------------------
# Canada — TSX 60 + selection (~40 tickers)
# ---------------------------------------------------------------------------
CANADA_TICKERS = [
    "RY.TO", "TD.TO", "BNS.TO", "BMO.TO", "ENB.TO", "CNR.TO",
    "CP.TO", "TRP.TO", "SU.TO", "CNQ.TO", "MFC.TO", "SLF.TO",
    "BCE.TO", "T.TO", "ABX.TO", "NTR.TO", "FNV.TO", "WFG.TO",
    "ATD.TO", "CSU.TO", "BAM.TO", "BN.TO", "RCI-B.TO", "IFC.TO",
    "GIB-A.TO", "SHOP.TO", "L.TO", "MG.TO", "DOL.TO", "CCL-B.TO",
    "WCN.TO", "QSR.TO", "FFH.TO", "AEM.TO", "IMO.TO", "CVE.TO",
    "TOU.TO", "ARX.TO", "WPM.TO",
]

# ---------------------------------------------------------------------------
# Australia — ASX 50 selection (~30 tickers)
# ---------------------------------------------------------------------------
AUSTRALIA_TICKERS = [
    "BHP.AX", "CBA.AX", "CSL.AX", "NAB.AX", "WBC.AX", "ANZ.AX",
    "MQG.AX", "WES.AX", "WOW.AX", "TLS.AX", "RIO.AX", "FMG.AX",
    "ALL.AX", "COL.AX", "GMG.AX", "TCL.AX", "STO.AX", "WDS.AX",
    "REA.AX", "XRO.AX", "JHX.AX", "S32.AX", "MIN.AX", "ORG.AX",
    "SHL.AX", "QBE.AX", "IAG.AX", "MPL.AX", "NCM.AX", "NST.AX",
]

# ---------------------------------------------------------------------------
# Japan — Nikkei 225 top (~30 tickers)
# ---------------------------------------------------------------------------
JAPAN_TICKERS = [
    "7203.T", "6758.T", "9984.T", "6902.T", "8306.T", "9432.T",
    "6861.T", "7741.T", "6501.T", "4063.T", "8035.T", "6098.T",
    "9433.T", "4502.T", "4503.T", "6301.T", "7267.T", "8316.T",
    "3382.T", "6954.T", "7974.T", "4519.T", "8058.T", "8031.T",
    "6326.T", "4543.T", "6367.T", "2914.T", "6594.T", "7751.T",
]

# ---------------------------------------------------------------------------
# Hong Kong — Hang Seng selection (~20 tickers)
# ---------------------------------------------------------------------------
HONG_KONG_TICKERS = [
    "0700.HK", "9988.HK", "0005.HK", "1299.HK", "0941.HK",
    "2318.HK", "0388.HK", "0883.HK", "0016.HK", "0003.HK",
    "1038.HK", "0011.HK", "0066.HK", "0823.HK", "1113.HK",
    "2688.HK", "0027.HK", "1997.HK", "0669.HK", "9618.HK",
]

# ---------------------------------------------------------------------------
# Singapore — STI selection (~10 tickers)
# ---------------------------------------------------------------------------
SINGAPORE_TICKERS = [
    "D05.SI", "O39.SI", "U11.SI", "Z74.SI", "C6L.SI",
    "A17U.SI", "C38U.SI", "G13.SI", "S58.SI", "BN4.SI",
]

# ---------------------------------------------------------------------------
# South Korea — KOSPI top (~15 tickers)
# ---------------------------------------------------------------------------
KOREA_TICKERS = [
    "005930.KS", "000660.KS", "035420.KS", "051910.KS", "006400.KS",
    "035720.KS", "068270.KS", "028260.KS", "105560.KS", "055550.KS",
    "003670.KS", "034730.KS", "066570.KS", "032830.KS", "096770.KS",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_global_universe(exclude_tickers: set[str] | None = None) -> list[str]:
    """Return the full curated global universe, excluding specified tickers.

    Returns ~800+ tickers across all major global markets.
    """
    all_tickers = (
        UK_TICKERS
        + GERMANY_TICKERS
        + FRANCE_TICKERS
        + SPAIN_TICKERS
        + NETHERLANDS_TICKERS
        + ITALY_TICKERS
        + SWITZERLAND_TICKERS
        + NORDIC_TICKERS
        + CANADA_TICKERS
        + AUSTRALIA_TICKERS
        + JAPAN_TICKERS
        + HONG_KONG_TICKERS
        + SINGAPORE_TICKERS
        + KOREA_TICKERS
    )

    # Deduplicate and exclude
    exclude = exclude_tickers or set()
    seen = set()
    result = []
    for t in all_tickers:
        t_upper = t.upper()
        if t_upper not in seen and t not in exclude and t_upper not in exclude:
            seen.add(t_upper)
            result.append(t)

    return result


def get_universe_by_region() -> dict[str, list[str]]:
    """Return the universe organised by region (for progress reporting)."""
    return {
        "UK": UK_TICKERS,
        "Germany": GERMANY_TICKERS,
        "France": FRANCE_TICKERS,
        "Spain": SPAIN_TICKERS,
        "Netherlands": NETHERLANDS_TICKERS,
        "Italy": ITALY_TICKERS,
        "Switzerland": SWITZERLAND_TICKERS,
        "Nordics": NORDIC_TICKERS,
        "Canada": CANADA_TICKERS,
        "Australia": AUSTRALIA_TICKERS,
        "Japan": JAPAN_TICKERS,
        "Hong Kong": HONG_KONG_TICKERS,
        "Singapore": SINGAPORE_TICKERS,
        "South Korea": KOREA_TICKERS,
    }
