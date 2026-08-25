"""Millennium-specific domain model.

A generic resume parser produces generic labels. The BD team does not search for
"finance experience" -- it searches for "healthcare-focused fundamental L/S with a
sell-side feeder path in APAC". Encoding that vocabulary is cheap and is the single
clearest signal that the tool was built for this business.

Every taxonomy is (a) versioned, (b) alias-mapped so surface variation collapses, and
(c) deliberately editable by a recruiter -- these are heuristics, not ground truth,
and the UI says so.

Firm names below are drawn from the supplied corpus plus the standard hedge-fund
feeder universe; the tier table is the piece a recruiter would most want to edit.
"""
from __future__ import annotations

import re
import unicodedata

TAXONOMY_VERSION = "1.2.0"

# ---------------------------------------------------------------- strategies
# label -> (display, lexical triggers, semantic exemplars used for embedding match)
STRATEGIES: dict[str, dict] = {
    "equity_long_short": {
        "display": "Equity Long/Short",
        "triggers": ["long/short", "long short", "l/s equity", "equity long-short",
                     "fundamental long/short", "long short fundamental"],
        "exemplars": ["fundamental long short equity portfolio management with single name alpha and sector hedges"],
    },
    "market_neutral": {
        "display": "Market Neutral",
        "triggers": ["market neutral", "beta neutral", "dollar neutral", "factor neutral"],
        "exemplars": ["beta and factor neutral equity book with tight net exposure limits"],
    },
    "statistical_arbitrage": {
        "display": "Statistical Arbitrage",
        "triggers": ["statistical arbitrage", "stat arb", "mean reversion", "pairs trading"],
        "exemplars": ["high turnover statistical arbitrage signals and mean reversion research"],
    },
    "quantitative_research": {
        "display": "Quantitative Research",
        "triggers": ["quantitative research", "quant research", "signal research", "alpha research",
                     "multi-factor", "multi factor model", "factor model", "backtest", "backtesting"],
        "exemplars": ["systematic alpha signal research, factor construction and backtesting"],
    },
    "systematic_macro": {
        "display": "Systematic Macro / CTA",
        "triggers": ["systematic macro", "managed futures", "cta", "trend following"],
        "exemplars": ["systematic macro and trend following across futures markets"],
    },
    "global_macro": {
        "display": "Global Macro",
        "triggers": ["global macro", "macro strategy", "rates and fx", "discretionary macro"],
        "exemplars": ["discretionary global macro across rates, fx and sovereign risk"],
    },
    "fixed_income_rv": {
        "display": "Fixed Income Relative Value",
        "triggers": ["fixed income relative value", "fi rv", "relative value", "yield curve",
                     "basis trading", "swap spread"],
        "exemplars": ["fixed income relative value on the yield curve, swap spreads and basis"],
    },
    "credit_long_short": {
        "display": "Credit Long/Short",
        "triggers": ["credit long/short", "high yield", "investment grade", "hy", "ig credit",
                     "corporate bonds", "credit research", "structured credit"],
        "exemplars": ["corporate credit research across high yield and investment grade issuers"],
    },
    "distressed": {
        "display": "Distressed / Special Situations",
        "triggers": ["distressed", "special situations", "restructuring", "bankruptcy", "workout"],
        "exemplars": ["distressed debt and restructuring special situations analysis"],
    },
    "event_driven": {
        "display": "Event Driven",
        "triggers": ["event driven", "event-driven", "catalyst", "soft catalyst", "hard catalyst"],
        "exemplars": ["catalyst driven equity investing around corporate events"],
    },
    "merger_arbitrage": {
        "display": "Merger Arbitrage",
        "triggers": ["merger arbitrage", "merger arb", "risk arbitrage", "deal spread"],
        "exemplars": ["announced deal merger arbitrage and spread risk assessment"],
    },
    "derivatives_pricing": {
        "display": "Derivatives / Pricing Quant",
        "triggers": ["pricing library", "derivatives pricing", "quantitative developer",
                     "quantitative development", "exotic", "payoff", "greeks", "stochastic calculus",
                     "monte carlo", "option pricing", "copula", "volatility surface"],
        "exemplars": ["derivatives pricing library development, exotic payoffs and greeks computation"],
    },
    "private_markets": {
        "display": "Private Markets / Growth",
        "triggers": ["private equity", "growth equity", "venture", "series b", "buyout", "lbo"],
        "exemplars": ["private equity and growth investing with diligence and LBO modelling"],
    },
    "multi_strategy": {
        "display": "Multi-Strategy",
        "triggers": ["multi-strategy", "multi strategy", "pod shop", "platform fund"],
        "exemplars": ["multi strategy platform allocating risk across independent pods"],
    },
}

# ---------------------------------------------------------------- sectors (GICS-lite)
SECTORS: dict[str, dict] = {
    "technology": {"display": "Technology",
                   "triggers": ["technology", "tmt", "software", "internet", "semiconductor", "cloud",
                                "saas", "enterprise software", "digital advertising", "e-commerce"]},
    "healthcare": {"display": "Healthcare",
                   "triggers": ["healthcare", "health care", "pharma", "pharmaceutical", "biotech",
                                "life sciences", "medtech", "medical device", "diagnostics",
                                "therapeutics", "hospital", "generics", "usfda", "oncology"]},
    "financials": {"display": "Financial Services",
                   # NOTE: "banking" is deliberately NOT a sector trigger. In this domain
                   # "investment banking" is a FEEDER PATH, not sector coverage, and
                   # treating it as a sector made "healthcare investment banking"
                   # demand financials coverage as a hard requirement.
                   "triggers": ["financials", "banks", "insurance", "asset management", "fintech",
                                "brokerage", "specialty finance"]},
    "energy": {"display": "Energy",
               "triggers": ["energy", "oil", "gas", "oil&gas", "oil & gas", "renewable", "utilities power",
                            "upstream", "refining", "solar"]},
    "industrials": {"display": "Industrials",
                    "triggers": ["industrials", "aerospace", "manufacturing", "logistics", "transport",
                                 "infrastructure", "machinery", "3d printing"]},
    "consumer": {"display": "Consumer",
                 "triggers": ["consumer", "retail", "consumer discretionary", "consumer staples",
                              "grocery", "restaurant", "alcobev", "apparel"]},
    "materials": {"display": "Materials",
                  "triggers": ["materials", "chemicals", "specialty chemical", "mining", "metals", "steel"]},
    "utilities": {"display": "Utilities", "triggers": ["utilities", "power generation", "regulated utility"]},
    "real_estate": {"display": "Real Estate", "triggers": ["real estate", "reit", "property", "mortgage", "cmbs", "rmbs"]},
    "communications": {"display": "Communications",
                       "triggers": ["telecom", "media", "communications", "streaming", "entertainment",
                                    "social media", "interactive entertainment"]},
    "macro_rates": {"display": "Macro / Rates",
                    "triggers": ["macro", "rates", "sovereign", "fx", "central bank", "inflation"]},
    "credit": {"display": "Credit",
               "triggers": ["credit", "high yield", "investment grade", "leveraged loans", "clo", "securitization"]},
}

# ---------------------------------------------------------------- skills
# canonical -> (aliases, category). Aliases are matched case-insensitively on word
# boundaries; short/ambiguous ones like "r" and "q" get special handling in match().
SKILLS: dict[str, dict] = {
    "python":            {"aliases": ["python", "python3", "py", "pandas", "numpy", "scipy"], "category": "programming"},
    "r_lang":            {"aliases": ["r"], "category": "programming", "strict": True},
    "cpp":               {"aliases": ["c++", "cpp"], "category": "programming"},
    "csharp":            {"aliases": ["c#", ".net", "c# / .net"], "category": "programming"},
    "java":              {"aliases": ["java"], "category": "programming"},
    "sql":               {"aliases": ["sql", "t-sql", "postgres", "mysql"], "category": "programming"},
    "kdb":               {"aliases": ["kdb", "kdb+", "kx", "q language"], "category": "programming"},
    "matlab":            {"aliases": ["matlab"], "category": "programming"},
    "vba":               {"aliases": ["vba", "excel with vba", "advanced excel/vba"], "category": "programming"},
    "mongodb":           {"aliases": ["mongodb", "mongo", "nosql"], "category": "tools"},
    "machine_learning":  {"aliases": ["machine learning", "ml", "neural network", "deep learning",
                                      "gradient descent", "random forest", "xgboost"], "category": "analytics"},
    "time_series":       {"aliases": ["time series", "arima", "garch", "signal research", "stochastic calculus",
                                      "monte carlo", "probability theory"], "category": "analytics"},
    "statistics":        {"aliases": ["statistics", "statistical analysis", "hypothesis testing",
                                      "econometrics", "regression"], "category": "analytics"},
    "backtesting":       {"aliases": ["backtesting", "backtest", "backtester", "performance attribution"], "category": "analytics"},
    "financial_modelling": {"aliases": ["financial modeling", "financial modelling", "three-statement",
                                        "three statement", "3-statement", "dcf", "lbo", "comparable company",
                                        "sum-of-the-parts", "valuation model", "operating model"], "category": "finance"},
    "equity_research":   {"aliases": ["equity research", "initiation report", "coverage initiation",
                                      "earnings estimates", "sector thematic"], "category": "finance"},
    "portfolio_construction": {"aliases": ["position sizing", "portfolio construction", "risk exposures",
                                           "hedging", "factor attribution", "risk-adjusted"], "category": "finance"},
    "due_diligence":     {"aliases": ["due diligence", "diligence", "expert network", "expert calls",
                                      "channel checks", "primary research"], "category": "finance"},
    "alternative_data":  {"aliases": ["alternative data", "alt data", "alternative datasets", "web scraping"], "category": "analytics"},
    "bloomberg":         {"aliases": ["bloomberg", "bloomberg terminal", "allq", "runz"], "category": "tools"},
    "factset":           {"aliases": ["factset"], "category": "tools"},
    "capital_iq":        {"aliases": ["capital iq", "s&p capital iq", "capiq"], "category": "tools"},
    "refinitiv":         {"aliases": ["reuters", "refinitiv", "eikon"], "category": "tools"},
    "wind_db":           {"aliases": ["wind database", "wind"], "category": "tools", "strict": True},
    "excel":             {"aliases": ["excel", "microsoft office", "ms office"], "category": "tools"},
    "gis":               {"aliases": ["gis", "satellite imagery", "geospatial"], "category": "analytics"},
}

# ---------------------------------------------------------------- firm tiers
# Tier materially changes what a title means: "Analyst" at Goldman and "Analyst" at a
# 5-person shop are different jobs. Tier feeds seniority normalisation below.
FIRM_TIERS: dict[str, list[str]] = {
    "bulge_bracket": ["goldman sachs", "morgan stanley", "j.p. morgan", "jp morgan", "jpmorgan",
                      "j.p.mogan", "jpmorgan chase", "bank of america", "merrill lynch", "citi",
                      "citigroup", "credit suisse", "ubs", "barclays", "deutsche bank",
                      "bnp paribas", "societe generale", "société générale", "nomura", "hsbc",
                      "wells fargo", "rbc"],
    "elite_boutique": ["evercore", "lazard", "centerview", "moelis", "perella weinberg", "pjt",
                       "guggenheim", "houlihan lokey", "jefferies", "william blair", "leerink",
                       "piper sandler", "raymond james", "baird"],
    "mbb": ["mckinsey", "bain & company", "bain and company", "boston consulting group", "bcg"],
    "big_four": ["pwc", "pricewaterhousecoopers", "deloitte", "ernst & young", "ey", "kpmg"],
    "pod_shop": ["millennium", "citadel", "point72", "p72", "balyasny", "exoduspoint", "schonfeld",
                 "verition", "walleye", "cinctive", "eisler", "brevan howard", "squarepoint",
                 "north53 capital", "meridian capital partners", "meridian capital"],
    "quant_fund": ["two sigma", "de shaw", "d. e. shaw", "renaissance technologies", "jane street",
                   "hudson river trading", "jump trading", "optiver", "imc", "drw", "aqr", "man group"],
    "long_only": ["fidelity", "vanguard", "blackrock", "t. rowe price", "capital group", "wellington",
                  "pimco", "invesco", "franklin templeton", "j.p. morgan asset management",
                  "axis mutual fund", "sbi mutual fund", "icici prudential"],
    "hedge_fund_other": ["coatue", "tiger global", "viking global", "lone pine", "third point",
                         "elliott", "baupost", "magnetar", "apollo global management", "blackstone",
                         "kkr", "carlyle", "prism asset management"],
    "regional_broker": ["icici securities", "kotak securities", "centrum broking", "anand rathi",
                        "motilal oswal", "edelweiss", "iifl", "jardine lloyd thompson",
                        "bank of china", "transparent value", "dataflow research"],
}


# Alias -> single display name, so 'J.P.Mogan', 'jpmorgan chase' and 'JP Morgan' all
# collapse to one employer for faceting and de-duplication.
EMPLOYER_DISPLAY: dict[str, str] = {
    "goldman sachs": "Goldman Sachs", "morgan stanley": "Morgan Stanley",
    "j.p. morgan": "J.P. Morgan", "jp morgan": "J.P. Morgan", "jpmorgan": "J.P. Morgan",
    "j.p.mogan": "J.P. Morgan", "jpmorgan chase": "J.P. Morgan",
    "j.p. morgan asset management": "J.P. Morgan Asset Management",
    "bank of america": "Bank of America", "merrill lynch": "Bank of America",
    "citi": "Citi", "citigroup": "Citi", "credit suisse": "Credit Suisse", "ubs": "UBS",
    "barclays": "Barclays", "deutsche bank": "Deutsche Bank", "bnp paribas": "BNP Paribas",
    "societe generale": "Societe Generale", "société générale": "Societe Generale",
    "nomura": "Nomura", "hsbc": "HSBC", "wells fargo": "Wells Fargo", "rbc": "RBC",
    "evercore": "Evercore", "lazard": "Lazard", "centerview": "Centerview",
    "moelis": "Moelis", "perella weinberg": "Perella Weinberg", "pjt": "PJT Partners",
    "guggenheim": "Guggenheim", "houlihan lokey": "Houlihan Lokey", "jefferies": "Jefferies",
    "william blair": "William Blair", "leerink": "Leerink Partners",
    "piper sandler": "Piper Sandler", "raymond james": "Raymond James", "baird": "Baird",
    "mckinsey": "McKinsey & Company", "bain & company": "Bain & Company",
    "bain and company": "Bain & Company", "boston consulting group": "BCG", "bcg": "BCG",
    "pwc": "PwC", "pricewaterhousecoopers": "PwC", "deloitte": "Deloitte",
    "ernst & young": "EY", "ey": "EY", "kpmg": "KPMG",
    "millennium": "Millennium Management", "citadel": "Citadel", "point72": "Point72",
    "p72": "Point72", "balyasny": "Balyasny", "exoduspoint": "ExodusPoint",
    "schonfeld": "Schonfeld", "verition": "Verition", "walleye": "Walleye",
    "cinctive": "Cinctive Capital", "eisler": "Eisler Capital",
    "brevan howard": "Brevan Howard", "squarepoint": "Squarepoint",
    "north53 capital": "North53 Capital", "meridian capital partners": "Meridian Capital Partners",
    "meridian capital": "Meridian Capital",
    "two sigma": "Two Sigma", "de shaw": "D. E. Shaw", "d. e. shaw": "D. E. Shaw",
    "renaissance technologies": "Renaissance Technologies", "jane street": "Jane Street",
    "hudson river trading": "Hudson River Trading", "jump trading": "Jump Trading",
    "optiver": "Optiver", "imc": "IMC", "drw": "DRW", "aqr": "AQR", "man group": "Man Group",
    "fidelity": "Fidelity", "vanguard": "Vanguard", "blackrock": "BlackRock",
    "t. rowe price": "T. Rowe Price", "capital group": "Capital Group",
    "wellington": "Wellington", "pimco": "PIMCO", "invesco": "Invesco",
    "franklin templeton": "Franklin Templeton", "axis mutual fund": "Axis Mutual Fund",
    "sbi mutual fund": "SBI Mutual Fund", "icici prudential": "ICICI Prudential",
    "coatue": "Coatue Management", "tiger global": "Tiger Global",
    "viking global": "Viking Global", "lone pine": "Lone Pine", "third point": "Third Point",
    "elliott": "Elliott Management", "baupost": "Baupost", "magnetar": "Magnetar Capital",
    "apollo global management": "Apollo Global Management", "blackstone": "Blackstone",
    "kkr": "KKR", "carlyle": "Carlyle", "prism asset management": "Prism Asset Management",
    "icici securities": "ICICI Securities", "kotak securities": "Kotak Securities",
    "centrum broking": "Centrum Broking", "anand rathi": "Anand Rathi",
    "motilal oswal": "Motilal Oswal", "edelweiss": "Edelweiss", "iifl": "IIFL",
    "jardine lloyd thompson": "Jardine Lloyd Thompson", "bank of china": "Bank of China",
    "transparent value": "Transparent Value", "dataflow research": "DataFlow Research",
}

TIER_DISPLAY = {
    "bulge_bracket": "Bulge Bracket", "elite_boutique": "Elite Boutique", "mbb": "MBB Consulting",
    "big_four": "Big 4", "pod_shop": "Multi-Manager Pod Shop", "quant_fund": "Quant Fund",
    "long_only": "Long-Only / Asset Mgmt", "hedge_fund_other": "Hedge Fund / PE",
    "regional_broker": "Regional Broker / Other", "unknown": "Unknown",
}

# Tier weight used when normalising seniority (higher = title inflation less likely).
TIER_RIGOR = {"bulge_bracket": 1.0, "elite_boutique": 0.95, "mbb": 0.95, "quant_fund": 1.0,
              "pod_shop": 1.0, "big_four": 0.8, "long_only": 0.9, "hedge_fund_other": 0.9,
              "regional_broker": 0.7, "unknown": 0.75}

# ---------------------------------------------------------------- seniority
SENIORITY_LEVELS = {
    1: "Intern / Trainee", 2: "Junior Analyst", 3: "Analyst", 4: "Senior Analyst / Associate",
    5: "Lead Analyst / VP", 6: "Portfolio Manager / Director", 7: "Head / CIO",
}
TITLE_LEVEL_RULES: list[tuple[str, int]] = [
    (r"\b(intern|trainee|summer analyst|apprentice)\b", 1),
    (r"\b(junior|jr\.?)\b", 2),
    (r"\b(research assistant|desk analyst|business analyst|capital markets analyst)\b", 3),
    (r"\b(analyst)\b", 3),
    (r"\b(associate|senior associate)\b", 4),
    (r"\b(senior analyst|investment analyst|equity research analyst|research analyst)\b", 4),
    (r"\b(lead analyst|vice president|vp|principal|senior investment)\b", 5),
    (r"\b(portfolio manager|pm|director|managing director|md|investment professional)\b", 6),
    (r"\b(head of|chief|cio|partner|founder|co-founder)\b", 7),
]

# ---------------------------------------------------------------- feeder paths
# How junior hedge-fund talent actually arrives. Recruiters think in these terms.
FEEDER_PATHS: dict[str, dict] = {
    "ibd_analyst_program": {
        "display": "IBD Analyst Program",
        "signals": ["investment banking", "ibd", "m&a", "leveraged finance", "coverage group",
                    "summer analyst", "analyst class"],
        "tiers": ["bulge_bracket", "elite_boutique"]},
    "sellside_research": {
        "display": "Sell-Side Equity Research",
        "signals": ["equity research", "sell-side", "sell side", "initiation", "institutional investor",
                    "coverage of", "under coverage", "brokerage"],
        "tiers": ["bulge_bracket", "elite_boutique", "regional_broker"]},
    "buyside_lateral": {
        "display": "Buy-Side Lateral / Pod Move",
        "signals": ["portfolio manager", "long/short", "pod", "book", "pnl", "p&l", "sizing"],
        "tiers": ["pod_shop", "hedge_fund_other", "long_only"]},
    "quant_technical": {
        "display": "Quant / Technical Pipeline",
        "signals": ["quantitative developer", "quantitative strategist", "pricing library", "c++",
                    "phd", "stochastic", "engineering", "msc", "financial engineering"],
        "tiers": ["bulge_bracket", "quant_fund"]},
    "consulting": {
        "display": "Consulting (MBB / Strategy)",
        "signals": ["consultant", "business analyst", "engagement", "client team", "case competition"],
        "tiers": ["mbb"]},
    "accounting_ta": {
        "display": "Big 4 / Transaction Advisory",
        "signals": ["transaction services", "transaction advisory", "audit", "cdts", "valuation services"],
        "tiers": ["big_four"]},
    "private_markets": {
        "display": "Private Equity / Growth",
        "signals": ["private equity", "growth equity", "portfolio company", "board observer", "buyout"],
        "tiers": ["hedge_fund_other"]},
    "industry_domain": {
        "display": "Industry / Domain Expert",
        "signals": ["mbbs", "md ", "physician", "biotech", "laboratory", "clinical", "engineer at",
                    "biomodeller", "research associate"],
        "tiers": []},
}

# ---------------------------------------------------------------- geography
GEO_MAP: dict[str, tuple[str, str]] = {}
_GEO_SEED = {
    "americas": {
        "United States": ["new york", "ny", "nyc", "boston", "chicago", "san francisco", "greenwich",
                          "connecticut", "ct", "cambridge, ma", "evanston", "ann arbor", "brooklyn",
                          "united states", "usa", "u.s.", "massachusetts", "illinois", "michigan"],
        "Brazil": ["sao paulo", "são paulo", "brazil"],
        "Canada": ["toronto", "montreal", "canada"],
    },
    "emea": {
        "United Kingdom": ["london", "united kingdom", "uk", "england"],
        "France": ["paris", "lyon", "france"],
        "Morocco": ["casablanca", "rabat", "morocco"],
        "Germany": ["frankfurt", "berlin", "munich", "germany"],
        "Switzerland": ["zurich", "geneva", "switzerland"],
        "UAE": ["dubai", "abu dhabi", "uae"],
    },
    "apac": {
        "Hong Kong": ["hong kong", "hk"],
        "India": ["mumbai", "noida", "pune", "navi mumbai", "bangalore", "bengaluru", "delhi",
                  "gurgaon", "india", "maharashtra"],
        "Singapore": ["singapore"],
        "China": ["shanghai", "beijing", "shenzhen", "jiangxi", "china", "greater china"],
        "Japan": ["tokyo", "japan"],
        "Australia": ["sydney", "melbourne", "australia"],
    },
}
for _region, _countries in _GEO_SEED.items():
    for _country, _cities in _countries.items():
        for _c in _cities:
            GEO_MAP[_c] = (_country, _region)

REGION_DISPLAY = {"americas": "Americas", "emea": "Europe / EMEA", "apac": "Asia-Pacific"}

# ---------------------------------------------------------------- certifications
CERTIFICATIONS: dict[str, dict] = {
    "cfa": {"display": "CFA", "aliases": ["cfa", "chartered financial analyst"],
            "levels": {"charterholder": ["charterholder", "charter holder", "level iii", "level 3",
                                          "passed level iii", "cfa®"],
                       "level_ii": ["level ii", "level 2"], "level_i": ["level i", "level 1"]}},
    "frm": {"display": "FRM", "aliases": ["frm", "financial risk manager"], "levels": {}},
    "cpa": {"display": "CPA", "aliases": ["cpa", "certified public accountant"], "levels": {}},
    "caia": {"display": "CAIA", "aliases": ["caia"], "levels": {}},
    "series_7": {"display": "Series 7", "aliases": ["series 7"], "levels": {}},
    "series_63": {"display": "Series 63", "aliases": ["series 63"], "levels": {}},
    "series_87": {"display": "Series 87", "aliases": ["series 87"], "levels": {}},
    "mbbs": {"display": "MBBS (Medical)", "aliases": ["mbbs", "m.b.b.s"], "levels": {}},
}

DEGREE_LEVELS: list[tuple[str, str]] = [
    (r"\b(ph\.?d|doctor of philosophy|dphil)\b", "phd"),
    (r"\b(m\.?b\.?a|master of business administration|pgdm|pgp)\b", "mba"),
    (r"\b(m\.?b\.?b\.?s|m\.?d\b|doctor of medicine)\b", "professional"),
    (r"\b(m\.?s\.?c?|master(?:'s)? (?:of|in|degree)|m\.?tech|m\.?com|masters?|diplôme d'ingénieur|diplome d'ingenieur|mfe)\b", "masters"),
    (r"\b(b\.?s\.?c?|b\.?a\b|bachelor|b\.?tech|b\.?com|b\.?m\.?s|bba)\b", "bachelors"),
    (r"\b(xii std|x std|hsc|ssc|high school|secondary|preparatory class)\b", "secondary"),
]

LANGUAGE_NAMES = ["english", "mandarin", "cantonese", "chinese", "french", "arabic", "spanish",
                  "portuguese", "german", "japanese", "hindi", "marathi", "italian", "russian"]
PROFICIENCY = ["native", "fluent", "professional", "conversational", "basic", "working"]

# ============================================================================ utils

_WS = re.compile(r"\s+")


def norm(s: str) -> str:
    """Aggressive normalisation for matching: unicode-fold, lowercase, squeeze space."""
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    return _WS.sub(" ", s.lower()).strip()


def _compile(aliases: list[str], strict: bool = False) -> re.Pattern:
    parts = sorted((re.escape(a) for a in aliases), key=len, reverse=True)
    body = "|".join(parts)
    # \b fails around '+' and '#', so use lookarounds on non-word-ish boundaries.
    return re.compile(rf"(?<![\w+#]) ?({body})(?![\w+#])", re.I)


_SKILL_RE = {k: _compile(v["aliases"], v.get("strict", False)) for k, v in SKILLS.items()}
_STRAT_RE = {k: _compile(v["triggers"]) for k, v in STRATEGIES.items()}
_SECTOR_RE = {k: _compile(v["triggers"]) for k, v in SECTORS.items()}


def find_skills(text: str) -> list[tuple[str, str, int, int]]:
    """-> [(canonical, surface, start, end)] over the *raw* text so offsets stay valid."""
    hits = []
    for canon, rx in _SKILL_RE.items():
        strict = SKILLS[canon].get("strict", False)
        for m in rx.finditer(text):
            surface = m.group(1)
            if strict and len(surface) <= 2:
                # 'R' and 'Wind' only count in an explicit skills/tools listing context.
                ctx = norm(text[max(0, m.start() - 90): m.end() + 90])
                if not any(w in ctx for w in ("skill", "programming", "language", "tool",
                                              "technical", "software", "database", "python")):
                    continue
            hits.append((canon, surface, m.start(1), m.end(1)))
    return hits


def find_strategies(text: str) -> list[tuple[str, str, int, int]]:
    return [(k, m.group(1), m.start(1), m.end(1))
            for k, rx in _STRAT_RE.items() for m in rx.finditer(text)]


def find_sectors(text: str) -> list[tuple[str, str, int, int]]:
    return [(k, m.group(1), m.start(1), m.end(1))
            for k, rx in _SECTOR_RE.items() for m in rx.finditer(text)]


def canonical_employer(raw: str) -> tuple[str, str]:
    """-> (canonical display name, tier). Falls back to a cleaned raw string.

    Deliberately tolerant of the typos present in real resumes: the corpus contains
    'J.P.Mogan', which must still resolve to the bulge-bracket tier.
    """
    n = norm(raw)
    n = re.sub(r"\b(ltd|limited|inc|llc|plc|pvt|corp|co|group|holdings?|partners?|management|"
               r"securities|capital|and co|& co)\b\.?", " ", n)
    n = _WS.sub(" ", n).strip(" ,.-")
    best_tier, best_alias = "unknown", ""
    for tier, names in FIRM_TIERS.items():
        for name in names:
            nn = norm(name)
            if nn and (nn in norm(raw) or nn in n) and len(nn) > len(best_alias):
                best_tier, best_alias = tier, nn
    if best_alias in EMPLOYER_DISPLAY:
        return EMPLOYER_DISPLAY[best_alias], best_tier
    display = " ".join(w.capitalize() if len(w) > 3 else w.upper() for w in (best_alias or n).split())
    return (display or raw.strip(), best_tier)


def title_to_level(title: str, tier: str = "unknown") -> tuple[int, str]:
    """Map raw title + employer tier -> level 1..7 with a stated rationale.

    Tier adjustment: at a low-rigor shop a senior-sounding title is discounted by one
    level (floored at 2); at a top-tier shop it is left alone. This is a documented,
    recruiter-editable heuristic, not a claim about individuals.
    """
    t = norm(title)
    level, why = 3, "default analyst level"
    for pattern, lvl in TITLE_LEVEL_RULES:
        if re.search(pattern, t):
            if lvl > level or lvl == 1:
                level, why = lvl, f"title matched /{pattern}/"
            if lvl == 1:
                break
    rigor = TIER_RIGOR.get(tier, 0.75)
    if rigor < 0.75 and level >= 4:
        level -= 1
        why += f"; discounted one level for {TIER_DISPLAY.get(tier, tier)} title inflation"
    return max(1, min(7, level)), why


def match_geography(text: str) -> list[tuple[str, str, str, int, int]]:
    """-> [(country, region, surface, start, end)] ordered by position."""
    out = []
    low = norm(text)
    for token, (country, region) in GEO_MAP.items():
        for m in re.finditer(rf"(?<![\w]){re.escape(token)}(?![\w])", low):
            out.append((country, region, token, m.start(), m.end()))
    return sorted(out, key=lambda x: x[3])


def degree_level(text: str) -> str | None:
    t = norm(text)
    for pattern, lvl in DEGREE_LEVELS:
        if re.search(pattern, t):
            return lvl
    return None


def match_certifications(text: str) -> list[tuple[str, str | None, str, int, int]]:
    """-> [(canonical, status, surface, start, end)]"""
    out = []
    for canon, spec in CERTIFICATIONS.items():
        rx = _compile(spec["aliases"])
        for m in rx.finditer(text):
            window = norm(text[max(0, m.start() - 60): m.end() + 140])
            status = None
            for st, markers in spec.get("levels", {}).items():
                if any(mk in window for mk in markers):
                    status = st
                    break
            out.append((canon, status, m.group(1), m.start(1), m.end(1)))
    return out


def display(kind: str, label, fallback: str = "—") -> str:
    """Human-readable name for a taxonomy label, tolerant of unknown values.

    Taxonomies are versioned, which means they evolve: a profile parsed under
    taxonomy 1.1 may carry a label that 1.3 has renamed or retired. Direct dictionary
    indexing turns that into a KeyError that takes down the whole page. Since a stale
    label is a display problem and not a correctness problem, this degrades to a
    humanised form of the raw label instead -- the recruiter sees something sensible
    and the System page's version banner explains why it looks unfamiliar.
    """
    if label is None or label == "":
        return fallback
    table = {
        "strategy": STRATEGIES, "sector": SECTORS, "feeder": FEEDER_PATHS,
        "certification": CERTIFICATIONS, "skill": SKILLS,
    }.get(kind)
    if table is not None:
        entry = table.get(label)
        if isinstance(entry, dict) and entry.get("display"):
            return entry["display"]
    elif kind == "tier":
        if label in TIER_DISPLAY:
            return TIER_DISPLAY[label]
    elif kind == "region":
        if label in REGION_DISPLAY:
            return REGION_DISPLAY[label]
    elif kind == "seniority":
        try:
            lvl = int(str(label).lstrip("Ll"))
        except (TypeError, ValueError):
            return str(label)
        return SENIORITY_LEVELS.get(lvl, f"Level {lvl}")
    return str(label).replace("_", " ").title()


ALL_STRATEGY_LABELS = list(STRATEGIES)
ALL_SECTOR_LABELS = list(SECTORS)
ALL_SKILL_LABELS = list(SKILLS)
