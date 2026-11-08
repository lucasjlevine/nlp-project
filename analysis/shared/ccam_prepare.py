"""CCAM wave prep: label-aware recodes, wave dates, weighted means.

The released CSV uses SPSS value *labels* (strings), not numeric codes.
Recodes follow the Yale CCAM data-release syntax semantics.
"""

from __future__ import annotations

import re
from datetime import datetime

import numpy as np
import pandas as pd

MONTHS = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}

WAVE_RE = re.compile(r"^([A-Za-z]{3})-(\d{2})$")

REFUSED = {"refused", "nan", ""}


def wave_to_date(wave: str) -> pd.Timestamp | pd.NaT:
    """Map labels like 'Apr-13' / 'Dec-25' to the first of that month."""
    if pd.isna(wave):
        return pd.NaT
    m = WAVE_RE.match(str(wave).strip())
    if not m:
        return pd.NaT
    mon, yy = m.group(1).title(), int(m.group(2))
    if mon not in MONTHS:
        return pd.NaT
    year = 2000 + yy
    return pd.Timestamp(datetime(year, MONTHS[mon], 1))


def _norm(x) -> str | None:
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s.lower() in REFUSED:
        return None
    return s


def _map_series(series: pd.Series, mapping: dict[str, float]) -> pd.Series:
    def one(v):
        s = _norm(v)
        if s is None:
            return np.nan
        if s in mapping:
            return mapping[s]
        # tolerate minor punctuation/spacing differences
        key = s.lower()
        for k, val in mapping.items():
            if k.lower() == key:
                return val
        return np.nan

    return series.map(one)


WORRY_MAP = {
    "Not at all worried": 1,
    "Not very worried": 1,
    "Somewhat worried": 2,
    "Very worried": 2,
}

HARM_MAP = {
    "Don't know": 0,
    "Dont know": 0,
    "Not at all": 1,
    "Only a little": 1,
    "A moderate amount": 2,
    "A great deal": 2,
}

WHEN_HARM_MAP = {
    "Never": 1,
    "In 100 years": 1,
    "In 50 years": 1,
    "In 25 years": 1,
    "In 10 years": 2,
    "They are being harmed right now": 2,
}

OPPOSE_SUPPORT_MAP = {
    "Strongly oppose": 1,
    "Somewhat oppose": 1,
    "Somewhat support": 2,
    "Strongly support": 2,
}

AGREE_MAP = {
    "Strongly disagree": 1,
    "Somewhat disagree": 1,
    "Somewhat agree": 2,
    "Strongly agree": 2,
}

PRIORITY_MAP = {
    "Low": 1,
    "Medium": 1,
    "High": 2,
    "Very high": 2,
}

DISCUSS_MAP = {
    "Never": 1,
    "Rarely": 1,
    "Occasionally": 2,
    "Often": 2,
}

HEAR_MEDIA_MAP = {
    "Never": 1,
    "Once a year or less often": 1,
    "Several times a year": 1,
    "At least once a month": 1,
    "0": 1,  # rare coding artifact in export
    "Don't know": np.nan,
    "At least once a week": 2,
}


def apply_recodes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["worry_r"] = _map_series(out["worry"], WORRY_MAP)

    for col in (
        "harm_personally",
        "harm_US",
        "harm_dev_countries",
        "harm_future_gen",
        "harm_plants_animals",
    ):
        if col in out.columns:
            out[f"{col}_r"] = _map_series(out[col], HARM_MAP)

    out["when_harm_US_r"] = _map_series(out["when_harm_US"], WHEN_HARM_MAP)

    for col in (
        "reg_CO2_pollutant",
        "fund_research",
        "transition_economy",
        "reduce_tax",
        "generate_renewable",
    ):
        if col in out.columns:
            out[f"{col}_r"] = _map_series(out[col], OPPOSE_SUPPORT_MAP)

    if "teach_gw" in out.columns:
        out["teach_gw_r"] = _map_series(out["teach_gw"], AGREE_MAP)
    if "priority" in out.columns:
        out["priority_r"] = _map_series(out["priority"], PRIORITY_MAP)
    if "priority_cleanenergy" in out.columns:
        out["priority_cleanenergy_r"] = _map_series(out["priority_cleanenergy"], PRIORITY_MAP)
    if "discuss_GW" in out.columns:
        out["discuss_GW_r"] = _map_series(out["discuss_GW"], DISCUSS_MAP)
    if "hear_GW_media" in out.columns:
        out["hear_GW_media_r"] = _map_series(out["hear_GW_media"], HEAR_MEDIA_MAP)

    out["wave_date"] = out["wave"].map(wave_to_date)
    return out


# Binary "high" indicators for weighted prevalence (recoded value == 2)
BINARY_OUTCOMES = {
    "worry_high": ("worry_r", 2),
    "harm_personally_high": ("harm_personally_r", 2),
    "harm_US_high": ("harm_US_r", 2),
    "when_harm_soon": ("when_harm_US_r", 2),
    "discuss_often": ("discuss_GW_r", 2),
    "hear_media_weekly": ("hear_GW_media_r", 2),
    "support_CO2_reg": ("reg_CO2_pollutant_r", 2),
    "support_fund_research": ("fund_research_r", 2),
    "support_transition": ("transition_economy_r", 2),
    "priority_high": ("priority_r", 2),
    "priority_cleanenergy_high": ("priority_cleanenergy_r", 2),
}


def weighted_prevalence(series: pd.Series, weights: pd.Series, high_value: float = 2) -> float:
    mask = series.notna() & weights.notna()
    if mask.sum() == 0:
        return np.nan
    s = series[mask]
    w = weights[mask]
    return float((w * (s == high_value).astype(float)).sum() / w.sum())


def wave_level_outcomes(ccam: pd.DataFrame) -> pd.DataFrame:
    """Weighted prevalence of high/support outcomes by wave."""
    rows = []
    for wave, g in ccam.groupby("wave", sort=False):
        row = {
            "wave": wave,
            "wave_date": g["wave_date"].iloc[0],
            "year": g["year"].iloc[0] if "year" in g.columns else np.nan,
            "n": len(g),
        }
        w = g["weight_wave"]
        for out_name, (col, high) in BINARY_OUTCOMES.items():
            if col not in g.columns:
                row[out_name] = np.nan
                continue
            row[out_name] = weighted_prevalence(g[col], w, high_value=high)
        rows.append(row)
    out = pd.DataFrame(rows).dropna(subset=["wave_date"]).sort_values("wave_date")
    return out.reset_index(drop=True)
