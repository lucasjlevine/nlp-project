"""Climate news frame dictionaries and lexicon scoring helpers.

Frame scores are keyword hit rates per 1,000 tokens in title+text.
Designed for RQ1 framing evolution (science/uncertainty, political,
economic, solutions) without embedding models.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

FRAME_LEXICONS: dict[str, tuple[str, ...]] = {
    "science_uncertainty": (
        "uncertainty",
        "uncertain",
        "unproven",
        "inconclusive",
        "contested",
        "debate",
        "debated",
        "skeptic",
        "skeptics",
        "sceptical",
        "skeptical",
        "hoax",
        "myth",
        "disputed",
        "controversial",
        "consensus",
        "scientific consensus",
        "peer reviewed",
        "peer-reviewed",
        "ipcc",
        "climate science",
        "climate scientist",
        "climate scientists",
        "study finds",
        "new study",
        "research shows",
        "evidence",
        "settled science",
        "denier",
        "deniers",
        "denial",
    ),
    "political": (
        "congress",
        "senate",
        "democrat",
        "democrats",
        "republican",
        "republicans",
        "partisan",
        "partisanship",
        "election",
        "elections",
        "campaign",
        "white house",
        "president",
        "presidential",
        "legislation",
        "legislators",
        "lawmakers",
        "bipartisan",
        "gop",
        "trump",
        "biden",
        "obama",
        "epa",
        "political",
        "politician",
        "politicians",
        "vote",
        "voters",
        "policy fight",
        "culture war",
    ),
    "economic": (
        "cost",
        "costs",
        "expensive",
        "affordable",
        "price",
        "prices",
        "jobs",
        "job",
        "employment",
        "unemployment",
        "tax",
        "taxes",
        "carbon tax",
        "gdp",
        "economy",
        "economic",
        "industry",
        "industries",
        "market",
        "markets",
        "fossil fuel",
        "fossil fuels",
        "oil",
        "gas",
        "coal",
        "subsidy",
        "subsidies",
        "investment",
        "investments",
        "consumer",
        "consumers",
        "business",
        "businesses",
        "profit",
        "profits",
        "recession",
        "inflation",
    ),
    "solutions": (
        "solution",
        "solutions",
        "renewable",
        "renewables",
        "solar",
        "wind power",
        "wind energy",
        "clean energy",
        "clean power",
        "net zero",
        "net-zero",
        "carbon neutral",
        "carbon-neutral",
        "mitigation",
        "mitigate",
        "adaptation",
        "adapt",
        "transition",
        "decarboniz",
        "decarbonis",
        "electrification",
        "electric vehicle",
        "electric vehicles",
        "evs",
        "paris agreement",
        "climate action",
        "actionable",
        "emissions reduction",
        "reduce emissions",
        "carbon capture",
        "green new deal",
        "inflation reduction act",
        "ira",
        "resilience",
        "resilient",
    ),
}

FRAME_NAMES = tuple(FRAME_LEXICONS.keys())

# Precompile: longer phrases first so multi-word matches win.
_FRAME_PATTERNS: dict[str, re.Pattern[str]] = {}
for _frame, _terms in FRAME_LEXICONS.items():
    ordered = sorted(set(_terms), key=len, reverse=True)
    parts = [re.escape(t) for t in ordered]
    _FRAME_PATTERNS[_frame] = re.compile(
        r"(?<![a-z0-9])(?:" + "|".join(parts) + r")(?![a-z0-9])",
        re.IGNORECASE,
    )

_TOKEN_RE = re.compile(r"[a-z0-9']+", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def token_count(text: str) -> int:
    return len(tokenize(text))


def score_frames(text: str) -> dict[str, float]:
    """Return frame hit rates per 1,000 tokens (0 if empty text)."""
    n = token_count(text)
    if n == 0:
        return {name: 0.0 for name in FRAME_NAMES}
    out: dict[str, float] = {}
    for name, pattern in _FRAME_PATTERNS.items():
        hits = len(pattern.findall(text))
        out[name] = 1000.0 * hits / n
    return out


def dominant_frame(scores: dict[str, float]) -> str:
    if not scores or max(scores.values()) <= 0:
        return "none"
    return max(scores, key=scores.get)


def batch_score_frames(texts: Iterable[str]) -> list[dict[str, float]]:
    return [score_frames(t) for t in texts]
