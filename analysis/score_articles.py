#!/usr/bin/env python3
"""Shim: lexicon scoring moved to analysis/lexicon_frames/score_articles.py."""

from pathlib import Path
import runpy
import sys

target = Path(__file__).resolve().parent / "lexicon_frames" / "score_articles.py"
sys.argv[0] = str(target)
runpy.run_path(str(target), run_name="__main__")
