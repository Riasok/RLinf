# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Tests for complete-episode seven-task filtering without simulator/GPU imports."""

import ast
from pathlib import Path

import pytest


def selector():
    source = Path(__file__).parents[2] / "rlinf/workers/sft/spatial_bc_worker.py"
    tree = ast.parse(source.read_text())
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "select_episodes"
    )
    scope = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), scope)
    return scope["select_episodes"]


def test_retains_all_allowed_and_excludes_mixed_episodes():
    episodes = [
        {"tasks": ["a"]},
        {"tasks": ["a"]},
        {"tasks": ["b"]},
        {"tasks": ["heldout"]},
        {"tasks": ["a", "heldout"]},
        {"tasks": []},
    ]
    assert selector()(episodes, {"a", "b"}) == episodes[:3]


def test_missing_task_fails_closed():
    with pytest.raises(ValueError):
        selector()([{"tasks": ["a"]}], {"a", "b"})
