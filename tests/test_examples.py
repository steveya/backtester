"""Smoke tests for repository examples."""

from __future__ import annotations

import io
import runpy
from contextlib import redirect_stdout
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestExamples:
    def test_execution_walkforward_synthetic_runs(self) -> None:
        example = REPO_ROOT / "examples" / "execution_walkforward_synthetic.py"
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            runpy.run_path(str(example), run_name="__main__")

        output = stdout.getvalue()
        assert "Walk-forward metrics" in output
        assert "Per-fold metrics" in output
        assert "Event log" in output
