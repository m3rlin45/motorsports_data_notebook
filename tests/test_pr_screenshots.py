"""Tests for scripts/pr_screenshots.py (the parts that do not call GitHub)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "pr_screenshots.py"
_spec = importlib.util.spec_from_file_location("pr_screenshots", _SCRIPT)
assert _spec is not None and _spec.loader is not None
pr_screenshots = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pr_screenshots)


class TestParseRepo:
    @pytest.mark.parametrize(
        "url",
        [
            "https://github.com/someone/some_repo.git",
            "https://github.com/someone/some_repo",
            "git@github.com:someone/some_repo.git",
            "ssh://git@github.com/someone/some_repo.git\n",
        ],
    )
    def test_github_urls(self, url: str) -> None:
        assert pr_screenshots.parse_repo(url) == "someone/some_repo"

    def test_rejects_other_hosts(self) -> None:
        with pytest.raises(ValueError):
            pr_screenshots.parse_repo("https://example.com/someone/some_repo.git")


class TestParseItem:
    def test_caption_and_file(self) -> None:
        caption, file = pr_screenshots.parse_item("Suspension tab=renders/suspension_dark.png")
        assert caption == "Suspension tab"
        assert file == Path("renders/suspension_dark.png")

    def test_caption_defaults_to_the_file_stem(self) -> None:
        caption, file = pr_screenshots.parse_item("renders/tire_grip.png")
        assert caption == "tire grip"
        assert file == Path("renders/tire_grip.png")


class TestMarkdown:
    def test_links_are_pinned_to_the_commit(self) -> None:
        lines = pr_screenshots.markdown(
            "someone/some_repo",
            "abc123",
            "desktop_app_cs",
            [("Suspension tab", Path("renders/suspension.png"))],
        )
        assert lines == [
            "![Suspension tab](https://raw.githubusercontent.com/someone/some_repo/abc123/"
            "desktop_app_cs/suspension.png)"
        ]
