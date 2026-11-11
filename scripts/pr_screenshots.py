"""Publish PR screenshots to the orphan ``pr-screenshots`` branch and print their Markdown.

Images shown in pull-request descriptions live on an orphan branch (no shared
history with ``master``, never merged) so feature diffs stay free of binaries.
This script adds the given files under ``PREFIX/`` in one commit on that branch,
creating the branch as a root commit on first use, and prints one Markdown
image line per file. The links are pinned to the new commit, so they keep
showing the same image even if the file is replaced later.

Everything goes through the GitHub API (``gh api``); the working copy is not
touched. Run from the repo root::

    uv run python scripts/pr_screenshots.py --prefix desktop_app_cs suspension.png "Suspension tab=other.png"

Each argument is ``FILE`` or ``CAPTION=FILE``. Paste the printed lines into the
commit message (``sl pr submit`` rebuilds the PR description from it).
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import subprocess
import sys
from pathlib import Path

BRANCH = "pr-screenshots"
README = """# PR screenshots

Images referenced from pull-request descriptions. This branch is an orphan: it
shares no history with `master` and is never merged. Add files with
`scripts/pr_screenshots.py` (see CLAUDE.md, "PR Screenshots").

Do not delete or force-push this branch: PR descriptions link to its commits.
"""


def parse_repo(url: str) -> str:
    """Return ``OWNER/REPO`` from a GitHub remote URL.

    Parameters
    ----------
    url : str
        An https or ssh GitHub remote URL, with or without ``.git``.

    Returns
    -------
    str
        The ``OWNER/REPO`` slug.

    Raises
    ------
    ValueError
        If the URL is not a GitHub repository URL.
    """
    match = re.search(r"github\.com[:/]+([^/\s]+)/([^/\s]+?)(?:\.git)?/?$", url.strip())
    if match is None:
        raise ValueError(f"not a GitHub repository URL: {url!r}")
    return f"{match.group(1)}/{match.group(2)}"


def parse_item(item: str) -> tuple[str, Path]:
    """Split a ``CAPTION=FILE`` argument; the caption defaults to the file stem."""
    caption, sep, path = item.rpartition("=")
    file = Path(path)
    return (caption if sep else file.stem.replace("_", " ")), file


def raw_url(repo: str, commit: str, path: str) -> str:
    """The raw-content URL of ``path`` at ``commit``."""
    return f"https://raw.githubusercontent.com/{repo}/{commit}/{path}"


def markdown(repo: str, commit: str, prefix: str, items: list[tuple[str, Path]]) -> list[str]:
    """One Markdown image line per item, pinned to ``commit``."""
    return [
        f"![{caption}]({raw_url(repo, commit, f'{prefix}/{file.name}')})" for caption, file in items
    ]


def gh_api(
    repo: str, endpoint: str, payload: dict | None = None, method: str | None = None
) -> dict:
    """Call ``gh api repos/REPO/ENDPOINT`` and return the parsed JSON response."""
    cmd = ["gh", "api", f"repos/{repo}/{endpoint}"]
    if method:
        cmd += ["--method", method]
    if payload is not None:
        cmd += ["--input", "-"]
    result = subprocess.run(
        cmd,
        input=json.dumps(payload) if payload is not None else None,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"gh api {endpoint} failed: {result.stderr.strip() or result.stdout}")
    data: dict = json.loads(result.stdout)
    return data


def branch_head(repo: str) -> tuple[str, str] | None:
    """The ``(commit, tree)`` at the tip of the screenshots branch, or None if it does not exist."""
    try:
        ref = gh_api(repo, f"git/ref/heads/{BRANCH}")
    except RuntimeError as error:
        if "Not Found" in str(error):
            return None
        raise
    commit = ref["object"]["sha"]
    return commit, gh_api(repo, f"git/commits/{commit}")["tree"]["sha"]


def publish(repo: str, prefix: str, items: list[tuple[str, Path]]) -> str:
    """Commit the files under ``prefix/`` on the screenshots branch; return the new commit."""
    head = branch_head(repo)
    entries = []
    for _, file in items:
        blob = gh_api(
            repo,
            "git/blobs",
            {"content": base64.b64encode(file.read_bytes()).decode(), "encoding": "base64"},
        )
        entries.append(
            {"path": f"{prefix}/{file.name}", "mode": "100644", "type": "blob", "sha": blob["sha"]}
        )
    tree: dict = {"tree": entries}
    if head is None:
        entries.append({"path": "README.md", "mode": "100644", "type": "blob", "content": README})
    else:
        tree["base_tree"] = head[1]
    new_tree = gh_api(repo, "git/trees", tree)["sha"]
    names = ", ".join(file.name for _, file in items)
    commit: str = gh_api(
        repo,
        "git/commits",
        {
            "message": f"{prefix}: {names}",
            "tree": new_tree,
            # No parent on first use: the branch starts as an orphan root commit.
            "parents": [] if head is None else [head[0]],
        },
    )["sha"]
    if head is None:
        gh_api(repo, "git/refs", {"ref": f"refs/heads/{BRANCH}", "sha": commit})
    else:
        gh_api(repo, f"git/refs/heads/{BRANCH}", {"sha": commit}, method="PATCH")
    return commit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--prefix", required=True, help="directory on the branch, e.g. desktop_app_cs"
    )
    parser.add_argument("--repo", help="OWNER/REPO (default: from `sl paths default`)")
    parser.add_argument("items", nargs="+", metavar="[CAPTION=]FILE")
    args = parser.parse_args(argv)

    repo = args.repo
    if repo is None:
        remote = subprocess.run(
            ["sl", "paths", "default"], capture_output=True, text=True, check=True
        ).stdout
        repo = parse_repo(remote)
    prefix = args.prefix.strip("/")
    items = [parse_item(item) for item in args.items]
    missing = [str(file) for _, file in items if not file.is_file()]
    if missing:
        parser.error(f"no such file: {', '.join(missing)}")

    commit = publish(repo, prefix, items)
    print(f"{BRANCH} @ {commit}", file=sys.stderr)
    print("\n\n".join(markdown(repo, commit, prefix, items)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
