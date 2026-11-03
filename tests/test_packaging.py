"""Packaging guard: the public package must not carry working-directory files.

Checks the directory tree AND git's view (a gitignored file hides from
`git ls-files` but still ships in a hand-made zip), per the research
standing rules. Also enforces the no-em-dash rule on released text files.
"""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_NAMES = {
    "CLAUDE.md", "MANUSCRIPT.md", "FINDINGS.md", "CORRECTIONS.md",
    "RESEARCH.md", "README-working.md", "LITERATURE.md", "PAPER-DRAFT.md",
}
FORBIDDEN_PREFIXES = ("COVER-LETTER", "SUBMITTED")
FORBIDDEN_DIRS = {"paper", "out", "data", "external", ".venv", "submission", "draft"}


def walk():
    for p in ROOT.rglob("*"):
        rel = p.relative_to(ROOT)
        if any(part in {".git", "__pycache__", ".pytest_cache"} for part in rel.parts):
            continue
        yield rel


def test_directory_clean():
    for rel in walk():
        assert rel.name not in FORBIDDEN_NAMES, f"forbidden file in package: {rel}"
        assert not rel.name.startswith(FORBIDDEN_PREFIXES), f"forbidden file in package: {rel}"
        if len(rel.parts) == 1 and (ROOT / rel).is_dir():
            assert rel.parts[0] not in FORBIDDEN_DIRS, f"forbidden directory in package: {rel}"


def test_git_view_clean():
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True)
    if out.returncode != 0:
        return  # not a git checkout (e.g. unpacked archive): directory check covers it
    for line in out.stdout.splitlines():
        name = Path(line).name
        top = Path(line).parts[0]
        assert name not in FORBIDDEN_NAMES, f"forbidden file tracked by git: {line}"
        assert not name.startswith(FORBIDDEN_PREFIXES), f"forbidden file tracked by git: {line}"
        assert top not in FORBIDDEN_DIRS, f"forbidden directory tracked by git: {line}"


def test_no_tracked_bytecode():
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True)
    if out.returncode != 0:
        return  # not a git checkout: test_no_stray_bytecode covers the directory
    for line in out.stdout.splitlines():
        assert not line.endswith(".pyc"), f"bytecode tracked by git: {line}"


def test_no_stray_bytecode():
    # walk() skips __pycache__, which every test run recreates; a .pyc anywhere else is stray.
    for rel in walk():
        assert rel.suffix != ".pyc", f"stray bytecode in package: {rel}"


def test_no_em_dashes_in_released_text():
    for rel in walk():
        p = ROOT / rel
        if p.suffix in {".md", ".cff", ".txt"} and p.is_file():
            text = p.read_text(encoding="utf-8", errors="ignore")
            assert "—" not in text, f"em dash in released file: {rel}"
