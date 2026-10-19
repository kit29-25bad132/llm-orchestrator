"""Validate built distribution artifacts against the project packaging contract.

Run after ``python -m build`` from the repository root::

    python scripts/validate_dist.py dist

Checks, without assuming a successful build means the artifacts are correct:

- exactly one wheel is present; its package files match ``src/llm_orchestrator``
  one-for-one and its metadata matches ``pyproject.toml`` (name, version,
  requires-python, ``Typing :: Typed``, extras and their dependency sets,
  license expression and license files);
- ``py.typed`` ships in the wheel; tests, docs, CI config, env files, caches,
  and other repository-only material do not (the sdist deliberately ships
  ``tests/`` per its declared policy — nothing else outside that policy);
- the MIT ``LICENSE`` file ships in both artifacts (PEP 639 layout in the
  wheel, root of the sdist);
- when an sdist is present, its contents match the declared sdist policy and
  include everything needed to rebuild the wheel (``--wheel-only`` skips this,
  e.g. when validating a wheel built *from* the sdist).

Exit status is non-zero if any check fails. Standard library only.
"""

from __future__ import annotations

import argparse
import email
import re
import sys
import tarfile
import tomllib
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_PREFIX = "llm_orchestrator/"

#: The packaging contract (kept in sync with pyproject.toml by the checks).
EXPECTED_EXTRAS: dict[str, set[str]] = {
    "gemini": {"google-genai", "httpx"},
    "groq": {"groq", "httpx"},
    "langgraph": {"langgraph"},
    "all": {"google-genai", "groq", "langgraph", "httpx"},
    "dev": {"pytest", "ruff", "mypy", "build"},
}
CORE_DEPENDENCIES = {"pydantic"}

#: License contract: PEP 639 SPDX expression in metadata, file in both
#: artifacts (wheel: ``*.dist-info/licenses/LICENSE``, sdist: root).
EXPECTED_LICENSE = "MIT"
LICENSE_FILE = "LICENSE"

#: Repository-only path segments that must never appear in any artifact.
FORBIDDEN_PARTS = frozenset(
    {
        "tests",
        "docs",
        "scripts",
        ".github",
        ".env",
        ".env.example",
        ".venv",
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".freebuff",
        ".kilo",
        "__pycache__",
        "dist",
        "build",
    }
)

#: The sdist intentionally ships the test suite (and no other repo material),
#: so ``tests`` is the one forbidden segment that does not apply there.
SDIST_FORBIDDEN_PARTS = FORBIDDEN_PARTS - {"tests"}

#: Deliberate sdist policy (mirrors [tool.hatch.build.targets.sdist]).
SDIST_REQUIRED = (
    "pyproject.toml",
    "README.md",
    "CHANGELOG.md",
    "LICENSE",
    "src/llm_orchestrator/__init__.py",
)
SDIST_ALLOWED_PREFIXES = ("src/", "tests/")
#: PKG-INFO is required in an sdist (PEP 517 / core metadata); .gitignore is
#: force-included by hatchling as the VCS exclusion file (like hatch.toml /
#: hatch_build.py would be, if they existed — they are deliberately not
#: pre-allowed here, so adding one is a conscious policy change).
SDIST_ALLOWED_FILES = frozenset(
    {"pyproject.toml", "README.md", "CHANGELOG.md", "PKG-INFO", ".gitignore"}
)
SDIST_ALLOWED_FILE_PREFIXES = ("LICENSE",)


def _require_name(requirement: str) -> str:
    return re.split(r"[<>=!~;\[\s]", requirement, maxsplit=1)[0].strip().lower()


def _require_extra(requirement: str) -> str | None:
    match = re.search(r"""extra\s*==\s*['"]([^'"]+)['"]""", requirement)
    return match.group(1) if match else None


def _expected_package_files() -> set[str]:
    """Package-relative paths that must be inside the wheel (incl. py.typed)."""
    files: set[str] = set()
    for path in (REPO_ROOT / "src").rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        files.add(path.relative_to(REPO_ROOT / "src").as_posix())
    return files


def _check_forbidden(
    names: list[str],
    label: str,
    errors: list[str],
    *,
    forbidden: frozenset[str] = FORBIDDEN_PARTS,
) -> None:
    for name in names:
        parts = Path(name).parts
        if forbidden.intersection(parts):
            errors.append(f"{label}: forbidden path included: {name}")
        if name.endswith((".pyc", ".pyo")):
            errors.append(f"{label}: bytecode file included: {name}")


def _load_pyproject() -> dict:
    with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)


def _validate_wheel(wheel_path: Path, project: dict, errors: list[str]) -> None:
    print(f"validating wheel: {wheel_path.name}")
    expected_files = _expected_package_files()
    with zipfile.ZipFile(wheel_path) as archive:
        names = archive.namelist()

        _check_forbidden(names, "wheel", errors)

        package_files = {name for name in names if name.startswith(PACKAGE_PREFIX)}
        dist_info = {name for name in names if ".dist-info/" in name}
        other = set(names) - package_files - dist_info
        for name in sorted(other):
            errors.append(f"wheel: unexpected top-level entry: {name}")
        missing = expected_files - package_files
        extra = package_files - expected_files
        for name in sorted(missing):
            errors.append(f"wheel: missing package file: {name}")
        for name in sorted(extra):
            errors.append(f"wheel: file not present in src/: {name}")
        if f"{PACKAGE_PREFIX}py.typed" not in package_files:
            errors.append("wheel: py.typed marker missing (Typing :: Typed would be a lie)")

        shipped_licenses = sorted(name for name in names if ".dist-info/licenses/" in name)
        if not any(name.endswith(f"/{LICENSE_FILE}") for name in shipped_licenses):
            errors.append(
                f"wheel: {LICENSE_FILE} missing from .dist-info/licenses/ "
                f"(found: {shipped_licenses or 'nothing'})"
            )

        metadata_files = [name for name in dist_info if name.endswith("/METADATA")]
        if len(metadata_files) != 1:
            errors.append(f"wheel: expected exactly one METADATA, found {len(metadata_files)}")
            return
        metadata = email.message_from_bytes(archive.read(metadata_files[0]))

    if metadata["Name"] != project["project"]["name"]:
        errors.append(
            f"wheel: Name {metadata['Name']!r} != pyproject {project['project']['name']!r}"
        )
    if metadata["Version"] != project["project"]["version"]:
        errors.append("wheel: Version does not match pyproject")
    if metadata["Requires-Python"] != project["project"]["requires-python"]:
        errors.append(
            f"wheel: Requires-Python {metadata['Requires-Python']!r} "
            f"!= pyproject {project['project']['requires-python']!r}"
        )
    if "Typing :: Typed" not in metadata.get_all("Classifier", []):
        errors.append("wheel: Typing :: Typed classifier missing")

    declared_license = project["project"].get("license")
    if isinstance(declared_license, dict):
        declared_license = declared_license.get("text")
    if declared_license != EXPECTED_LICENSE:
        errors.append(f"pyproject: license {declared_license!r} != {EXPECTED_LICENSE!r}")
    if metadata["License-Expression"] != EXPECTED_LICENSE:
        errors.append(
            f"wheel: License-Expression {metadata['License-Expression']!r} != {EXPECTED_LICENSE!r}"
        )
    shipped_license_files = metadata.get_all("License-File") or []
    if LICENSE_FILE not in shipped_license_files:
        errors.append(
            f"wheel: License-File metadata missing {LICENSE_FILE!r} "
            f"(found {sorted(shipped_license_files)})"
        )

    core: set[str] = set()
    by_extra: dict[str, set[str]] = {}
    for requirement in metadata.get_all("Requires-Dist", []):
        extra = _require_extra(requirement)
        if extra is None:
            core.add(_require_name(requirement))
        else:
            by_extra.setdefault(extra, set()).add(_require_name(requirement))
    if core != CORE_DEPENDENCIES:
        errors.append(f"wheel: core Requires-Dist {sorted(core)} != {sorted(CORE_DEPENDENCIES)}")
    if set(by_extra) != set(EXPECTED_EXTRAS):
        errors.append(f"wheel: extras {sorted(by_extra)} != {sorted(EXPECTED_EXTRAS)}")
    for extra, expected in EXPECTED_EXTRAS.items():
        actual = by_extra.get(extra, set())
        if actual != expected:
            errors.append(f"wheel: extra {extra!r} deps {sorted(actual)} != {sorted(expected)}")
    print(f"  {len(package_files)} package files, extras: {sorted(by_extra)}")


def _validate_sdist(sdist_path: Path, errors: list[str]) -> None:
    print(f"validating sdist: {sdist_path.name}")
    expected_files = _expected_package_files()
    with tarfile.open(sdist_path, "r:gz") as archive:
        members = [member.name for member in archive.getmembers() if member.isfile()]

    roots = {name.split("/", 1)[0] for name in members}
    if len(roots) != 1:
        errors.append(f"sdist: expected a single root directory, found {sorted(roots)}")
        return
    root = roots.pop()
    relative = [name.split("/", 1)[1] for name in members if name.startswith(f"{root}/")]

    _check_forbidden(relative, "sdist", errors, forbidden=SDIST_FORBIDDEN_PARTS)

    for name in sorted(relative):
        if (
            not name.startswith(SDIST_ALLOWED_PREFIXES)
            and name not in SDIST_ALLOWED_FILES
            and not name.startswith(SDIST_ALLOWED_FILE_PREFIXES)
        ):
            errors.append(f"sdist: file outside declared policy: {name}")

    for required in SDIST_REQUIRED:
        if required not in relative:
            errors.append(f"sdist: required file missing: {required}")
    if not any(name.startswith("tests/") for name in relative):
        errors.append("sdist: tests/ missing (sdist policy includes the test suite)")
    for path in sorted(expected_files):
        if f"src/{path}" not in relative:
            errors.append(f"sdist: package file missing (needed to rebuild the wheel): src/{path}")
    print(f"  {len(relative)} files under {root}/")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dist_dir", type=Path, help="directory containing the built artifacts")
    parser.add_argument(
        "--wheel-only",
        action="store_true",
        help="skip sdist checks (for wheels rebuilt from the sdist)",
    )
    args = parser.parse_args(argv)

    if not args.dist_dir.is_dir():
        print(f"FAIL: {args.dist_dir} is not a directory")
        return 2

    wheels = sorted(args.dist_dir.glob("*.whl"))
    sdists = sorted(args.dist_dir.glob("*.tar.gz"))
    if len(wheels) != 1:
        print(f"FAIL: expected exactly one wheel in {args.dist_dir}, found {len(wheels)}")
        return 2

    project = _load_pyproject()
    errors: list[str] = []
    _validate_wheel(wheels[0], project, errors)

    if not args.wheel_only:
        if sdists:
            if len(sdists) != 1:
                errors.append(f"sdist: expected exactly one, found {len(sdists)}")
            else:
                _validate_sdist(sdists[0], errors)
        else:
            errors.append("sdist: none found (run `python -m build` or pass --wheel-only)")

    if errors:
        print("\nFAILED:")
        for error in errors:
            print(f"  - {error}")
        return 1
    print("\nOK: all distribution checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
