#!/usr/bin/env python3
"""Run pytest-based spatial test directories discovered from the current repository."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SPATIAL_DIR = ROOT / "skills" / "spatial"
EXTRA_TEST_DIRS = [
    ROOT / "tests" / "spatial",
]


def discover_test_dirs() -> list[Path]:
    discovered = [
        path
        for path in SPATIAL_DIR.glob("*/tests")
        if path.is_dir()
    ]
    discovered.extend(path for path in EXTRA_TEST_DIRS if path.is_dir())
    return sorted(set(discovered))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run pytest for discovered spatial test directories")
    parser.add_argument(
        "--filter",
        default=None,
        help="Only run test directories whose path contains this substring",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first failing test directory",
    )
    args = parser.parse_args()

    test_dirs = discover_test_dirs()
    if args.filter:
        test_dirs = [path for path in test_dirs if args.filter in str(path)]

    if not test_dirs:
        print("No spatial test directories found.")
        return 1

    print("Discovered spatial test directories:")
    for path in test_dirs:
        print(f"  - {path}")

    failures: list[Path] = []
    for test_dir in test_dirs:
        print(f"\n--- Running tests in {test_dir} ---")
        result = subprocess.run(
            [sys.executable, "-m", "pytest", str(test_dir)],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )
        if result.returncode == 0:
            print("SUCCESS")
        else:
            print("FAILED")
            print(result.stdout)
            if result.stderr:
                print("STDERR:")
                print(result.stderr)
            failures.append(test_dir)
            if args.fail_fast:
                break

    if failures:
        print("\nFailing directories:")
        for path in failures:
            print(f"  - {path}")
        return 1

    print("\nAll discovered spatial test directories passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
