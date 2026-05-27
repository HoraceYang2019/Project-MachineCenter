from __future__ import annotations

"""
validate_hybrid_stack.py

Purpose
-------
Orchestration script for the TMV-720 hybrid stack.

It runs:
    1. validation/validate_knowledge_layer.py
    2. validation/validate_runtime_layer.py

This file does not directly validate a combined ontology + knowledge + runtime
graph, because that caused false-positive SHACL violations by applying runtime
constraints to static ontology/demo individuals.

Run from project root:

    python validation/validate_hybrid_stack.py
"""

from pathlib import Path
import subprocess
import sys

BASE_DIR = Path(__file__).resolve().parents[1]
VALIDATION_DIR = BASE_DIR / "validation"

SCRIPTS = [
    VALIDATION_DIR / "validate_knowledge_layer.py",
    VALIDATION_DIR / "validate_runtime_layer.py",
]

def run_script(script: Path) -> bool:
    print("\n" + "=" * 80)
    print(f"[RUN] {script}")
    print("=" * 80)

    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(BASE_DIR),
        text=True,
    )

    if result.returncode == 0:
        print(f"[PASS] {script.name}")
        return True

    print(f"[FAIL] {script.name}")
    return False


def main() -> None:
    all_ok = True

    for script in SCRIPTS:
        if not script.exists():
            print(f"[FAIL] Missing script: {script}")
            all_ok = False
            continue

        ok = run_script(script)
        all_ok = all_ok and ok

    print("\n" + "=" * 80)
    print("Hybrid Stack Validation Summary")
    print("=" * 80)
    print(f"Knowledge layer: {'see knowledge report'}")
    print(f"Runtime layer:   {'see runtime report'}")
    print(f"Overall:         {'PASS' if all_ok else 'FAIL'}")

    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
