"""Accept no tests only while the Phase 0 suite is empty."""
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
result = subprocess.run([sys.executable, "-m", "pytest", *sys.argv[1:]], cwd=root)
has_tests = any((root / "tests").rglob("test_*.py")) or any(
    (root / "tests").rglob("*_test.py")
)
if result.returncode == 5 and not has_tests:
    print("Phase 0: empty suite verified; no tests exist yet.")
    sys.exit(0)
sys.exit(result.returncode)
