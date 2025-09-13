import sys
from pathlib import Path
import pytest

# Ensure the `src/` directory is on sys.path for imports like `import microlux`
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TEST_DIR = ROOT / "test"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))
