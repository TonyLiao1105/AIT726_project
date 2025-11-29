import sys
from pathlib import Path


def pytest_configure(config):
    """Ensure the project root is on sys.path so tests can import `src`.

    This is necessary when tests are executed from the project directory
    but the interpreter doesn't automatically include the repo root on
    sys.path (common in some virtualenv/IDE setups).
    """
    # project root is the parent of the tests/ directory
    repo_root = Path(__file__).resolve().parent.parent
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
