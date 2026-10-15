import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from j1939pf.config import load_defaults  # noqa: E402
from j1939pf.spn_table import load_spn_table  # noqa: E402


@pytest.fixture(scope="session")
def table():
    return load_spn_table()


@pytest.fixture
def cfg():
    return load_defaults()
