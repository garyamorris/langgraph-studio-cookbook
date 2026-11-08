"""Test configuration: every test runs offline (no network, no cost) and starts from clean stub handlers."""

from __future__ import annotations

import os

# Must be set before deepdive modules are imported anywhere.
os.environ["DEEPDIVE_OFFLINE"] = "1"

import pytest  # noqa: E402

from deepdive.common.decisions import clear_decision_overrides  # noqa: E402
from deepdive.common.fakes import clear_handlers  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_stubs():
    clear_handlers()
    clear_decision_overrides()
    yield
    clear_handlers()
    clear_decision_overrides()
