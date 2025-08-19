import os
import pytest

# Set environment variable before any imports
os.environ["SKIP_MODEL_LOADING"] = "true"

@pytest.fixture(autouse=True)
def skip_model_loading():
    """Automatically set environment variable to skip model loading for all tests."""
    # Environment variable is already set above
    yield
    # Keep the variable set for the entire test session
