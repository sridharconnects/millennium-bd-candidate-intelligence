import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import pytest

RESUMES = sorted(p for p in ROOT.iterdir()
                 if p.suffix.lower() in (".pdf", ".docx") and not p.name.startswith("~$"))


@pytest.fixture(scope="session")
def root():
    return ROOT


@pytest.fixture(scope="session")
def resumes():
    return RESUMES


@pytest.fixture(scope="session")
def profiles():
    """Parsed profiles from the committed export. Tests that need them skip if the
    pipeline has not been run, so the suite is still useful on a fresh checkout."""
    from millennium import app_data
    ps, _ = app_data.load_profiles_from_artifact()
    if not ps:
        pytest.skip("no parsed profiles; run scripts/run_pipeline.py first")
    app_data.load_raw_texts(ps)
    return ps
