"""Load every Streamlit entry point and fail on uncaught page exceptions."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from streamlit.testing.v1 import AppTest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils import user_profile  # noqa: E402


ENTRY_POINTS = [
    REPO_ROOT / "app" / "Main.py",
    *sorted((REPO_ROOT / "app" / "pages").glob("*.py")),
]


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="restaurant-guide-smoke-") as temp_dir:
        user_profile.USER_PROFILES_PATH = Path(temp_dir) / "profiles.json"
        profile = user_profile._default_profile("Smoke Test", "smoke-test")
        user_profile.save_profiles({profile["id"]: profile})

        failures = []
        for entry_point in ENTRY_POINTS:
            app = AppTest.from_file(str(entry_point), default_timeout=120)
            app.session_state["authenticated_profile_id"] = profile["id"]
            app.run()
            exceptions = [str(exception.value) for exception in app.exception]
            relative_path = entry_point.relative_to(REPO_ROOT)
            if exceptions:
                failures.append((relative_path, exceptions))
                print(f"FAIL {relative_path}: {'; '.join(exceptions)}")
            else:
                print(f"PASS {relative_path}")

        if failures:
            raise SystemExit(f"{len(failures)} Streamlit page(s) failed smoke testing.")


if __name__ == "__main__":
    main()
