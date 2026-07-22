import hashlib
import json

import pytest

from utils import auth, user_profile


@pytest.fixture
def isolated_profile_store(tmp_path, monkeypatch):
    profile_path = tmp_path / "profiles" / "users.json"
    monkeypatch.setattr(user_profile, "USER_PROFILES_PATH", profile_path)
    return profile_path


def test_registration_uses_pbkdf2_and_authenticates(isolated_profile_store):
    success, profile_id = auth.register_user("Demo User", "correct-horse")

    assert success is True
    profiles = user_profile.load_profiles()
    profile = profiles[profile_id]
    assert profile["password_scheme"] == auth.PASSWORD_SCHEME
    assert profile["password_iterations"] == auth.PBKDF2_ITERATIONS
    assert profile["password_hash"] != "correct-horse"
    assert auth.authenticate_user("demo user", "correct-horse") == (True, profile_id)
    assert auth.authenticate_user("Demo User", "wrong-password") == (False, None)


def test_short_password_is_rejected(isolated_profile_store):
    success, message = auth.register_user("Demo User", "short")

    assert success is False
    assert "at least 8" in message
    assert not isolated_profile_store.exists()


def test_legacy_hash_is_upgraded_after_successful_login(isolated_profile_store):
    salt = "00112233445566778899aabbccddeeff"
    password = "legacy-password"
    profile = user_profile._default_profile("Legacy User", "legacy-user")
    profile["salt"] = salt
    profile["password_hash"] = hashlib.sha256(
        (password + salt).encode("utf-8")
    ).hexdigest()
    user_profile.save_profiles({profile["id"]: profile})

    assert auth.authenticate_user("Legacy User", password) == (True, "legacy-user")
    migrated = user_profile.load_profiles()["legacy-user"]
    assert migrated["password_scheme"] == auth.PASSWORD_SCHEME
    assert migrated["password_hash"] != profile["password_hash"]


def test_profile_writes_replace_valid_json_atomically(isolated_profile_store):
    first = {"one": user_profile._default_profile("One", "one")}
    second = {"two": user_profile._default_profile("Two", "two")}

    user_profile.save_profiles(first)
    user_profile.save_profiles(second)

    with isolated_profile_store.open(encoding="utf-8") as file:
        assert json.load(file) == second
    assert list(isolated_profile_store.parent.glob("*.tmp")) == []
