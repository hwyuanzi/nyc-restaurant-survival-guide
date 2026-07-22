import hashlib
import hmac
import os
from utils.user_profile import (
    PROFILE_STORE_LOCK,
    _default_profile,
    _slugify,
    find_profile_by_name,
    load_profiles,
    save_profiles,
    upsert_profile,
)

PASSWORD_SCHEME = "pbkdf2_sha256"
PBKDF2_ITERATIONS = 600_000
MIN_PASSWORD_LENGTH = 8


def hash_password(password, salt=None, iterations=PBKDF2_ITERATIONS):
    """Hash a password with salted PBKDF2-HMAC-SHA256."""
    if salt is None:
        salt = os.urandom(16).hex()

    salt_bytes = bytes.fromhex(salt)
    hashed = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt_bytes,
        int(iterations),
    ).hex()
    return hashed, salt


def _legacy_hash_password(password, salt):
    """Reproduce hashes created by older releases for login migration."""
    return hashlib.sha256((password + salt).encode("utf-8")).hexdigest()


def _password_is_valid(profile, password):
    stored_hash = profile.get("password_hash")
    stored_salt = profile.get("salt")
    if not stored_hash or not stored_salt:
        return False

    if profile.get("password_scheme") == PASSWORD_SCHEME:
        try:
            iterations = int(profile.get("password_iterations", PBKDF2_ITERATIONS))
            candidate_hash, _ = hash_password(password, stored_salt, iterations)
        except (TypeError, ValueError):
            return False
    else:
        candidate_hash = _legacy_hash_password(password, stored_salt)

    return hmac.compare_digest(candidate_hash, stored_hash)


def _set_password(profile, password):
    hashed_password, salt = hash_password(password)
    profile["password_hash"] = hashed_password
    profile["salt"] = salt
    profile["password_scheme"] = PASSWORD_SCHEME
    profile["password_iterations"] = PBKDF2_ITERATIONS


def _validate_new_password(password):
    if len(password or "") < MIN_PASSWORD_LENGTH:
        return False, f"Password must be at least {MIN_PASSWORD_LENGTH} characters."
    return True, ""


def authenticate_user(username, password):
    """Authenticate a user by checking their username and password."""
    with PROFILE_STORE_LOCK:
        profile = find_profile_by_name(username)
        if not profile or not _password_is_valid(profile, password):
            return False, None

        # Existing local users keep working. A successful login transparently
        # upgrades the earlier single-SHA256 format to PBKDF2.
        if profile.get("password_scheme") != PASSWORD_SCHEME:
            _set_password(profile, password)
            upsert_profile(profile)
        return True, profile["id"]


def register_user(username, password):
    """Register a new user with a hashed password."""
    username = username.strip()
    if not username:
        return False, "Username cannot be empty."
    password_ok, message = _validate_new_password(password)
    if not password_ok:
        return False, message

    with PROFILE_STORE_LOCK:
        if find_profile_by_name(username):
            return False, "Username already exists. Please choose a different name or log in."

        profiles = load_profiles()
        base_profile_id = _slugify(username)
        profile_id = base_profile_id
        suffix = 2
        while profile_id in profiles:
            profile_id = f"{base_profile_id}-{suffix}"
            suffix += 1

        new_profile = _default_profile(name=username, profile_id=profile_id)
        _set_password(new_profile, password)
        upsert_profile(new_profile)
        return True, profile_id


def change_password(profile_id, current_password, new_password):
    """Change the password for an existing authenticated profile."""
    password_ok, message = _validate_new_password(new_password)
    if not password_ok:
        return False, message

    with PROFILE_STORE_LOCK:
        profiles = load_profiles()
        profile = profiles.get(profile_id)
        if not profile:
            return False, "Profile not found."
        if profile.get("password_hash") and not _password_is_valid(profile, current_password or ""):
            return False, "Current password is incorrect."

        _set_password(profile, new_password)
        upsert_profile(profile)
        return True, "Password updated."


def delete_user_account(profile_id, current_password):
    """Delete a user account after password confirmation."""
    with PROFILE_STORE_LOCK:
        profiles = load_profiles()
        if profile_id not in profiles:
            return False, "Profile not found."

        profile = profiles[profile_id]
        if profile.get("password_hash") and not _password_is_valid(profile, current_password or ""):
            return False, "Current password is incorrect."

        del profiles[profile_id]
        save_profiles(profiles)
        return True, "Profile deleted."


def require_auth():
    """Call at the top of any Streamlit page to enforce authentication.
    
    If the user is not authenticated, renders a Login / Sign Up form
    and calls st.stop() to prevent the rest of the page from running.
    Returns the authenticated profile_id if successful.
    """
    import streamlit as st
    
    if "authenticated_profile_id" not in st.session_state:
        st.session_state["authenticated_profile_id"] = None
    
    if st.session_state["authenticated_profile_id"]:
        return st.session_state["authenticated_profile_id"]
    
    st.title("🔒 Login Required")
    st.markdown("Please log in or create an account to access this page.")
    
    tab_login, tab_signup = st.tabs(["Login", "Sign Up"])
    
    with tab_login:
        login_username = st.text_input("Username", key="page_login_username")
        login_password = st.text_input("Password", type="password", key="page_login_password")
        if st.button("Login", width="stretch", key="page_login_btn"):
            success, profile_id = authenticate_user(login_username, login_password)
            if success:
                st.session_state["authenticated_profile_id"] = profile_id
                st.rerun()
            else:
                st.error("Invalid username or password.")
                
    with tab_signup:
        signup_username = st.text_input("Choose a Username", key="page_signup_username")
        signup_password = st.text_input(
            "Choose a Password",
            type="password",
            key="page_signup_password",
            help=f"Use at least {MIN_PASSWORD_LENGTH} characters.",
        )
        if st.button("Sign Up", width="stretch", key="page_signup_btn"):
            success, result = register_user(signup_username, signup_password)
            if success:
                st.success("Account created! Logging you in...")
                st.session_state["authenticated_profile_id"] = result
                st.rerun()
            else:
                st.error(result)
    
    st.stop()
