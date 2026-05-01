import hashlib
import os
from utils.user_profile import (
    _default_profile,
    _slugify,
    find_profile_by_name,
    get_profile,
    load_profiles,
    save_profiles,
    upsert_profile,
)

def hash_password(password, salt=None):
    """Securely hash a password with a salt."""
    if salt is None:
        salt = os.urandom(16).hex()
    
    salted_password = password + salt
    hashed = hashlib.sha256(salted_password.encode('utf-8')).hexdigest()
    return hashed, salt

def authenticate_user(username, password):
    """Authenticate a user by checking their username and password."""
    profile = find_profile_by_name(username)
    if not profile:
        return False, None
    
    stored_hash = profile.get("password_hash")
    stored_salt = profile.get("salt")
    
    if not stored_hash:
        return False, None
        
    hashed, _ = hash_password(password, stored_salt)
    if hashed == stored_hash:
        return True, profile["id"]
    return False, None

def register_user(username, password):
    """Register a new user with a hashed password."""
    username = username.strip()
    if not username:
        return False, "Username cannot be empty."
    if not password:
        return False, "Password cannot be empty."
        
    if find_profile_by_name(username):
        return False, "Username already exists. Please choose a different name or log in."
        
    hashed_pwd, salt = hash_password(password)
    
    profile_id = _slugify(username)
    new_profile = _default_profile(name=username, profile_id=profile_id)
    new_profile["password_hash"] = hashed_pwd
    new_profile["salt"] = salt
    
    upsert_profile(new_profile)
    return True, profile_id


def change_password(profile_id, current_password, new_password):
    """Change the password for an existing authenticated profile."""
    profile = get_profile(profile_id=profile_id)
    if not profile:
        return False, "Profile not found."
    if not new_password:
        return False, "New password cannot be empty."

    stored_hash = profile.get("password_hash")
    stored_salt = profile.get("salt")
    if stored_hash:
        hashed_current, _ = hash_password(current_password or "", stored_salt)
        if hashed_current != stored_hash:
            return False, "Current password is incorrect."

    hashed_pwd, salt = hash_password(new_password)
    profile["password_hash"] = hashed_pwd
    profile["salt"] = salt
    upsert_profile(profile)
    return True, "Password updated."


def delete_user_account(profile_id, current_password):
    """Delete a user account after password confirmation."""
    profiles = load_profiles()
    if profile_id not in profiles:
        return False, "Profile not found."

    profile = profiles[profile_id]
    stored_hash = profile.get("password_hash")
    stored_salt = profile.get("salt")
    if stored_hash:
        hashed_current, _ = hash_password(current_password or "", stored_salt)
        if hashed_current != stored_hash:
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
        signup_password = st.text_input("Choose a Password", type="password", key="page_signup_password")
        if st.button("Sign Up", width="stretch", key="page_signup_btn"):
            success, result = register_user(signup_username, signup_password)
            if success:
                st.success("Account created! Logging you in...")
                st.session_state["authenticated_profile_id"] = result
                st.rerun()
            else:
                st.error(result)
    
    st.stop()
