# utils/auth_utils.py
import time
from typing import Dict, Any, Optional

# --- Simulated TinyAuth Database (In-memory for demonstration) ---
# In a real TinyAuth setup, this data would be persisted in their database.
# Passwords are NOT hashed here for simplicity, but MUST be hashed in production.
_users_db = {
    "adminuser": {"password": "adminpassword", "role": "admin"},
    "annotator1": {"password": "annotatorpassword", "role": "annotator"},
    "viewer1": {"password": "viewerpassword", "role": "viewer"}
}

# --- Dummy Token Generation (Insecure for demonstration) ---
# In a real system, this would be a signed JWT or a secure session token.
def _generate_dummy_token(username: str, role: str) -> str:
    """Generates a simple dummy token for demonstration."""
    return f"dummy_token_{username}_{role}_{int(time.time())}"

# --- Simulated TinyAuth API Functions ---

def register_user(username: str, password: str, role: str) -> Optional[Dict[str, Any]]:
    """Simulates user registration with TinyAuth."""
    if username in _users_db:
        return None  # User already exists
    
    _users_db[username] = {"password": password, "role": role}
    access_token = _generate_dummy_token(username, role) # Changed 'token' to 'access_token'
    print(f"Simulated TinyAuth: Registered user '{username}' with role '{role}'")
    return {"username": username, "role": role, "access_token": access_token} # Changed 'token' to 'access_token'

def login_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    """Simulates user login with TinyAuth."""
    user_data = _users_db.get(username)
    if user_data and user_data["password"] == password:
        access_token = _generate_dummy_token(username, user_data["role"]) # Changed 'token' to 'access_token'
        print(f"Simulated TinyAuth: User '{username}' logged in with role '{user_data['role']}'")
        return {"username": username, "role": user_data["role"], "access_token": access_token} # Changed 'token' to 'access_token'
    return None # Invalid credentials

def get_user_from_token(token: str) -> Optional[Dict[str, Any]]:
    """Simulates token validation and user/role retrieval from TinyAuth."""
    if not token.startswith("dummy_token_"):
        return None # Invalid dummy token format

    parts = token.split('_')
    if len(parts) < 4:
        return None # Malformed dummy token

    username = parts[2]
    role = parts[3]

    # In a real system, you'd verify the token's signature, expiration, etc.
    # Here, we just check if the user exists in our dummy DB with that role.
    user_data = _users_db.get(username)
    if user_data and user_data["role"] == role:
        return {"username": username, "role": role}
    return None

