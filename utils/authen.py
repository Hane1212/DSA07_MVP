from fastapi import HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer
from database.models import UserPayload

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")
# --- Authentication Dependencies ---
# def get_current_user(token: str = Depends(oauth2_scheme)) -> UserPayload:
#     """Authenticates user based on token and returns user payload."""
#     user = auth_utils.get_user_from_token(token) # Simulate TinyAuth token validation
#     if not user:
#         raise HTTPException(
#             status_code=status.HTTP_401_UNAUTHORIZED,
#             detail="Invalid authentication credentials",
#             headers={"WWW-Authenticate": "Bearer"},
#         )
#     return UserPayload(**user)

# def get_admin_user(current_user: UserPayload = Depends(get_current_user)):
#     """Ensures the current user has 'admin' role."""
#     if current_user.role != "admin":
#         raise HTTPException(
#             status_code=status.HTTP_403_FORBIDDEN,
#             detail="Not enough permissions: Admin role required",
#         )
#     return current_user

# def get_annotator_user(current_user: UserPayload = Depends(get_current_user)):
#     """Ensures the current user has 'annotator' or 'admin' role."""
#     if current_user.role not in ["annotator", "admin"]:
#         raise HTTPException(
#             status_code=status.HTTP_403_FORBIDDEN,
#             detail="Not enough permissions: Annotator or Admin role required",
#         )
#     return current_user

# def get_viewer_user(current_user: UserPayload = Depends(get_current_user)):
#     """Ensures the current user has 'viewer', 'annotator', or 'admin' role."""
#     if current_user.role not in ["viewer", "annotator", "admin"]:
#         raise HTTPException(
#             status_code=status.HTTP_403_FORBIDDEN,
#             detail="Not enough permissions: Viewer, Annotator, or Admin role required",
#         )
#     return current_user