import firebase_admin
from firebase_admin import credentials, auth
import os
from datetime import datetime
from typing import Dict, Optional, Tuple
import logging
from config import FIREBASE_CONFIG

logger = logging.getLogger(__name__)

class AuthManager:
    def __init__(self):
        """Initialize Firebase Auth connection."""
        try:
            # Check if credentials file exists
            cred_path = FIREBASE_CONFIG["credentials_path"]
            if not os.path.exists(cred_path):
                logger.warning(f"Firebase credentials file not found at {cred_path}")
                logger.info("Firebase Auth will be disabled")
                self.auth = None
                return

            # Initialize Firebase Admin SDK if not already initialized
            if not firebase_admin._apps:
                cred = credentials.Certificate(cred_path)
                firebase_admin.initialize_app(cred)
            self.auth = auth
            logger.info("Firebase Auth initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Auth Manager: {str(e)}")
            self.auth = None

    def verify_token(self, token: str) -> dict:
        """
        Verify Firebase ID token.
        
        Args:
            token: Firebase ID token
            
        Returns:
            Decoded token claims or None if verification fails
        """
        if self.auth is None:
            logger.warning("Firebase Auth is not initialized")
            return None

        try:
            decoded_token = self.auth.verify_id_token(token)
            return decoded_token
        except Exception as e:
            logger.error(f"Error verifying token: {str(e)}")
            return None

    def create_user(self, email: str, password: str) -> dict:
        """
        Create a new user in Firebase Auth.
        
        Args:
            email: User's email
            password: User's password
            
        Returns:
            User record or None if creation fails
        """
        if self.auth is None:
            logger.warning("Firebase Auth is not initialized")
            return None

        try:
            user = self.auth.create_user(
                email=email,
                password=password,
                email_verified=False
            )
            return user
        except Exception as e:
            logger.error(f"Error creating user: {str(e)}")
            return None

    def delete_user(self, uid: str) -> bool:
        """
        Delete a user from Firebase Auth.
        
        Args:
            uid: User ID
            
        Returns:
            True if deletion was successful, False otherwise
        """
        if self.auth is None:
            logger.warning("Firebase Auth is not initialized")
            return False

        try:
            self.auth.delete_user(uid)
            return True
        except Exception as e:
            logger.error(f"Error deleting user: {str(e)}")
            return False

    def update_user(self, uid: str, **kwargs) -> dict:
        """
        Update user properties in Firebase Auth.
        
        Args:
            uid: User ID
            **kwargs: User properties to update
            
        Returns:
            Updated user record or None if update fails
        """
        if self.auth is None:
            logger.warning("Firebase Auth is not initialized")
            return None

        try:
            user = self.auth.update_user(uid, **kwargs)
            return user
        except Exception as e:
            logger.error(f"Error updating user: {str(e)}")
            return None

    def send_password_reset_email(self, email: str) -> bool:
        """
        Send password reset email using Firebase's built-in functionality.
        
        Args:
            email: User's email address
            
        Returns:
            True if email sent successfully, False otherwise
        """
        try:
            # Generate password reset link using Firebase
            auth.generate_password_reset_link(email)
            logger.info(f"Password reset email sent to: {email}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending password reset email: {str(e)}")
            return False

    def verify_email(self, oob_code: str) -> bool:
        """
        Verify user's email using Firebase's OOB code.
        
        Args:
            oob_code: Firebase's out-of-band code from email verification link
            
        Returns:
            True if verification successful, False otherwise
        """
        try:
            # Verify email using Firebase's OOB code
            auth.verify_email(oob_code)
            logger.info("Email verified successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error verifying email: {str(e)}")
            return False

    def login(self, email: str, password: str) -> Optional[Dict]:
        """
        Login user using Firebase Authentication.
        
        Args:
            email: User's email
            password: User's password
            
        Returns:
            User data if successful, None otherwise
        """
        try:
            # Get user by email
            user = auth.get_user_by_email(email)
            
            user_data = {
                "uid": user.uid,
                "email": user.email,
                "display_name": user.display_name,
                "email_verified": user.email_verified
            }
            
            logger.info(f"User logged in successfully: {user.uid}")
            return user_data
            
        except Exception as e:
            logger.error(f"Error logging in: {str(e)}")
            return None

    def reset_password(self, oob_code: str, new_password: str) -> bool:
        """
        Reset user's password using Firebase's OOB code.
        
        Args:
            oob_code: Firebase's out-of-band code from password reset link
            new_password: New password to set
            
        Returns:
            True if password reset successful, False otherwise
        """
        try:
            # Reset password using Firebase's OOB code
            auth.confirm_password_reset(oob_code, new_password)
            logger.info("Password reset successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error resetting password: {str(e)}")
            return False 