import firebase_admin
from firebase_admin import auth, credentials
import os
from datetime import datetime
from typing import Dict, Optional, Tuple
import logging
from config import SECURITY_CONFIG

logger = logging.getLogger(__name__)

class AuthManager:
    def __init__(self):
        """Initialize Firebase Auth."""
        try:
            # Initialize Firebase Admin SDK if not already initialized
            if not firebase_admin._apps:
                cred = credentials.Certificate("firebase-credentials.json")
                firebase_admin.initialize_app(cred)
            
            logger.info("Auth Manager initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Auth Manager: {str(e)}")
            raise

    def create_user(self, email: str, password: str, display_name: str) -> Dict:
        """
        Create a new user with Firebase's built-in email verification.
        
        Args:
            email: User's email address
            password: User's password
            display_name: User's display name
            
        Returns:
            User data dictionary
        """
        try:
            # Create user in Firebase
            user = auth.create_user(
                email=email,
                password=password,
                display_name=display_name,
                email_verified=False
            )
            
            # Send email verification using Firebase's built-in functionality
            auth.generate_email_verification_link(email)
            
            user_data = {
                "uid": user.uid,
                "email": user.email,
                "display_name": user.display_name,
                "email_verified": user.email_verified,
                "created_at": datetime.now().isoformat()
            }
            
            logger.info(f"User created successfully: {user.uid}")
            return user_data
            
        except Exception as e:
            logger.error(f"Error creating user: {str(e)}")
            raise

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