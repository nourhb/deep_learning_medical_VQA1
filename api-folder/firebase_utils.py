import firebase_admin
from firebase_admin import credentials, storage
import os
from datetime import datetime
import uuid
from typing import Optional, Dict, Any
import logging
from config import FIREBASE_CONFIG

logger = logging.getLogger(__name__)

class FirebaseStorageManager:
    def __init__(self):
        """Initialize Firebase Storage connection."""
        try:
            # Check if credentials file exists
            cred_path = FIREBASE_CONFIG["credentials_path"]
            if not os.path.exists(cred_path):
                logger.warning(f"Firebase credentials file not found at {cred_path}")
                logger.info("Firebase Storage will be disabled")
                self.bucket = None
                return

            # Initialize Firebase Admin SDK
            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred, {
                'storageBucket': FIREBASE_CONFIG["storage_bucket"]
            })
            self.bucket = storage.bucket()
            logger.info("Firebase Storage initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Firebase Storage: {str(e)}")
            self.bucket = None

    def upload_image(self, image_path: str, user_id: str) -> Dict[str, Any]:
        """
        Upload an image to Firebase Storage.
        
        Args:
            image_path: Path to the local image file
            user_id: ID of the user uploading the image
            
        Returns:
            Dict containing upload metadata
        """
        if self.bucket is None:
            logger.warning("Firebase Storage is not initialized")
            return {
                "filename": os.path.basename(image_path),
                "url": None,
                "uploaded_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "user_id": user_id,
                "original_path": image_path
            }

        try:
            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())
            filename = f"images/{user_id}/{timestamp}_{unique_id}.jpg"
            
            # Upload file
            blob = self.bucket.blob(filename)
            blob.upload_from_filename(image_path)
            
            # Make the file publicly accessible
            blob.make_public()
            
            # Get the public URL
            public_url = blob.public_url
            
            # Create metadata
            metadata = {
                "filename": filename,
                "url": public_url,
                "uploaded_at": timestamp,
                "user_id": user_id,
                "original_path": image_path
            }
            
            logger.info(f"Image uploaded successfully: {filename}")
            return metadata
            
        except Exception as e:
            logger.error(f"Error uploading image: {str(e)}")
            return {
                "filename": os.path.basename(image_path),
                "url": None,
                "uploaded_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "user_id": user_id,
                "original_path": image_path
            }

    def get_image_url(self, filename: str) -> Optional[str]:
        """
        Get the public URL of an uploaded image.
        
        Args:
            filename: Name of the file in Firebase Storage
            
        Returns:
            Public URL of the image or None if not found
        """
        if self.bucket is None:
            logger.warning("Firebase Storage is not initialized")
            return None

        try:
            blob = self.bucket.blob(filename)
            if not blob.exists():
                logger.warning(f"Image not found: {filename}")
                return None
            return blob.public_url
        except Exception as e:
            logger.error(f"Error getting image URL: {str(e)}")
            return None

    def delete_image(self, filename: str) -> bool:
        """
        Delete an image from Firebase Storage.
        
        Args:
            filename: Name of the file to delete
            
        Returns:
            True if deletion was successful, False otherwise
        """
        if self.bucket is None:
            logger.warning("Firebase Storage is not initialized")
            return False

        try:
            blob = self.bucket.blob(filename)
            if blob.exists():
                blob.delete()
                logger.info(f"Image deleted successfully: {filename}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting image: {str(e)}")
            return False

    def list_user_images(self, user_id: str) -> list:
        """
        List all images uploaded by a specific user.
        
        Args:
            user_id: ID of the user
            
        Returns:
            List of image metadata
        """
        if self.bucket is None:
            logger.warning("Firebase Storage is not initialized")
            return []

        try:
            prefix = f"images/{user_id}/"
            blobs = self.bucket.list_blobs(prefix=prefix)
            
            images = []
            for blob in blobs:
                metadata = {
                    "filename": blob.name,
                    "url": blob.public_url,
                    "uploaded_at": blob.metadata.get("uploaded_at", ""),
                    "size": blob.size,
                    "content_type": blob.content_type
                }
                images.append(metadata)
            
            return images
        except Exception as e:
            logger.error(f"Error listing user images: {str(e)}")
            return [] 