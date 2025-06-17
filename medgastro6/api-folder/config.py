import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "saved_models"
DATASET_DIR = BASE_DIR / "dataset_chunks"
CACHE_DIR = BASE_DIR / "dataset_cache"

# API Configuration
API_VERSION = "v1"
API_PREFIX = f"/api/{API_VERSION}"
HOST = "0.0.0.0"
PORT = 5000
DEBUG = False

# Model Configuration
MODEL_CONFIG = {
    "default_model": "model_weights.h5",
    "model_dir": str(MODELS_DIR),
    "cache_size": 100,  # Number of images to cache
    "batch_size": 32,
    "image_size": (224, 224),
}

# Dataset Configuration
DATASET_CONFIG = {
    "images_dir": str(DATASET_DIR / "images"),
    "questions_dir": str(DATASET_DIR / "questions"),
    "answers_dir": str(DATASET_DIR / "answers"),
    "cache_dir": str(CACHE_DIR),
}

# Security Configuration
SECURITY_CONFIG = {
    "frontend_url": os.getenv("FRONTEND_URL", "http://localhost:3000"),
    "jwt_secret": os.getenv("JWT_SECRET", "your-secret-key"),
    "jwt_algorithm": "HS256",
    "jwt_expiration": 7 * 24 * 60 * 60,  # 7 days in seconds
    "email_verification_expiration": 24 * 60 * 60,  # 24 hours in seconds
}

# Email Configuration
EMAIL_CONFIG = {
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587,
    "smtp_username": os.getenv("SMTP_USERNAME"),
    "smtp_password": os.getenv("SMTP_PASSWORD"),
    "sender_email": os.getenv("SENDER_EMAIL"),
}

# Firebase Configuration
FIREBASE_CONFIG = {
    "credentials_path": "firebase-credentials.json",
    "storage_bucket": os.getenv("FIREBASE_STORAGE_BUCKET"),
}

# API Configuration
API_CONFIG = {
    "host": os.getenv("API_HOST", "0.0.0.0"),
    "port": int(os.getenv("API_PORT", "5000")),
    "debug": os.getenv("API_DEBUG", "False").lower() == "true",
}

# Logging Configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "stream": "ext://sys.stdout"
        },
        "file": {
            "class": "logging.FileHandler",
            "formatter": "default",
            "filename": str(BASE_DIR / "logs" / "app.log")
        }
    },
    "root": {
        "level": "INFO",
        "handlers": ["console", "file"]
    }
}

# Create necessary directories
for directory in [MODELS_DIR, DATASET_DIR, CACHE_DIR, BASE_DIR / "logs"]:
    directory.mkdir(parents=True, exist_ok=True) 