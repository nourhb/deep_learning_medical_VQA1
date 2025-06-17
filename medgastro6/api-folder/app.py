from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_swagger_ui import get_swaggerui_blueprint
import tensorflow as tf
import numpy as np
from config import (
    API_PREFIX, HOST, PORT, DEBUG,
    MODEL_CONFIG, SECURITY_CONFIG, API_CONFIG
)
from utils import (
    ModelCache, timing_decorator, validate_image,
    preprocess_image, rate_limit_decorator,
    error_handler, validate_model_input,
    get_model_metrics
)
from model_manager import ModelManager
from firebase_utils import FirebaseStorageManager
import logging
import os
from auth_utils import AuthManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": SECURITY_CONFIG["cors_origins"]}})

# Initialize managers
model_manager = ModelManager()
model_cache = ModelCache(max_size=MODEL_CONFIG["cache_size"])
firebase_storage = FirebaseStorageManager()
auth_manager = AuthManager()

# Load default model
try:
    model_manager.load_model(MODEL_CONFIG["default_model"])
    logger.info("Default model loaded successfully")
except Exception as e:
    logger.error(f"Error loading default model: {str(e)}")
    raise

# Swagger configuration
SWAGGER_URL = '/api/docs'
API_URL = '/static/swagger.json'

swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "Medical VQA API"
    }
)
app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

@app.route(f"{API_PREFIX}/health", methods=["GET"])
def health_check():
    """
    Health check endpoint
    ---
    responses:
      200:
        description: System health status
        schema:
          type: object
          properties:
            status:
              type: string
              example: healthy
            model_loaded:
              type: boolean
              example: true
            cache_size:
              type: integer
              example: 0
    """
    return jsonify({
        "status": "healthy",
        "model_loaded": model_manager.get_current_model() is not None,
        "cache_size": len(model_cache.cache)
    })

@app.route(f"{API_PREFIX}/predict", methods=["POST"])
@error_handler
@rate_limit_decorator
@timing_decorator
def predict():
    """
    Predict endpoint
    ---
    parameters:
      - in: body
        name: body
        schema:
          type: object
          required:
            - image_path
            - question
            - user_id
          properties:
            image_path:
              type: string
              description: Path to the image file
            question:
              type: string
              description: Question about the image
            user_id:
              type: string
              description: ID of the user making the request
    responses:
      200:
        description: Prediction result
        schema:
          type: object
          properties:
            error:
              type: boolean
            prediction:
              type: array
            status:
              type: string
            cached:
              type: boolean
            image_url:
              type: string
    """
    if not request.is_json:
        return jsonify({
            "error": True,
            "message": "Request must be JSON",
            "status": "error"
        }), 400

    data = request.get_json()
    
    if not validate_model_input(data):
        return jsonify({
            "error": True,
            "message": "Missing required fields",
            "status": "error"
        }), 400

    image_path = data["image_path"]
    question = data["question"]
    user_id = data.get("user_id", "anonymous")

    if not validate_image(image_path):
        return jsonify({
            "error": True,
            "message": "Invalid image file",
            "status": "error"
        }), 400

    # Upload image to Firebase Storage
    try:
        image_metadata = firebase_storage.upload_image(image_path, user_id)
        image_url = image_metadata["url"]
    except Exception as e:
        logger.error(f"Error uploading image to Firebase: {str(e)}")
        image_url = None

    cache_key = f"{image_path}_{question}"
    cached_result = model_cache.get(cache_key)
    if cached_result is not None:
        logger.info("Cache hit")
        return jsonify({
            "error": False,
            "prediction": cached_result,
            "status": "success",
            "cached": True,
            "image_url": image_url
        })

    try:
        processed_image = preprocess_image(
            image_path,
            target_size=MODEL_CONFIG["image_size"]
        )
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        return jsonify({
            "error": True,
            "message": "Error processing image",
            "status": "error"
        }), 500

    try:
        model = model_manager.get_current_model()
        if model is None:
            raise Exception("No model loaded")

        prediction = model.predict(
            np.expand_dims(processed_image, axis=0),
            batch_size=MODEL_CONFIG["batch_size"]
        )
        
        model_cache.set(cache_key, prediction.tolist())
        
        return jsonify({
            "error": False,
            "prediction": prediction.tolist(),
            "status": "success",
            "cached": False,
            "image_url": image_url
        })
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        return jsonify({
            "error": True,
            "message": "Error making prediction",
            "status": "error"
        }), 500

@app.route(f"{API_PREFIX}/images/<user_id>", methods=["GET"])
@error_handler
def get_user_images(user_id):
    """
    Get all images uploaded by a user
    ---
    parameters:
      - name: user_id
        in: path
        required: true
        type: string
    responses:
      200:
        description: List of user's images
        schema:
          type: object
          properties:
            error:
              type: boolean
            images:
              type: array
              items:
                type: object
            status:
              type: string
    """
    try:
        images = firebase_storage.list_user_images(user_id)
        return jsonify({
            "error": False,
            "images": images,
            "status": "success"
        })
    except Exception as e:
        logger.error(f"Error getting user images: {str(e)}")
        return jsonify({
            "error": True,
            "message": "Error retrieving images",
            "status": "error"
        }), 500

@app.route(f"{API_PREFIX}/images/<user_id>/<filename>", methods=["DELETE"])
@error_handler
def delete_image(user_id, filename):
    """
    Delete a specific image
    ---
    parameters:
      - name: user_id
        in: path
        required: true
        type: string
      - name: filename
        in: path
        required: true
        type: string
    responses:
      200:
        description: Deletion status
        schema:
          type: object
          properties:
            error:
              type: boolean
            status:
              type: string
    """
    try:
        full_filename = f"images/{user_id}/{filename}"
        success = firebase_storage.delete_image(full_filename)
        if success:
            return jsonify({
                "error": False,
                "status": "success"
            })
        else:
            return jsonify({
                "error": True,
                "message": "Image not found",
                "status": "error"
            }), 404
    except Exception as e:
        logger.error(f"Error deleting image: {str(e)}")
        return jsonify({
            "error": True,
            "message": "Error deleting image",
            "status": "error"
        }), 500

@app.route(f"{API_PREFIX}/metrics", methods=["GET"])
@error_handler
def get_metrics():
    """
    Get system metrics
    ---
    responses:
      200:
        description: System metrics
        schema:
          type: object
          properties:
            error:
              type: boolean
            metrics:
              type: object
              properties:
                model:
                  type: object
                cache:
                  type: object
            status:
              type: string
    """
    current_model = model_manager.get_current_model()
    if current_model is None:
        return jsonify({
            "error": True,
            "message": "No model loaded",
            "status": "error"
        }), 500

    return jsonify({
        "error": False,
        "metrics": {
            "model": get_model_metrics(current_model),
            "cache": {
                "size": len(model_cache.cache),
                "max_size": model_cache.max_size
            }
        },
        "status": "success"
    })

@app.route(f"{API_PREFIX}/models", methods=["GET"])
@error_handler
def list_models():
    """
    List available model versions
    ---
    responses:
      200:
        description: List of available models
        schema:
          type: object
          properties:
            error:
              type: boolean
            models:
              type: array
              items:
                type: object
            status:
              type: string
    """
    return jsonify({
        "error": False,
        "models": model_manager.get_model_versions(),
        "status": "success"
    })

@app.route('/api/auth/register', methods=['POST'])
def register():
    """Register a new user."""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['email', 'password', 'display_name']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'error': f'Missing required field: {field}'
                }), 400
        
        # Create user
        user_data = auth_manager.create_user(
            email=data['email'],
            password=data['password'],
            display_name=data['display_name']
        )
        
        return jsonify({
            'message': 'User registered successfully. Please check your email for verification.',
            'user': user_data
        }), 201
        
    except Exception as e:
        logger.error(f"Error in register endpoint: {str(e)}")
        return jsonify({
            'error': 'Failed to register user'
        }), 500

@app.route('/api/auth/verify-email', methods=['POST'])
def verify_email():
    """Verify user's email."""
    try:
        data = request.get_json()
        
        if 'oob_code' not in data:
            return jsonify({
                'error': 'Missing verification code'
            }), 400
        
        # Verify email
        if auth_manager.verify_email(data['oob_code']):
            return jsonify({
                'message': 'Email verified successfully'
            }), 200
        else:
            return jsonify({
                'error': 'Invalid or expired verification code'
            }), 400
            
    except Exception as e:
        logger.error(f"Error in verify-email endpoint: {str(e)}")
        return jsonify({
            'error': 'Failed to verify email'
        }), 500

@app.route('/api/auth/login', methods=['POST'])
def login():
    """Login user."""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['email', 'password']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'error': f'Missing required field: {field}'
                }), 400
        
        # Login user
        user_data = auth_manager.login(
            email=data['email'],
            password=data['password']
        )
        
        if user_data:
            return jsonify({
                'message': 'Login successful',
                'user': user_data
            }), 200
        else:
            return jsonify({
                'error': 'Invalid credentials'
            }), 401
            
    except Exception as e:
        logger.error(f"Error in login endpoint: {str(e)}")
        return jsonify({
            'error': 'Failed to login'
        }), 500

@app.route('/api/auth/reset-password', methods=['POST'])
def reset_password():
    """Send password reset email."""
    try:
        data = request.get_json()
        
        if 'email' not in data:
            return jsonify({
                'error': 'Missing email address'
            }), 400
        
        # Send password reset email
        if auth_manager.send_password_reset_email(data['email']):
            return jsonify({
                'message': 'Password reset email sent successfully'
            }), 200
        else:
            return jsonify({
                'error': 'Failed to send password reset email'
            }), 400
            
    except Exception as e:
        logger.error(f"Error in reset-password endpoint: {str(e)}")
        return jsonify({
            'error': 'Failed to send password reset email'
        }), 500

@app.route('/api/auth/confirm-reset', methods=['POST'])
def confirm_reset():
    """Confirm password reset."""
    try:
        data = request.get_json()
        
        required_fields = ['oob_code', 'new_password']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'error': f'Missing required field: {field}'
                }), 400
        
        # Reset password
        if auth_manager.reset_password(data['oob_code'], data['new_password']):
            return jsonify({
                'message': 'Password reset successfully'
            }), 200
        else:
            return jsonify({
                'error': 'Failed to reset password'
            }), 400
            
    except Exception as e:
        logger.error(f"Error in confirm-reset endpoint: {str(e)}")
        return jsonify({
            'error': 'Failed to reset password'
        }), 500

if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=DEBUG)