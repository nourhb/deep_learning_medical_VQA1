import logging
import time
from functools import wraps
from typing import Any, Callable, Dict, List, Optional
import numpy as np
from PIL import Image
import tensorflow as tf
from config import LOGGING_CONFIG, SECURITY_CONFIG

# Configure logging
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

class ModelCache:
    """Cache for model predictions to improve performance."""
    def __init__(self, max_size: int = 100):
        self.cache: Dict[str, Any] = {}
        self.max_size = max_size
        self.access_times: Dict[str, float] = {}

    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            self.access_times[key] = time.time()
            return self.cache[key]
        return None

    def set(self, key: str, value: Any) -> None:
        if len(self.cache) >= self.max_size:
            # Remove least recently used item
            lru_key = min(self.access_times.items(), key=lambda x: x[1])[0]
            del self.cache[lru_key]
            del self.access_times[lru_key]
        self.cache[key] = value
        self.access_times[key] = time.time()

def timing_decorator(func: Callable) -> Callable:
    """Decorator to measure function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"{func.__name__} took {end_time - start_time:.2f} seconds to execute")
        return result
    return wrapper

def validate_image(image_path: str) -> bool:
    """Validate if the image file is valid and can be processed."""
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception as e:
        logger.error(f"Invalid image file {image_path}: {str(e)}")
        return False

def preprocess_image(image_path: str, target_size: tuple = (224, 224)) -> np.ndarray:
    """Preprocess image for model input."""
    try:
        with Image.open(image_path) as img:
            img = img.resize(target_size)
            img_array = np.array(img) / 255.0
            return img_array
    except Exception as e:
        logger.error(f"Error preprocessing image {image_path}: {str(e)}")
        raise

def rate_limit_decorator(func: Callable) -> Callable:
    """Decorator to implement rate limiting."""
    request_counts: Dict[str, List[float]] = {}
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        client_ip = kwargs.get('client_ip', 'default')
        current_time = time.time()
        
        # Clean old requests
        if client_ip in request_counts:
            request_counts[client_ip] = [
                t for t in request_counts[client_ip]
                if current_time - t < SECURITY_CONFIG['rate_limit']['period']
            ]
        
        # Check rate limit
        if client_ip in request_counts and len(request_counts[client_ip]) >= SECURITY_CONFIG['rate_limit']['requests']:
            raise Exception("Rate limit exceeded")
        
        # Add current request
        if client_ip not in request_counts:
            request_counts[client_ip] = []
        request_counts[client_ip].append(current_time)
        
        return func(*args, **kwargs)
    return wrapper

def error_handler(func: Callable) -> Callable:
    """Decorator to handle errors and provide consistent error responses."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            return {
                "error": True,
                "message": str(e),
                "status": "error"
            }
    return wrapper

def validate_model_input(data: Dict[str, Any]) -> bool:
    """Validate model input data."""
    required_fields = ['image_path', 'question']
    return all(field in data for field in required_fields)

def get_model_metrics(model: tf.keras.Model) -> Dict[str, float]:
    """Get model performance metrics."""
    return {
        "parameters": model.count_params(),
        "layers": len(model.layers),
        "trainable_parameters": sum(
            np.prod(w.shape) for w in model.trainable_weights
        )
    } 