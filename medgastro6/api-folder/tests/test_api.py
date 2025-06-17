import pytest
from app import app
import json
import os
from PIL import Image
import numpy as np

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_health_check(client):
    """Test health check endpoint"""
    response = client.get('/api/v1/health')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'healthy'
    assert 'model_loaded' in data
    assert 'cache_size' in data

def test_predict_invalid_input(client):
    """Test predict endpoint with invalid input"""
    response = client.post('/api/v1/predict',
                          json={})
    assert response.status_code == 400
    data = json.loads(response.data)
    assert data['error'] is True
    assert 'message' in data

def test_predict_valid_input(client, tmp_path):
    """Test predict endpoint with valid input"""
    # Create a test image
    img = Image.new('RGB', (224, 224), color='red')
    img_path = tmp_path / "test_image.jpg"
    img.save(img_path)

    # Test prediction
    response = client.post('/api/v1/predict',
                          json={
                              'image_path': str(img_path),
                              'question': 'What color is this image?'
                          })
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'prediction' in data
    assert data['error'] is False

def test_metrics_endpoint(client):
    """Test metrics endpoint"""
    response = client.get('/api/v1/metrics')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'metrics' in data
    assert 'model' in data['metrics']
    assert 'cache' in data['metrics']

def test_rate_limiting(client, tmp_path):
    """Test rate limiting"""
    # Create a test image
    img = Image.new('RGB', (224, 224), color='red')
    img_path = tmp_path / "test_image.jpg"
    img.save(img_path)

    # Make multiple requests
    for _ in range(101):  # Assuming rate limit is 100
        response = client.post('/api/v1/predict',
                             json={
                                 'image_path': str(img_path),
                                 'question': 'What color is this image?'
                             })

    # The last request should be rate limited
    assert response.status_code == 429
    data = json.loads(response.data)
    assert data['error'] is True
    assert 'rate limit' in data['message'].lower() 