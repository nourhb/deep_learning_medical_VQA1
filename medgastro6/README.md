# Medical Visual Question Answering (VQA) System

## Overview
This project is a comprehensive Medical Visual Question Answering (VQA) system that allows users to upload medical images and ask questions about them. The system features a PyTorch-based backend with BLIP model, Flask API, and Firebase integration, along with a Flutter web frontend.

### 🚀 **Key Features**
- **Advanced VQA Model**: BLIP (Bootstrapping Language-Image Pre-training) for medical image understanding
- **PyTorch Backend**: Robust, GPU-optimized deep learning pipeline
- **Medical Domain Focus**: Specialized processing for medical terminology and context
- **Real-time Training**: Comprehensive training script with attention visualization
- **Flutter Frontend**: Modern, responsive web interface
- **Firebase Integration**: Secure authentication and storage (optional)

---

## Directory Structure
```
medgastro6/
├── api-folder/           # Python backend (Flask API, PyTorch models)
│   ├── app.py           # Main Flask application
│   ├── model.py         # VQA model implementation
│   ├── model_manager.py # Model loading and management
│   ├── firebase_utils.py # Firebase integration
│   ├── auth_utils.py    # Authentication utilities
│   ├── config.py        # Configuration settings
│   ├── requirements.txt # Python dependencies
│   └── saved_models/    # Trained model weights
├── lib/                 # Flutter frontend (Dart code)
├── web/                 # Flutter web build output
├── train_vqa_cleaned.py # Advanced training script
└── README.md           # This file
```

---

## Quick Start

### 1. **Environment Setup**
```bash
# Clone the repository
git clone <your-repo-url>
cd medgastro6

# Create and activate virtual environment (Python 3.10+ recommended)
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 2. **Install Dependencies**
```bash
# Install Python dependencies
cd api-folder
pip install -r requirements.txt

# Install Flutter dependencies
cd ..
flutter pub get
```

### 3. **Run the System**
```bash
# Terminal 1: Start the backend
cd api-folder
python app.py

# Terminal 2: Start the frontend
cd ..
flutter run -d chrome
```

---

## Backend Setup (Flask API + PyTorch)

### **Dependencies**
The backend uses PyTorch with the following key packages:
- `torch` - Deep learning framework
- `transformers` - Hugging Face transformers (BLIP model)
- `flask` - Web framework
- `firebase-admin` - Firebase integration (optional)
- `PIL` - Image processing
- `numpy` - Numerical computing

### **Model Training**
Use the advanced training script for custom model training:

```bash
# Train a new VQA model
python train_vqa_cleaned.py
```

**Training Features:**
- ✅ **Medical Domain Processing**: Custom medical term extraction
- ✅ **GPU Optimization**: Automatic device detection and memory management
- ✅ **Attention Visualization**: Model interpretability features
- ✅ **Checkpointing**: Automatic model saving during training
- ✅ **Progress Monitoring**: Comprehensive logging and metrics
- ✅ **Error Recovery**: Robust error handling and recovery

### **Configuration**
Update `api-folder/config.py` for your environment:
```python
# API Configuration
API_HOST = "127.0.0.1"
API_PORT = 5000
DEBUG = True

# Model Configuration
MODEL_PATH = "saved_models/your_model.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Firebase Configuration (Optional)
FIREBASE_CONFIG = {
    "project_id": "your-project-id",
    "storage_bucket": "your-bucket.appspot.com"
}
```

### **Firebase Setup (Optional)**
1. Go to [Firebase Console](https://console.firebase.google.com/)
2. Create a project
3. Download service account key as `firebase-credentials.json`
4. Place in `api-folder/`
5. Update `FIREBASE_CONFIG` in `config.py`

---

## Frontend Setup (Flutter Web)

### **Requirements**
- Flutter SDK (latest stable version)
- Chrome browser for web development

### **Configuration**
Update API endpoint in `lib/config.dart`:
```dart
static const apiBaseUrl = "http://127.0.0.1:5000/api/v1/predict";
```

### **Build and Run**
```bash
# Development mode
flutter run -d chrome

# Production build
flutter build web
```

---

## API Endpoints

### **Health Check**
```
GET /api/v1/health
```
Returns system status and model loading state.

### **Prediction**
```
POST /api/v1/predict
Content-Type: multipart/form-data

Parameters:
- image: Medical image file
- question: Text question about the image
```

**Response:**
```json
{
  "answer": "Model prediction",
  "confidence": 0.95,
  "processing_time": 1.23
}
```

---

## Troubleshooting

### **Backend Issues**

#### **Python Version Compatibility**
```bash
# Check Python version
python --version

# Recommended: Python 3.10 or 3.11
# If using Python 3.13+, some packages may need updates
```

#### **Missing Dependencies**
```bash
# Install missing packages
pip install torch torchvision torchaudio
pip install transformers datasets
pip install flask flask-cors
pip install firebase-admin pillow numpy
```

#### **GPU Issues**
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# If no GPU, the system will automatically use CPU
```

#### **Model Loading Errors**
- Ensure model file exists in `saved_models/`
- Check file permissions
- Verify model format compatibility

### **Frontend Issues**

#### **API Connection Errors**
- Verify backend is running on correct port
- Check CORS configuration in `config.py`
- Ensure API endpoint matches in `lib/config.dart`

#### **404 Errors**
- Backend endpoint must be `/api/v1/predict` (not `/predict`)
- Check network tab in browser developer tools

### **Training Issues**

#### **Memory Errors**
- Reduce batch size in `train_vqa_cleaned.py`
- Use gradient accumulation for large models
- Enable mixed precision training

#### **Dataset Issues**
- Ensure sufficient disk space (3GB+ recommended)
- Check internet connection for dataset download
- Verify dataset format compatibility

---

## Development

### **Adding New Features**
1. **Backend**: Add routes in `app.py`
2. **Model**: Extend `model.py` or `model_manager.py`
3. **Frontend**: Update Flutter widgets in `lib/`

### **Testing**
```bash
# Backend tests
cd api-folder
python -m pytest tests/

# Frontend tests
flutter test
```

### **Deployment**
1. **Backend**: Deploy to cloud platform (Heroku, AWS, etc.)
2. **Frontend**: Build and deploy to web hosting
3. **Update**: API endpoints in production configuration

---

## Performance Optimization

### **Backend**
- Use GPU acceleration when available
- Enable model caching
- Implement request queuing for high load
- Use async processing for long operations

### **Frontend**
- Implement image compression
- Add loading states and progress indicators
- Cache API responses
- Optimize bundle size

---

## Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

---

## License
MIT License - see LICENSE file for details

---

## Support
For issues and questions:
1. Check the troubleshooting section above
2. Review backend logs for error details
3. Check browser console for frontend errors
4. Open an issue on GitHub with detailed information

**Last Updated**: December 2024 