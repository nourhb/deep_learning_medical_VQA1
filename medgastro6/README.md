# medgastro6

A Flutter project for medical image analysis and VQA.

## Getting Started

This project is a starting point for a Flutter application.

A few resources to get you started if this is your first Flutter project:

- [Lab: Write your first Flutter app](https://docs.flutter.dev/get-started/codelab)
- [Cookbook: Useful Flutter samples](https://docs.flutter.dev/cookbook)

For help getting started with Flutter development, view the
[online documentation](https://docs.flutter.dev/), which offers tutorials,
samples, guidance on mobile development, and a full API reference.

## Dataset Setup

The project uses a large dataset that is not included in the repository due to size constraints. To set up the dataset:

1. The dataset chunks are stored in `api-folder/dataset_chunks/` and are gitignored
2. To rebuild the dataset on a new machine:
   - Run the data processing scripts in the `api-folder` directory
   - The dataset will be automatically generated in the correct location
   - Make sure you have all required Python dependencies installed

## Project Structure

- `api-folder/`: Contains the backend API and data processing code
  - `dataset_chunks/`: Directory for storing processed dataset chunks (gitignored)
  - `kvasir_data/`: Original dataset images
  - Various Python scripts for data processing and API endpoints

## Development Setup

1. Clone the repository
2. Install Flutter dependencies:
   ```bash
   flutter pub get
   ```
3. Install Python dependencies:
   ```bash
   pip install -r api-folder/requirements.txt
   ```
4. Run the data processing scripts to generate the dataset
5. Start the Flutter application:
   ```bash
   flutter run
   ```

## Running the Project in Another Environment

### 1. Clone the Repository
```bash
git clone https://github.com/nourhb/deep_learning_medical_VQA1.git
cd deep_learning_medical_VQA1/medgastro6
```

### 2. Set Up the Environment
#### Option 1: Using the Setup Script (Recommended)
- For Windows:
```bash
.\setup.bat
```
- For Linux/Mac:
```bash
chmod +x setup.sh
./setup.sh
```

#### Option 2: Manual Setup
1. Create a virtual environment:
```bash
python -m venv venv
```

2. Activate the virtual environment:
- Windows:
```bash
.\venv\Scripts\activate
```
- Linux/Mac:
```bash
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### 3. Model Setup
1. **Download Required Model Files**
   - The model requires specific weights and configuration files
   - Place these files in the `api-folder/models` directory:
     - `model_weights.h5` (or your specific model file)
     - Any additional model configuration files
   - If you don't have the model files, you can:
     - Train the model using the provided training scripts in `api-folder/training/`
     - Or contact the project maintainers for pre-trained weights

2. **Verify Model Files**
   - Check that all required model files are present in `api-folder/models/`
   - Ensure file permissions are correct
   - Verify the model file paths in `api-folder/config.py`
   - The model should be compatible with TensorFlow 2.x

### 4. Data Setup
1. **Prepare the Dataset**
   - Follow the dataset preparation steps in the "Dataset Setup" section above
   - Ensure all required data files are in place in `api-folder/dataset_chunks/`
   - Verify the data paths in `api-folder/config.py`
   - The dataset should be organized as follows:
     ```
     api-folder/
     ├── dataset_chunks/
     │   ├── images/
     │   ├── questions/
     │   └── answers/
     ├── kvasir_data/
     └── ...
     ```

### 5. Running the Application
1. **Start the API Server**
```bash
cd api-folder
python app.py
```

2. **Access the Application**
   - Open your web browser
   - Navigate to `http://localhost:5000`
   - The application should be ready to use
   - For Flutter app development:
     ```bash
     flutter run
     ```

### Troubleshooting
If you encounter any issues:

1. **Model Loading Issues**
   - Verify model file paths in `api-folder/config.py`
   - Check model file integrity
   - Ensure all dependencies are installed
   - Check model version compatibility
   - Common error: "Model file not found" - Check if model files are in correct location
   - Common error: "Incompatible model version" - Ensure TensorFlow version matches model requirements

2. **Data Loading Issues**
   - Verify data file paths in `api-folder/config.py`
   - Check data file formats
   - Ensure all required data files are present
   - Common error: "Dataset not found" - Run data processing scripts first
   - Common error: "Invalid data format" - Check data structure matches requirements

3. **Environment Issues**
   - Verify Python version (3.8+ recommended)
   - Check all dependencies are installed
   - Ensure virtual environment is activated
   - Check system requirements
   - Common error: "Module not found" - Run `pip install -r requirements.txt`
   - Common error: "CUDA not available" - Check GPU drivers and TensorFlow installation

4. **API Server Issues**
   - Check port availability (default: 5000)
   - Verify API configuration in `api-folder/config.py`
   - Check server logs for errors
   - Common error: "Port already in use" - Change port in config or kill existing process
   - Common error: "Connection refused" - Check if server is running

For additional help:
- Check the project's GitHub issues
- Contact the project maintainers
- Refer to the API documentation in `api-folder/docs/`
