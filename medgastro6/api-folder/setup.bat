@echo off
echo Setting up MedGastro6 API environment...

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed! Please install Python 3.8 or higher.
    exit /b 1
)

:: Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

:: Activate virtual environment and install dependencies
echo Activating virtual environment and installing dependencies...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt

:: Create necessary directories if they don't exist
if not exist "dataset_chunks" mkdir dataset_chunks
if not exist "dataset_chunks\metadata" mkdir dataset_chunks\metadata
if not exist "dataset_chunks\processed" mkdir dataset_chunks\processed
if not exist "saved_models" mkdir saved_models

echo Setup completed successfully!
echo To activate the environment, run: venv\Scripts\activate.bat
echo To start the API server, run: python app.py 