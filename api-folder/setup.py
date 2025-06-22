#!/usr/bin/env python3
import os
import sys
import subprocess
import venv
from pathlib import Path

def check_python_version():
    """Check if Python version is 3.8 or higher."""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        sys.exit(1)

def create_virtual_environment():
    """Create a virtual environment if it doesn't exist."""
    venv_path = Path("venv")
    if not venv_path.exists():
        print("Creating virtual environment...")
        venv.create(venv_path, with_pip=True)

def get_venv_python():
    """Get the path to the Python executable in the virtual environment."""
    if sys.platform == "win32":
        return Path("venv/Scripts/python.exe")
    return Path("venv/bin/python")

def get_venv_pip():
    """Get the path to pip in the virtual environment."""
    if sys.platform == "win32":
        return Path("venv/Scripts/pip.exe")
    return Path("venv/bin/pip")

def install_dependencies():
    """Install required packages using pip."""
    pip_path = get_venv_pip()
    print("Installing dependencies...")
    subprocess.run([str(pip_path), "install", "--upgrade", "pip"])
    subprocess.run([str(pip_path), "install", "-r", "requirements.txt"])

def create_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        "dataset_chunks",
        "dataset_chunks/metadata",
        "dataset_chunks/processed",
        "saved_models"
    ]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

def main():
    print("Setting up MedGastro6 API environment...")
    
    # Check Python version
    check_python_version()
    
    # Create virtual environment
    create_virtual_environment()
    
    # Install dependencies
    install_dependencies()
    
    # Create necessary directories
    create_directories()
    
    print("\nSetup completed successfully!")
    if sys.platform == "win32":
        print("To activate the environment, run: venv\\Scripts\\activate.bat")
    else:
        print("To activate the environment, run: source venv/bin/activate")
    print("To start the API server, run: python app.py")

if __name__ == "__main__":
    main() 