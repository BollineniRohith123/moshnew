#!/usr/bin/env python3
"""
Perfect Voice Assistant Setup Script
Automated installation and configuration for the enhanced speech-to-speech system
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(command, check=True):
    """Run shell command with error handling"""
    try:
        logger.info(f"Running: {command}")
        result = subprocess.run(command, shell=True, check=check, capture_output=True, text=True)
        if result.stdout:
            logger.info(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {e}")
        if e.stderr:
            logger.error(e.stderr)
        raise

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        logger.error("Python 3.8 or higher is required")
        sys.exit(1)
    logger.info(f"✅ Python {sys.version} detected")

def install_system_dependencies():
    """Install system-level dependencies"""
    logger.info("🔧 Installing system dependencies...")
    
    # Detect OS and install dependencies
    if sys.platform.startswith('linux'):
        commands = [
            "apt-get update",
            "apt-get install -y python3-pip python3-dev build-essential",
            "apt-get install -y libsndfile1-dev ffmpeg portaudio19-dev",
            "apt-get install -y libasound2-dev libpulse-dev"
        ]
        for cmd in commands:
            run_command(cmd)
    elif sys.platform == 'darwin':  # macOS
        commands = [
            "brew install portaudio",
            "brew install ffmpeg"
        ]
        for cmd in commands:
            run_command(cmd, check=False)  # Don't fail if brew not available
    elif sys.platform.startswith('win'):
        logger.info("Windows detected. Please ensure you have Visual Studio Build Tools installed.")
    
    logger.info("✅ System dependencies installed")

def create_virtual_environment():
    """Create and activate virtual environment"""
    logger.info("🐍 Creating virtual environment...")
    
    venv_path = Path("venv")
    if venv_path.exists():
        logger.info("Virtual environment already exists")
        return
    
    run_command(f"{sys.executable} -m venv venv")
    logger.info("✅ Virtual environment created")

def install_python_dependencies():
    """Install Python packages"""
    logger.info("📦 Installing Python dependencies...")
    
    # Determine pip path
    if sys.platform.startswith('win'):
        pip_path = "venv\\Scripts\\pip"
    else:
        pip_path = "venv/bin/pip"
    
    commands = [
        f"{pip_path} install --upgrade pip wheel setuptools",
        f"{pip_path} install -r requirements.txt"
    ]
    
    for cmd in commands:
        run_command(cmd)
    
    logger.info("✅ Python dependencies installed")

def create_directories():
    """Create necessary directories"""
    logger.info("📁 Creating directories...")
    
    directories = [
        "models/stt",
        "models/tts", 
        "models/llm",
        "logs",
        "cache"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    logger.info("✅ Directories created")

def download_models():
    """Download AI models"""
    logger.info("🤖 Downloading AI models...")
    logger.info("This may take a while depending on your internet connection...")
    
    # Determine python path
    if sys.platform.startswith('win'):
        python_path = "venv\\Scripts\\python"
    else:
        python_path = "venv/bin/python"
    
    try:
        run_command(f"{python_path} download_models.py")
        logger.info("✅ Models downloaded successfully")
    except subprocess.CalledProcessError:
        logger.warning("⚠️ Model download failed. The system will use fallback models.")

def create_startup_scripts():
    """Create startup scripts for different platforms"""
    logger.info("📝 Creating startup scripts...")
    
    # Unix/Linux/macOS startup script
    with open("start.sh", "w") as f:
        f.write("""#!/bin/bash
# Perfect Voice Assistant Startup Script

echo "🚀 Starting Perfect Voice Assistant..."

# Activate virtual environment
source venv/bin/activate

# Start the application
python app.py

echo "👋 Perfect Voice Assistant stopped"
""")
    
    # Make executable
    if not sys.platform.startswith('win'):
        run_command("chmod +x start.sh")
    
    # Windows startup script
    with open("start.bat", "w") as f:
        f.write("""@echo off
REM Perfect Voice Assistant Startup Script

echo 🚀 Starting Perfect Voice Assistant...

REM Activate virtual environment
call venv\\Scripts\\activate.bat

REM Start the application
python app.py

echo 👋 Perfect Voice Assistant stopped
pause
""")
    
    logger.info("✅ Startup scripts created")

def run_tests():
    """Run basic system tests"""
    logger.info("🧪 Running system tests...")
    
    # Determine python path
    if sys.platform.startswith('win'):
        python_path = "venv\\Scripts\\python"
    else:
        python_path = "venv/bin/python"
    
    # Test imports
    test_script = """
import sys
try:
    import torch
    import numpy as np
    import fastapi
    import uvicorn
    import librosa
    import scipy
    print("✅ All core dependencies imported successfully")
    
    # Test device detection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ PyTorch device: {device}")
    
    print("✅ System tests passed")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
"""
    
    with open("test_setup.py", "w") as f:
        f.write(test_script)
    
    try:
        run_command(f"{python_path} test_setup.py")
        os.remove("test_setup.py")
        logger.info("✅ System tests passed")
    except subprocess.CalledProcessError:
        logger.error("❌ System tests failed")
        raise

def main():
    """Main setup function"""
    logger.info("🎙️ Perfect Voice Assistant Setup")
    logger.info("=" * 50)
    
    try:
        check_python_version()
        install_system_dependencies()
        create_virtual_environment()
        create_directories()
        install_python_dependencies()
        run_tests()
        download_models()
        create_startup_scripts()
        
        logger.info("=" * 50)
        logger.info("🎉 Setup completed successfully!")
        logger.info("")
        logger.info("To start the application:")
        if sys.platform.startswith('win'):
            logger.info("  Windows: double-click start.bat or run 'start.bat'")
        else:
            logger.info("  Unix/Linux/macOS: run './start.sh'")
        logger.info("")
        logger.info("The application will be available at: http://localhost:8000")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"❌ Setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()