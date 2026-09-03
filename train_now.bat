@echo off
echo ====================================================================
echo MetriGuard — GPU Acceleration Setup (Anaconda)
echo ====================================================================
echo.
echo Since your system Python is 3.14 (which lacks PyTorch CUDA wheels),
echo this script will use your Anaconda installation to create a Python 3.11
echo environment with full GPU support, and then start YOLO training on the 
echo massive 2.8 Lakh dataset.
echo.

set CONDA_PATH=
if exist "%USERPROFILE%\anaconda3\Scripts\conda.exe" set CONDA_PATH="%USERPROFILE%\anaconda3\Scripts\conda.exe"
if exist "%USERPROFILE%\miniconda3\Scripts\conda.exe" set CONDA_PATH="%USERPROFILE%\miniconda3\Scripts\conda.exe"
if exist "C:\ProgramData\Anaconda3\Scripts\conda.exe" set CONDA_PATH="C:\ProgramData\Anaconda3\Scripts\conda.exe"
if exist "C:\ProgramData\Miniconda3\Scripts\conda.exe" set CONDA_PATH="C:\ProgramData\Miniconda3\Scripts\conda.exe"

if "%CONDA_PATH%"=="" (
    echo [!] ERROR: Could not locate conda.exe in standard directories.
    echo Please open your "Anaconda Prompt" manually and run:
    echo   conda create -n metriguard_gpu python=3.11 -y
    echo   conda activate metriguard_gpu
    echo   conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
    echo   pip install ultralytics rapidocr_onnxruntime opencv-python-headless
    echo   python train_yolo_only.py
    
    exit /b
)

echo [*] Found Conda at: %CONDA_PATH%
echo [*] Creating conda environment 'metriguard_gpu' with Python 3.11...
call %CONDA_PATH% create -n metriguard_gpu python=3.11 -y

echo [*] Activating environment...
call "%~dp0..\..\..\anaconda3\Scripts\activate.bat" metriguard_gpu 2>nul || call "%USERPROFILE%\anaconda3\Scripts\activate.bat" metriguard_gpu 2>nul || echo [!] Warning: Activation script path might differ.

echo [*] Installing PyTorch with CUDA 12.1 support...
call conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

echo [*] Installing YOLOv11 and dependencies...
call pip install ultralytics rapidocr_onnxruntime opencv-python-headless flask

echo [*] All GPU dependencies installed! 
echo [*] Starting the Massive 2.8 Lakh Dataset YOLO11m Training on GPU...
python train_yolo_only.py


