@echo off
REM Quick Start Script for DOJ Press Release NER Project
REM Run this script to set up and train the model

echo ================================================================================
echo DOJ Press Release NER Model - Quick Start
echo ================================================================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

echo Step 1: Installing dependencies...
echo --------------------------------------------------------------------------------
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo Step 2: Downloading spaCy English model...
echo --------------------------------------------------------------------------------
python -m spacy download en_core_web_sm
if errorlevel 1 (
    echo ERROR: Failed to download spaCy model
    pause
    exit /b 1
)

echo.
echo Step 3: Preparing training data...
echo --------------------------------------------------------------------------------
cd scripts
python prepare_data.py
if errorlevel 1 (
    echo ERROR: Failed to prepare data
    cd ..
    pause
    exit /b 1
)

echo.
echo Step 4: Training the model...
echo --------------------------------------------------------------------------------
echo This may take 15-30 minutes depending on your hardware...
python train_model.py
if errorlevel 1 (
    echo ERROR: Failed to train model
    cd ..
    pause
    exit /b 1
)

echo.
echo Step 5: Evaluating the model...
echo --------------------------------------------------------------------------------
python evaluate_model.py
if errorlevel 1 (
    echo ERROR: Failed to evaluate model
    cd ..
    pause
    exit /b 1
)

cd ..

echo.
echo ================================================================================
echo Setup Complete!
echo ================================================================================
echo.
echo Your NER model has been trained and evaluated.
echo.
echo Next steps:
echo   1. Check the evaluation results above
echo   2. Run 'python scripts\inference.py' for interactive testing
echo   3. Process new data: 'python scripts\inference.py path\to\data.jsonl output.json'
echo.
echo ================================================================================

pause
