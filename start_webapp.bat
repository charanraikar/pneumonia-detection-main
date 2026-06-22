@echo off
cd /d C:\Users\Akshay\pneumonia_project
call venv\Scripts\activate

echo ============================================
echo  PneumoScan Web App
echo ============================================
echo.

REM Install web dependencies if needed
pip install fastapi uvicorn[standard] python-multipart --quiet

echo  Starting server...
echo  Open your browser at: http://localhost:8000
echo  Press Ctrl+C to stop
echo.

set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True
set CUDA_VISIBLE_DEVICES=0

uvicorn app:app --host 0.0.0.0 --port 8000 --reload