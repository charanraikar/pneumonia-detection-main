@echo off
echo ============================================
echo  PNEUMONIA PROJECT - FULL TRAINING PIPELINE
echo ============================================

call venv\Scripts\activate

echo.
echo [1/3] Starting DDPM training...
python main.py --mode ddpm
if %errorlevel% neq 0 ( echo DDPM FAILED & pause & exit )

echo.
echo [2/3] Starting ConvNeXt training...
python main.py --mode train
if %errorlevel% neq 0 ( echo TRAINING FAILED & pause & exit )

echo.
echo [3/3] Running Grad-CAM analysis...
python main.py --mode gradcam
if %errorlevel% neq 0 ( echo GRADCAM FAILED & pause & exit )

echo.
echo ============================================
echo  ALL DONE - Results saved in results/ folder
echo ============================================
pause