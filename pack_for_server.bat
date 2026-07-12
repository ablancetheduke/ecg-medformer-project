@echo off
REM =============================================================================
REM Windows-side Packing Script — prepares files for server upload
REM Run this on your Windows machine in:
REM   E:\deeplearning\ecg_time_frequency_dual_branch_classification
REM =============================================================================

echo ============================================
echo  Packing files for server deployment...
echo ============================================

set PACK_DIR=E:\deeplearning\ecg_time_frequency_dual_branch_classification\server_deploy
set CODE_DIR=E:\deeplearning\ecg_time_frequency_dual_branch_classification\Medformer\Medformer-main

echo.
echo [1/4] Copying Medformer code (excluding dataset, checkpoints, results)...
robocopy "%CODE_DIR%" "%PACK_DIR%\Medformer-main" /E /XD dataset checkpoints results logs __pycache__ .git /NFL /NDL /NJH /NJS

echo.
echo [2/4] Copying PTB-XL data...
copy "E:\deeplearning\ecg_time_frequency_dual_branch_classification\PTB-XL.zip" "%PACK_DIR%\" >nul

echo.
echo [3/4] Copying FFT model files (verify they exist)...
if exist "%CODE_DIR%\models\MedformerFFT.py" (
    echo   MedformerFFT.py ✓
) else (
    echo   WARNING: MedformerFFT.py NOT FOUND!
)
if exist "%CODE_DIR%\models\FrequencyOnly.py" (
    echo   FrequencyOnly.py ✓
) else (
    echo   WARNING: FrequencyOnly.py NOT FOUND!
)

echo.
echo [4/4] Creating upload tarball...
cd /d E:\deeplearning\ecg_time_frequency_dual_branch_classification
tar -czf medformer_server_pack.tar.gz server_deploy\

echo.
echo ============================================
echo  DONE!
echo ============================================
echo.
echo Upload this file to your server:
echo   E:\deeplearning\ecg_time_frequency_dual_branch_classification\medformer_server_pack.tar.gz
echo.
echo On the server, extract and start:
echo   tar -xzf medformer_server_pack.tar.gz
echo   cd server_deploy
echo   bash setup.sh
echo   bash run_all.sh
echo.
echo Size check:
dir medformer_server_pack.tar.gz
echo.
pause
