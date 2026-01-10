@echo off
set IMAGE_NAME=ml-api
set PORT=8000

REM Train
python3 -m app.train_models.train

REM Build
docker build -t %IMAGE_NAME% -f docker\Dockerfile .

REM Run
docker run --rm -p %PORT%:%PORT% ^
  -e MODEL_PATH=/app/artifacts/xgboost_model.pkl ^
  -v "%cd%\artifacts:/app/artifacts" ^
  %IMAGE_NAME%