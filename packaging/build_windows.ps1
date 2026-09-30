# Genera dist\Smart Sentinel\ (carpeta), dist\SmartSentinel-Windows.zip y, si Inno Setup está instalado,
# dist\SmartSentinel-Setup.exe. Ejecutar desde la raíz del proyecto en PowerShell.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

if (-not (Test-Path yolov8n.onnx)) { throw "Falta yolov8n.onnx (python scripts/export_model.py)" }
if (-not (Test-Path hand_landmarker.task)) {
    Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" -OutFile hand_landmarker.task
}

python -m PyInstaller --noconfirm --clean packaging/smart_sentinel.spec
Compress-Archive -Path "dist\Smart Sentinel" -DestinationPath dist\SmartSentinel-Windows.zip -Force
Write-Host "OK dist\SmartSentinel-Windows.zip"

$iscc = Get-Command iscc -ErrorAction SilentlyContinue
if (-not $iscc) { $iscc = Get-Item "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe" -ErrorAction SilentlyContinue }
if ($iscc) {
    & $iscc.Source packaging\installer.iss
    Write-Host "OK dist\SmartSentinel-Setup.exe"
} else {
    Write-Host "Inno Setup no encontrado: solo se generó el .zip (instálalo con 'choco install innosetup' para el instalador)."
}
