#!/usr/bin/env bash
# Genera dist/SmartSentinel-macOS.dmg. Ejecutar desde cualquier sitio: bash packaging/build_mac.sh
#
# Se compila y firma en una carpeta temporal FUERA del proyecto: si el proyecto vive en una carpeta
# sincronizada con iCloud (Documentos/Escritorio), macOS añade atributos protegidos a los archivos y
# codesign falla con "resource fork, Finder information, or similar detritus not allowed".
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"

PY="${PYTHON:-python3}"
WORK="${SENTINEL_BUILD_DIR:-${TMPDIR:-/tmp}/smart-sentinel-build}"
APP="$WORK/dist/Smart Sentinel.app"

[ -f yolov8n.onnx ] || { echo "Falta yolov8n.onnx (python scripts/export_model.py)"; exit 1; }
[ -f hand_landmarker.task ] || curl -fL -o hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

rm -rf "$WORK"; mkdir -p "$WORK"
"$PY" -m PyInstaller --noconfirm --clean --distpath "$WORK/dist" --workpath "$WORK/work" packaging/smart_sentinel.spec

xattr -cr "$APP" 2>/dev/null || true
# Firma ad-hoc: obligatoria para que arranque en Apple Silicon (no sustituye a un Developer ID de Apple).
codesign --force --deep --sign - "$APP"
codesign --verify --deep --strict "$APP"

mkdir -p "$ROOT/dist"
rm -f "$ROOT/dist/SmartSentinel-macOS.dmg"
hdiutil create -volname "Smart Sentinel" -srcfolder "$APP" -ov -format UDZO "$WORK/SmartSentinel-macOS.dmg"
cp "$WORK/SmartSentinel-macOS.dmg" "$ROOT/dist/"
echo "✅ dist/SmartSentinel-macOS.dmg  (app sin empaquetar en: $APP)"
