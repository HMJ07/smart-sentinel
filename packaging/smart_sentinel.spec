# PyInstaller: `pyinstaller packaging/smart_sentinel.spec` (usar los scripts build_mac.sh / build_windows.ps1)
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all

ROOT = Path(SPECPATH).parent
APP_NAME = "Smart Sentinel"

datas = [(str(ROOT / "yolov8n.onnx"), "."), (str(ROOT / "hand_landmarker.task"), ".")]
binaries, hiddenimports = [], []
for pkg in ("mediapipe", "onnxruntime"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

a = Analysis(
    [str(ROOT / "main.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    # torch/ultralytics solo se usan para exportar el modelo en desarrollo: fuera del instalador.
    excludes=["torch", "torchvision", "torchaudio", "ultralytics", "tkinter", "IPython", "pytest"],
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name=APP_NAME,
    console=False,          # sin ventana negra; el registro va a sentinel.log
)
coll = COLLECT(exe, a.binaries, a.datas, name=APP_NAME)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name=f"{APP_NAME}.app",
        bundle_identifier="com.smartsentinel.app",
        info_plist={
            "CFBundleDisplayName": APP_NAME,
            "CFBundleShortVersionString": "1.0.0",
            "NSHighResolutionCapable": True,
            # Sin esta clave macOS deniega la cámara sin ni siquiera preguntar.
            "NSCameraUsageDescription": "Smart Sentinel usa la cámara para detectar personas, gestos y anomalías. "
                                        "Las imágenes se procesan localmente.",
        },
    )
