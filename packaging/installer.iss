; Inno Setup: instalador de Windows (doble clic -> Siguiente -> Instalar), sin permisos de administrador.
[Setup]
AppName=Smart Sentinel
AppVersion=1.0.0
DefaultDirName={autopf}\Smart Sentinel
DefaultGroupName=Smart Sentinel
PrivilegesRequired=lowest
OutputDir=..\dist
OutputBaseFilename=SmartSentinel-Setup
Compression=lzma2
SolidCompression=yes
UninstallDisplayName=Smart Sentinel

[Files]
Source: "..\dist\Smart Sentinel\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Smart Sentinel"; Filename: "{app}\Smart Sentinel.exe"
Name: "{autodesktop}\Smart Sentinel"; Filename: "{app}\Smart Sentinel.exe"

[Run]
Filename: "{app}\Smart Sentinel.exe"; Description: "Iniciar Smart Sentinel"; Flags: nowait postinstall skipifsilent
