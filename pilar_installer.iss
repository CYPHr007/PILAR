; PILAR Desktop — Inno Setup Installer Script
; =============================================
; Compile with:  iscc pilar_installer.iss
; Requires Inno Setup 6.x — https://jrsoftware.org/isinfo.php
;
; No code signing: users must accept Windows SmartScreen warning on first run.
; To suppress SmartScreen in production, sign the exe with a trusted cert.

#define AppName       "PILAR"
#define AppVersion    "1.0.0"
#define AppPublisher  "PILAR — Predictive Maintenance"
#define AppURL        "https://pilar.app"
#define AppExeName    "PILAR.exe"
#define AppId         "{{6A1F7E2C-3B4D-4E5F-8A9B-0C1D2E3F4A5B}"

[Setup]
AppId={#AppId}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisherURL={#AppURL}
AppSupportURL={#AppURL}
AppUpdatesURL={#AppURL}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
AllowNoIcons=yes
; Output directory (relative to script location)
OutputDir=dist
OutputBaseFilename=PILAR_Setup_{#AppVersion}
; Optional: set this to your .ico file path
; SetupIconFile=pilar.ico
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
; Minimum Windows version: Windows 10
MinVersion=10.0.17763
; Allow both 32-bit and 64-bit
ArchitecturesInstallIn64BitMode=x64
; Do NOT require admin (installs per-user if admin not available)
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
; Show license on install
; LicenseFile=LICENSE.txt
UninstallDisplayIcon={app}\{#AppExeName}
UninstallDisplayName={#AppName} {#AppVersion}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "french";  MessagesFile: "compiler:Languages\French.isl"

[Tasks]
Name: "desktopicon";     Description: "{cm:CreateDesktopIcon}";       GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "startupentry";    Description: "Launch PILAR when Windows starts"; GroupDescription: "Startup";              Flags: unchecked

[Files]
; Copy the entire PyInstaller output directory
Source: "dist\pilar\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
; Start Menu
Name: "{group}\{#AppName}";            Filename: "{app}\{#AppExeName}"
Name: "{group}\Uninstall {#AppName}";  Filename: "{uninstallexe}"
; Desktop shortcut (optional task)
Name: "{autodesktop}\{#AppName}";      Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Registry]
; Startup entry (optional task)
Root: HKCU; Subkey: "SOFTWARE\Microsoft\Windows\CurrentVersion\Run"; ValueType: string; ValueName: "{#AppName}"; ValueData: """{app}\{#AppExeName}"""; Flags: uninsdeletevalue; Tasks: startupentry

[Run]
; Offer to launch PILAR after install
Filename: "{app}\{#AppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(AppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[UninstallRun]
; Kill running PILAR process before uninstall
Filename: "taskkill"; Parameters: "/F /IM {#AppExeName}"; Flags: runhidden; RunOnceId: "KillPILAR"

[Code]
// Custom install code — check for existing running instance
function InitializeSetup(): Boolean;
var
  ResultCode: Integer;
begin
  // Attempt to terminate any running instance gracefully
  Exec('taskkill', '/IM {#AppExeName} /F', '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
  Result := True;
end;
