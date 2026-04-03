; PILAR Desktop — Inno Setup Installer Script
; =============================================
; Compile:  iscc pilar_installer.iss
; Requires Inno Setup 6.x or 7.x

#define AppName       "PILAR"
#define AppVersion    "1.2.16"
#define AppPublisher  "PILAR Predictive Maintenance"
#define AppURL        "https://pilarapp.up.railway.app"
#define AppExeName    "PILAR.exe"
#define AppId         "{{6A1F7E2C-3B4D-4E5F-8A9B-0C1D2E3F4A5B}"

[Setup]
AppId={#AppId}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppURL}
AppSupportURL={#AppURL}
AppUpdatesURL={#AppURL}
; Install per-user in %LOCALAPPDATA%\Programs\PILAR (no admin required)
DefaultDirName={localappdata}\Programs\{#AppName}
DefaultGroupName={#AppName}
AllowNoIcons=yes
OutputDir=dist
OutputBaseFilename=PILAR_Setup_{#AppVersion}
SetupIconFile=pilar.ico
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
MinVersion=10.0.17763
ArchitecturesInstallIn64BitMode=x64compatible
; No admin required — installs silently for current user
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
UninstallDisplayIcon={app}\{#AppExeName}
UninstallDisplayName={#AppName} {#AppVersion}
; Disable SmartScreen nag on update (silent mode)
DisableWelcomePage=yes
DisableDirPage=yes
DisableProgramGroupPage=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "french";  MessagesFile: "compiler:Languages\French.isl"

[Tasks]
Name: "desktopicon";  Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: checkedonce
Name: "startupentry"; Description: "Lancer PILAR au démarrage de Windows"; GroupDescription: "Démarrage"; Flags: unchecked
Name: "addtopath";    Description: "Ajouter 'pilar' au PATH (utilisation en terminal)"; GroupDescription: "Terminal"; Flags: unchecked

[Files]
; The entire PyInstaller output (PILAR.exe + _internal/)
Source: "dist\pilar\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
; Terminal wrapper
Source: "pilar.bat"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
; Start Menu
Name: "{group}\{#AppName}";           Filename: "{app}\{#AppExeName}"; IconFilename: "{app}\{#AppExeName}"
Name: "{group}\{#AppName} (terminal)"; Filename: "{app}\{#AppExeName}"; Parameters: "--cli"; IconFilename: "{app}\{#AppExeName}"; Comment: "Start PILAR in terminal / server mode"
Name: "{group}\Désinstaller {#AppName}"; Filename: "{uninstallexe}"
; Desktop shortcut (checked by default on first install)
Name: "{autodesktop}\{#AppName}";     Filename: "{app}\{#AppExeName}"; IconFilename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Registry]
; Optional: launch on Windows startup
Root: HKCU; Subkey: "SOFTWARE\Microsoft\Windows\CurrentVersion\Run"; ValueType: string; ValueName: "{#AppName}"; ValueData: """{app}\{#AppExeName}"""; Flags: uninsdeletevalue; Tasks: startupentry
; Add app dir to user PATH so 'pilar' works from any terminal
Root: HKCU; Subkey: "Environment"; ValueType: expandsz; ValueName: "Path"; \
  ValueData: "{olddata};{app}"; \
  Flags: uninsdeletevalue preservestringtype; \
  Check: PathNotAdded; \
  Tasks: addtopath

[Run]
; Launch after install
Filename: "{app}\{#AppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(AppName, '&', '&&')}}"; Flags: nowait postinstall

[UninstallRun]
; Kill PILAR before uninstall
Filename: "taskkill"; Parameters: "/F /IM {#AppExeName}"; Flags: runhidden; RunOnceId: "KillPILAR"

[Code]
// Kill any running PILAR instance before installing (supports silent update)
function InitializeSetup(): Boolean;
var
  ResultCode: Integer;
begin
  Exec('taskkill', '/IM {#AppExeName} /F', '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
  Sleep(1200);
  Result := True;
end;

// Check that the app dir is not already in PATH (avoids duplicates)
function PathNotAdded(): Boolean;
var
  CurrentPath: string;
  AppDir: string;
begin
  AppDir := ExpandConstant('{app}');
  if RegQueryStringValue(HKCU, 'Environment', 'Path', CurrentPath) then
    Result := Pos(Lowercase(AppDir), Lowercase(CurrentPath)) = 0
  else
    Result := True;
end;
