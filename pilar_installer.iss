; PILAR Desktop — Inno Setup Installer Script
; =============================================
; Compile:  iscc pilar_installer.iss
; Requires Inno Setup 6.x or 7.x

#define AppName       "PILAR"
#define AppVersion    "1.5.1"
#define AppPublisher  "PILAR Predictive Maintenance"
#define AppURL        "https://github.com/CYPHR007/PILAR"
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
; Classic style to enable the full-height side image
WizardStyle=classic
WizardImageFile=wizard_image.bmp
WizardSmallImageFile=wizard_small.bmp
WizardImageStretch=no
WizardImageBackColor=$0c0908
ShowLanguageDialog=no
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
Name: "installollama"; Description: "Télécharger Ollama (requis pour les agents IA - diagnostic et maintenance)"; GroupDescription: "Agents IA (recommandé)"; Flags: checkedonce

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
; Add app dir to user PATH so 'pilar' works from any terminal.
; Two variants: preserve olddata if present, else set directly (avoids leading ';').
Root: HKCU; Subkey: "Environment"; ValueType: expandsz; ValueName: "Path"; \
  ValueData: "{olddata};{app}"; \
  Flags: uninsdeletevalue preservestringtype; \
  Check: PathNotAdded and HasExistingPath; \
  Tasks: addtopath
Root: HKCU; Subkey: "Environment"; ValueType: expandsz; ValueName: "Path"; \
  ValueData: "{app}"; \
  Flags: uninsdeletevalue preservestringtype; \
  Check: PathNotAdded and (not HasExistingPath); \
  Tasks: addtopath

[Run]
; Open Ollama download page if user opted in (auto-detected: skipped if already installed)
Filename: "https://ollama.com/download/windows"; Description: "Télécharger Ollama (site officiel)"; Flags: shellexec nowait postinstall skipifsilent; Tasks: installollama; Check: NeedsOllama
; Launch after install
Filename: "{app}\{#AppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(AppName, '&', '&&')}}"; Flags: nowait postinstall

[UninstallRun]
; Kill PILAR before uninstall
Filename: "taskkill"; Parameters: "/F /IM {#AppExeName}"; Flags: runhidden; RunOnceId: "KillPILAR"

[Code]
// ── PILAR theme: dark navy + teal ────────────────────────────────────────────
procedure ApplyTheme;
begin
  // Main wizard background
  WizardForm.Color                           := $110E18;
  WizardForm.MainPanel.Color                 := $110E18;
  WizardForm.InnerPage.Color                 := $110E18;
  // Page title and description
  WizardForm.PageNameLabel.Font.Color        := $ECF0A6;  // light teal  ($A6F0EC in RGB)
  WizardForm.PageDescriptionLabel.Font.Color := $AE9B8A;  // muted
  // Lists / memo boxes
  WizardForm.TasksList.Color                 := $110E18;
  WizardForm.ReadyMemo.Color                 := $110E18;
  // Status labels on progress page
  WizardForm.FilenameLabel.Font.Color        := $AE9B8A;
  WizardForm.StatusLabel.Font.Color          := $AE9B8A;
end;

procedure InitializeWizard;
begin
  ApplyTheme;
end;

procedure CurPageChanged(CurPageID: Integer);
begin
  ApplyTheme;
end;

// Kill any running PILAR instance before installing (supports silent update)
function InitializeSetup(): Boolean;
var
  ResultCode: Integer;
begin
  Exec('taskkill', '/IM {#AppExeName} /F', '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
  Sleep(1200);
  Result := True;
end;

// Check if Ollama is already installed — skip the download prompt if so
function NeedsOllama(): Boolean;
var
  Paths: array[0..4] of string;
  I: Integer;
begin
  Paths[0] := ExpandConstant('{localappdata}\Programs\Ollama\ollama.exe');
  Paths[1] := ExpandConstant('{pf}\Ollama\ollama.exe');
  Paths[2] := ExpandConstant('{pf32}\Ollama\ollama.exe');
  Paths[3] := ExpandConstant('{userpf}\Ollama\ollama.exe');
  // winget default install location
  Paths[4] := ExpandConstant('{localappdata}\Microsoft\WinGet\Links\ollama.exe');
  Result := True;
  for I := 0 to 4 do
    if FileExists(Paths[I]) then
    begin
      Result := False;
      Exit;
    end;
end;

// True if HKCU Environment\Path is set and non-empty.
function HasExistingPath(): Boolean;
var
  CurrentPath: string;
begin
  if RegQueryStringValue(HKCU, 'Environment', 'Path', CurrentPath) then
    Result := Trim(CurrentPath) <> ''
  else
    Result := False;
end;

// Check that the app dir is not already in PATH (avoids duplicates).
// Splits on ';' and does full-entry compare so "C:\PILAR" does not
// falsely match "C:\PILARdev".
function PathNotAdded(): Boolean;
var
  CurrentPath, AppDir, Needle, Entry: string;
  StartPos, SemiPos: Integer;
begin
  AppDir := ExpandConstant('{app}');
  if not RegQueryStringValue(HKCU, 'Environment', 'Path', CurrentPath) then
  begin
    Result := True;
    Exit;
  end;
  Needle := Lowercase(Trim(AppDir));
  CurrentPath := Lowercase(CurrentPath);
  StartPos := 1;
  while StartPos <= Length(CurrentPath) do
  begin
    SemiPos := Pos(';', Copy(CurrentPath, StartPos, Length(CurrentPath)));
    if SemiPos = 0 then
    begin
      Entry := Trim(Copy(CurrentPath, StartPos, Length(CurrentPath)));
      if Entry = Needle then
      begin
        Result := False;
        Exit;
      end;
      Break;
    end;
    Entry := Trim(Copy(CurrentPath, StartPos, SemiPos - 1));
    if Entry = Needle then
    begin
      Result := False;
      Exit;
    end;
    StartPos := StartPos + SemiPos;
  end;
  Result := True;
end;
