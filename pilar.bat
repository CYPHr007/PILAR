@echo off
:: PILAR — terminal / server launcher
:: ====================================
:: Place this file (or a shortcut to it) somewhere on your PATH,
:: then type  pilar  in any terminal to start PILAR in server mode.
::
:: Options:
::   pilar                    — start on port 5000, open browser
::   pilar --port 8080        — use a different port
::   pilar --no-browser       — don't open the browser automatically
::
"%~dp0PILAR.exe" --cli %*
