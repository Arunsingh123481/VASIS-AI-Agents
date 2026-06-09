@echo off
chcp 65001 >nul
cd /d "%~dp0"

:: Automatically activate virtual environment if it exists
if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
) else if exist "venv\Scripts\activate.bat" (
    call "venv\Scripts\activate.bat"
)

title VASIS AI-RE-MSE CRDB System Control Panel
color 0d
cls

:menu
cls
echo ┌──────────────────────────────────────────────────────────────┐
echo │               VASIS AI-RE-MSE - CONTROL PANEL                │
echo └──────────────────────────────────────────────────────────────┘
echo.
echo       ██╗   ██╗ █████╗ ███████╗██╗███████╗     █████╗ ██╗
echo       ██║   ██║██╔══██╗██╔════╝██║██╔════╝    ██╔══██╗██║
echo       ██║   ██║███████║███████╗██║███████╗    ███████║██║
echo       ╚██╗ ██╔╝██╔══██║╚════██║██║╚════██║    ██╔══██║██║
echo        ╚████╔╝ ██║  ██║███████║██║███████║    ██║  ██║██║
echo         ╚═══╝  ╚═╝  ╚═╝╚══════╝╚═╝╚══════╝    ╚═╝  ╚═╝╚═╝
echo.
echo   [1] Start Web RAG App Server (FastAPI + Chat UI)
echo   [2] Launch Interactive CLI RAG Chat
echo   [3] Index a New PDF Document
echo   [4] List Indexed Documents in Local Vault
echo   [5] Check Ollama Local Server Status
echo   [6] Exit
echo.
echo ────────────────────────────────────────────────────────────────
echo.
set /p choice="  ╰──▶ Enter your selection (1-6): "

if "%choice%"=="1" goto start_api
if "%choice%"=="2" goto start_chat
if "%choice%"=="3" goto index_doc
if "%choice%"=="4" goto list_docs
if "%choice%"=="5" goto check_ollama
if "%choice%"=="6" goto exit
goto menu

:start_api
cls
echo ┌──────────────────────────────────────────────────────────────┐
echo │                   STARTING FASTAPI SERVER                    │
echo └──────────────────────────────────────────────────────────────┘
echo.
echo   ● REST API Docs:   http://localhost:8001/docs
echo   ● Interactive UI:  http://localhost:8001/ui
echo.
echo   * Press Ctrl+C in this terminal to stop server.
echo.
echo ────────────────────────────────────────────────────────────────
echo.
python api.py
pause
goto menu

:start_chat
cls
echo ┌──────────────────────────────────────────────────────────────┐
echo │               LAUNCHING INTERACTIVE CLI CHAT                 │
echo └──────────────────────────────────────────────────────────────┘
echo.
set /p pdf="  ╰──▶ Enter absolute or relative path of PDF: "
set "pdf=%pdf:"=%"
if not exist "%pdf%" (
    echo.
    echo   [ERROR] File "%pdf%" does not exist! Please check the path.
    echo.
    pause
    goto menu
)
python main.py chat "%pdf%"
pause
goto menu

:index_doc
cls
echo ┌──────────────────────────────────────────────────────────────┐
echo │          INDEXING NEW DOCUMENT (BUILD CAUSAL GRAPH)          │
echo └──────────────────────────────────────────────────────────────┘
echo.
set /p pdf="  ╰──▶ Enter the path of the PDF to index: "
set "pdf=%pdf:"=%"
if not exist "%pdf%" (
    echo.
    echo   [ERROR] File "%pdf%" does not exist! Please check the path.
    echo.
    pause
    goto menu
)
python main.py index "%pdf%" --show-tree
pause
goto menu

:list_docs
cls
echo ┌──────────────────────────────────────────────────────────────┐
echo │            INDEXED DOCUMENTS IN LOCAL CACHE VAULT            │
echo └──────────────────────────────────────────────────────────────┘
echo.
python main.py list
echo.
pause
goto menu

:check_ollama
cls
echo ┌──────────────────────────────────────────────────────────────┐
echo │              OLLAMA LOCAL SERVER SERVICE CHECK               │
echo └──────────────────────────────────────────────────────────────┘
echo.
echo   Checking connection to Ollama at http://127.0.0.1:11435...
echo.
powershell -Command "try { $r = Invoke-RestMethod -Uri http://127.0.0.1:11435/api/tags; echo '  [STATUS] local Ollama service is ACTIVE'; echo ''; echo '  Downloaded Models:'; foreach($m in $r.models) { echo ('    - ' + $m.name + ' (' + [math]::round($m.size / 1GB, 2) + ' GB)') } } catch { echo '  [STATUS] Ollama unreachable on port 11435!'; echo '  Please run: ollama serve' }"
echo.
echo ────────────────────────────────────────────────────────────────
pause
goto menu

:exit
cls
echo ┌──────────────────────────────────────────────────────────────┐
echo │               EXITING CONTROL PANEL... GOODBYE!              │
echo └──────────────────────────────────────────────────────────────┘
echo.
timeout /t 2 >nul
exit
