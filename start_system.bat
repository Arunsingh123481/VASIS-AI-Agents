@echo off
chcp 65001 >nul
cd /d "%~dp0"

:: Automatically activate virtual environment if it exists
if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
) else if exist "venv\Scripts\activate.bat" (
    call "venv\Scripts\activate.bat"
)

title VASIS AI System Control Panel
cls

:menu
cls
echo ┌──────────────────────────────────────────────────────────────┐
echo │                 VASIS AI - SYSTEM CONTROL PANEL              │
echo └──────────────────────────────────────────────────────────────┘
echo.
echo       ██╗   ██╗ █████╗ ███████╗██╗███████╗     █████╗ ██╗
echo       ██║   ██║██╔══██╗██╔════╝██║██╔════╝    ██╔══██╗██║
echo       ██║   ██║███████║███████╗██║███████╗    ███████║██║
echo       ╚██╗ ██╔╝██╔══██║╚════██║██║╚════██║    ██╔══██║██║
echo        ╚████╔╝ ██║  ██║███████║██║███████║    ██║  ██║██║
echo         ╚═══╝  ╚═╝  ╚═╝╚══════╝╚═╝╚══════╝    ╚═╝  ╚═╝╚═╝
echo.
echo   [1] Launch Interactive CLI RAG Chat     (single PDF)
echo   [2] Index a New PDF Document            (build causal graph)
echo   [3] Vault Multi-Paper Chat              (cross-paper contradictions)
echo   [4] List Indexed Documents in Local Vault
echo   [5] Check Ollama Local Server Status
echo   [6] Run Tests                           (pytest, 32 tests)
echo   [7] Launch API Server                   (FastAPI on port 8000)
echo   [8] Exit
echo.
echo ────────────────────────────────────────────────────────────────
echo.
set /p choice="  Enter your selection (1-8): "

if "%choice%"=="1" goto start_chat
if "%choice%"=="2" goto index_doc
if "%choice%"=="3" goto vault_chat
if "%choice%"=="4" goto list_docs
if "%choice%"=="5" goto check_ollama
if "%choice%"=="6" goto run_tests
if "%choice%"=="7" goto launch_api
if "%choice%"=="8" goto exit
goto menu

:start_chat
cls
echo ┌──────────────────────────────────────────────────────────────┐
echo │               LAUNCHING INTERACTIVE CLI CHAT                 │
echo └──────────────────────────────────────────────────────────────┘
echo.
set /p pdf="  Enter absolute or relative path of PDF: "
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
set /p pdf="  Enter the path of the PDF to index: "
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

:vault_chat
cls
echo ┌──────────────────────────────────────────────────────────────┐
echo │         VAULT MODE - MULTI-PAPER CROSS-DOCUMENT CHAT        │
echo └──────────────────────────────────────────────────────────────┘
echo.
echo   Load 2+ PDFs into one session. Ask cross-paper questions
echo   like "Do these papers contradict each other on accuracy?"
echo.
echo   Enter PDF paths separated by spaces:
echo.
set /p pdfs="  > "
set "pdfs=%pdfs:"=%"
if "%pdfs%"=="" (
    echo.
    echo   [ERROR] No PDFs provided!
    echo.
    pause
    goto menu
)
python main.py vault %pdfs%
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

:run_tests
cls
echo ┌──────────────────────────────────────────────────────────────┐
echo │                    RUNNING TEST SUITE                        │
echo └──────────────────────────────────────────────────────────────┘
echo.
python -m pytest tests/ -v
echo.
echo ────────────────────────────────────────────────────────────────
pause
goto menu

:launch_api
cls
echo ┌──────────────────────────────────────────────────────────────┐
echo │              LAUNCHING FASTAPI SERVER (PORT 8000)            │
echo └──────────────────────────────────────────────────────────────┘
echo.
echo   API will be available at http://127.0.0.1:8000
echo   Docs at http://127.0.0.1:8000/docs
echo   Press Ctrl+C to stop the server.
echo.
python api.py
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
