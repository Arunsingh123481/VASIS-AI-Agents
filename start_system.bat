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
echo [38;5;208m┌──────────────────────────────────────────────────────────────┐[0m
echo [38;5;208m│                 [1mVASIS AI - SYSTEM CONTROL PANEL[0m[38;5;208m              │[0m
echo [38;5;208m└──────────────────────────────────────────────────────────────┘[0m
echo.
echo       [38;5;208m██╗   ██╗ █████╗ ███████╗██╗███████╗     █████╗ ██╗[0m
echo       [38;5;208m██║   ██║██╔══██╗██╔════╝██║██╔════╝    ██╔══██╗██║[0m
echo       [38;5;208m██║   ██║███████║███████╗██║███████╗    ███████║██║[0m
echo       [38;5;208m╚██╗ ██╔╝██╔══██║╚════██║██║╚════██║    ██╔══██║██║[0m
echo       [38;5;208m ╚████╔╝ ██║  ██║███████║██║███████║    ██║  ██║██║[0m
echo       [38;5;208m  ╚═══╝  ╚═╝  ╚═╝╚══════╝╚═╝╚══════╝    ╚═╝  ╚═╝╚═╝[0m
echo.
echo   [38;5;45m[1][0m [1mLaunch Interactive CLI RAG Chat[0m  [38;5;244m(python main.py chat)[0m
echo   [38;5;45m[2][0m [1mIndex a New PDF Document[0m        [38;5;244m(python main.py index)[0m
echo   [38;5;45m[3][0m [1mList Indexed Documents in Local Vault[0m
echo   [38;5;45m[4][0m [1mCheck Ollama Local Server Status[0m
echo   [38;5;45m[5][0m [38;5;196mExit[0m
echo.
echo [38;5;208m────────────────────────────────────────────────────────────────[0m
echo.
set /p choice="  [38;5;208m╰──▶[0m [1mEnter your selection (1-5):[0m "

if "%choice%"=="1" goto start_chat
if "%choice%"=="2" goto index_doc
if "%choice%"=="3" goto list_docs
if "%choice%"=="4" goto check_ollama
if "%choice%"=="5" goto exit
goto menu

:start_chat
cls
echo [38;5;208m┌──────────────────────────────────────────────────────────────┐[0m
echo [38;5;208m│               LAUNCHING INTERACTIVE CLI CHAT                 │[0m
echo [38;5;208m└──────────────────────────────────────────────────────────────┘[0m
echo.
set /p pdf="  [38;5;208m╰──▶[0m [1mEnter absolute or relative path of PDF:[0m "
set "pdf=%pdf:"=%"
if not exist "%pdf%" (
    echo.
    echo   [38;5;196m[ERROR] File "%pdf%" does not exist! Please check the path.[0m
    echo.
    pause
    goto menu
)
python main.py chat "%pdf%"
pause
goto menu

:index_doc
cls
echo [38;5;208m┌──────────────────────────────────────────────────────────────┐[0m
echo [38;5;208m│          INDEXING NEW DOCUMENT (BUILD CAUSAL GRAPH)          │[0m
echo [38;5;208m└──────────────────────────────────────────────────────────────┘[0m
echo.
set /p pdf="  [38;5;208m╰──▶[0m [1mEnter the path of the PDF to index:[0m "
set "pdf=%pdf:"=%"
if not exist "%pdf%" (
    echo.
    echo   [38;5;196m[ERROR] File "%pdf%" does not exist! Please check the path.[0m
    echo.
    pause
    goto menu
)
python main.py index "%pdf%" --show-tree
pause
goto menu

:list_docs
cls
echo [38;5;208m┌──────────────────────────────────────────────────────────────┐[0m
echo [38;5;208m│            INDEXED DOCUMENTS IN LOCAL CACHE VAULT            │[0m
echo [38;5;208m└──────────────────────────────────────────────────────────────┘[0m
echo.
python main.py list
echo.
pause
goto menu

:check_ollama
cls
echo [38;5;208m┌──────────────────────────────────────────────────────────────┐[0m
echo [38;5;208m│              OLLAMA LOCAL SERVER SERVICE CHECK               │[0m
echo [38;5;208m└──────────────────────────────────────────────────────────────┘[0m
echo.
echo   Checking connection to Ollama at http://127.0.0.1:11435...
echo.
powershell -Command "try { $r = Invoke-RestMethod -Uri http://127.0.0.1:11435/api/tags; echo '  [STATUS] local Ollama service is ACTIVE'; echo ''; echo '  Downloaded Models:'; foreach($m in $r.models) { echo ('    - ' + $m.name + ' (' + [math]::round($m.size / 1GB, 2) + ' GB)') } } catch { echo '  [STATUS] Ollama unreachable on port 11435!'; echo '  Please run: ollama serve' }"
echo.
echo [38;5;208m────────────────────────────────────────────────────────────────[0m
pause
goto menu

:exit
cls
echo [38;5;208m┌──────────────────────────────────────────────────────────────┐[0m
echo [38;5;208m│               EXITING CONTROL PANEL... GOODBYE!              │[0m
echo [38;5;208m└──────────────────────────────────────────────────────────────┘[0m
echo.
timeout /t 2 >nul
exit
