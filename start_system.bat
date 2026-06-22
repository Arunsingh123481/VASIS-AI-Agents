@echo off
chcp 65001 >nul
cd /d "%~dp0"

:: Generate ESC character for ANSI color codes
for /F %%a in ('echo prompt $E ^| cmd') do set "ESC=%%a"

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
echo %ESC%[38;5;208m================================================================%ESC%[0m
echo.
echo       %ESC%[38;5;208m%ESC%[1m V  A  S  I  S      A  I%ESC%[0m
echo.
echo       %ESC%[38;5;45m14-Agent Consensus Engine - Control Panel%ESC%[0m
echo.
echo %ESC%[38;5;208m================================================================%ESC%[0m
echo.
echo   %ESC%[38;5;45m[1]%ESC%[0m %ESC%[1mLaunch Interactive CLI RAG Chat%ESC%[0m     %ESC%[38;5;244m(single PDF)%ESC%[0m
echo   %ESC%[38;5;45m[2]%ESC%[0m %ESC%[1mIndex a New PDF Document%ESC%[0m            %ESC%[38;5;244m(build causal graph)%ESC%[0m
echo   %ESC%[38;5;45m[3]%ESC%[0m %ESC%[1mVault Multi-Paper Chat%ESC%[0m              %ESC%[38;5;244m(cross-paper contradictions)%ESC%[0m
echo   %ESC%[38;5;45m[4]%ESC%[0m %ESC%[1mList Indexed Documents in Local Vault%ESC%[0m
echo   %ESC%[38;5;45m[5]%ESC%[0m %ESC%[1mCheck Ollama Local Server Status%ESC%[0m
echo   %ESC%[38;5;45m[6]%ESC%[0m %ESC%[1mRun Tests%ESC%[0m                           %ESC%[38;5;244m(pytest, 32 tests)%ESC%[0m
echo   %ESC%[38;5;45m[7]%ESC%[0m %ESC%[1mLaunch API Server%ESC%[0m                   %ESC%[38;5;244m(FastAPI on port 8000)%ESC%[0m
echo   %ESC%[38;5;45m[8]%ESC%[0m %ESC%[38;5;196mExit%ESC%[0m
echo.
echo %ESC%[38;5;208m----------------------------------------------------------------%ESC%[0m
echo.
set /p choice="  %ESC%[38;5;208m^>%ESC%[0m %ESC%[1mEnter your selection (1-8):%ESC%[0m "

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
echo %ESC%[38;5;208m================================================================%ESC%[0m
echo %ESC%[38;5;208m             LAUNCHING INTERACTIVE CLI CHAT                     %ESC%[0m
echo %ESC%[38;5;208m================================================================%ESC%[0m
echo.
set /p pdf="  %ESC%[38;5;208m^>%ESC%[0m %ESC%[1mEnter absolute or relative path of PDF:%ESC%[0m "
set "pdf=%pdf:"=%"
if not exist "%pdf%" (
    echo.
    echo   %ESC%[38;5;196m[ERROR] File "%pdf%" does not exist! Please check the path.%ESC%[0m
    echo.
    pause
    goto menu
)
python main.py chat "%pdf%"
pause
goto menu

:index_doc
cls
echo %ESC%[38;5;208m================================================================%ESC%[0m
echo %ESC%[38;5;208m          INDEXING NEW DOCUMENT (BUILD CAUSAL GRAPH)            %ESC%[0m
echo %ESC%[38;5;208m================================================================%ESC%[0m
echo.
set /p pdf="  %ESC%[38;5;208m^>%ESC%[0m %ESC%[1mEnter the path of the PDF to index:%ESC%[0m "
set "pdf=%pdf:"=%"
if not exist "%pdf%" (
    echo.
    echo   %ESC%[38;5;196m[ERROR] File "%pdf%" does not exist! Please check the path.%ESC%[0m
    echo.
    pause
    goto menu
)
python main.py index "%pdf%" --show-tree
pause
goto menu

:vault_chat
cls
echo %ESC%[38;5;208m================================================================%ESC%[0m
echo %ESC%[38;5;208m       VAULT MODE - MULTI-PAPER CROSS-DOCUMENT CHAT            %ESC%[0m
echo %ESC%[38;5;208m================================================================%ESC%[0m
echo.
echo   Load 2+ PDFs into one session. Ask cross-paper questions
echo   like "Do these papers contradict each other on accuracy?"
echo.
echo   %ESC%[38;5;244mEnter ONE PDF path per line. Type DONE when finished.%ESC%[0m
echo   %ESC%[38;5;244mPaths with spaces are handled automatically.%ESC%[0m
echo.
set "vault_cmd=python main.py vault"
set "pdf_count=0"

:vault_add_pdf
set /p pdf="  %ESC%[38;5;208m^>%ESC%[0m %ESC%[1mPDF path (or DONE to start):%ESC%[0m "
:: Strip surrounding quotes the user may have added
set "pdf=%pdf:"=%"
:: Check if user is finished
if /i "%pdf%"=="done" goto vault_run
if /i "%pdf%"=="DONE" goto vault_run
:: Ignore blank input
if "%pdf%"=="" goto vault_add_pdf
:: Validate the file exists
if not exist "%pdf%" (
    echo.
    echo   %ESC%[38;5;196m[ERROR] File not found: %pdf%%ESC%[0m
    echo   %ESC%[38;5;244mCheck the path and try again.%ESC%[0m
    echo.
    goto vault_add_pdf
)
:: Append this quoted path to the running command
set "vault_cmd=%vault_cmd% "%pdf%""
set /a pdf_count+=1
echo   %ESC%[38;5;82m[OK]%ESC%[0m Paper %pdf_count% added.
echo.
goto vault_add_pdf

:vault_run
if %pdf_count% LSS 2 (
    echo.
    echo   %ESC%[38;5;196m[ERROR] Need at least 2 PDFs to use Vault mode!%ESC%[0m
    echo.
    pause
    goto menu
)
echo.
echo   %ESC%[38;5;45mStarting Vault with %pdf_count% papers...%ESC%[0m
echo.
%vault_cmd%
pause
goto menu

:list_docs
cls
echo %ESC%[38;5;208m================================================================%ESC%[0m
echo %ESC%[38;5;208m            INDEXED DOCUMENTS IN LOCAL CACHE VAULT              %ESC%[0m
echo %ESC%[38;5;208m================================================================%ESC%[0m
echo.
python main.py list
echo.
pause
goto menu

:check_ollama
cls
echo %ESC%[38;5;208m================================================================%ESC%[0m
echo %ESC%[38;5;208m              OLLAMA LOCAL SERVER SERVICE CHECK                 %ESC%[0m
echo %ESC%[38;5;208m================================================================%ESC%[0m
echo.
echo   Checking connection to Ollama at http://127.0.0.1:11435...
echo.
powershell -Command "try { $r = Invoke-RestMethod -Uri http://127.0.0.1:11435/api/tags; echo '  [STATUS] local Ollama service is ACTIVE'; echo ''; echo '  Downloaded Models:'; foreach($m in $r.models) { echo ('    - ' + $m.name + ' (' + [math]::round($m.size / 1GB, 2) + ' GB)') } } catch { echo '  [STATUS] Ollama unreachable on port 11435!'; echo '  Please run: ollama serve' }"
echo.
echo %ESC%[38;5;208m----------------------------------------------------------------%ESC%[0m
pause
goto menu

:run_tests
cls
echo %ESC%[38;5;208m================================================================%ESC%[0m
echo %ESC%[38;5;208m                    RUNNING TEST SUITE                          %ESC%[0m
echo %ESC%[38;5;208m================================================================%ESC%[0m
echo.
python -m pytest tests/ -v
echo.
echo %ESC%[38;5;208m----------------------------------------------------------------%ESC%[0m
pause
goto menu

:launch_api
cls
echo %ESC%[38;5;208m================================================================%ESC%[0m
echo %ESC%[38;5;208m              LAUNCHING FASTAPI SERVER (PORT 8000)              %ESC%[0m
echo %ESC%[38;5;208m================================================================%ESC%[0m
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
echo %ESC%[38;5;208m================================================================%ESC%[0m
echo %ESC%[38;5;208m               EXITING CONTROL PANEL... GOODBYE!               %ESC%[0m
echo %ESC%[38;5;208m================================================================%ESC%[0m
echo.
timeout /t 2 >nul
exit
