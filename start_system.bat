@echo off
chcp 65001 >nul
cd /d "%~dp0"

:: Force Python to run in UTF-8 mode on Windows (avoids UnicodeEncodeError on block chars/agents)
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

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
echo   %ESC%[38;5;45m[1]%ESC%[0m %ESC%[1mLaunch Interactive CLI Shell%ESC%[0m          %ESC%[38;5;244m(Rich + Prompt Toolkit - New!)%ESC%[0m
echo   %ESC%[38;5;45m[2]%ESC%[0m %ESC%[1mLaunch Interactive CLI RAG Chat%ESC%[0m     %ESC%[38;5;244m(single or multi-PDF)%ESC%[0m
echo   %ESC%[38;5;45m[3]%ESC%[0m %ESC%[1mIndex a New PDF Document%ESC%[0m            %ESC%[38;5;244m(build causal graph)%ESC%[0m
echo   %ESC%[38;5;45m[4]%ESC%[0m %ESC%[1mList Indexed Documents in Local Vault%ESC%[0m
echo   %ESC%[38;5;45m[5]%ESC%[0m %ESC%[1mCheck Ollama Local Server Status%ESC%[0m
echo   %ESC%[38;5;45m[6]%ESC%[0m %ESC%[1mRun Tests%ESC%[0m                           %ESC%[38;5;244m(pytest, 43 tests)%ESC%[0m
echo   %ESC%[38;5;45m[7]%ESC%[0m %ESC%[1mLaunch API Server%ESC%[0m                   %ESC%[38;5;244m(FastAPI on port 8000)%ESC%[0m
echo   %ESC%[38;5;45m[8]%ESC%[0m %ESC%[38;5;196mExit%ESC%[0m
echo.
echo %ESC%[38;5;208m----------------------------------------------------------------%ESC%[0m
echo.
set /p choice="  %ESC%[38;5;208m^>%ESC%[0m %ESC%[1mEnter your selection (1-8):%ESC%[0m "

if "%choice%"=="1" goto start_tui
if "%choice%"=="2" goto start_chat
if "%choice%"=="3" goto index_doc
if "%choice%"=="4" goto list_docs
if "%choice%"=="5" goto check_ollama
if "%choice%"=="6" goto run_tests
if "%choice%"=="7" goto launch_api
if "%choice%"=="8" goto exit
goto menu

:start_tui
cls
echo %ESC%[38;5;208m================================================================%ESC%[0m
echo %ESC%[38;5;208m             LAUNCHING INTERACTIVE CLI SHELL                     %ESC%[0m
echo %ESC%[38;5;208m================================================================%ESC%[0m
echo.
echo   Checking for required packages...
python -c "import prompt_toolkit" 2>nul
if %errorlevel% neq 0 (
    echo.
    echo   %ESC%[38;5;220m[WARNING] prompt_toolkit is not installed. Installing it now...%ESC%[0m
    pip install prompt_toolkit
)
python -c "import rich" 2>nul
if %errorlevel% neq 0 (
    echo.
    echo   %ESC%[38;5;220m[WARNING] rich is not installed. Installing it now...%ESC%[0m
    pip install rich
)
echo.
echo   Starting Interactive CLI Shell...
chcp 65001 >nul 2>&1
set PYTHONIOENCODING=utf-8
call python vasis_cli.py
pause
goto menu

:start_chat
cls
echo %ESC%[38;5;208m================================================================%ESC%[0m
echo %ESC%[38;5;208m             LAUNCHING INTERACTIVE CLI CHAT                     %ESC%[0m
echo %ESC%[38;5;208m================================================================%ESC%[0m
echo.
echo   Load 1 or more PDFs into the session. Ask questions about
echo   any paper, or cross-paper comparisons with 2+ papers loaded.
echo.
echo   %ESC%[38;5;244mEnter ONE PDF path per line. Type DONE when finished (max 6).%ESC%[0m
echo   %ESC%[38;5;244mDo NOT add quotes - paths with spaces work automatically.%ESC%[0m
echo.
set "pdf_count=0"
set "pdf1="
set "pdf2="
set "pdf3="
set "pdf4="
set "pdf5="
set "pdf6="

:chat_add_pdf
set /p pdf="  %ESC%[38;5;208m^>%ESC%[0m %ESC%[1mPDF path (or DONE to start):%ESC%[0m "
:: Strip any surrounding quotes the user may have typed
set "pdf=%pdf:"=%"
:: Blank input - ignore and re-prompt
if "%pdf%"=="" goto chat_add_pdf
:: Done - launch chat
if /i "%pdf%"=="done" goto chat_run
:: Already at max papers
if %pdf_count% GEQ 6 (
    echo.
    echo   %ESC%[38;5;196m[ERROR] Maximum 6 PDFs supported. Type DONE to start.%ESC%[0m
    echo.
    goto chat_add_pdf
)
:: Validate file exists
if not exist "%pdf%" (
    echo.
    echo   %ESC%[38;5;196m[ERROR] File not found: "%pdf%"%ESC%[0m
    echo   %ESC%[38;5;244mCheck the path and try again - no quotes needed.%ESC%[0m
    echo.
    goto chat_add_pdf
)
:: Store in numbered variable
set /a pdf_count+=1
if %pdf_count%==1 set "pdf1=%pdf%"
if %pdf_count%==2 set "pdf2=%pdf%"
if %pdf_count%==3 set "pdf3=%pdf%"
if %pdf_count%==4 set "pdf4=%pdf%"
if %pdf_count%==5 set "pdf5=%pdf%"
if %pdf_count%==6 set "pdf6=%pdf%"
echo   %ESC%[38;5;82m[OK]%ESC%[0m Paper %pdf_count% added.
echo.
goto chat_add_pdf

:chat_run
if %pdf_count% LSS 1 (
    echo.
    echo   %ESC%[38;5;196m[ERROR] Please enter at least one PDF path before typing DONE.%ESC%[0m
    echo.
    pause
    goto menu
)
echo.
if %pdf_count%==1 (
    echo   %ESC%[38;5;45mStarting Interactive Chat with 1 paper...%ESC%[0m
    echo.
    call python main.py chat "%pdf1%"
) else (
    echo   %ESC%[38;5;45mStarting Vault with %pdf_count% papers...%ESC%[0m
    echo.
    if %pdf_count%==2 call python main.py vault "%pdf1%" "%pdf2%"
    if %pdf_count%==3 call python main.py vault "%pdf1%" "%pdf2%" "%pdf3%"
    if %pdf_count%==4 call python main.py vault "%pdf1%" "%pdf2%" "%pdf3%" "%pdf4%"
    if %pdf_count%==5 call python main.py vault "%pdf1%" "%pdf2%" "%pdf3%" "%pdf4%" "%pdf5%"
    if %pdf_count%==6 call python main.py vault "%pdf1%" "%pdf2%" "%pdf3%" "%pdf4%" "%pdf5%" "%pdf6%"
)
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
call python main.py index "%pdf%" --show-tree
pause
goto menu



:list_docs
cls
echo %ESC%[38;5;208m================================================================%ESC%[0m
echo %ESC%[38;5;208m            INDEXED DOCUMENTS IN LOCAL CACHE VAULT              %ESC%[0m
echo %ESC%[38;5;208m================================================================%ESC%[0m
echo.
call python main.py list
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
call python -m pytest tests/ -v
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
call python api.py
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
