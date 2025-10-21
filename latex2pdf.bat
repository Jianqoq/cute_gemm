@echo off

REM ========== MiKTeX 路径 ==========
set MIKTEX_BIN=C:\Users\%USERNAME%\AppData\Local\Programs\MiKTeX\miktex\bin\x64

REM ========================================

if "%1"=="" (
    echo Usage: latex2pdf.bat [filename without .tex]
    echo Example: latex2pdf.bat gemm-mma
    pause
    exit /b 1
)

set FILENAME=%1
echo ========================================
echo MiKTeX PDF Generator
echo ========================================
echo Input: %FILENAME%.tex
echo Output: %FILENAME%.pdf
echo ========================================
echo.

REM 检查文件是否存在
if not exist %FILENAME%.tex (
    echo ❌ Error: %FILENAME%.tex not found!
    pause
    exit /b 1
)

REM 编译 LaTeX
echo Compiling LaTeX...
"%MIKTEX_BIN%\pdflatex.exe" -interaction=nonstopmode %FILENAME%.tex

if exist %FILENAME%.pdf (
    echo.
    echo ========================================
    echo ✅ Success!
    echo ========================================
    echo PDF: %FILENAME%.pdf
    
    echo Cleaning up temporary files...
    del %FILENAME%.aux 2>nul
    del %FILENAME%.log 2>nul
    del %FILENAME%.out 2>nul
    echo Temporary files removed.
    
    echo Opening PDF...
    start %FILENAME%.pdf
    
    echo ========================================
) else (
    echo.
    echo ========================================
    echo ❌ PDF Generation Failed!
    echo ========================================
    echo.
    if exist %FILENAME%.log (
        echo Error details:
        type %FILENAME%.log | findstr /i "error" /i "undefined"
        echo.
        echo Full log: %FILENAME%.log
    )
    echo ========================================
)

pause