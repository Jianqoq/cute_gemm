@echo off

if "%1"=="" (
    echo Usage: compile.bat [filename without .cu]
    echo Example: compile.bat gemm-simple
    pause
    exit /b 1
)

set FILENAME=%1
echo compiling %FILENAME%...

nvcc -std=c++17 -arch=sm_80 -I./include %FILENAME%.cu -o %FILENAME%.exe

if %errorlevel% equ 0 (
    echo compile success!
    echo running program...
    
    REM 把输出保存到文件
    %FILENAME%.exe > %FILENAME%.tex
    
    echo.
    echo Output saved to: %FILENAME%.tex
) else (
    echo compile failed!
)

pause