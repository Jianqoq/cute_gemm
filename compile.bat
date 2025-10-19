@echo off
echo compiling gemm-simple...

nvcc -std=c++17 -arch=sm_80 -I./include gemm-simple.cu -o gemm-simple.exe

if %errorlevel% equ 0 (
    echo compile success!
    echo running program...
    gemm-simple.exe
) else (
    echo compile failed!
)

pause
