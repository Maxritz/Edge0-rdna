@echo off
rem Put the ROCm 10.2 runtime beside strata.exe.  REQUIRED on this box: C:\Windows\System32\amdhip64.dll is
rem a stale HIP 5.7 runtime, and the loader finds it before PATH, so the engine reports "no ROCm-capable device".
rem build_hip_official.bat wipes build\, so re-run this after every build.
setlocal
set "DST=C:\Users\rr\Desktop\Edge0-rdna\edge0-strata\build"
if not exist "%DST%" (echo no build dir & exit /b 1)

for %%N in (amdhip64_7.dll amd_comgr.dll rocm_kpack.dll hipblas.dll rocblas.dll libhipblaslt.dll rocsolver.dll
            rocsparse.dll rocrand.dll hiprand.dll rocm-openblas.dll rocm-openblas64.dll) do (
  if exist "C:\Rocm10\bin\%%N" copy /y "C:\Rocm10\bin\%%N" "%DST%" >nul
)
rem rocblas.dll looks for its Tensile library under <its own dir>\rocblas\library
if not exist "%DST%\rocblas" xcopy /s /i /q /y "C:\Rocm10\bin\rocblas" "%DST%\rocblas" >nul
if not exist "%DST%\hipblaslt" xcopy /s /i /q /y "C:\Rocm10\bin\hipblaslt" "%DST%\hipblaslt" >nul
rem the import is the unversioned name; the exe's own folder is searched before System32
copy /y "%DST%\amdhip64_7.dll" "%DST%\amdhip64.dll" >nul
echo ROCm runtime staged into %DST%
endlocal
