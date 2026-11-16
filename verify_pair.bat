@echo off
rem 1) the constraint that must never break, with the staged U tile on by default
rem 2) the chunk-1024 question, which the stage_n/batch fix invalidated (A-stage tile fill)
cd /d C:\Users\rr\Desktop\Edge0-rdna
powershell -File run_512k_mtp.ps1 -Embed 1 -Tag ut
echo ctx=%ERRORLEVEL%
cmd /c chunk_ab2.bat
echo ALL_DONE
