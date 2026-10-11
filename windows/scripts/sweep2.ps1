$ErrorActionPreference = "Continue"
$bin = "D:\edge0\wt\win\build-hip\bin"
$log = "D:\edge0\sweep2.log"; $lerr = "$log.err"
foreach ($f in @($log,$lerr)) { if (Test-Path $f) { Remove-Item $f } }

# Match the working S21 config exactly, sweep ncmoe to expose the fit threshold.
$a = @("-m","C:\x\Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf",
       "-ngl","99","-ncmoe","0,2,4,6,8,10,12,14,16,18,20,24,28,32",
       "-fa","auto","-ctk","q8_0","-ctv","q4_0","-t","8",
       "-p","512","-n","128","-r","2","-o","csv")
$p = Start-Process -FilePath "$bin\llama-bench.exe" -ArgumentList $a `
     -RedirectStandardOutput $log -RedirectStandardError $lerr -PassThru -WindowStyle Hidden
$p | Wait-Process -Timeout 1500 -ErrorAction SilentlyContinue
if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue }
Write-Output "--- RESULTS (ncmoe, test, t/s) ---"
Get-Content $log -ErrorAction SilentlyContinue | ForEach-Object { Write-Output $_ }
