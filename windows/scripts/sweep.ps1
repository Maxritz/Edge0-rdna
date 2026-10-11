$ErrorActionPreference = "Continue"
$bin = "D:\edge0\wt\win\build-hip\bin"
$log = "D:\edge0\sweep.log"; $lerr = "$log.err"
foreach ($f in @($log,$lerr)) { if (Test-Path $f) { Remove-Item $f } }

# -ngl 99 pins all layers to ROCm; --n-cpu-moe N moves the first N MoE LAYERS' expert
# tensors to host. Multiple comma values = llama-bench runs each arm.
$a = @("-m", "C:\x\Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf",
       "-ngl", "99", "-ncmoe", "0,4,8,16,20,24,28",
       "-fa", "auto", "-p", "512", "-n", "128", "-r", "3", "-t", "32",
       "--load-mode", "none", "-o", "md")
$p = Start-Process -FilePath "$bin\llama-bench.exe" -ArgumentList $a `
     -RedirectStandardOutput $log -RedirectStandardError $lerr -PassThru -WindowStyle Hidden
$p.WaitForExit(2400000) | Out-Null
Write-Output "EXIT=$($p.ExitCode)"
Get-Content $log -ErrorAction SilentlyContinue | ForEach-Object { Write-Output $_ }
Get-Content $lerr -ErrorAction SilentlyContinue |
  Select-String "CPU model size|offloaded|failed|abort|error" |
  Select-Object -Last 8 | ForEach-Object { Write-Output "  $($_.Line.Trim())" }
