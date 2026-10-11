$ErrorActionPreference = "Continue"
$bin   = "D:\edge0\wt\win\build-hip\bin"
$log = "D:\edge0\llamabench.log"; $lerr = "$log.err"
foreach ($f in @($log,$lerr)) { if (Test-Path $f) { Remove-Item $f } }

# Standard pp512 / tg128. -ngl is OMITTED so common_fit_params can auto-fit layers.
# --load-mode none is the validated lever for a model that fits RAM (RUN-014: pp2048 434 -> 772).
$m = "C:\x\Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf"

$a = @("-m", $m, "-fa", "auto", "-p", "512", "-n", "128", "-r", "3",
       "-t", "32", "--load-mode", "none")
$p = Start-Process -FilePath "$bin\llama-bench.exe" -ArgumentList $a `
     -RedirectStandardOutput $log -RedirectStandardError $lerr -PassThru -WindowStyle Hidden
$p.WaitForExit(1800000) | Out-Null
Write-Output "EXIT=$($p.ExitCode)"
Get-Content $log -ErrorAction SilentlyContinue | ForEach-Object { Write-Output $_ }
Get-Content $lerr -ErrorAction SilentlyContinue |
  Select-String "CPU model size|offloaded|n_gpu_layers|load-mode|KV self|failed" |
  Select-Object -Last 10 | ForEach-Object { Write-Output "  $($_.Line.Trim())" }
