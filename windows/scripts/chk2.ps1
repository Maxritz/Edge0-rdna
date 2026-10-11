$s = "D:\edge0\sweep.log.err"
Write-Output '=== PLACEMENT (first arm) ==='
Get-Content $s -ErrorAction SilentlyContinue |
  Select-String "CPU model size|offloaded|n_gpu_layers|rocm|ROCm|VRAM|CPU_Mapped|buffer size|failed|abort|n_cpu_moe|overrides" |
  Select-Object -First 25 | ForEach-Object { Write-Output $_.Line.Trim() }
Write-Output '=== tail ==='
Get-Content $s -ErrorAction SilentlyContinue | Select-Object -Last 6 | ForEach-Object { Write-Output $_.Line.Trim() }
