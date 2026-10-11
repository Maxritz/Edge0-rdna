$ErrorActionPreference = "Continue"
$bin   = "D:\edge0\wt\win\build-hip\bin"
$model = "C:\x\qwen3.8-flash-next-reap-288-Q4_K_M.gguf"
$port  = 8092
$log = "D:\edge0\probe-hook.log"; $lerr = "$log.err"
foreach ($f in @($log,$lerr)) { if (Test-Path $f) { Remove-Item $f } }

# No -cmoe: first N layers' experts stay in host RAM, the rest go to VRAM if they fit.
# 4 tokens only: this is an engagement probe, not a throughput measurement.
$a = @("-m", $model, "-ngl", "99", "--n-cpu-moe", "44", "-c", "2048", "-t", "32",
       "--no-webui", "--port", "$port",
       "--reasoning-format", "none", "--no-reasoning-preserve")
$env:E0_VRAM_MB          = "1024"
$env:E0_VRAM_TRACE_N     = "1"
$env:E0_VRAM_HOT_N       = "0"
$env:E0_VRAM_MAX_SWAPS   = "8"

$p = Start-Process -FilePath "$bin\llama-server.exe" -ArgumentList $a `
     -RedirectStandardOutput $log -RedirectStandardError $lerr -PassThru -WindowStyle Hidden
$ok = $false
for ($i = 0; $i -lt 300; $i++) {
    Start-Sleep -Seconds 2
    if ($p.HasExited) { break }
    try { $h = Invoke-RestMethod "http://127.0.0.1:$port/health" -TimeoutSec 3
          if ($h.status -eq "ok") { $ok = $true; break } } catch {}
    $t = (Get-Content $lerr -Tail 1 -ErrorAction SilentlyContinue)
    if ($t -match "error|failed|assert") { Write-Output "LOAD-ERR: $t"; exit }
}
Write-Output "LOAD_OK=$ok after ~$([int]($i*2))s"
if ($ok) {
    $body = @{ prompt = "Count to five."; n_predict = 4; temperature = 0.7; top_p = 0.80; top_k = 20; presence_penalty = 1.5 } | ConvertTo-Json -Compress
    try { Invoke-RestMethod "http://127.0.0.1:$port/completion" -Method Post -Body $body -ContentType "application/json" -TimeoutSec 3600 | Out-Null }
    catch { Write-Output "GENFAIL: $_" }
    Start-Sleep -Seconds 3
    Write-Output "--- vram-trace lines ---"
    Get-Content $lerr | Select-String "vram-trace" | Select-Object -First 3 | ForEach-Object { Write-Output $_.Line.Trim() }
    Write-Output "... last trace ..."
    Get-Content $lerr | Select-String "vram-trace" | Select-Object -Last 3 | ForEach-Object { Write-Output $_.Line.Trim() }
    Write-Output "--- what landed where ---"
    Get-Content $lerr | Select-String "offloaded|CPU model|CUDA_Host|buffer size|recompute" | Select-Object -Last 8 | ForEach-Object { Write-Output $_.Line.Trim() }
}
Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
