$ErrorActionPreference = "Continue"
$bin = "D:\edge0\wt\win\build-hip\bin"
$log = "D:\edge0\diag.log"; $lerr = "$log.err"
foreach ($f in @($log,$lerr)) { if (Test-Path $f) { Remove-Item $f } }

# EXACT S21 config recorded at 33.10 tok/s.
$a = @("-m","C:\x\Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf",
       "-ngl","99","--n-cpu-moe","20","-fa","auto","--cache-type-k","q8_0",
       "--cache-type-v","q4_0","--ctx-size","32768","-t","8","--no-webui","--port","8099",
       "--reasoning-format","none")
$p = Start-Process -FilePath "$bin\llama-server.exe" -ArgumentList $a `
     -RedirectStandardOutput $log -RedirectStandardError $lerr -PassThru -WindowStyle Hidden
$ok = $false
for ($i=0; $i -lt 200; $i++) {
    Start-Sleep -Seconds 2
    if ($p.HasExited) { break }
    try { if ((Invoke-RestMethod "http://127.0.0.1:8099/health" -TimeoutSec 3).status -eq "ok") { $ok=$true; break } } catch {}
}
Write-Output "LOADED=$ok"
if ($ok) {
    $b = @{ prompt = ("Detail the history of computing at length. " * 60); n_predict = 48;
            temperature = 0.7; top_p = 0.80; top_k = 20; presence_penalty = 1.5 } | ConvertTo-Json -Compress
    $r = Invoke-RestMethod "http://127.0.0.1:8099/completion" -Method Post -Body $b -ContentType "application/json" -TimeoutSec 600
    Write-Output ("PP  = {0:N2} tok/s ({1:N0} tok in {2:N0} ms)" -f ($r.timings.prompt_per_second), $r.timings.prompt_n, $r.timings.prompt_ms)
    Write-Output ("TG  = {0:N2} tok/s ({1:N0} tok in {2:N0} ms)" -f ($r.timings.predicted_per_second), $r.timings.predicted_n, $r.timings.predicted_ms)
} else {
    Get-Content $lerr -ErrorAction SilentlyContinue | Select-Object -Last 5 | ForEach-Object { Write-Output "  $_" }
}
Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue

Write-Output '=== LOAD / PLACEMENT ==='
Get-Content $lerr -ErrorAction SilentlyContinue |
  Select-String "CPU model size|offloaded|ROC|VRAM|CPU_Mapped|buffer|n_gpu_layers|n_cpu_moe|overrides|mapped|mmap|MAP_HOST|load-mode" |
  ForEach-Object { Write-Output $_.Line.Trim() }
