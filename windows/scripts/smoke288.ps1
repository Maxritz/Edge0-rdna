$ErrorActionPreference = "Continue"
$bin   = "D:\edge0\wt\win\build-hip\bin"
$model = "C:\x\qwen3.8-flash-next-reap-288-Q4_K_M.gguf"
$port  = 8091
$log = "D:\edge0\smoke288.log"; $lerr = "$log.err"
foreach ($f in @($log,$lerr)) { if (Test-Path $f) { Remove-Item $f } }

# QSA/GDN shell to VRAM, expert weights stream from host. thinking off.
$a = @("-m", $model, "-ngl", "99", "-cmoe", "-c", "4096",
       "--no-webui", "--port", "$port", "-t", "32",
       "--reasoning-format", "none")
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
    # non-thinking (instruct) params per Qwen3.8-Flash-Next recommendation
    $body = @{ prompt = "In one sentence, what is a mixture of experts network?";
               n_predict = 32; temperature = 0.7; top_p = 0.80; top_k = 20;
               min_p = 0.0; presence_penalty = 1.5; repetition_penalty = 1.0 } | ConvertTo-Json -Compress
    try { Invoke-RestMethod "http://127.0.0.1:$port/completion" -Method Post -Body $body -ContentType "application/json" -TimeoutSec 1800 | Out-Null }
    catch { Write-Output "GENFAIL: $_" }
    Start-Sleep -Seconds 2
    Get-Content $lerr |
        Select-String "n_expert|n_embd|n_layer|expert|eval time|prompt eval|CPU model|VRAM|memory|offload|buffer|reasoning" |
        Select-Object -Last 30 |
        ForEach-Object { Write-Output $_.Line.Trim() }
}
Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
