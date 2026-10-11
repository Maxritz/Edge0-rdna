$ErrorActionPreference = "Continue"
$bin   = "D:\edge0\wt\win\build-hip\bin"
$model = "C:\x\Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf"
$tr    = "D:\edge0\hip-tracer"
$port  = 8097
$csv   = "D:\edge0\hip-kernels.csv"
$log = "D:\edge0\hip.log"; $lerr = "$log.err"
foreach ($f in @($log,$lerr,$csv)) { if (Test-Path $f) { Remove-Item $f } }

# The proxy must sit in the SAME directory as llama-server.exe: DLL search starts with the
# child exe's directory, ahead of PATH, and Copy-HipRuntime put the real amdhip64_7.dll there.
$env:HIP_TRACER_OUTPUT = $csv
$a = @("-m", $model, "-ngl", "99", "--n-cpu-moe", "20", "-fa", "auto",
       "--ctx-size", "4096", "--no-webui", "--port", "$port",
       "--reasoning-format", "none")
$p = Start-Process -FilePath "$bin\llama-server.exe" -ArgumentList $a `
     -RedirectStandardOutput $log -RedirectStandardError $lerr -PassThru -WindowStyle Hidden
$ok = $false
for ($i = 0; $i -lt 150; $i++) {
    Start-Sleep -Seconds 2
    if ($p.HasExited) { break }
    try { if ((Invoke-RestMethod "http://127.0.0.1:$port/health" -TimeoutSec 3).status -eq "ok") { $ok = $true; break } } catch {}
    $t = (Get-Content $lerr -Tail 1 -ErrorAction SilentlyContinue)
    if ($t -match "error|failed|assert") { Write-Output "LOAD-ERR: $t"; exit }
}
Write-Output "LOAD_OK=$ok after ~$([int]($i*2))s"
if ($ok) {
    $body = @{ prompt = "Write a detailed essay about the history of computing."; n_predict = 32;
               temperature = 0.7; top_p = 0.80; top_k = 20; presence_penalty = 1.5 } | ConvertTo-Json -Compress
    try { Invoke-RestMethod "http://127.0.0.1:$port/completion" -Method Post -Body $body -ContentType "application/json" -TimeoutSec 900 | Out-Null } catch { Write-Output "GENFAIL: $_" }
    Start-Sleep -Seconds 3
    Get-Content $lerr | Select-String "vram-trace. FINAL|pref-trace. POOL2 step|hip_tracer" | Select-Object -Last 3 | ForEach-Object { Write-Output $_.Line.Trim() }
    Get-Content $lerr | Select-String "eval time =" | Select-Object -Last 1 | ForEach-Object { Write-Output $_.Line.Trim() }
}
Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
