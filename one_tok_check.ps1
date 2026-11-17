# Isolate PREFILL correctness: one emitted token, spec 2 (native packs require >= 2).
# A = canonical q3k pack.  B = raw3k experts.bin + raw3k GPU decode arm.
# STRATA_NATIVE_LOGITS dumps the full head logits row per emitted token; compare byte-exactly.
$ErrorActionPreference='Stop'
$root='C:\Users\rr\Desktop\Edge0-rdna'
$build=Join-Path $root 'edge0-strata\build'
$env:PATH='C:\Rocm10\bin;'+$env:PATH
$q='E:\Edge0-120B-q3k'
$r='E:\Edge0-120B-raw3k'

$common=@{
  STRATA_LATENT_A="$q/experts.bin"
  STRATA_LATENT_LABELS="$q/labels.bin"
  STRATA_LATENT_DENSE="$q/latent_dense_q4.bin"
  STRATA_LATENT_DENSE_FMT=''
  STRATA_EMBED_VRAM='1'; STRATA_LATENT_FUSED='1'; STRATA_LATENT_FUSED_DU='1'
  STRATA_LATENT_U_GPU='1'; STRATA_LAT_FUSED_Q3K='1'; STRATA_MTP_SIDE_STREAM='1'
  STRATA_PREFILL_TIMING='1'
}
function Run-One([string]$name,[string]$pack,[string]$raw3k) {
  foreach($k in $common.Keys){ if($common[$k] -eq ''){Remove-Item "Env:\$k" -ErrorAction SilentlyContinue}else{Set-Item "Env:\$k" $common[$k]} }
  Remove-Item 'Env:\STRATA_A_RAW3K' -ErrorAction SilentlyContinue
  if($raw3k -eq '1'){
    Set-Item 'Env:\STRATA_A_RAW3K' '1'
    Set-Item 'Env:\STRATA_LATENT_A' ($pack+'/experts.bin')
  }
  $lg=Join-Path $root "onelog_${name}.bin"
  Remove-Item $lg -ErrorAction SilentlyContinue
  Set-Item 'Env:\STRATA_NATIVE_LOGITS' $lg
  $args=@('--pack',$pack,'--native',"$q\model-skeleton.gguf",'--native-dense-gguf',"$q\model-skeleton.gguf",
    '--ple-gguf',"$q\ple-fp8.gguf",'--ple-io','direct','--expert-profile',"$q\profile.bin",
    '--expert-cache','auto','--prefill','832','--spec','2',
    '--mtp',"$q\mtp",'--tokens-file',(Join-Path $root 'prompt4k.txt'),
    '--max-context','4096','--kv','int8','--pcie-frac','0.1','--gpu','0','--max-new','1','--stats','--vram-reserve-mib','512')
  $log=Join-Path $root "one_${name}.log"
  $p=Start-Process -FilePath (Join-Path $build 'strata.exe') -ArgumentList $args -WorkingDirectory $build -NoNewWindow -Wait -PassThru -RedirectStandardOutput "$log.out" -RedirectStandardError "$log.err"
  Get-Content "$log.err" | Set-Content $log; Get-Content "$log.out" | Add-Content $log
  Remove-Item "$log.out","$log.err" -ErrorAction SilentlyContinue
  $lines=@(Get-Content $log | Where-Object { $_ -match '^\s*output\s*:' })
  if($lines.Count -eq 0){ $tok='(no output line)' } else { $tok=(($lines[0] -replace '^\s*output\s*:\s*','').Trim()) }
  [pscustomobject]@{name=$name;exit=$p.ExitCode;tokens=$tok;logits=$lg}
}
Remove-Item 'Env:\STRATA_NATIVE_LOGITS' -ErrorAction SilentlyContinue
$rows=@()
$rows+=Run-One 'a_q3k'   $q '0'
$rows+=Run-One 'b_raw3k' $r '1'
Remove-Item 'Env:\STRATA_NATIVE_LOGITS' -ErrorAction SilentlyContinue
$rows | Format-Table -AutoSize | Out-String -Width 220 | Write-Output
