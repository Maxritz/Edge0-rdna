# Self-contained Windows distribution (RDNA2 / gfx1031)

The engine is a native binary (`strata.exe`), but the HTTP API is Python. A proper
distribution therefore ships a **folder** containing the engine, its ROCm runtime,
a portable Python, and the server modules. It does not need the source checkout,
a system Python, or `C:\Rocm10` on `PATH`.

A single frozen `.exe` is deliberately not used: the server imports its own
`serve.*` / `strata_control.*` modules dynamically, so PyInstaller-style freezing
needs a large hidden-import list and produces an unsigned binary that Smart App
Control may block. The folder below is a normal, testable distribution.

## Layout (built at `E:\edge-0`)

```
E:\edge-0\
  strata.exe, strata-device.exe
  *.dll, rocblas\                 ROCm 10.2, staged beside the exes
  config-rdna2.json               engine contract, exe/cwd/log point here
  runtime\python\                 standalone CPython 3.11.15 + site-packages
  runtime\app\                    serve\, strata_control\, tools\
  START-API.bat, README.txt
E:\.kpack\                        gfx1031 blas/fft/rand kpacks
```

## Build steps

1. Build with `tools\hip\build_windows_rdna2.bat`. It stages the exes, all
   `%ROCM_ROOT%\bin\*.dll` and `rocblas\` into `edge0-strata\build`.

2. Copy the staged runtime, not the whole build tree (which also has probes,
   CMake caches and ISA dumps):

   ```bat
   robocopy <repo>\edge0-strata\build E:\edge-0 strata.exe strata-device.exe
   robocopy C:\Rocm10\bin E:\edge-0 *.dll
   robocopy C:\Rocm10\bin\rocblas E:\edge-0\rocblas /E
   ```

3. Bundle Python. Copy the uv-managed base interpreter and copy the packages from
   the checkout's `.venv\Lib\site-packages` into it:

   ```bat
   robocopy C:\Users\<you>\AppData\Roaming\uv\python\cpython-3.11-windows-x86_64-none E:\edge-0\runtime\python /E
   robocopy <repo>\edge0-strata\.venv\Lib\site-packages E:\edge-0\runtime\python\Lib\site-packages /E
   robocopy <repo>\edge0-strata\serve E:\edge-0\runtime\app\serve /E
   robocopy <repo>\edge0-strata\strata_control E:\edge-0\runtime\app\strata_control /E
   robocopy <repo>\edge0-strata\tools E:\edge-0\runtime\app\tools /E
   ```

   Use `runtime\python\python.exe` **directly**. A copied `.venv\Scripts\python.exe`
   is a uv shim that chains back to the original base interpreter on `C:`.

4. Copy `serve\config-rdna2.json` and rewrite `exe`, `cwd` and `log` to absolute
   paths in the distribution folder. Delete the other `config-*.json` from
   `runtime\app\serve` so stale absolute paths do not ship.

5. `rocBLAS` looks for `../.kpack` relative to its DLL folder. On NTFS the build
   script creates a junction; on exFAT copy the gfx1031 kpacks as real files into
   the parent of the distribution folder (`E:\.kpack` here).

6. Launcher:

   ```bat
   set "DEPLOY=%~dp0"
   set "PY=%DEPLOY%runtime\python\python.exe"
   set "PYTHONPATH=%DEPLOY%runtime\app"
   "%PY%" -m serve.server --engine strata --config "%DEPLOY%config-rdna2.json" --host 0.0.0.0 --port 8080
   ```

   Pass the key through `STRATA_API_KEY`, not `--api-key`, so it stays out of the
   process list. Do not add `C:\Rocm10` to `PATH`; the DLLs are beside the exe.

## Verification checklist

- `strata-device.exe --list-devices` reports the expected `gfx1031` card.
- `python -c "import sys,serve.server; print(sys.base_prefix, serve.server.__file__)"`
  prints paths under the distribution folder, with no checkout path on `sys.path`.
- `GET /health` returns `loaded:true`; `GET /v1/models` returns the configured
  model name; `POST /v1/chat/completions` works streamed and non-streamed from
  both loopback and the LAN IP.
- A generate run from the distribution folder produces the canonical byte-exact
  tokens for the benchmark prompt.

## Caveats

- The model pack is not part of the distribution: `E:\Edge0-120B-q3k` must stay at
  the path named in the config.
- `E:` is exFAT, so no junctions or symlinks can be used there.
- Windows Firewall must allow inbound TCP on the chosen port; an unsigned exe may
  also be blocked by Smart App Control.
