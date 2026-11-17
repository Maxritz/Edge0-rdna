# Remote API endpoint (Windows / RDNA2)

`START-API.bat` starts Edge0-120B as a resident, OpenAI/Anthropic-compatible HTTP
server. It loads the same `serve/config-rdna2.json` engine contract that was used
for the byte-exact smoke test (`gfx1031`, 12 GiB, q3k pack + original MTP).

## Start

Double-click `edge0-strata\START-API.bat`, or from a command prompt:

```bat
cd /d C:\Users\rr\Desktop\Edge0-rdna\edge0-strata
START-API.bat
```

The console shows a `ready:` line when the model is loaded, for example:

```
ready: http://127.0.0.1:8080/v1  (OpenAI: /v1/chat/completions, Anthropic: /v1/messages, context 2048 tokens, API key required)
       open http://127.0.0.1:8080/ in a browser to chat; close this window to stop the model
       from other devices: http://192.168.29.228:8080/   (API: http://192.168.29.228:8080/v1)
```

Press Ctrl+C or close the window to stop the server and free the GPU.

## Defaults and overrides

| Setting | Default | Override |
|---|---|---|
| Engine config | `serve\config-rdna2.json` | `STRATA_CONFIG` |
| Listen address | `0.0.0.0` (all IPv4) | `STRATA_API_HOST` |
| Port | `8080` | `STRATA_API_PORT` |
| API key | `edge0-rdna2-2026` | `STRATA_API_KEY` |

Examples:

```bat
set STRATA_API_PORT=8090
set STRATA_API_KEY=my-secret-key
START-API.bat
```

For a file log, redirect when launching: `START-API.bat >> api-server.log 2>&1`.
The engine's own log is `engine-serve.log` (from the config's `log` field).

## Connect

Base URL from another device on the LAN:

```
http://192.168.29.228:8080/v1
```

Set the API key on every request (either header works):

```
Authorization: Bearer edge0-rdna2-2026
x-api-key: edge0-rdna2-2026
```

Quick checks:

```bat
curl http://192.168.29.228:8080/health
curl -H "Authorization: Bearer edge0-rdna2-2026" http://192.168.29.228:8080/v1/models
curl -H "Authorization: Bearer edge0-rdna2-2026" -H "Content-Type: application/json" ^
  -d "{\"model\":\"edge0-120b\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"max_tokens\":32}" ^
  http://192.168.29.228:8080/v1/chat/completions
```

Endpoints: `GET /health`, `GET /v1/models`, `GET /props`, `GET /slots`,
`POST /v1/chat/completions` (stream and non-stream), `POST /v1/messages`
(Anthropic), `GET /` (browser chat UI). The model name reported by `/v1/models`
is `edge0-120b`.

## Windows Firewall

`0.0.0.0` binds all interfaces, but Windows Firewall blocks the first inbound
connection. Either accept the Windows prompt for Python, or add a rule from an
elevated PowerShell (and set the active network profile to **Private**):

```powershell
New-NetFirewallRule -DisplayName "Strata 8080" -Direction Inbound -Protocol TCP -LocalPort 8080 -Action Allow -Profile Private
```

The server prints this hint itself at startup when bound beyond loopback.

## Notes

- `--host 0.0.0.0` without an API key is allowed but warned about; the launcher
  always sets a key. Do not reuse the default key on an untrusted network.
- DNS rebinding protection: requests with an unexpected `Host` header are refused
  unless the host is in `allowed_hosts` / `STRATA_ALLOWED_HOSTS` or an API key is
  supplied ( supplying the key is enough).
- The context is 2048 tokens in this config (server-managed window); the engine
  still supports 512K with the appropriate config/flags, but that is not the
  default here because of 12 GiB VRAM / commit safety on this card.
- `C:\Rocm10\bin` and `C:\Rocm10\lib\llvm\bin` are prepended to `PATH`, and the
  ROCm DLLs are staged beside `build\strata.exe`, so no extra setup is needed.
