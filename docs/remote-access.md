# Remote Access

OmniInfer can expose the desktop inference API through Cloudflare Quick Tunnel for temporary remote access without router port forwarding, a public IP address, or a Cloudflare account.

Quick Tunnel is best for demos, testing, and short-lived personal access. Cloudflare assigns a random `trycloudflare.com` URL and does not guarantee uptime for this mode. For best compatibility, use non-streaming requests; streaming over Quick Tunnel is best-effort only.

## Cloudflare Quick Tunnel

Start the gateway with Cloudflare mode:

```sh
./omniinfer serve --cloudflare
```

Windows:

```powershell
.\omniinfer.ps1 serve --cloudflare --window-mode visible
```

OmniInfer will:

- keep the local gateway bound to `127.0.0.1`
- download and update a managed `cloudflared` binary under `.local/tools/cloudflared`
- require an API key for requests arriving through Cloudflare
- generate a session API key when `--api-key` or `OMNIINFER_API_KEY` is not set
- keep `/omni/*` management endpoints local-only
- start `cloudflared tunnel --url http://127.0.0.1:<port>`
- stop `cloudflared` when OmniInfer exits

The startup output includes:

```text
Cloudflare Quick Tunnel: https://example.trycloudflare.com
OpenAI Base URL: https://example.trycloudflare.com/v1
Health URL: https://example.trycloudflare.com/health
API key: oi_example
```

Use the OpenAI Base URL and API key in remote clients.

## Managed cloudflared

OmniInfer uses its own managed `cloudflared` binary by default instead of relying on a system-wide install. This avoids old system versions and keeps Cloudflare mode self-contained.

On first `--cloudflare` use, OmniInfer queries the latest Cloudflare GitHub release, selects the asset for the current operating system and CPU architecture, downloads it to:

```text
.local/tools/cloudflared/
```

The download is verified with the release asset SHA-256 digest when GitHub provides one. OmniInfer records the installed version and asset metadata in `.local/tools/cloudflared/manifest.json` and checks for newer releases on later Cloudflare starts.

Supported automatic assets include:

| System | Architecture | Asset |
|---|---|---|
| Linux | x86_64 / amd64 | `cloudflared-linux-amd64` |
| Linux | aarch64 / arm64 | `cloudflared-linux-arm64` |
| Linux | x86 / 386 | `cloudflared-linux-386` |
| Linux | ARMv7 hard-float | `cloudflared-linux-armhf` |
| Linux | other 32-bit ARM | `cloudflared-linux-arm` |
| macOS | Apple Silicon | `cloudflared-darwin-arm64.tgz` |
| macOS | Intel | `cloudflared-darwin-amd64.tgz` |
| Windows | x64 / amd64 | `cloudflared-windows-amd64.exe` |
| Windows | x86 / 386 | `cloudflared-windows-386.exe` |

Advanced users can override the managed binary:

Example:

```sh
./omniinfer serve --cloudflare --cloudflared-path /usr/local/bin/cloudflared
```

If a default `~/.cloudflared/config.yaml` exists, Quick Tunnel may fail because account-managed tunnel configuration can conflict with account-less Quick Tunnel usage. OmniInfer warns about this but does not modify user Cloudflare files.

## Remote Requests

Health:

```sh
curl https://example.trycloudflare.com/health \
  -H "Authorization: Bearer oi_example"
```

OpenAI-compatible chat:

```sh
curl https://example.trycloudflare.com/v1/chat/completions \
  -H "Authorization: Bearer oi_example" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local",
    "messages": [{"role": "user", "content": "Introduce OmniInfer in one sentence."}],
    "stream": false,
    "max_tokens": 64
  }'
```

Tokenize or detokenize:

```sh
curl https://example.trycloudflare.com/tokenize \
  -H "Authorization: Bearer oi_example" \
  -H "Content-Type: application/json" \
  -d '{"content":"Hello","with_pieces":false}'
```

You can also send the key as:

```text
x-api-key: oi_example
```

## Security Model

Treat the Quick Tunnel URL as public. Anyone who has the URL can reach the gateway, so OmniInfer requires the API key for inference endpoints exposed through Cloudflare.

Cloudflare terminates HTTPS for the public URL and forwards traffic to the local gateway. Do not describe this mode as end-to-end encrypted against Cloudflare.

The following combinations are intentionally rejected:

- `--cloudflare --lan`
- `--cloudflare --host 0.0.0.0`
- `--cloudflare --allow-insecure-lan`
- `--cloudflare --allow-remote-management`

## Streaming Boundary

Cloudflare Quick Tunnel is intended for testing and has product limitations. Use non-streaming chat for the most reliable remote behavior. Standard OpenAI SSE and OmniInfer line streaming may work in some environments, but they are not guaranteed in Quick Tunnel mode.
