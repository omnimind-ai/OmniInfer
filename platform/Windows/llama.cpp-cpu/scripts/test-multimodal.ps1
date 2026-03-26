param(
    [string]$GatewayBaseUrl = "http://127.0.0.1:9000",
    [string]$Backend = "llama.cpp-cpu",
    [string]$Model = "Qwen3.5-0.8B-Q4_K_M/Qwen3.5-0.8B-Q4_K_M.gguf",
    [string]$Mmproj = "Qwen3.5-0.8B-Q4_K_M/mmproj-F32.gguf",
    [string]$ImagePath = ".\\tests\\pictures\\test1.png"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $ImagePath)) {
    throw "Image not found: $ImagePath"
}

Invoke-RestMethod `
    -Method Post `
    -Uri "$GatewayBaseUrl/omni/model/select" `
    -ContentType "application/json" `
    -Body (@{ backend = $Backend; model = $Model; mmproj = $Mmproj } | ConvertTo-Json -Compress) | Out-Null

$bytes = [System.IO.File]::ReadAllBytes((Resolve-Path $ImagePath))
$base64 = [System.Convert]::ToBase64String($bytes)
$imageDataUrl = "data:image/png;base64,$base64"

$payload = @{
    model = $Model
    backend = $Backend
    messages = @(
        @{
            role = "user"
            content = @(
                @{
                    type = "text"
                    text = "Describe this image in Chinese in one short paragraph."
                },
                @{
                    type = "image_url"
                    image_url = @{
                        url = $imageDataUrl
                    }
                }
            )
        }
    )
    temperature = 0.2
    max_tokens = 256
} | ConvertTo-Json -Depth 10

Invoke-RestMethod `
    -Method Post `
    -Uri "$GatewayBaseUrl/v1/chat/completions" `
    -ContentType "application/json" `
    -Body $payload
