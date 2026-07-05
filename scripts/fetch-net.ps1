# Download the FullThreats NNUE net into nets/main.nnue (gitignored, ~106 MB).
# Required before publish if you want the net embedded in Eonego.exe.
#
#   pwsh ./scripts/fetch-net.ps1
#   pwsh ./scripts/fetch-net.ps1 -Version v0.0.4
#
# The net is attached to GitHub releases as main.nnue (not in git — too large).
# Runtime override without rebuild: EONEGO_NET=/path/to/net.nnue

param(
    [string]$Version = "latest"
)

$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$netDir = Join-Path $root 'nets'
$netPath = Join-Path $netDir 'main.nnue'

if (Test-Path $netPath) {
    $mb = [math]::Round((Get-Item $netPath).Length / 1MB, 1)
    Write-Host "OK: $netPath already present ($mb MB)" -ForegroundColor Green
    exit 0
}

New-Item -ItemType Directory -Force -Path $netDir | Out-Null

$url =
    if ($Version -eq 'latest') {
        'https://github.com/Houijasu/Eonego/releases/latest/download/main.nnue'
    } else {
        $tag = if ($Version.StartsWith('v')) { $Version } else { "v$Version" }
        "https://github.com/Houijasu/Eonego/releases/download/$tag/main.nnue"
    }

Write-Host "Downloading main.nnue from $url ..." -ForegroundColor Cyan
try {
    Invoke-WebRequest -Uri $url -OutFile $netPath -UseBasicParsing
} catch {
    Remove-Item $netPath -ErrorAction SilentlyContinue
    Write-Error @"
Failed to download main.nnue ($($_.Exception.Message)).

The trained weights are not stored in git. Either:
  1. Run this script again after a release with a main.nnue asset is published, or
  2. Copy any compatible FullThreats net (version 0x6A448AFA) to nets/main.nnue, or
  3. Set EONEGO_NET=<path> at runtime (no embed; search still works).
"@
}

$mb = [math]::Round((Get-Item $netPath).Length / 1MB, 1)
if ((Get-Item $netPath).Length -lt 1MB) {
    Remove-Item $netPath -Force
    throw "Download looks truncated (< 1 MB); delete nets/main.nnue and retry."
}

Write-Host "OK: $netPath ($mb MB)" -ForegroundColor Green
