# Publish the NativeAOT release binary (the Fritz/ChessBase deliverable).
#
# NativeAOT links with the MSVC toolchain; the link fails with an opaque "exit code 123" unless the Visual
# Studio Installer directory (which holds vswhere.exe) is on PATH. This script puts it there, then publishes.
#
#   pwsh ./publish.ps1
#
# Output: Eonego/bin/Release/net10.0/win-x64/publish/Eonego.exe  (self-contained, ~42 MB; embeds the NNUE net
# if nets/main.nnue is present). Requires the "Desktop development with C++" VS workload (link.exe + libs).

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot

$vsInstaller = 'C:\Program Files (x86)\Microsoft Visual Studio\Installer'
if (Test-Path (Join-Path $vsInstaller 'vswhere.exe')) {
    if (($env:PATH -split ';') -notcontains $vsInstaller) {
        $env:PATH = "$vsInstaller;$env:PATH"
    }
} else {
    Write-Warning "VS Installer not found at '$vsInstaller'; if AOT link fails with code 123, install VS with the C++ workload."
}

$proj = Join-Path $root 'Eonego/Eonego.fsproj'
Write-Host "Publishing NativeAOT (Release, win-x64)..." -ForegroundColor Cyan
dotnet publish $proj -c Release -r win-x64
if ($LASTEXITCODE -ne 0) { throw "publish failed (exit $LASTEXITCODE)" }

$exe = Join-Path $root 'Eonego/bin/Release/net10.0/win-x64/publish/Eonego.exe'
if (Test-Path $exe) {
    $mb = [math]::Round((Get-Item $exe).Length / 1MB, 1)
    Write-Host "OK: $exe ($mb MB)" -ForegroundColor Green
} else {
    throw "publish reported success but $exe is missing"
}
