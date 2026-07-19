# Publish the NativeAOT release binary (the shipping UCI deliverable).
#
# NativeAOT links with the MSVC toolchain. Instead of relying on whatever PATH the caller happens to
# have, this script initializes the full x64 MSVC developer environment (vcvars64.bat) in a child
# process whose PATH is sanitized down to the OS + dotnet + VS Installer directories, then imports
# the resulting variables. That keeps stray toolchains (mingw link.exe, older VS, shim managers)
# from leaking into the link step, and makes the "exit code 123" class of failures deterministic.
#
#   pwsh ./publish.ps1
#
# Output: Eonego/bin/Release/net10.0/win-x64/publish/Eonego.exe (~110 MB with embedded net).
# Auto-runs scripts/fetch-net.ps1 when nets/main.nnue is absent (e.g. LFS not pulled).
# Requires the "Desktop development with C++" VS workload (link.exe + libs).

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$startedAt = Get-Date

$netPath = Join-Path $root 'nets/main.nnue'
if (-not (Test-Path $netPath)) {
    Write-Host "nets/main.nnue missing — fetching release net for embed ..." -ForegroundColor Yellow
    & (Join-Path $root 'scripts/fetch-net.ps1')
}

# --- MSVC developer environment ----------------------------------------------------------------
$vsInstaller = 'C:\Program Files (x86)\Microsoft Visual Studio\Installer'
$vswhere = Join-Path $vsInstaller 'vswhere.exe'
if (-not (Test-Path $vswhere)) {
    throw "vswhere.exe not found at '$vswhere'; install Visual Studio with the C++ workload."
}
$vsRoot = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vsRoot) { throw "no VS installation with the C++ toolset found (vswhere returned nothing)" }
$vcvars = Join-Path $vsRoot 'VC\Auxiliary\Build\vcvars64.bat'
if (-not (Test-Path $vcvars)) { throw "vcvars64.bat not found under '$vsRoot'" }

$dotnetDir = Split-Path -Parent (Get-Command dotnet).Source
$basePath = "$env:SystemRoot\System32;$env:SystemRoot;$dotnetDir;$vsInstaller"
$marker = '__EONEGO_VCVARS_OK__'
$tmpCmd = Join-Path ([IO.Path]::GetTempPath()) "eonego-vcvars-$PID.cmd"
@"
@echo off
set "PATH=$basePath"
call "$vcvars" >nul 2>&1
if errorlevel 1 exit /b 1
echo $marker
set
"@ | Set-Content -Path $tmpCmd -Encoding ASCII
try {
    $cmdOut = & cmd.exe /d /c $tmpCmd
    if ($LASTEXITCODE -ne 0 -or $cmdOut -notcontains $marker) {
        throw "vcvars64.bat failed (exit $LASTEXITCODE)"
    }
} finally {
    Remove-Item $tmpCmd -Force -ErrorAction SilentlyContinue
}
$import = $false
foreach ($line in $cmdOut) {
    if ($line -eq $marker) { $import = $true; continue }
    if ($import -and $line -match '^([^=]+)=(.*)$') {
        # vcvars sets Platform=x64, which MSBuild honors and silently redirects output to bin\x64\.
        # Configuration would do the same. Neither is needed by link.exe — drop both.
        if ($Matches[1] -in @('Platform', 'Configuration')) { continue }
        Set-Item -Path ('Env:' + $Matches[1]) -Value $Matches[2]
    }
}
Write-Host "MSVC x64 environment imported from $vcvars" -ForegroundColor DarkGray

# --- publish -----------------------------------------------------------------------------------
$proj = Join-Path $root 'Eonego/Eonego.fsproj'
$exe = Join-Path $root 'Eonego/bin/Release/net10.0/win-x64/publish/Eonego.exe'
# A stale artifact must not survive a failed link and masquerade as a fresh build.
if (Test-Path $exe) { Remove-Item $exe -Force }

Write-Host "Publishing NativeAOT (Release, win-x64)..." -ForegroundColor Cyan
dotnet publish $proj -c Release -r win-x64
if ($LASTEXITCODE -ne 0) { throw "publish failed (exit $LASTEXITCODE)" }

if (-not (Test-Path $exe)) { throw "publish reported success but $exe is missing" }
$exeItem = Get-Item $exe
if ($exeItem.LastWriteTime -lt $startedAt) {
    throw "$exe was not rebuilt (timestamp predates this run) — treat as a failed publish"
}

# --- smoke test --------------------------------------------------------------------------------
$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = $exe
$psi.RedirectStandardInput = $true
$psi.RedirectStandardOutput = $true
$psi.UseShellExecute = $false
$p = [System.Diagnostics.Process]::Start($psi)
$outTask = $p.StandardOutput.ReadToEndAsync()
$p.StandardInput.WriteLine('uci')
$p.StandardInput.WriteLine('isready')
$p.StandardInput.WriteLine('quit')
$p.StandardInput.Close()
if (-not $p.WaitForExit(20000)) {
    $p.Kill()
    throw 'smoke test: engine did not exit within 20s'
}
$smoke = $outTask.Result
if ($smoke -notmatch 'uciok') { throw 'smoke test: no uciok in engine output' }
if ($smoke -notmatch 'readyok') { throw 'smoke test: no readyok in engine output' }

$mb = [math]::Round($exeItem.Length / 1MB, 1)
Write-Host "OK: $exe ($mb MB) — uciok/readyok smoke test passed" -ForegroundColor Green
