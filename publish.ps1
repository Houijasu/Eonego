# Publish a shipping UCI binary. Cross-platform (Windows / Linux / macOS) — run with pwsh 7+.
#
#   pwsh ./publish.ps1                                   # portable AOT for the host OS (AVX2 baseline)
#   pwsh ./publish.ps1 -Mode jit                         # single-file JIT build: runtime CPU dispatch
#   pwsh ./publish.ps1 -Mode jit -Rid linux-x64          # Linux tester build, produced from Windows
#   pwsh ./publish.ps1 -InstructionSet native            # private max-speed build for THIS CPU only
#   pwsh ./publish.ps1 -CopyTo artifacts/dist/Eonego.exe # drop a named copy next to the publish dir
#
# Auto-runs scripts/fetch-net.ps1 when nets/main.nnue is absent (e.g. LFS not pulled).
#
# --- Why the default is AVX2 and not "whatever this machine has" ------------------------------------
# NativeAOT resolves `Avx2.IsSupported` / `Bmi2.X64.IsSupported` at COMPILE time - there is no runtime
# CPU dispatch in an AOT image. `IlcInstructionSet=native` therefore bakes in this build machine's ISA
# (AVX-512, AVX-VNNI on a 13980HX) and the binary faults on any lesser CPU, so it must never be the
# default for a binary handed to a tester. Measured 2026-07-27 on the audit bench (d14 midgame, 4 runs,
# a loaded laptop - treat the spread as noise, only the order of magnitude is meaningful):
#
#   build                        midgame nps          runs on
#   --------------------------   ------------------   ------------------------------------------
#   AOT, no ISA (ILC baseline)     34k                anything x64, but ~17x slower: no SIMD NNUE
#   AOT, x86-64-v3 (default)      541k - 651k         Haswell 2013+ / Zen1 2017+
#   jit, self-contained           537k - 587k         anything x64, AVX-512 included
#   AOT, native                   n/a                 the build host only
#
# Node counts were byte-identical (186731 startpos / 366639 midgame) across every build above: the ISA
# and the AOT/JIT choice never change the search tree, so the nodesweep contract survives all of them.
#
# The engine has AVX2 and AVX-512 NNUE kernels and a scalar fallback, but no SSE path - below AVX2 it
# collapses to scalar, which is why the ILC baseline is unusable and x86-64-v3 is the practical floor.
# v3 covers essentially every tester's CPU, so AOT/v3 is the default.
#
# -Mode jit is the only build with genuine runtime hardware detection (the JIT sees the real CPU and
# picks AVX-512 / AVX2 / scalar per method). It measured on par with AOT/v3 within noise, is ~70 MB
# larger, and is the fallback for a tester whose machine predates AVX2 or for cross-OS publishing.

[CmdletBinding()]
param(
    # aot: NativeAOT native image, ISA fixed at compile time (see table above).
    # jit: self-contained single-file managed build, CPU detected at runtime.
    [ValidateSet('aot', 'jit')]
    [string]$Mode = 'aot',

    # Target runtime identifier. Defaults to the host. AOT cannot cross-compile between operating
    # systems (it links with the host's native toolchain); jit can, so a Linux tester binary can be
    # produced from Windows with -Mode jit -Rid linux-x64.
    [string]$Rid = '',

    # AOT only. Overrides the ISA baseline; 'native' pins to the build machine's CPU.
    [string]$InstructionSet = '',

    [string]$CopyTo = '',
    [switch]$NoSmoke
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$startedAt = Get-Date

$netPath = Join-Path $root 'nets/main.nnue'
if (-not (Test-Path $netPath)) {
    Write-Host "nets/main.nnue missing - fetching release net for embed ..." -ForegroundColor Yellow
    & (Join-Path $root 'scripts/fetch-net.ps1')
}

# --- target resolution ---------------------------------------------------------------------------
$hostOs =
    if ($IsWindows) { 'win' }
    elseif ($IsLinux) { 'linux' }
    elseif ($IsMacOS) { 'osx' }
    else { throw "unsupported host OS" }

$hostArch = switch ([System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture) {
    'X64' { 'x64' }
    'Arm64' { 'arm64' }
    default { throw "unsupported host architecture: $_" }
}

if (-not $Rid) { $Rid = "$hostOs-$hostArch" }
$targetOs = ($Rid -split '-')[0]
$targetArch = ($Rid -split '-')[-1]
$crossOs = $targetOs -ne $hostOs

if ($Mode -eq 'aot' -and $crossOs) {
    throw "NativeAOT cannot cross-compile from '$hostOs' to '$targetOs' (it links with the host toolchain). " +
          "Either run this script on a $targetOs machine, or use -Mode jit -Rid $Rid, which cross-publishes fine."
}

# x64 needs an explicit floor because the ILC baseline has no SIMD NNUE path (see the table above).
# arm64's baseline already includes NEON, which is what the engine's vector kernels need there.
if ($Mode -eq 'aot' -and -not $InstructionSet -and $targetArch -eq 'x64') {
    $InstructionSet = 'x86-64-v3'
}
if ($Mode -eq 'jit' -and $InstructionSet) {
    throw "-InstructionSet applies to -Mode aot only; a jit build detects the CPU at runtime."
}

$exeName = if ($targetOs -eq 'win') { 'Eonego.exe' } else { 'Eonego' }
$exe = Join-Path $root "Eonego/bin/Release/net10.0/$Rid/publish/$exeName"

# --- native toolchain ----------------------------------------------------------------------------
# AOT links with a platform linker. jit builds pull a prebuilt apphost from the runtime pack and need
# no native toolchain at all, which is what makes cross-OS publishing possible.
if ($Mode -eq 'aot' -and $IsWindows) {
    # Instead of relying on whatever PATH the caller happens to have, initialize the full x64 MSVC
    # developer environment (vcvars64.bat) in a child process whose PATH is sanitized down to the
    # OS + dotnet + VS Installer directories, then import the resulting variables. That keeps stray
    # toolchains (mingw link.exe, older VS, shim managers) from leaking into the link step, and makes
    # the "exit code 123" class of failures deterministic.
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
            # Configuration would do the same. Neither is needed by link.exe - drop both.
            if ($Matches[1] -in @('Platform', 'Configuration')) { continue }
            Set-Item -Path ('Env:' + $Matches[1]) -Value $Matches[2]
        }
    }
    Write-Host "MSVC x64 environment imported from $vcvars" -ForegroundColor DarkGray
} elseif ($Mode -eq 'aot') {
    # Linux/macOS AOT needs a C toolchain plus objcopy; the ILC error when they are missing is opaque.
    $missing = @('clang', 'objcopy') | Where-Object { -not (Get-Command $_ -ErrorAction SilentlyContinue) }
    # macOS ships objcopy-equivalent tooling inside the Xcode command line tools, not as `objcopy`.
    if ($IsMacOS) { $missing = $missing | Where-Object { $_ -ne 'objcopy' } }
    if ($missing) {
        throw "NativeAOT prerequisites missing: $($missing -join ', '). On Debian/Ubuntu: " +
              "sudo apt-get install clang zlib1g-dev binutils. On Fedora: sudo dnf install clang zlib-devel binutils."
    }
}

# --- publish -------------------------------------------------------------------------------------
$proj = Join-Path $root 'Eonego/Eonego.fsproj'
# A stale artifact must not survive a failed link and masquerade as a fresh build.
if (Test-Path $exe) { Remove-Item $exe -Force }

$publishArgs = @('-c', 'Release', '-r', $Rid)
if ($Mode -eq 'aot') {
    if ($InstructionSet) { $publishArgs += "-p:IlcInstructionSet=$InstructionSet" }
    $isaLabel = if ($InstructionSet) { $InstructionSet } else { 'ILC baseline' }
    Write-Host "Publishing NativeAOT ($Rid, ISA=$isaLabel)..." -ForegroundColor Cyan
} else {
    # PublishAot is set in the fsproj; a jit build must switch it off explicitly. Self-contained +
    # single-file so the tester needs no .NET install and gets exactly one file to run.
    $publishArgs += @('-p:PublishAot=false', '-p:SelfContained=true', '-p:PublishSingleFile=true')
    Write-Host "Publishing self-contained single-file JIT ($Rid, runtime CPU dispatch)..." -ForegroundColor Cyan
}

dotnet publish $proj @publishArgs
if ($LASTEXITCODE -ne 0) { throw "publish failed (exit $LASTEXITCODE)" }

if (-not (Test-Path $exe)) { throw "publish reported success but $exe is missing" }
$exeItem = Get-Item $exe
if ($exeItem.LastWriteTime -lt $startedAt) {
    throw "$exe was not rebuilt (timestamp predates this run) - treat as a failed publish"
}

# --- smoke test ----------------------------------------------------------------------------------
# Only meaningful when the artifact can run here: a cross-published binary is for another OS, and an
# ISA-pinned binary may require instructions this host lacks.
$canRun = -not $crossOs -and -not $NoSmoke
if ($canRun) {
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $exeItem.FullName
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
}

$mb = [math]::Round($exeItem.Length / 1MB, 1)
$verdict = if ($canRun) { ' - uciok/readyok smoke test passed' } else { ' - smoke test skipped (cross-OS target)' }
Write-Host "OK: $exe ($mb MB)$verdict" -ForegroundColor Green

if ($CopyTo) {
    $dest = if ([IO.Path]::IsPathRooted($CopyTo)) { $CopyTo } else { Join-Path $root $CopyTo }
    $destDir = Split-Path -Parent $dest
    if ($destDir -and -not (Test-Path $destDir)) { New-Item -ItemType Directory -Path $destDir -Force | Out-Null }
    Copy-Item $exeItem.FullName $dest -Force
    Write-Host "Copied to $dest" -ForegroundColor Green
}

if ($targetOs -ne 'win' -and $IsWindows) {
    Write-Host "Note: the executable bit cannot be set from Windows - the tester must run 'chmod +x' on the file." -ForegroundColor Yellow
}
