# Fixed-position bench (audit 2026-07-01 protocol).
# Usage:  pwsh scripts/bench.ps1 [-Exe <path>] [-Depth 15] [-Prof] [-TimeoutSec 600]
#
# Holds stdin open until bestmove (a naive pipe kills the background search thread on EOF), then
# quits and drains remaining output so every `info string prof*` line is captured regardless of
# whether the engine prints them before or after bestmove. A hung engine is killed at -TimeoutSec.
param(
    [string]$Exe = "$PSScriptRoot\..\Eonego\bin\Release\net10.0\Eonego.exe",
    [int]$Depth = 15,
    [switch]$Prof,          # sets EONEGO_PROF=1 (child process only) to print the phase-counter lines
    [int]$TimeoutSec = 600
)

$positions = @(
    @{ Name = "startpos"; Cmd = "position startpos" },
    @{ Name = "midgame";  Cmd = "position fen r1bq1rk1/pp2bppp/2n1pn2/3p4/3P4/2NBPN2/PP3PPP/R1BQ1RK1 w - - 0 9" }
)

if (-not (Test-Path $Exe)) { throw "engine not found: $Exe (build Release first)" }
# ProcessStartInfo.FileName resolves a relative path against [Environment]::CurrentDirectory, which is
# NOT the PowerShell location - Process.Start then returns null and every later call NREs. Pin it now.
$Exe = (Resolve-Path $Exe).Path

# Reads one line with a hard deadline; $null means timeout or EOF (caller kills on timeout).
function Read-EngineLine([System.Diagnostics.Process]$proc, [datetime]$deadline) {
    $task = $proc.StandardOutput.ReadLineAsync()
    $remainMs = [int][math]::Max(1, ($deadline - (Get-Date)).TotalMilliseconds)
    if (-not $task.Wait($remainMs)) { return $null }
    return $task.Result
}

foreach ($pos in $positions) {
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $Exe
    $psi.RedirectStandardInput = $true
    $psi.RedirectStandardOutput = $true
    $psi.UseShellExecute = $false
    if ($Prof) { $psi.Environment['EONEGO_PROF'] = '1' }
    $proc = [System.Diagnostics.Process]::Start($psi)
    $deadline = (Get-Date).AddSeconds($TimeoutSec)

    $null = $proc.StandardInput.WriteLine("uci")
    $null = $proc.StandardInput.WriteLine("isready")
    $null = $proc.StandardInput.WriteLine($pos.Cmd)
    $null = $proc.StandardInput.WriteLine("go depth $Depth")

    $lastInfo = ""
    $profLines = @()
    $sawBestmove = $false
    while ($true) {
        $line = Read-EngineLine $proc $deadline
        if ($null -eq $line) { break }
        if ($line -match "^info depth $Depth ") { $lastInfo = $line }
        if ($line.StartsWith("info string prof")) { $profLines += $line }
        if ($line.StartsWith("bestmove")) { $sawBestmove = $true; break }
    }

    if ($sawBestmove) {
        # Drain post-bestmove output (prof lines may follow), then quit.
        $null = $proc.StandardInput.WriteLine("quit")
        while ($true) {
            $line = Read-EngineLine $proc $deadline
            if ($null -eq $line) { break }
            if ($line.StartsWith("info string prof")) { $profLines += $line }
        }
    }

    if (-not $proc.HasExited -and -not $proc.WaitForExit(3000)) { $proc.Kill() }

    if (-not $sawBestmove) {
        Write-Warning "[$($pos.Name)] no bestmove within ${TimeoutSec}s — engine killed"
        continue
    }
    Write-Host "[$($pos.Name)] $lastInfo"
    foreach ($pl in $profLines) { Write-Host "[$($pos.Name)] $pl" }
}
