# Fixed-position bench (audit 2026-07-01 protocol).
# Usage:  pwsh scripts/bench.ps1 [-Exe <path>] [-Depth 15] [-Prof]
param(
    [string]$Exe = "$PSScriptRoot\..\Eonego\bin\Release\net10.0\Eonego.exe",
    [int]$Depth = 15,
    [switch]$Prof          # sets EONEGO_PROF=1 to print the phase-counter line
)

$positions = @(
    @{ Name = "startpos"; Cmd = "position startpos" },
    @{ Name = "midgame";  Cmd = "position fen r1bq1rk1/pp2bppp/2n1pn2/3p4/3P4/2NBPN2/PP3PPP/R1BQ1RK1 w - - 0 9" }
)

if (-not (Test-Path $Exe)) { throw "engine not found: $Exe (build Release first)" }
if ($Prof) { $env:EONEGO_PROF = "1" } else { Remove-Item Env:EONEGO_PROF -ErrorAction SilentlyContinue }

foreach ($p in $positions) {
    $inp = "uci`nisready`n$($p.Cmd)`ngo depth $Depth`n"
    # go-depth terminates on its own; quit after bestmove via a generous timeout pipe.
    $out = $inp + "`n" | & $Exe 2>&1 | ForEach-Object { $_ }
    $last = $out | Where-Object { $_ -match "info depth $Depth " } | Select-Object -Last 1
    $prof = $out | Where-Object { $_ -match "info string prof" } | Select-Object -Last 1
    Write-Host "[$($p.Name)] $last"
    if ($prof) { Write-Host "[$($p.Name)] $prof" }
}
