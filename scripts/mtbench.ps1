# Offline fixed-node multi-thread bench: wall time to search a fixed node budget at 1/4/8 threads.
# Node counts at Threads>1 are nondeterministic, so the byte-identity gate (nodesweep.ps1) cannot
# compare multi-thread builds; fixed-node wall time is the throughput comparison that can.
#
#   pwsh scripts/mtbench.ps1 [-Exe <path>] [-Nodes 5000000] [-Threads "1,4,8"] [-Runs 3]
#                            [-EnvSpec "NAME=VAL,NAME2=VAL2"] [-Json]
#
# Per (thread count, run): a fresh engine process searches every position with `go nodes N`; the
# run's NPS = sum(reported nodes) / sum(go->bestmove wall). Reports per-run NPS and the median.
# -Threads and -EnvSpec are comma-separated STRINGS: under `pwsh -File`, an [int[]] param binds
# "1,4" through invariant-culture int parsing as 14 (comma = thousands separator).
param(
    [string]$Exe = "$PSScriptRoot\..\Eonego\bin\Release\net10.0\Eonego.exe",
    [long]$Nodes = 5000000,
    [string]$Threads = "1,4,8",
    [int]$Runs = 3,
    [string]$EnvSpec = "",
    [int]$TimeoutSec = 600,
    [switch]$Json
)

$positions = @(
    @{ Name = "startpos"; Pos = "startpos" },
    @{ Name = "midgame";  Pos = "fen r1bq1rk1/pp2bppp/2n1pn2/3p4/3P4/2NBPN2/PP3PPP/R1BQ1RK1 w - - 0 9" },
    @{ Name = "kiwipete"; Pos = "fen r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1" }
)

if (-not (Test-Path $Exe)) { throw "engine not found: $Exe (build Release first)" }

$envPairs = @{}
foreach ($part in $EnvSpec.Split(',')) {
    $part = $part.Trim()
    if ($part -and $part.Contains('=')) {
        $k, $v = $part.Split('=', 2)
        $envPairs[$k.Trim()] = $v.Trim()
    }
}

$threadList = @(
    foreach ($part in $Threads.Split(',')) {
        $part = $part.Trim()
        if ($part) { [int]::Parse($part, [System.Globalization.CultureInfo]::InvariantCulture) }
    }
)
if ($threadList.Count -eq 0) { throw "no thread counts parsed from -Threads '$Threads'" }

function Read-EngineLine([System.Diagnostics.Process]$proc, [datetime]$deadline) {
    $task = $proc.StandardOutput.ReadLineAsync()
    $remainMs = [int][math]::Max(1, ($deadline - (Get-Date)).TotalMilliseconds)
    if (-not $task.Wait($remainMs)) { return $null }
    return $task.Result
}

$results = @()
foreach ($t in $threadList) {
    $runNps = @()
    for ($run = 1; $run -le $Runs; $run++) {
        $psi = New-Object System.Diagnostics.ProcessStartInfo
        $psi.FileName = $Exe
        $psi.RedirectStandardInput = $true
        $psi.RedirectStandardOutput = $true
        $psi.UseShellExecute = $false
        foreach ($k in $envPairs.Keys) { $psi.Environment[$k] = [string]$envPairs[$k] }
        $proc = [System.Diagnostics.Process]::Start($psi)
        $deadline = (Get-Date).AddSeconds($TimeoutSec)

        $null = $proc.StandardInput.WriteLine("uci")
        if ($t -gt 1) { $null = $proc.StandardInput.WriteLine("setoption name Threads value $t") }
        $null = $proc.StandardInput.WriteLine("isready")
        while ($true) {
            $line = Read-EngineLine $proc $deadline
            if ($null -eq $line) { break }
            if ($line -eq "readyok") { break }
        }

        $totalNodes = 0L
        $totalSec = 0.0
        $failed = $false
        foreach ($p in $positions) {
            $null = $proc.StandardInput.WriteLine("position " + $p.Pos)
            $sw = [System.Diagnostics.Stopwatch]::StartNew()
            $null = $proc.StandardInput.WriteLine("go nodes $Nodes")
            $lastInfo = ""
            $saw = $false
            while ($true) {
                $line = Read-EngineLine $proc $deadline
                if ($null -eq $line) { break }
                if ($line.StartsWith("info depth")) { $lastInfo = $line }
                if ($line.StartsWith("bestmove")) { $saw = $true; break }
            }
            $sw.Stop()
            if (-not $saw) { $failed = $true; break }
            $totalSec += $sw.Elapsed.TotalSeconds
            if ($lastInfo -match "nodes (\d+)") { $totalNodes += [long]$Matches[1] }
        }

        $null = $proc.StandardInput.WriteLine("quit")
        if (-not $proc.WaitForExit(3000)) { $proc.Kill() }

        if ($failed -or $totalSec -le 0 -or $totalNodes -le 0) {
            Write-Warning "threads=$t run=$run failed (timeout or no node info) — skipped"
            continue
        }
        $nps = [long]($totalNodes / $totalSec)
        $runNps += $nps
        $results += [pscustomobject]@{ threads = $t; run = $run; nodes = $totalNodes; sec = [math]::Round($totalSec, 2); nps = $nps }
    }

    if ($runNps.Count -gt 0) {
        $sorted = $runNps | Sort-Object
        $median = $sorted[[int](($sorted.Count - 1) / 2)]
        $results += [pscustomobject]@{ threads = $t; run = "median"; nodes = $null; sec = $null; nps = $median }
    }
}

if ($Json) {
    $results | ConvertTo-Json -Compress
} else {
    $results | Format-Table -AutoSize | Out-String | Write-Host
}
