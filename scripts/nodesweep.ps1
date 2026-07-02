# Deterministic node-count sweep over the 6-FEN audit suite (1T, fixed depth). The campaign's
# byte-identity gate: node counts are load- and thermal-independent, so two binaries (or one binary
# under different EONEGO_* env vars) can be compared exactly even on a busy machine.
#
#   pwsh scripts/nodesweep.ps1 -Exe <path> [-Depths 13,14,15] [-EnvSpec "EONEGO_T_RFP_MARGIN=121,..."]
#
# Holds stdin open until bestmove (a naive pipe kills the background search thread on `quit`).
# -EnvSpec uses match.py's NAME=VAL,NAME2=VAL2 grammar (a hashtable param would not survive `pwsh -File`).
param(
    [Parameter(Mandatory)][string]$Exe,
    [int[]]$Depths = @(13, 14, 15),
    [string]$EnvSpec = "",
    [switch]$Json
)

$envPairs = @{}
foreach ($part in $EnvSpec.Split(',')) {
    $part = $part.Trim()
    if ($part -and $part.Contains('=')) {
        $k, $v = $part.Split('=', 2)
        $envPairs[$k.Trim()] = $v.Trim()
    }
}

$fens = @(
    @{ name = "startpos";  pos = "startpos" },
    @{ name = "midgame";   pos = "fen r1bq1rk1/pp2bppp/2n1pn2/3p4/3P4/2NBPN2/PP3PPP/R1BQ1RK1 w - - 0 9" },
    @{ name = "kiwipete";  pos = "fen r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1" },
    @{ name = "cpw3-end";  pos = "fen 8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1" },
    @{ name = "cpw4-tact"; pos = "fen r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1" },
    @{ name = "cpw6-mid";  pos = "fen r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 1" }
)

$results = @()
foreach ($depth in $Depths) {
    foreach ($f in $fens) {
        $psi = New-Object System.Diagnostics.ProcessStartInfo
        $psi.FileName = $Exe
        $psi.RedirectStandardInput = $true
        $psi.RedirectStandardOutput = $true
        $psi.UseShellExecute = $false
        foreach ($k in $envPairs.Keys) { $psi.Environment[$k] = [string]$envPairs[$k] }
        $p = [System.Diagnostics.Process]::Start($psi)
        $null = $p.StandardInput.WriteLine("uci")
        $null = $p.StandardInput.WriteLine("isready")
        while (($line = $p.StandardOutput.ReadLine()) -ne $null) { if ($line -eq "readyok") { break } }
        $p.StandardInput.WriteLine("position " + $f.pos)
        $p.StandardInput.WriteLine("go depth $depth")
        $last = ""; $best = ""
        while (($line = $p.StandardOutput.ReadLine()) -ne $null) {
            if ($line.StartsWith("info depth")) { $last = $line }
            if ($line.StartsWith("bestmove")) { $best = $line.Split(' ')[1]; break }
        }
        $p.StandardInput.WriteLine("quit")
        if (-not $p.WaitForExit(3000)) { $p.Kill() }

        $nodes = 0; $score = ""
        if ($last -match "nodes (\d+)") { $nodes = [long]$Matches[1] }
        if ($last -match "score (\S+ \S+)") { $score = $Matches[1] }
        $results += [pscustomobject]@{ depth = $depth; pos = $f.name; nodes = $nodes; score = $score; best = $best }
    }
}

if ($Json) {
    $results | ConvertTo-Json -Compress
} else {
    $results | Format-Table -AutoSize | Out-String | Write-Host
    $total = ($results | Measure-Object -Property nodes -Sum).Sum
    Write-Host "total nodes: $total"
}
