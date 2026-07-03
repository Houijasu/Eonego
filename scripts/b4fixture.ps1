# Time-to-truth: seconds until the after-b4 position's score crosses cp <= -300 (the win becoming
# visible to the engine), capped at 90s. Lower = better. 1T deterministic modulo clock granularity.
$exe = 'C:\Users\Samaritan\Projects\Eonego\Eonego\bin\Release\net10.0\win-x64\publish\Eonego.exe'
$configs = @(
    @{ n = 'baseline';      e = '' },
    @{ n = 'cont4';         e = 'EONEGO_CONT4=1' },
    @{ n = 'lmp-old(3)';    e = 'EONEGO_T_LMP_BASE=3' },
    @{ n = 'cont4+lmp3';    e = 'EONEGO_CONT4=1,EONEGO_T_LMP_BASE=3' },
    @{ n = 'rfp-soft';      e = 'EONEGO_T_RFP_MARGIN=200' },
    @{ n = 'corrhist-off';  e = 'EONEGO_CORRHIST=0' }
)
foreach ($c in $configs) {
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $exe
    $psi.RedirectStandardInput = $true
    $psi.RedirectStandardOutput = $true
    $psi.UseShellExecute = $false
    foreach ($part in $c.e.Split(',')) {
        if ($part -and $part.Contains('=')) { $k, $v = $part.Split('=', 2); $psi.Environment[$k] = $v }
    }
    $p = [System.Diagnostics.Process]::Start($psi)
    $in = $p.StandardInput; $out = $p.StandardOutput
    $in.WriteLine('uci'); $in.WriteLine('setoption name Threads value 1'); $in.WriteLine('isready'); $in.Flush()
    while (($l = $out.ReadLine()) -ne $null) { if ($l -eq 'readyok') { break } }
    $in.WriteLine('position fen 6k1/8/1pK4p/bPp5/8/1P6/P1B2P2/8 w - - 0 4 moves b3b4')
    $in.WriteLine('go movetime 90000'); $in.Flush()
    $hitMs = -1; $hitDepth = -1
    while (($l = $out.ReadLine()) -ne $null) {
        if ($l -match 'info depth (\d+) .*score cp (-?\d+) .*time (\d+) ') {
            if ([int]$Matches[2] -le -300 -and $hitMs -lt 0) { $hitDepth = [int]$Matches[1]; $hitMs = [int]$Matches[3]; break }
        }
        if ($l.StartsWith('bestmove')) { break }
    }
    if ($hitMs -ge 0) { $in.WriteLine('stop'); $in.Flush() }
    $in.WriteLine('quit'); $in.Flush(); if (-not $p.WaitForExit(5000)) { $p.Kill() }
    if ($hitMs -ge 0) { Write-Host ("{0,-13} sees the win at {1,6} ms (d{2})" -f $c.n, $hitMs, $hitDepth) }
    else { Write-Host ("{0,-13} NOT within 90s" -f $c.n) }
}
