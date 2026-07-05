# Detached, resumable Eonego-teacher labeling run (d25, 16 workers, checkpointed).
# Start:  pwsh trainer/run_label.ps1        (safe to rerun any time — resumes from checkpoint)
# Pause:  kill the python tree (or the Eonego teacher engines); rerun to resume.
# Watch:  line count of trainer/data/kga26_eteacher2.txt, or tail trainer/label_run.log
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$py = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
$env:TEACHER_ENGINE = Join-Path (Split-Path -Parent $here) "Eonego\bin\Release\net10.0\win-x64\publish\Eonego.exe"
Set-Location $here
& $py label.py --in data\kga26_labeled2.txt --out data\kga26_eteacher2.txt `
    --depth 25 --max-time 180 --workers 16 --hash 256 --batch 32 `
    *>> label_run.log
