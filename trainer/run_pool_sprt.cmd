@echo off
rem Night queue (runs via Task Scheduler, outside any session job tree). Sequential:
rem 1) Pool SPRT: EONEGO_POOL=1 vs default, 8T/Hash=512, mt100, c1     -> sprt_pool_8t.log
rem 2) PARTIAL SPRT at real game clocks 10+0.1 (its natural habitat)   -> sprt_partial_tc.log
rem 3) TM shape A/B at 10+0.1: spend-faster (mtg 22, hard 50%)         -> ab_tm_fast.log
rem 4) Quiet checks in qsearch: EONEGO_QSCHECKS=1 vs default, mt100    -> sprt_qschecks.log
rem 5) cont4 extended run (best rider, +24.4+-38.4 @200g needs more)   -> sprt_cont4_ext.log
set PY=%LOCALAPPDATA%\Programs\Python\Python312\python.exe
set MATCH=C:\Users\Samaritan\Projects\Eonego\trainer\match.py
set EXE=C:\Users\Samaritan\Projects\Eonego\Eonego\bin\Release\net10.0\Eonego.exe
set DIR=C:\Users\Samaritan\Projects\Eonego\trainer

"%PY%" "%MATCH%" --a=EONEGO_POOL=1 --b= "--exe=%EXE%" --options=Threads=8,Hash=512 --movetime=100 --openings=125 --sprt --concurrency=1 > "%DIR%\sprt_pool_8t.log" 2>&1

"%PY%" "%MATCH%" --a=EONEGO_PARTIAL=1 --b= "--exe=%EXE%" --tc=10+0.1 --openings=75 --sprt --concurrency=8 > "%DIR%\sprt_partial_tc.log" 2>&1

"%PY%" "%MATCH%" --a=EONEGO_T_TM_MTG=22,EONEGO_T_TM_HARDPCT=50 --b= "--exe=%EXE%" --tc=10+0.1 --openings=75 --sprt --concurrency=8 > "%DIR%\ab_tm_fast.log" 2>&1

"%PY%" "%MATCH%" --a=EONEGO_QSCHECKS=1 --b= "--exe=%EXE%" --movetime=100 --openings=200 --sprt --concurrency=8 > "%DIR%\sprt_qschecks.log" 2>&1

"%PY%" "%MATCH%" --a=EONEGO_CONT4=1,EONEGO_T_CONT4_DIV=1 --b= "--exe=%EXE%" --movetime=100 --openings=400 --sprt --concurrency=8 > "%DIR%\sprt_cont4_ext.log" 2>&1

rem 6) Aspiration fail-high depth-reduction OFF (the b3-b4 slow-win suppressor) -> sprt_fhred.log
"%PY%" "%MATCH%" --a=EONEGO_T_ASP_FHRED=0 --b= "--exe=%EXE%" --movetime=100 --openings=200 --sprt --concurrency=8 > "%DIR%\sprt_fhred.log" 2>&1

rem 7) Root effort ordering -> sprt_rooteffort.log
"%PY%" "%MATCH%" --a=EONEGO_ROOTEFFORT=1 --b= "--exe=%EXE%" --movetime=100 --openings=200 --sprt --concurrency=8 > "%DIR%\sprt_rooteffort.log" 2>&1

echo QUEUE DONE > "%DIR%\night_queue_done.txt"
