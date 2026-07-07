/// Eonego — minimal UCI driver. Console.Out/Console.In ONLY (never printfn — the documented AOT crash).
/// The search runs on its own master thread so the read loop stays responsive to stop/quit; go, quit,
/// ucinewgame, setoption and position all stop+join any active search first, so the TT is never
/// resized/cleared under a live probe and exactly one `bestmove` is emitted per `go`.
module Eonego.UCI

#nowarn "9" // NativePtr.stackalloc in move re-stamping

open System
open System.Threading
open Microsoft.FSharp.NativeInterop
open Eonego.Move
open Eonego.Position
open Eonego.MoveGeneration
open Eonego.Transposition
open Eonego.NNUE
open Eonego.Search

let private writeLine (s: string) = Console.Out.WriteLine(s)

/// Default transposition-table size (MB); tunable via `setoption name Hash`.
[<Literal>]
let private DefaultHashMb = 256

[<Literal>]
let private MaxHashMb = 65536

/// Read an embedded manifest resource (the baked-in nets) fully into a byte array; None if absent.
let private readEmbedded (name: string) : byte[] option =
    let asm = System.Reflection.Assembly.GetExecutingAssembly()

    match asm.GetManifestResourceStream(name) with
    | null -> None
    | stream ->
        use s = stream
        use ms = new System.IO.MemoryStream()
        s.CopyTo ms
        Some(ms.ToArray())

/// Tunable options: Threads, Hash, MultiPV, Move Overhead; every search toggle is
/// hardwired ON. The NNUE net is embedded in the binary (see Eonego.fsproj), so there is no EvalFile
/// UCI option.
type private UCIState =
    { mutable Threads: int
      mutable HashMb: int
      mutable MultiPv: int
      mutable MoveOverhead: int
      Net: Network option
      Policy: Policy.PolicyNetwork option
      OwnPolicy: Policy.OwnNetwork option
      Tt: TranspositionTable
      mutable RootFen: string
      mutable RootMoves: Move[]
      mutable Control: SearchControl option
      mutable SearchThread: Thread option
      // Worker pool (EONEGO_POOL=1): reused across `go` calls — dropped on ucinewgame (fresh-history
      // semantics) and recreated whenever the length stops matching Threads. Only touched on the UCI
      // thread with no live search (go/ucinewgame stop+join first), and goCore holds its own reference.
      mutable Pool: Worker[] }

let private tryInt (s: string) : int =
    match Int32.TryParse s with
    | true, v -> v
    | _ -> 0

let private tryInt64 (s: string) : int64 =
    match Int64.TryParse s with
    | true, v -> v
    | _ -> 0L

/// UCI moves are castling/en-passant flag-lossy via parseUCI, so re-stamp each against legal generation.
let private matchMove (pos: Position) (uci: string) : Move =
    let t = parseUCI uci

    if t = MoveNone then
        MoveNone
    else
        let p = NativePtr.stackalloc<Move> MaxMoves
        let buf = Span<Move>(NativePtr.toVoidPtr p, MaxMoves)
        let n = generateLegal pos buf
        let mutable found = MoveNone

        for i in 0 .. n - 1 do
            let m = buf.[i]

            if
                fromSq m = fromSq t
                && toSq m = toSq t
                && (not (isPromotion t) || promoType m = promoType t)
            then
                found <- m

        found

/// Legal-move count at the root (Syzygy root-filter reporting only).
let private countLegalRoot (pos: Position) : int =
    let p = NativePtr.stackalloc<Move> MaxMoves
    let buf = Span<Move>(NativePtr.toVoidPtr p, MaxMoves)
    generateLegal pos buf

let private stopAndJoin (st: UCIState) =
    (match st.Control with
     | Some c -> c.Stop()
     | None -> ())

    (match st.SearchThread with
     | Some t -> t.Join()
     | None -> ())

    st.SearchThread <- None

// position [startpos | fen <fields...>] [moves m1 m2 ...]
let private parsePosition (st: UCIState) (tokens: string[]) =
    let movesIdx = Array.tryFindIndex ((=) "moves") tokens
    let specEnd = defaultArg movesIdx tokens.Length

    let fen =
        if tokens.Length = 0 then
            StartPosFen
        elif tokens.[0] = "startpos" then
            StartPosFen
        elif tokens.[0] = "fen" && specEnd > 1 then
            String.Join(" ", tokens.[1 .. specEnd - 1])
        else
            StartPosFen

    let moveToks =
        match movesIdx with
        | Some mi -> tokens.[mi + 1 ..]
        | None -> [||]

    let scratch = Position.OfFen fen
    let acc = ResizeArray<Move>()

    for tok in moveToks do
        let m = matchMove scratch tok

        if m <> MoveNone then
            acc.Add m
            scratch.Make m

    st.RootFen <- fen
    st.RootMoves <- acc.ToArray()

// go [depth d | nodes n | movetime t | wtime .. winc .. btime .. binc .. movestogo .. | infinite |
//     mate n | searchmoves m1 m2 ...]. Returns the limits plus the RAW searchmoves tokens (the go
// handler stamps them against the current position — parseGo has no board).
let private goKeywords =
    [| "depth"; "nodes"; "movetime"; "wtime"; "btime"; "winc"; "binc"; "movestogo"; "mate"
       "infinite"; "ponder"; "searchmoves" |]

let private parseGo (tokens: string[]) : SearchLimits * string[] =
    let mutable lim = defaultLimits
    let mutable i = 0
    let searchMoves = System.Collections.Generic.List<string>()

    let arg () =
        if i + 1 < tokens.Length then tryInt tokens.[i + 1] else 0

    while i < tokens.Length do
        match tokens.[i] with
        | "searchmoves" ->
            i <- i + 1

            while i < tokens.Length && not (Array.contains tokens.[i] goKeywords) do
                searchMoves.Add tokens.[i]
                i <- i + 1
        | "depth" ->
            lim <- { lim with Depth = arg () }
            i <- i + 2
        | "nodes" ->
            // Parse as int64 directly: `int64 (arg ())` went through the 32-bit parser first, so a
            // budget above 2^31-1 parsed to 0 = "no node limit" = an unbounded search.
            lim <-
                { lim with
                    Nodes = (if i + 1 < tokens.Length then tryInt64 tokens.[i + 1] else 0L) }

            i <- i + 2
        | "movetime" ->
            lim <- { lim with MoveTime = arg () }
            i <- i + 2
        | "wtime" ->
            lim <- { lim with WTime = arg () }
            i <- i + 2
        | "btime" ->
            lim <- { lim with BTime = arg () }
            i <- i + 2
        | "winc" ->
            lim <- { lim with WInc = arg () }
            i <- i + 2
        | "binc" ->
            lim <- { lim with BInc = arg () }
            i <- i + 2
        | "movestogo" ->
            lim <- { lim with MovesToGo = arg () }
            i <- i + 2
        | "mate" ->
            lim <- { lim with Mate = arg () }
            i <- i + 2
        | "infinite" ->
            lim <- { lim with Infinite = true }
            i <- i + 1
        | "ponder" ->
            lim <- { lim with Ponder = true }
            i <- i + 1
        | _ -> i <- i + 1

    (lim, searchMoves.ToArray())

let private startSearch (st: UCIState) (lim: SearchLimits) =
    stopAndJoin st

    match st.Net with
    | None ->
        writeLine "info string no NNUE net embedded; cannot search"
        let p = Position()
        p.LoadFen st.RootFen
        for m in st.RootMoves do p.Make m
        writeLine ("bestmove " + toUCI (Search.firstLegalMove p))
    | Some _ ->
        // All search toggles are hardwired ON; Threads/Hash/MultiPV/MoveOverhead come from state.
        let cfg =
            { Threads = st.Threads
              HashMb = st.HashMb
              UseTt = true
              UsePruning = true
              UseProbCut = true
              UseIir = true
              UseRazoring = true
              UseHistoryPruning = true
              UseHistPruneCombined = (Environment.GetEnvironmentVariable("EONEGO_HISTCOMBINED") = "1")
              UseDeltaPruning = true
              UseQsDeltaCorrected = (Environment.GetEnvironmentVariable("EONEGO_QSDELTACORR") = "1")
              UseContHist = true
              UseSingular = true
              UseNmpVerify = true
              UseLmrTweaks = true
              UseAspTweaks = true
              // A/B env knobs for match.py per-player overrides (campaign step B4): defaults preserve
              // release behaviour; each flag flips for one player without a rebuild.
              UseQsTt = (Environment.GetEnvironmentVariable("EONEGO_QSTT") <> "1")
              UseTtEvalAdjust = (Environment.GetEnvironmentVariable("EONEGO_TTEVADJ") <> "1")
              UseCheckExt = (Environment.GetEnvironmentVariable("EONEGO_CHECKEXT") = "1")
              UseOneReplyExt = (Environment.GetEnvironmentVariable("EONEGO_ONEREPLY") = "1")
              UseQsEvasionCap = (Environment.GetEnvironmentVariable("EONEGO_QSEVCAP") = "1")
              UseTtCapture = (Environment.GetEnvironmentVariable("EONEGO_TTCAPTURE") = "1")
              UseCorrHist = (Environment.GetEnvironmentVariable("EONEGO_CORRHIST") <> "1")
              UseCorrMinor = (Environment.GetEnvironmentVariable("EONEGO_CORRMINOR") = "1")
              UseCorrMajor = (Environment.GetEnvironmentVariable("EONEGO_CORRMAJOR") = "1")
              UseCorrNonPawn = (Environment.GetEnvironmentVariable("EONEGO_CORRNONPAWN") = "1")
              UseCorrCont = (Environment.GetEnvironmentVariable("EONEGO_CORRCONT") = "1")
              UseCaptFut = (Environment.GetEnvironmentVariable("EONEGO_CAPFUT") = "1")
              UsePartialCommit = (Environment.GetEnvironmentVariable("EONEGO_PARTIAL") = "1")
              UseCont4 = (Environment.GetEnvironmentVariable("EONEGO_CONT4") <> "1")
              UseR50Damp = (Environment.GetEnvironmentVariable("EONEGO_R50DAMP") <> "1")
              UseQsChecks = (Environment.GetEnvironmentVariable("EONEGO_QSCHECKS") = "1")
              UseRootEffort = (Environment.GetEnvironmentVariable("EONEGO_ROOTEFFORT") = "1")
              UseRootVerify = (Environment.GetEnvironmentVariable("EONEGO_ROOTVERIFY") = "1")
              UseRetro = (Environment.GetEnvironmentVariable("EONEGO_RETRO") <> "0")
              // Syzygy WDL probe: inert until `setoption name SyzygyPath` loads tables
              // (Syzygy.Largest = 0 gates every probe); EONEGO_SYZYGY=0 is the kill switch.
              // (Was `<> "1"` — that inverted the documented kill switch: =1 disabled, =0 didn't.)
              UseSyzygy = (Environment.GetEnvironmentVariable("EONEGO_SYZYGY") <> "0")
              // df-pn mate oracle: default OFF pre-SPRT (the CHECKEXT/CAPFUT class); flip to
              // <> "0" only after a passing match verdict.
              UseDFPN = (Environment.GetEnvironmentVariable("EONEGO_DFPN") = "1")
              // Policy sidecar: ON iff startup actually loaded one (EONEGO_POLICY gate; see run()).
              UsePolicy = st.Policy.IsSome || st.OwnPolicy.IsSome
              // Dynamic time management (the TM campaign; default OFF pre-SPRT): each component is
              // independently A/B-able per player without a rebuild. Game clocks only — movetime
              // matches make all of these inert (soft = 0). EONEGO_TMLOG=1 adds per-move telemetry.
              UseTmMtgHarden = (Environment.GetEnvironmentVariable("EONEGO_TMMTG") = "1")
              UseTmStability = (Environment.GetEnvironmentVariable("EONEGO_TMSTAB") = "1")
              UseTmTrend = (Environment.GetEnvironmentVariable("EONEGO_TMTREND") = "1")
              UseTmFailLow = (Environment.GetEnvironmentVariable("EONEGO_TMFAILLOW") = "1")
              UseTmEffort = (Environment.GetEnvironmentVariable("EONEGO_TMEFFORT") = "1")
              MoveOverhead = st.MoveOverhead
              // NNUE accumulator checkpoint cache (AccumulatorCache.fs): fully built but inert at
              // 0 MiB. EONEGO_ACCMB=<MiB> arms it for SPRT (per-search table shared across the
              // LazySMP workers); unset keeps the byte-identical default.
              AccCheckpointMb =
                (match Int32.TryParse(Environment.GetEnvironmentVariable("EONEGO_ACCMB")) with
                 | true, v -> max 0 (min 1024 v)
                 | _ -> 0)
              MultiPv = st.MultiPv }

        // Syzygy DTZ root filter: restrict the root to the TB-best-preserving move set (rides the
        // SearchMoves mechanism, so the search itself is untouched). Skipped when the user supplied
        // `go searchmoves` (their restriction wins) and inert without tables (Largest = 0).
        // EONEGO_SYZYGYDTZ=0 is the kill switch. One probe pass per `go`, on the UCI thread.
        let lim =
            if
                cfg.UseSyzygy
                && Syzygy.Largest > 0
                && lim.SearchMoves.Length = 0
                && Environment.GetEnvironmentVariable("EONEGO_SYZYGYDTZ") <> "0"
            then
                let p = Position()
                p.LoadFen st.RootFen

                for mv in st.RootMoves do
                    p.Make mv

                let tbMoves = Syzygy.probeRoot p

                if tbMoves.Length > 0 then
                    writeLine (
                        "info string Syzygy root filter: "
                        + string tbMoves.Length
                        + " of "
                        + string (countLegalRoot p)
                        + " moves kept"
                    )

                    { lim with SearchMoves = tbMoves }
                else
                    lim
            else
                lim

        let control =
            SearchControl(
                cfg,
                lim,
                st.Tt,
                st.RootFen,
                st.RootMoves,
                ?net = st.Net,
                ?policy = st.Policy,
                ?ownPolicy = st.OwnPolicy
            )

        // Arm on the UCI thread BEFORE the search thread exists: a `stop`/`quit` arriving right
        // after t.Start() must find the stop flag armed-and-clear, not race the thread's own Reset
        // (which either erased the stop — unbounded search — or aborted depth 1 into first-legal).
        Search.arm control
        st.Control <- Some control

        // Worker pool: DEFAULT ON since 2026-07-05 (EONEGO_POOL=0 restores the legacy fresh-workers
        // path). Warm per-worker gravity history is the only LazySMP divergence source besides the
        // shared TT — measured +19.7±29.8 and +20.9±29.1 vs fresh workers over two independent
        // 300-game 8T movetime-100 books (pooled ~+20±21), plus the 26ms/move allocation saving at
        // 16T. First search from a cold pool is byte-identical to the legacy path (proven; pool
        // tests pin the warm divergence). Ensure/recreate on the UCI thread (no live search here);
        // goPooled rebinds each worker to the fresh control.
        let usePool = Environment.GetEnvironmentVariable("EONEGO_POOL") <> "0"

        if usePool && st.Pool.Length <> cfg.Threads then
            st.Pool <- Array.init cfg.Threads (fun i -> Worker(i, (i = 0), control))

        let pool = st.Pool

        let t =
            Thread(
                ThreadStart(fun () ->
                    if usePool then
                        Search.goPooledArmed control pool |> ignore
                    else
                        Search.goArmed control |> ignore),
                16 * 1024 * 1024
            )

        t.IsBackground <- true
        st.SearchThread <- Some t
        t.Start()

let private handleSetOption (st: UCIState) (tokens: string[]) =
    // setoption name <Name...> value <Value> (the name is the tokens between `name` and `value` —
    // Move Overhead / Clear Hash are two-word names); button options arrive with no `value` at all.
    // stop+join FIRST, unconditionally: every branch below mutates state a live search reads
    // (helper slots, the TT, Syzygy tables) — the module header's documented invariant, previously
    // upheld only by the Hash branch (Threads raced resizeHelpers' Shutdown/Join against a running
    // dispatch; SyzygyPath re-pointed tables under live probes).
    stopAndJoin st
    let nameIdx = Array.tryFindIndex ((=) "name") tokens
    let valIdx = Array.tryFindIndex ((=) "value") tokens

    match nameIdx, valIdx with
    | Some ni, Some vi when ni < vi && vi + 1 < tokens.Length ->
        let name = String.Join(" ", tokens.[ni + 1 .. vi - 1])
        let v = tryInt tokens.[vi + 1]

        if String.Equals(name, "Threads", StringComparison.OrdinalIgnoreCase) then
            st.Threads <- max 1 (min 256 v)
            Search.resizeHelpers st.Threads
        elif String.Equals(name, "Hash", StringComparison.OrdinalIgnoreCase) then
            st.HashMb <- max 1 (min MaxHashMb v)
            st.Tt.Resize st.HashMb
        elif String.Equals(name, "MultiPV", StringComparison.OrdinalIgnoreCase) then
            st.MultiPv <- max 1 (min 256 v)
        elif String.Equals(name, "Move Overhead", StringComparison.OrdinalIgnoreCase) then
            st.MoveOverhead <- max 0 (min 5000 v)
        elif String.Equals(name, "RowPrefetch", StringComparison.OrdinalIgnoreCase) then
            // Weight-row prefetch mode (0 = off, 1 = leading lines, 2 = +L2 tail). Semantics-free —
            // node counts are byte-identical in all modes; a pure speed A/B knob (see Accumulator.fs).
            Eonego.Accumulator.RowPrefetchMode <- max 0 (min 2 v)
        elif String.Equals(name, "SyzygyPath", StringComparison.OrdinalIgnoreCase) then
            let pathVal = String.Join(" ", tokens.[vi + 1 ..])
            if Syzygy.init pathVal then
                writeLine ("info string Syzygy tables loaded, largest " + string Syzygy.Largest + " pieces")
            else
                writeLine "info string Syzygy init failed"
    // `Ponder value true/false` needs no state (the GUI drives pondering via `go ponder`) and
    // falls through here by design — declaring the option is what unlocks the GUI checkbox.
    | Some ni, None when ni + 1 < tokens.Length ->
        // Button options: no `value` token.
        let name = String.Join(" ", tokens.[ni + 1 ..])

        if String.Equals(name, "Clear Hash", StringComparison.OrdinalIgnoreCase) then
            st.Tt.Clear()
    | _ -> ()

let run () =
    // Startup banner (conventional for UCI engines; GUIs log it and show it in their engine lists).
    writeLine (Engine.Name + " " + Engine.Version + " by " + Engine.Author)

    // The NNUE net is embedded in the binary (see Eonego.fsproj <EmbeddedResource>); load it once at
    // startup. EONEGO_NET=<path> overrides with a net file from disk (same architecture required) —
    // the A/B channel for net experiments (blends, candidates) without a rebuild per arm.
    let net =
        match Environment.GetEnvironmentVariable("EONEGO_NET") with
        | null
        | "" ->
            match readEmbedded "eval.nnue" with
            | Some bytes -> (match NNUE.loadBytes bytes with Loaded n -> Some n | Failed _ -> None)
            | None -> None
        | path ->
            match NNUE.load path with
            | Loaded n ->
                writeLine ("info string net override: " + path)
                Some n
            | Failed why ->
                writeLine ("info string EONEGO_NET load FAILED (" + why + "); falling back to embedded")

                match readEmbedded "eval.nnue" with
                | Some bytes -> (match NNUE.loadBytes bytes with Loaded n -> Some n | Failed _ -> None)
                | None -> None

    // Policy net (EONEGO_POLICY). Two formats share the one env var, dispatched by file magic:
    //   EONPOL02 sidecar — reads the NNUE trunk; ftHash-bound; any position.
    //   EONPOL03 own-trunk — its own board-feature net; only fires at ≤6 pieces (endgames).
    // Values: unset => classic search (byte-identical); "1" => embedded sidecar `policy.dat`;
    //   "own1" => embedded own-trunk `ownpolicy.dat`; <path> => the file (magic picks the format).
    let policyEnv = Environment.GetEnvironmentVariable("EONEGO_POLICY")

    let ownPolicy =
        match policyEnv, net with
        | (null | ""), _ -> None
        | _, None -> None
        | "own1", Some _ ->
            match readEmbedded "ownpolicy.dat" with
            | Some bytes ->
                match Policy.loadOwnBytes bytes with
                | Policy.OwnLoaded o ->
                    writeLine "info string own-trunk policy: embedded"
                    Some o
                | Policy.OwnFailed why ->
                    writeLine ("info string embedded own policy FAILED (" + why + "); policy off")
                    None
            | None ->
                writeLine "info string EONEGO_POLICY=own1 but no ownpolicy.dat embedded; policy off"
                None
        | path, Some _ when IO.File.Exists path ->
            match Policy.loadOwn path with
            | Policy.OwnLoaded o ->
                writeLine ("info string own-trunk policy: " + path)
                Some o
            | Policy.OwnFailed _ -> None // not an own-trunk file; fall through to the sidecar loader
        | _ -> None

    let policy =
        if ownPolicy.IsSome then
            None
        else
            match policyEnv, net with
            | (null | ""), _ -> None
            | _, None -> None
            | "own1", _ -> None
            | "1", Some n ->
                match readEmbedded "policy.dat" with
                | Some bytes ->
                    match Policy.loadBytes bytes n.FtHash with
                    | Policy.PolicyLoaded p -> Some p
                    | Policy.PolicyFailed why ->
                        writeLine ("info string embedded policy load FAILED (" + why + "); policy off")
                        None
                | None ->
                    writeLine "info string EONEGO_POLICY=1 but no policy.dat embedded; policy off"
                    None
            | path, Some n ->
                match Policy.load path n.FtHash with
                | Policy.PolicyLoaded p ->
                    writeLine ("info string policy sidecar: " + path)
                    Some p
                | Policy.PolicyFailed why ->
                    writeLine ("info string EONEGO_POLICY load FAILED (" + why + "); policy off")
                    None

    let st =
        { Threads = 1
          HashMb = DefaultHashMb
          MultiPv = 1
          MoveOverhead = 10
          Net = net
          Policy = policy
          OwnPolicy = ownPolicy
          Tt = TranspositionTable(DefaultHashMb)
          RootFen = StartPosFen
          RootMoves = [||]
          Control = None
          SearchThread = None
          Pool = [||] }

    let mutable running = true

    while running do
        match Console.In.ReadLine() with
        | null -> running <- false
        | line ->
            let tokens = line.Split([| ' '; '\t' |], StringSplitOptions.RemoveEmptyEntries)

            if tokens.Length > 0 then
                match tokens.[0] with
                | "uci" ->
                    writeLine ("id name " + Engine.Name + " " + Engine.Version)
                    writeLine ("id author " + Engine.Author)
                    writeLine "option name Threads type spin default 1 min 1 max 256"
                    writeLine "option name Hash type spin default 256 min 1 max 65536"
                    writeLine "option name Clear Hash type button"
                    writeLine "option name MultiPV type spin default 1 min 1 max 256"
                    writeLine "option name Move Overhead type spin default 10 min 0 max 5000"
                    writeLine "option name RowPrefetch type spin default 0 min 0 max 2"
                    // Pondering is fully wired (`go ponder`/`ponderhit`); GUIs gate their ponder
                    // checkbox on this declaration, so without it the feature is unreachable.
                    writeLine "option name Ponder type check default false"
                    writeLine "option name SyzygyPath type string default <empty>"
                    writeLine "uciok"
                | "isready" ->
                    writeLine "readyok"
                | "ucinewgame" ->
                    stopAndJoin st
                    st.Tt.Clear()
                    st.Pool <- [||] // new game = fresh history (pooled workers carry warm tables)
                    st.RootFen <- StartPosFen
                    st.RootMoves <- [||]
                | "position" ->
                    stopAndJoin st
                    parsePosition st tokens.[1..]

                    // Retrograde root trigger: on a low-material root, start solving the reachable
                    // signatures in the background NOW — during the opponent's think time — so the
                    // search's probes are usually live before the ending matters. Same kill-switch
                    // as the probe's config rider. The replay Position is net-free (no acc frames);
                    // the cap keeps a pathological >900-ply history away from the state-stack limit
                    // (skipping only delays solving — the search stays correct without it).
                    if
                        Environment.GetEnvironmentVariable "EONEGO_RETRO" <> "0"
                        && st.RootMoves.Length <= 900
                    then
                        let p = Position()
                        p.LoadFen st.RootFen

                        for mv in st.RootMoves do
                            p.Make mv

                        Eonego.Retrograde.requestSolveFor p
                | "go" ->
                    let (lim, smUCI) = parseGo tokens.[1..]

                    // Stamp `searchmoves` tokens against the actual position (parseUCI is castling/
                    // en-passant flag-lossy, same reason parsePosition re-stamps). Unmatched tokens drop.
                    let lim =
                        if smUCI.Length = 0 then
                            lim
                        else
                            let p = Position()
                            p.LoadFen st.RootFen

                            for mv in st.RootMoves do
                                p.Make mv

                            let stamped =
                                smUCI |> Array.map (matchMove p) |> Array.filter (fun m -> m <> MoveNone)

                            { lim with SearchMoves = stamped }

                    startSearch st lim
                | "stop" ->
                    (match st.Control with
                     | Some c -> c.Stop()
                     | None -> ())
                | "ponderhit" ->
                    // the opponent played the predicted move: arm the running ponder search's clock now.
                    (match st.Control with
                     | Some c -> c.PonderHit()
                     | None -> ())
                | "setoption" -> handleSetOption st tokens
                | "quit" ->
                    stopAndJoin st
                    Search.shutdownHelpers ()
                    running <- false
                | _ -> ()
