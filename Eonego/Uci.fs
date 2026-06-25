/// Eonego — minimal UCI driver. Console.Out/Console.In ONLY (never printfn — the documented AOT crash).
/// The search runs on its own master thread so the read loop stays responsive to stop/quit; go, quit,
/// ucinewgame, setoption Hash and position all stop+join any active search first, so the TT is never
/// resized/cleared under a live probe and exactly one `bestmove` is emitted per `go`.
module Eonego.Uci

#nowarn "9" // NativePtr.stackalloc in move re-stamping

open System
open System.Threading
open Microsoft.FSharp.NativeInterop
open Eonego.Move
open Eonego.Position
open Eonego.MoveGeneration
open Eonego.Transposition
open Eonego.SfNnue
open Eonego.Search

let private writeLine (s: string) = Console.Out.WriteLine(s)

/// Hardwired transposition-table size (MB). Hash is no longer a UCI option.
[<Literal>]
let private HashMb = 256

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

/// Only Threads and Move Overhead remain tunable; every other option is hardwired ON / fixed. The SF leaf
/// net is embedded in the binary (see Eonego.fsproj); the optional Lc0 policy net loads from disk
/// (EONEGO_LC0 / auto-discovery) — so there is no Eval/Policy file UCI option for either.
type private UciState =
    { mutable Threads: int
      mutable MoveOverhead: int
      Net: SfNetwork option
      Lc0Net: Lc0Proto.Lc0Net option // Lc0 CNN priors (root); gated by EONEGO_LC0 env var, else history fallback
      Lc0Int8: Lc0Net.Lc0Int8 option // int8 companion of Lc0Net (~2.77x forward); built once at load unless EONEGO_LC0_FP32
      Tt: TranspositionTable
      mutable RootFen: string
      mutable RootMoves: Move[]
      mutable Control: SearchControl option
      mutable SearchThread: Thread option
      Reuse: Mcts.MctsReuse } // MCTS tree-reuse state, persisted across `go` (cleared on ucinewgame)

let private tryInt (s: string) : int =
    match Int32.TryParse s with
    | true, v -> v
    | _ -> 0

/// An MCTS tuning parameter overridable via an env var (for SPRT/self-play sweeps without unlocking the
/// release UCI surface). Defaults to the shipped value when unset/invalid.
let private envInt (name: string) (defaultValue: int) : int =
    match System.Environment.GetEnvironmentVariable name with
    | null
    | "" -> defaultValue
    | s ->
        match Int32.TryParse s with
        | true, v -> v
        | _ -> defaultValue

/// UCI moves are castling/en-passant flag-lossy via parseUci, so re-stamp each against legal generation.
let private matchMove (pos: Position) (uci: string) : Move =
    let t = parseUci uci

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

let private stopAndJoin (st: UciState) =
    (match st.Control with
     | Some c -> c.Stop()
     | None -> ())

    (match st.SearchThread with
     | Some t -> t.Join()
     | None -> ())

    st.SearchThread <- None

// position [startpos | fen <fields...>] [moves m1 m2 ...]
let private parsePosition (st: UciState) (tokens: string[]) =
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

// go [depth d | nodes n | movetime t | wtime .. winc .. btime .. binc .. movestogo .. | infinite | mate n]
let private parseGo (tokens: string[]) : SearchLimits =
    let mutable lim = defaultLimits
    let mutable i = 0

    let arg () =
        if i + 1 < tokens.Length then tryInt tokens.[i + 1] else 0

    while i < tokens.Length do
        match tokens.[i] with
        | "depth" ->
            lim <- { lim with Depth = arg () }
            i <- i + 2
        | "nodes" ->
            lim <- { lim with Nodes = int64 (arg ()) }
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

    lim

let private startSearch (st: UciState) (lim: SearchLimits) =
    stopAndJoin st

    match st.Net with
    | None ->
        writeLine "info string no NNUE net embedded; cannot search"
        let p = Position()
        p.LoadFen st.RootFen
        for m in st.RootMoves do p.Make m
        writeLine ("bestmove " + toUci (Search.firstLegalMove p))
    | Some _ ->
        // All toggles hardwired ON; HashMb fixed; MCTS is the production search (Lc0 priors when a net is
        // found, else history fallback); only Threads/MoveOverhead come from state.
        let cfg =
            { Threads = st.Threads
              HashMb = HashMb
              UseTt = true
              UsePruning = true
              UseProbCut = true
              UseIir = true
              UseRazoring = true
              UseHistoryPruning = true
              UseDeltaPruning = true
              UseContHist = true
              UseSingular = true
              UseNmpVerify = true
              UseLmrTweaks = true
              UseAspTweaks = true
              MoveOverhead = st.MoveOverhead
              // MCTS is the production search; EONEGO_MCTS=0 routes `go` through plain alpha-beta (Search.go)
              // so one binary can A/B the hybrid vs alpha-beta at equal movetime (the dispatch is at startSearch).
              UseMcts = (envInt "EONEGO_MCTS" 1) <> 0
              // MCTS tuning knobs, env-overridable for SPRT sweeps (defaults are the shipped values).
              MctsCpuct = envInt "EONEGO_CPUCT" 150
              MctsLeafDepth = envInt "EONEGO_LEAFDEPTH" 8
              MctsK = envInt "EONEGO_K" 200
              // Blend the Lc0 value head into the leaf q (x100): 0 = pure negamax+SF leaf (default/today),
              // 100 = pure Lc0 value. Experimental strength lever; SPRT-gate it. Only meaningful with Lc0.
              MctsValueBlend = envInt "EONEGO_VALUE_BLEND" 0
              // Lc0 (if loaded via EONEGO_LC0) drives priors; else the history-softmax fallback.
              UseLc0 = st.Lc0Net.IsSome
              // Batched Lc0 forwards per worker (EXPERIMENTAL, default OFF=1): on this hardware the larger
              // batched activations spill cache and it runs SLOWER than single eval, so it is opt-in via the
              // EONEGO_BATCH env var (for other hardware/nets). Only meaningful when Lc0 is active.
              MctsBatchSize =
                  if st.Lc0Net.IsSome then
                      match System.Environment.GetEnvironmentVariable "EONEGO_BATCH" with
                      | null
                      | "" -> 1
                      | s ->
                          match System.Int32.TryParse s with
                          | true, v when v >= 1 -> v
                          | _ -> 1
                  else
                      1
              // No-Lc0 fallback only: EONEGO_EVAL_PRIORS uses SF-NNUE static-eval priors instead of history
              // (principled-but-tactically-blind, experimental, SPRT-gated). Ignored when Lc0 drives priors.
              UseEvalPriors =
                  (not st.Lc0Net.IsSome)
                  && (match System.Environment.GetEnvironmentVariable "EONEGO_EVAL_PRIORS" with
                      | null
                      | "" -> false
                      | _ -> true) }

        let control =
            SearchControl(cfg, lim, st.Tt, st.RootFen, st.RootMoves, ?net = st.Net, ?lc0Net = st.Lc0Net, ?lc0Int8 = st.Lc0Int8)
        st.Control <- Some control

        let t =
            Thread(
                ThreadStart(fun () -> (if cfg.UseMcts then Mcts.mctsSearch control st.Reuse else Search.go control) |> ignore),
                16 * 1024 * 1024
            )
        t.IsBackground <- true
        st.SearchThread <- Some t
        t.Start()

let private handleSetOption (st: UciState) (tokens: string[]) =
    // setoption name <Name...> value <Value> — only Threads and Move Overhead are tunable (the latter is a
    // two-word name, so the name is the tokens between `name` and `value`); every other option is ignored.
    let nameIdx = Array.tryFindIndex ((=) "name") tokens
    let valIdx = Array.tryFindIndex ((=) "value") tokens

    match nameIdx, valIdx with
    | Some ni, Some vi when ni < vi && vi + 1 < tokens.Length ->
        let name = String.Join(" ", tokens.[ni + 1 .. vi - 1])
        let v = tryInt tokens.[vi + 1]

        if String.Equals(name, "Threads", StringComparison.OrdinalIgnoreCase) then
            st.Threads <- max 1 (min 256 v)
        elif String.Equals(name, "Move Overhead", StringComparison.OrdinalIgnoreCase) then
            st.MoveOverhead <- max 0 (min 5000 v)
    // Any other (legacy/hardwired) option is silently ignored.
    | _ -> ()

let run () =
    // Nets are embedded in the binary (see Eonego.fsproj <EmbeddedResource>); load both once at startup.
    let net =
        match readEmbedded "sf16.nnue" with
        | Some bytes -> (match SfNnue.loadBytes bytes with Loaded n -> Some n | Failed _ -> None)
        | None -> None

    // Lc0 CNN policy (root). The net is loaded from DISK (113 MB, not embedded): use EONEGO_LC0 if set,
    // otherwise AUTO-DISCOVER the first *.pb next to the exe or in the working dir, so the Lc0+SF hybrid
    // works in a GUI without any env var. If no net is found, MCTS falls back to weak history priors.
    let lc0Path =
        match Environment.GetEnvironmentVariable "EONEGO_LC0" with
        | null | "" ->
            [ AppContext.BaseDirectory; Environment.CurrentDirectory ]
            |> List.tryPick (fun d ->
                try
                    System.IO.Directory.GetFiles(d, "*.pb") |> Array.sort |> Array.tryHead
                with _ -> None)
            |> Option.defaultValue ""
        // Explicit disable sentinel: EONEGO_LC0=none/off/0 forces the history-prior fallback WITHOUT auto-
        // discovery, so an ablation (Lc0 policy vs history priors) can be run on one binary by env alone.
        | p when (let l = p.Trim().ToLowerInvariant() in l = "none" || l = "off" || l = "0" || l = "disable") -> ""
        | p -> p

    let lc0Net =
        if lc0Path = "" then
            writeLine "info string no Lc0 .pb found (set EONEGO_LC0 or place a .pb next to the exe); using history priors"
            None
        else
            match Lc0Proto.load lc0Path with
            | Lc0Proto.Loaded n ->
                writeLine ("info string Lc0 net loaded: " + lc0Path + " (" + string n.Tower.Length + " blocks)")
                Some n
            | Lc0Proto.Failed r ->
                writeLine ("info string Lc0 net load failed (" + r + "); using history priors")
                None

    // int8 companion (~2.77x faster forward, parity-validated): built once at load when an Lc0 net is present,
    // unless EONEGO_LC0_FP32 forces the fp32 path (debug/parity escape hatch).
    let lc0Int8 =
        match lc0Net with
        | Some n when (match Environment.GetEnvironmentVariable "EONEGO_LC0_FP32" with null | "" -> true | _ -> false) ->
            let q = Lc0Net.quantize n
            writeLine "info string Lc0 int8 forward enabled (set EONEGO_LC0_FP32=1 to force fp32)"
            Some q
        | _ -> None

    let st =
        { Threads = 1
          MoveOverhead = 10
          Net = net
          Lc0Net = lc0Net
          Lc0Int8 = lc0Int8
          Tt = TranspositionTable(HashMb)
          RootFen = StartPosFen
          RootMoves = [||]
          Control = None
          SearchThread = None
          Reuse = Mcts.MctsReuse() }

    let mutable running = true

    while running do
        match Console.In.ReadLine() with
        | null -> running <- false
        | line ->
            let tokens = line.Split([| ' '; '\t' |], StringSplitOptions.RemoveEmptyEntries)

            if tokens.Length > 0 then
                match tokens.[0] with
                | "uci" ->
                    writeLine "id name Eonego"
                    writeLine "id author Houijasu"
                    writeLine "option name Threads type spin default 1 min 1 max 256"
                    writeLine "option name Move Overhead type spin default 10 min 0 max 5000"
                    writeLine "uciok"
                | "isready" ->
                    writeLine "readyok"
                | "ucinewgame" ->
                    stopAndJoin st
                    st.Tt.Clear()
                    st.Reuse.Clear()
                    st.RootFen <- StartPosFen
                    st.RootMoves <- [||]
                | "position" ->
                    stopAndJoin st
                    parsePosition st tokens.[1..]
                | "go" -> startSearch st (parseGo tokens.[1..])
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
                    running <- false
                | _ -> ()
