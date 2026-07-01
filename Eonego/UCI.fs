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
open Eonego.Nnue
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

/// Tunable options: Threads, Hash, MultiPV, Move Overhead, Use Work Queue; every search toggle is
/// hardwired ON. The NNUE net is embedded in the binary (see Eonego.fsproj), so there is no EvalFile
/// UCI option.
type private UCIState =
    { mutable Threads: int
      mutable HashMb: int
      mutable MultiPv: int
      mutable MoveOverhead: int
      mutable UseWorkQueue: bool
      Net: Network option
      Tt: TranspositionTable
      mutable RootFen: string
      mutable RootMoves: Move[]
      mutable Control: SearchControl option
      mutable SearchThread: Thread option }

let private tryInt (s: string) : int =
    match Int32.TryParse s with
    | true, v -> v
    | _ -> 0

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

let private startSearch (st: UCIState) (lim: SearchLimits) =
    stopAndJoin st

    match st.Net with
    | None ->
        writeLine "info string no NNUE net embedded; cannot search"
        let p = Position()
        p.LoadFen st.RootFen
        for m in st.RootMoves do p.Make m
        writeLine ("bestmove " + toUci (Search.firstLegalMove p))
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
              UseDeltaPruning = true
              UseContHist = true
              UseSingular = true
              UseNmpVerify = true
              UseLmrTweaks = true
              UseAspTweaks = true
              MoveOverhead = st.MoveOverhead
              AccCheckpointMb = 0
              DagHashMb = 2
              UseWorkQueue = st.UseWorkQueue
              MultiPv = st.MultiPv }

        let control =
            SearchControl(cfg, lim, st.Tt, st.RootFen, st.RootMoves, ?net = st.Net)
        st.Control <- Some control

        let t =
            Thread(
                ThreadStart(fun () -> Search.go control |> ignore),
                16 * 1024 * 1024
            )

        t.IsBackground <- true
        st.SearchThread <- Some t
        t.Start()

let private handleSetOption (st: UCIState) (tokens: string[]) =
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
        elif String.Equals(name, "Hash", StringComparison.OrdinalIgnoreCase) then
            // Resize must never run under a live probe: stop+join any active search first (same
            // guarantee ucinewgame provides for Clear).
            stopAndJoin st
            st.HashMb <- max 1 (min MaxHashMb v)
            st.Tt.Resize st.HashMb
        elif String.Equals(name, "MultiPV", StringComparison.OrdinalIgnoreCase) then
            st.MultiPv <- max 1 (min 256 v)
        elif String.Equals(name, "Move Overhead", StringComparison.OrdinalIgnoreCase) then
            st.MoveOverhead <- max 0 (min 5000 v)
        elif String.Equals(name, "Use Work Queue", StringComparison.OrdinalIgnoreCase) then
            st.UseWorkQueue <- (v <> 0)
    // Any other (legacy/hardwired) option is silently ignored.
    | _ -> ()

let run () =
    // The NNUE net is embedded in the binary (see Eonego.fsproj <EmbeddedResource>); load it once at startup.
    let net =
        match readEmbedded "eval.nnue" with
        | Some bytes -> (match Nnue.loadBytes bytes with Loaded n -> Some n | Failed _ -> None)
        | None -> None

    let st =
        { Threads = 1
          HashMb = DefaultHashMb
          MultiPv = 1
          MoveOverhead = 10
          UseWorkQueue = false
          Net = net
          Tt = TranspositionTable(DefaultHashMb)
          RootFen = StartPosFen
          RootMoves = [||]
          Control = None
          SearchThread = None }

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
                    writeLine "option name Hash type spin default 256 min 1 max 65536"
                    writeLine "option name MultiPV type spin default 1 min 1 max 256"
                    writeLine "option name Move Overhead type spin default 10 min 0 max 5000"
                    writeLine "option name Use Work Queue type check default false"
                    writeLine "uciok"
                | "isready" ->
                    writeLine "readyok"
                | "ucinewgame" ->
                    stopAndJoin st
                    st.Tt.Clear()
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
