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
open Eonego.Search

let private writeLine (s: string) = Console.Out.WriteLine(s)

type private UciState =
    { mutable Threads: int
      mutable HashMb: int
      mutable UseProbCut: bool
      mutable Tt: TranspositionTable
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
        | _ -> i <- i + 1

    lim

let private startSearch (st: UciState) (lim: SearchLimits) =
    stopAndJoin st

    let cfg =
        { Threads = st.Threads
          HashMb = st.HashMb
          UseTt = true
          UsePruning = true
          UseProbCut = st.UseProbCut }

    let control = SearchControl(cfg, lim, st.Tt, st.RootFen, st.RootMoves)
    st.Control <- Some control
    let t = Thread(ThreadStart(fun () -> Search.go control |> ignore), 16 * 1024 * 1024)
    t.IsBackground <- true
    st.SearchThread <- Some t
    t.Start()

let private handleSetOption (st: UciState) (tokens: string[]) =
    // setoption name <Name> value <Value>
    let nameIdx = Array.tryFindIndex ((=) "name") tokens
    let valIdx = Array.tryFindIndex ((=) "value") tokens

    match nameIdx, valIdx with
    | Some ni, Some vi when ni + 1 < tokens.Length && vi + 1 < tokens.Length ->
        let name = tokens.[ni + 1]
        let v = tryInt tokens.[vi + 1]

        if String.Equals(name, "Hash", StringComparison.OrdinalIgnoreCase) then
            stopAndJoin st
            st.HashMb <- max 1 v
            st.Tt.Resize st.HashMb
        elif String.Equals(name, "Threads", StringComparison.OrdinalIgnoreCase) then
            st.Threads <- max 1 v
        elif String.Equals(name, "UseProbCut", StringComparison.OrdinalIgnoreCase) then
            match Boolean.TryParse tokens.[vi + 1] with
            | true, b -> st.UseProbCut <- b
            | _ -> ()
    | _ -> ()

let run () =
    let st =
        { Threads = 1
          HashMb = 16
          UseProbCut = true
          Tt = TranspositionTable(16)
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
                    writeLine "option name Hash type spin default 16 min 1 max 65536"
                    writeLine "option name Threads type spin default 1 min 1 max 256"
                    writeLine "option name UseProbCut type check default true"
                    writeLine "uciok"
                | "isready" -> writeLine "readyok"
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
                | "setoption" -> handleSetOption st tokens
                | "quit" ->
                    stopAndJoin st
                    running <- false
                | _ -> ()
