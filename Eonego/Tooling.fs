/// Eonego — offline tooling subcommands (gen) for self-play data generation.
/// Console.Out only (AOT-safe); parse numbers with InvariantCulture.
module Eonego.Tooling

#nowarn "9"

open System
open System.Collections.Generic
open System.Globalization
open System.IO
open System.Text
open Microsoft.FSharp.NativeInterop
open Eonego.Bitboard
open Eonego.Move
open Eonego.Position
open Eonego.MoveGeneration
open Eonego.NNUE
open Eonego.Search

let private inv = CultureInfo.InvariantCulture

let private writeLine (s: string) = Console.Out.WriteLine(s)

let private errLine (s: string) = Console.Error.WriteLine(s)

let private usage () =
    writeLine "Eonego tooling subcommands:"
    writeLine "  gen --start <fen> --games N --out <file> --net <path> [--depth D | --nodes K] [--temp T] [--seed S] [--random-plies P] [--max-plies M]"
    writeLine "  dumpft --net <path> --in <fens> --out <bin>   (trainer FT dump: 1034-byte records bucket/stm/psqt/eval/ft[1024])"
    writeLine "  dumppolicy --net <nnue> --policy <sidecar> --in <fens> --out <txt>   (policy parity dump: fen TAB 64 from-logits TAB 64 to-logits [TAB w d l])"
    writeLine "  tbgen --tb <dir[;dir2]> --out <file> --signatures <list> [--per-signature N] [--total N] [--seed S]   (native Syzygy-labeled records with WDL-preserving move sets)"
    writeLine "  retro <fen> [--verify]   (solve the position's retrograde signatures, print stats + its value; --verify runs the full self-consistency proof)"

/// Unknown-subcommand entry (Program.fs wildcard): fail fast with usage instead of silently
/// dropping into the interactive UCI read loop, which blocks forever on stdin with zero output —
/// a typo'd subcommand in a script/CI otherwise just hangs.
let runUnknown (cmd: string) : int =
    errLine ("unknown subcommand: " + cmd)
    usage ()
    2

/// Hand-rolled `--key value` parser; returns a map of flag -> optional value.
let private parseFlags (args: string[]) : Map<string, string option> =
    let m = Dictionary<string, string option>(StringComparer.OrdinalIgnoreCase)
    let mutable i = 0

    while i < args.Length do
        let tok = args.[i]

        if tok.StartsWith("--") then
            let key = tok.[2..]

            if i + 1 < args.Length && not (args.[i + 1].StartsWith "--") then
                m.[key] <- Some args.[i + 1]
                i <- i + 2
            else
                m.[key] <- None
                i <- i + 1
        else
            i <- i + 1

    m |> Seq.map (|KeyValue|) |> Map.ofSeq

let private flag (m: Map<string, string option>) (name: string) : string option =
    Map.tryFind name m |> Option.flatten

let private flagInt (m: Map<string, string option>) (name: string) (defaultVal: int) : int =
    match flag m name with
    | Some s ->
        match Int32.TryParse(s, NumberStyles.Integer, inv) with
        | true, v -> v
        | _ -> defaultVal
    | None -> defaultVal

let private flagInt64 (m: Map<string, string option>) (name: string) (defaultVal: int64) : int64 =
    match flag m name with
    | Some s ->
        match Int64.TryParse(s, NumberStyles.Integer, inv) with
        | true, v -> v
        | _ -> defaultVal
    | None -> defaultVal

let private flagFloat (m: Map<string, string option>) (name: string) (defaultVal: float) : float =
    match flag m name with
    | Some s ->
        match Double.TryParse(s, NumberStyles.Float, inv) with
        | true, v -> v
        | _ -> defaultVal
    | None -> defaultVal

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

[<Literal>]
let DefaultKgaFen = "rnbqkbnr/pppp1ppp/8/8/4Pp2/8/PPPP2PP/RNBQKBNR w KQkq - 0 3"

let private kgaBookLines =
    [| [| "g1f3"; "g7g5" |]
       [| "g1f3"; "d7d6" |]
       [| "f1c4" |]
       [| "g1f3"; "g7g5"; "h2h4" |]
       [| "g1f3"; "g7g5"; "f1c4"; "g5g4" |]
       [| "d2d4" |]
       [| "b1c3"; "g8f6" |]
       [| "g1f3"; "g7g5"; "e1g1" |] |]

let private collectLegal (pos: Position) : Move[] =
    let p = NativePtr.stackalloc<Move> MaxMoves
    let buf = Span<Move>(NativePtr.toVoidPtr p, MaxMoves)
    let n = generateLegal pos buf
    let moves = Array.zeroCreate n

    for i in 0 .. n - 1 do
        moves.[i] <- buf.[i]

    moves

let private isNoisyBest (pos: Position) (m: Move) : bool =
    isPromotion m || isEnPassant m || not (pos.IsEmpty(toSq m))

let private whiteRelEval (net: Network) (pos: Position) : int =
    let e = NNUE.evalCp net pos
    if pos.SideToMove = White then e else -e

let private softmaxPick (rng: Random) (temp: float) (moves: Move[]) (scores: int[]) : Move =
    if moves.Length = 1 then
        moves.[0]
    else
        let t = max 1.0 temp
        let maxS = scores |> Array.max
        let weights = scores |> Array.map (fun s -> exp (float (s - maxS) / t))
        let sum = weights |> Array.sum
        let r = rng.NextDouble() * sum
        let mutable acc = 0.0

        let mutable pick = moves.[moves.Length - 1]
        let mutable i = 0
        let mutable found = false

        while not found && i < moves.Length do
            acc <- acc + weights.[i]

            if r <= acc then
                pick <- moves.[i]
                found <- true

            i <- i + 1

        pick

let private gameResult (net: Network) (pos: Position) (ply: int) (maxPlies: int) : float =
    let legal = collectLegal pos

    if legal.Length = 0 then
        if pos.InCheck then
            if pos.SideToMove = White then 0.0 else 1.0
        else
            0.5
    elif isImmediateDraw pos then
        0.5
    elif ply >= maxPlies then
        let w = whiteRelEval net pos

        if w > 150 then 1.0
        elif w < -150 then 0.0
        else 0.5
    else
        nan // game continues

let private searchMove
    (pos: Position)
    (rootMoves: Move[])
    (depth: int option)
    (nodes: int64 option)
    (cfg: SearchConfig)
    (net: Network option)
    : struct (int * Move) =
    let fenNow = pos.ToFen()

    match depth, nodes with
    | Some d, _ ->
        let struct (score, _, mv) = searchToDepthNet fenNow [||] d cfg net
        struct (score, mv)
    | None, Some n ->
        let struct (score, _, mv) = searchToNodesNet fenNow [||] n cfg net
        struct (score, mv)
    | _ ->
        let struct (score, _, mv) = searchToDepthNet fenNow [||] 6 cfg net
        struct (score, mv)

type private PendingRec =
    { Fen: string
      Key: uint64
      CpWhite: int
      // gen v2 (policy campaign Phase 2): the search best move at this position (STM UCI). Was
      // computed and DISCARDED in v1 — free policy training labels. 4th `;`-field, so every v1
      // parser (split-and-take-3) still reads v2 files.
      BestUci: string }

let private playOneGame
    (writer: StreamWriter)
    (startFen: string)
    (rng: Random)
    (temp: float)
    (randomPlies: int)
    (maxPlies: int)
    (depthOpt: int option)
    (nodesOpt: int64 option)
    (cfg: SearchConfig)
    (net: Network)
    : unit =
    let pos = Position.OfFen startFen
    let mutable rootMoves = [||]
    let book = kgaBookLines.[rng.Next kgaBookLines.Length]
    let prefixTarget = max randomPlies book.Length
    let mutable prefixDone = 0

    while prefixDone < prefixTarget do
        let legal = collectLegal pos

        if legal.Length = 0 then
            prefixDone <- prefixTarget
        else
            let mvFromBook =
                if prefixDone < book.Length then matchMove pos book.[prefixDone] else MoveNone

            let mv =
                if mvFromBook <> MoveNone then
                    mvFromBook
                else
                    // v1 bug: one root search score was broadcast to every legal move, making
                    // softmaxPick uniform-random regardless of --temp. Score each move by the
                    // static eval after making it (STM-negated) — cheap (from-scratch eval,
                    // prefix plies only) and gives --temp its intended eval-weighted meaning.
                    let scores =
                        legal
                        |> Array.map (fun m ->
                            pos.Make m
                            let s = -NNUE.evalCp net pos
                            pos.Unmake m
                            s)

                    softmaxPick rng temp legal scores

            if mv = MoveNone then
                prefixDone <- prefixTarget
            else
                pos.Make mv
                rootMoves <- Array.append rootMoves [| mv |]
                prefixDone <- prefixDone + 1

    let mutable ply = prefixDone
    let keys = HashSet<uint64>()
    let pending = ResizeArray<PendingRec>()
    let mutable finished = false

    while not finished do
        let result = gameResult net pos ply maxPlies

        if not (Double.IsNaN result) then
            for entry in pending do
                writer.WriteLine(
                    entry.Fen + ";" + string entry.CpWhite + ";" + result.ToString("0.0", inv) + ";" + entry.BestUci
                )

            finished <- true
        else
            let struct (scoreStm, best) = searchMove pos rootMoves depthOpt nodesOpt cfg (Some net)

            if best = MoveNone then
                let r = gameResult net pos ply maxPlies

                for entry in pending do
                    writer.WriteLine(
                        entry.Fen + ";" + string entry.CpWhite + ";" + r.ToString("0.0", inv) + ";" + entry.BestUci
                    )

                finished <- true
            else
                if not pos.InCheck && keys.Add pos.Key && not (isNoisyBest pos best) then
                    let cpWhite =
                        if pos.SideToMove = White then scoreStm else -scoreStm

                    pending.Add
                        { Fen = pos.ToFen()
                          Key = pos.Key
                          CpWhite = cpWhite
                          BestUci = toUCI best }

                pos.Make best
                rootMoves <- Array.append rootMoves [| best |]
                ply <- ply + 1

let runGen (args: string[]) : int =
    let runGenGames outPath startFen games depthOpt nodesOpt temp seed randomPlies maxPlies (net: Network) =
        let cfg = { defaultConfig with Threads = 1 }

        use writer = new StreamWriter(outPath, false, Encoding.UTF8)
        writer.WriteLine("# eonego gen v2 fen;cp_white;result_white;best_uci")
        writer.WriteLine("# start=" + startFen)
        writer.WriteLine("# games=" + string games)
        writer.WriteLine("# seed=" + string seed)
        writer.WriteLine("# depth=" + (match depthOpt with Some d -> string d | None -> "-"))
        writer.WriteLine("# nodes=" + (match nodesOpt with Some n -> string n | None -> "-"))
        writer.Flush()

        for g in 0 .. games - 1 do
            let rng = Random(seed + g)

            playOneGame writer startFen rng temp randomPlies maxPlies depthOpt nodesOpt cfg net

        writer.Flush()
        0

    let m = parseFlags args

    match flag m "out" with
    | None ->
        errLine "gen requires --out <file>"
        1
    | Some outPath ->
        let startFen = flag m "start" |> Option.defaultValue DefaultKgaFen
        let games = flagInt m "games" 1

        let depthOpt =
            match flagInt m "depth" 0 with
            | 0 -> None
            | d -> Some d

        let nodesOpt =
            match flagInt64 m "nodes" 0L with
            | 0L -> None
            | n -> Some n

        let temp = flagFloat m "temp" 50.0
        let seed = flagInt m "seed" 42
        let randomPlies = flagInt m "random-plies" 8
        let maxPlies = flagInt m "max-plies" 200

        match flag m "net" with
        | Some path ->
            match NNUE.load path with
            | NNUE.Failed r ->
                errLine ("failed to load net: " + r)
                1
            | NNUE.Loaded n -> runGenGames outPath startFen games depthOpt nodesOpt temp seed randomPlies maxPlies n
        | None ->
            errLine "gen requires --net <path> (NNUE file)"
            1

/// Trainer FT dump: for each input FEN (bare fen, or `fen;...` records), write one binary record
/// [bucket u8][stm u8][psqtInternal i32][evalInternal i32][ft u8 x L1] (little-endian, 1034 bytes).
/// Record order matches input line order so the trainer can join labels by index.
let runDumpFt (args: string[]) : int =
    let m = parseFlags args

    match flag m "net", flag m "in", flag m "out" with
    | Some netPath, Some inPath, Some outPath ->
        match NNUE.load netPath with
        | NNUE.Failed r ->
            errLine ("failed to load net: " + r)
            1
        | NNUE.Loaded net ->
            use writer = new BinaryWriter(File.Create outPath)
            let ft = Array.zeroCreate<byte> NNUE.L1
            let mutable count = 0

            for line in File.ReadLines inPath do
                let t = line.Trim()

                if t.Length > 0 && not (t.StartsWith "#") then
                    let fen = t.Split(';').[0].Trim()
                    let pos = Position.OfFen fen
                    let struct (bucket, psqtInternal, evalInt) = NNUE.dumpFeatures net pos ft
                    writer.Write(byte bucket)
                    writer.Write(if pos.SideToMove = White then 0uy else 1uy)
                    writer.Write(psqtInternal)
                    writer.Write(evalInt)
                    writer.Write(ft)
                    count <- count + 1

            writer.Flush()
            writeLine ("dumped " + string count + " records to " + outPath)
            0
    | _ ->
        errLine "dumpft requires --net <path> --in <fens> --out <bin>"
        1

/// Policy parity dump (trainer/policy_parity.py's engine side): for each input FEN, run the sidecar
/// head forward from scratch and emit one text record `fen \t f0..f63 \t t0..t63 [\t w d l]`
/// (raw i32 logits, STM-relative square order; per-mille WDL when the sidecar carries the section).
/// The parity gate MUST include Black-to-move and promotion FENs — STM mirroring is the classic
/// silent engine/trainer skew.
let runDumpPolicy (args: string[]) : int =
    let m = parseFlags args

    match flag m "net", flag m "policy", flag m "in", flag m "out" with
    | Some netPath, Some polPath, Some inPath, Some outPath ->
        match NNUE.load netPath with
        | NNUE.Failed r ->
            errLine ("failed to load net: " + r)
            1
        | NNUE.Loaded net ->
            match Policy.load polPath net.FtHash with
            | Policy.PolicyFailed r ->
                errLine ("failed to load policy: " + r)
                1
            | Policy.PolicyLoaded pnet ->
                use writer = new StreamWriter(File.Create outPath)
                let fromL = Array.zeroCreate<int> Policy.HeadOut
                let toL = Array.zeroCreate<int> Policy.HeadOut
                let mutable count = 0

                for line in File.ReadLines inPath do
                    let t = line.Trim()

                    if t.Length > 0 && not (t.StartsWith "#") then
                        let fen = t.Split(';').[0].Trim()
                        let pos = Position.OfFen fen
                        Policy.fillLogits net pnet pos (fromL.AsSpan()) (toL.AsSpan())
                        let sb = Text.StringBuilder(1024)
                        sb.Append(fen).Append('\t') |> ignore

                        for i in 0 .. Policy.HeadOut - 1 do
                            (if i > 0 then sb.Append(' ') else sb).Append(fromL.[i]) |> ignore

                        sb.Append('\t') |> ignore

                        for i in 0 .. Policy.HeadOut - 1 do
                            (if i > 0 then sb.Append(' ') else sb).Append(toL.[i]) |> ignore

                        if pnet.HasWdl then
                            let struct (w, d, l) = Policy.evalWDL net pnet pos
                            sb.Append('\t').Append(w).Append(' ').Append(d).Append(' ').Append(l) |> ignore

                        writer.WriteLine(sb.ToString())
                        count <- count + 1

                writer.Flush()
                writeLine ("dumped " + string count + " policy records to " + outPath)
                0
    | _ ->
        errLine "dumppolicy requires --net <nnue> --policy <sidecar> --in <fens> --out <txt>"
        1

/// Syzygy-labeled endgame generator (`tbgen`) — NATIVE probing via Syzygy.fs (the python-chess
/// prober in trainer/policy_endgen.py is orders of magnitude slower and dies natively on rare
/// blocks; this replaces it for bulk generation). Samples random legal positions per material
/// signature, probes every child, and emits gen-v2 records PLUS a 5th field: the FULL set of
/// WDL-preserving moves. That set is the policy head's trainable definition of "100% correct
/// play" — never worsen the theoretical result — as opposed to DTZ-argmax, which no shippable
/// net can memorize. Cursed wins / blessed losses count as draws (50-move-aware).
///   tbgen --tb <dir[;dir2]> --out <file> --signatures "KQvK,KRvK,..." [--per-signature N]
///         [--total N] [--seed S]
/// Record: fen;cp_white;result_white;best_uci;good1 good2 ...   (best = first of the good set)
let runTbGen (args: string[]) : int =
    let m = parseFlags args

    match flag m "tb", flag m "out", flag m "signatures" with
    | Some tbPath, Some outPath, Some sigSpec ->
        if not (Syzygy.init tbPath) || Syzygy.Largest = 0 then
            errLine ("tbgen: no tables found under " + tbPath)
            1
        else
            let perSig = flagInt m "per-signature" 5000
            let totalCap = flagInt m "total" 1_000_000
            let seed = flagInt m "seed" 42
            let rng = Random(seed)

            let ptOfChar c =
                match c with
                | 'P' -> Pawn
                | 'N' -> Knight
                | 'B' -> Bishop
                | 'R' -> Rook
                | 'Q' -> Queen
                | _ -> failwith ("tbgen: bad piece char " + string c)

            let fenChar (white: bool) (pt: int) =
                let c = "pnbrqk".[pt]
                if white then Char.ToUpperInvariant c else c

            // Build a FEN for a random placement; None when the placement is structurally illegal.
            let tryBuildFen (strong: int[]) (weak: int[]) : string option =
                let n = 2 + strong.Length + weak.Length
                let squares = HashSet<int>()

                while squares.Count < n do
                    squares.Add(rng.Next 64) |> ignore

                let sq = Array.ofSeq squares
                let wk = sq.[0]
                let bk = sq.[1]
                let board = Array.create 64 ' '
                board.[wk] <- 'K'
                board.[bk] <- 'k'
                let mutable ok = abs (fileOf wk - fileOf bk) > 1 || abs (rankOf wk - rankOf bk) > 1

                let place (idx: int) (white: bool) (pt: int) =
                    let s = sq.[idx]

                    if pt = Pawn && (s < 8 || s >= 56) then
                        ok <- false
                    else
                        board.[s] <- fenChar white pt

                strong |> Array.iteri (fun i pt -> place (2 + i) true pt)
                weak |> Array.iteri (fun i pt -> place (2 + strong.Length + i) false pt)

                if not ok then
                    None
                else
                    let sb = StringBuilder(80)

                    for r in 7 .. -1 .. 0 do
                        let mutable empty = 0

                        for f in 0 .. 7 do
                            let c = board.[r * 8 + f]

                            if c = ' ' then
                                empty <- empty + 1
                            else
                                if empty > 0 then
                                    sb.Append(empty) |> ignore
                                    empty <- 0

                                sb.Append(c) |> ignore

                        if empty > 0 then
                            sb.Append(empty) |> ignore

                        if r > 0 then
                            sb.Append('/') |> ignore

                    sb.Append(if rng.Next 2 = 0 then " w - - 0 1" else " b - - 0 1") |> ignore
                    Some(sb.ToString())

            use writer = new StreamWriter(outPath, false, Encoding.UTF8)
            writer.WriteLine("# eonego tbgen v2 fen;cp_white;result_white;best_uci;good_ucis;quiet_ucis")
            let mutable keptTotal = 0

            for sigRaw in sigSpec.Split(',') do
                let sigName = sigRaw.Trim()

                if sigName.Length > 0 && keptTotal < totalCap then
                    let halves = sigName.Split('v')
                    let strong = halves.[0].[1..].ToCharArray() |> Array.map ptOfChar
                    let weak = halves.[1].[1..].ToCharArray() |> Array.map ptOfChar
                    let seen = HashSet<uint64>()
                    let mutable kept = 0
                    let mutable tries = 0
                    let budget = perSig * 60

                    while kept < perSig && tries < budget && keptTotal < totalCap do
                        tries <- tries + 1

                        match tryBuildFen strong weak with
                        | None -> ()
                        | Some fen ->
                            let pos = Position.OfFen fen
                            let us = pos.SideToMove
                            let them = us ^^^ 1

                            let illegal =
                                pos.AttackersTo (pos.KingSquare them) pos.Occupied &&& pos.ColorBB(us) <> 0UL

                            if not illegal && seen.Add pos.Key then
                                let legal = collectLegal pos

                                if legal.Length >= 2 && Syzygy.probeWDL pos <> Int32.MinValue then
                                    // probeWDL is Fathom's -2..+2 scale, SIDE-TO-MOVE relative
                                    // (-2 loss, 0 draw, +2 win, +/-1 cursed/blessed — exactly how
                                    // Search.fs consumes it). Value of each move from the MOVER's
                                    // perspective = the negated child probe. A bare KvK child has
                                    // no table: exact draw (0).
                                    let vals = Array.zeroCreate legal.Length
                                    let mutable allOk = true

                                    for i in 0 .. legal.Length - 1 do
                                        pos.Make legal.[i]

                                        vals.[i] <-
                                            if popCount pos.Occupied = 2 then
                                                0
                                            else
                                                let w = Syzygy.probeWDL pos
                                                if w = Int32.MinValue then allOk <- false
                                                -w

                                        pos.Unmake legal.[i]

                                    if allOk then
                                        let bestVal = Array.max vals

                                        let good =
                                            [| for i in 0 .. legal.Length - 1 do
                                                   if vals.[i] = bestVal then yield legal.[i] |]

                                        // stm-relative outcome under best play; cursed/blessed = draw.
                                        let resStm =
                                            if bestVal >= 2 then 1.0
                                            elif bestVal <= -2 then 0.0
                                            else 0.5

                                        let resWhite = if us = White then resStm else 1.0 - resStm
                                        let cpStm = int ((resStm - 0.5) * 2000.0)
                                        let cpWhite = if us = White then cpStm else -cpStm
                                        let goods = good |> Array.map toUCI |> String.concat " "

                                        // 6th field: ALL legal quiets (engine isQuiet) — lets the
                                        // trainer build the quiet-conditional softmax support by
                                        // pure text parsing (no python-side movegen over millions
                                        // of rows).
                                        let quiets =
                                            legal
                                            |> Array.filter (fun q ->
                                                pos.PieceOn(toSq q) = NoPiece
                                                && not (isEnPassant q)
                                                && not (isPromotion q))
                                            |> Array.map toUCI
                                            |> String.concat " "

                                        writer.WriteLine(
                                            fen + ";" + string cpWhite + ";" + resWhite.ToString("0.0", inv)
                                            + ";" + toUCI good.[0] + ";" + goods + ";" + quiets
                                        )

                                        kept <- kept + 1
                                        keptTotal <- keptTotal + 1

                    writeLine (sigName + ": kept " + string kept + " (" + string tries + " tries)")

            writer.Flush()
            writeLine ("TOTAL " + string keptTotal + " records -> " + outPath)
            0
    | _ ->
        errLine "tbgen requires --tb <dir[;dir2]> --out <file> --signatures <list>"
        1

/// Own-trunk policy parity dump (`dumppolicyown`): for each input FEN emit `fen \t f0..f383 \t
/// t0..t383` — the EONPOL03 net's scaled-int logits (only meaningful at ≤6 pieces). The Python
/// reference (policy_own.forward) is compared against this to validate the F# feature extraction.
let runDumpPolicyOwn (args: string[]) : int =
    let m = parseFlags args

    match flag m "policy", flag m "in", flag m "out" with
    | Some polPath, Some inPath, Some outPath ->
        match Policy.loadOwn polPath with
        | Policy.OwnFailed r ->
            errLine ("failed to load own policy: " + r)
            1
        | Policy.OwnLoaded onet ->
            use writer = new StreamWriter(File.Create outPath)
            let fromL = Array.zeroCreate<int> Policy.HeadOut
            let toL = Array.zeroCreate<int> Policy.HeadOut
            let mutable count = 0

            for line in File.ReadLines inPath do
                let t = line.Trim()

                if t.Length > 0 && not (t.StartsWith "#") then
                    let fen = t.Split(';').[0].Trim()
                    let pos = Position.OfFen fen

                    if Policy.ownApplies pos then
                        Policy.fillLogitsOwn onet pos (fromL.AsSpan()) (toL.AsSpan())
                        let sb = Text.StringBuilder(4096)
                        sb.Append(fen).Append('\t') |> ignore

                        for i in 0 .. Policy.HeadOut - 1 do
                            (if i > 0 then sb.Append(' ') else sb).Append(fromL.[i]) |> ignore

                        sb.Append('\t') |> ignore

                        for i in 0 .. Policy.HeadOut - 1 do
                            (if i > 0 then sb.Append(' ') else sb).Append(toL.[i]) |> ignore

                        writer.WriteLine(sb.ToString())
                        count <- count + 1

            writer.Flush()
            writeLine ("dumped " + string count + " own-policy records to " + outPath)
            0
    | _ ->
        errLine "dumppolicyown requires --policy <eonpol03> --in <fens> --out <txt>"
        1

/// `tbprobe --tb <dir> <fen>` — raw Syzygy.probeWDL of the position and every legal child
/// (debug surface for tbgen label verification; child values shown MOVER-relative, 4 - w).
let runTbProbe (args: string[]) : int =
    let m = parseFlags args

    match flag m "tb" with
    | Some tbPath when Syzygy.init tbPath ->
        // Everything after `--tb <dir>` is the FEN.
        let tbIdx = Array.findIndex ((=) "--tb") args
        let fen = args.[tbIdx + 2 ..] |> String.concat " "
        let pos = Position.OfFen fen
        writeLine ("root probeWDL (stm-rel, -2=loss..+2=win): " + string (Syzygy.probeWDL pos))

        for mv in collectLegal pos do
            pos.Make mv
            let w = if popCount pos.Occupied = 2 then 0 else Syzygy.probeWDL pos
            pos.Unmake mv
            writeLine (toUCI mv + ": child=" + string w + " moverVal=" + string (-w))

        0
    | _ ->
        errLine "tbprobe requires --tb <dir> <fen>"
        1

/// `retro <fen> [--verify]` — synchronously solve the position's retrograde signature closure,
/// print per-signature stats and the position's own value + root score mapping. `--verify` runs
/// the full self-consistency proof on each solved signature (exit 1 on the first mismatch).
let runRetro (args: string[]) : int =
    let verify =
        args
        |> Array.exists (fun a -> a.Equals("--verify", StringComparison.OrdinalIgnoreCase))

    let fenToks = args |> Array.filter (fun a -> not (a.StartsWith "--"))

    if fenToks.Length = 0 then
        usage ()
        1
    else
        let pos = Position.OfFen(String.Join(" ", fenToks))
        let sigs = Eonego.Retrograde.signatureClosure pos

        if List.isEmpty sigs then
            errLine "retro: no solvable signature (need a 3- or 4-man position)"
            1
        else
            let sigName (pce: Piece) =
                "K"
                + string ("PNBRQK".[pieceType pce])
                + "K "
                + (if pieceColor pce = White then "(white)" else "(black)")

            let promoTablesOf (pce: Piece) : sbyte[][] =
                if pieceType pce = Pawn then
                    let owner = pieceColor pce
                    let t: sbyte[][] = Array.zeroCreate 6
                    t.[Knight] <- Eonego.Retrograde.solvedTable (makePiece owner Knight)
                    t.[Bishop] <- Eonego.Retrograde.solvedTable (makePiece owner Bishop)
                    t.[Rook] <- Eonego.Retrograde.solvedTable (makePiece owner Rook)
                    t.[Queen] <- Eonego.Retrograde.solvedTable (makePiece owner Queen)
                    t
                else
                    Array.empty

            let mutable exitCode = 0

            for pce in sigs do
                let sw = System.Diagnostics.Stopwatch.StartNew()
                Eonego.Retrograde.ensureSolved pce
                sw.Stop()
                let tbl = Eonego.Retrograde.solvedTable pce

                let struct (legal, wins, losses, maxWin, maxLoss) = Eonego.Retrograde.statsOf tbl

                writeLine (
                    "signature "
                    + sigName pce
                    + ": legal "
                    + string legal
                    + " win "
                    + string wins
                    + " loss "
                    + string losses
                    + " draw "
                    + string (legal - wins - losses)
                    + " maxWinDtm "
                    + string maxWin
                    + " maxLossDtm "
                    + string maxLoss
                    + " solveMs "
                    + string sw.ElapsedMilliseconds
                )

                if verify then
                    match Eonego.Retrograde.verifySignature pce tbl (promoTablesOf pce) with
                    | None -> writeLine ("verify " + sigName pce + ": OK")
                    | Some err ->
                        errLine ("verify " + sigName pce + ": FAILED - " + err)
                        exitCode <- 1

            match Eonego.Retrograde.probe pos with
            | ValueSome v ->
                let desc =
                    if v = 0y then "draw"
                    elif v > 0y then "win in " + string (Eonego.Retrograde.retroDtm v) + " plies"
                    else "loss in " + string (Eonego.Retrograde.retroDtm v) + " plies"

                writeLine ("value: " + desc + " | root score " + string (retroScoreAt pos 0))
            | ValueNone -> writeLine "value: not covered (4-man input pre-solved only, or probe guards declined)"

            exitCode
