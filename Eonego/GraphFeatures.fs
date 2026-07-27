/// Fixed graph-feature extraction for the CUDA-batched graph evaluator.
///
/// This module is intentionally independent from NNUE and search. It is the runtime/trainer schema
/// contract for `dumpgraph`, the CUDA bridge, and parity tests.
module Eonego.GraphFeatures

open System
open System.IO
open System.Runtime.CompilerServices
open Eonego.Bitboard
open Eonego.Position

[<Literal>]
let MaxGraphNodes = 32

[<Literal>]
let NodeFeatureCount = 12

[<Literal>]
let EdgeFeatureCount = 8

[<Literal>]
let GlobalFeatureCount = 32

[<Literal>]
let GraphVersion = 1

/// Bump whenever the byte layout below changes.
[<Literal>]
let SchemaHash = 0x47463101u

[<Literal>]
let RecordBytes =
    4 + MaxGraphNodes * NodeFeatureCount + MaxGraphNodes * MaxGraphNodes * EdgeFeatureCount + GlobalFeatureCount

let Magic: byte[] = Text.Encoding.ASCII.GetBytes "EONGF01!"

[<Sealed>]
type GraphRecord(nodeCount: int, stm: int, inCheck: bool, checkerCount: int, nodes: byte[], edges: byte[], globals: byte[]) =
    member _.NodeCount = nodeCount
    member _.SideToMove = stm
    member _.InCheck = inCheck
    member _.CheckerCount = checkerCount
    member _.Nodes = nodes
    member _.Edges = edges
    member _.Globals = globals

[<Struct>]
type private NodeRef =
    { Square: Square
      Piece: Piece
      Own: int
      Rel: int }

[<MethodImpl(MethodImplOptions.AggressiveInlining)>]
let relSq (stm: Color) (sq: Square) : Square = if stm = White then sq else sq ^^^ 56

[<MethodImpl(MethodImplOptions.AggressiveInlining)>]
let private b (v: int) : byte = byte (max 0 (min 255 v))

[<MethodImpl(MethodImplOptions.AggressiveInlining)>]
let private attackMask (pc: Piece) (sq: Square) (occ: Bitboard) : Bitboard =
    let c = pieceColor pc

    match pieceType pc with
    | Pawn -> pawnAttacks c sq
    | Knight -> knightAttacks sq
    | Bishop -> bishopAttacks sq occ
    | Rook -> rookAttacks sq occ
    | Queen -> queenAttacks sq occ
    | _ -> kingAttacks sq

let private directionCode (from: Square) (dst: Square) : int =
    let df = fileOf dst - fileOf from
    let dr = rankOf dst - rankOf from

    if df = 0 && dr > 0 then 1
    elif df = 0 && dr < 0 then 2
    elif dr = 0 && df > 0 then 3
    elif dr = 0 && df < 0 then 4
    elif abs df = abs dr && df > 0 && dr > 0 then 5
    elif abs df = abs dr && df < 0 && dr > 0 then 6
    elif abs df = abs dr && df > 0 && dr < 0 then 7
    elif abs df = abs dr && df < 0 && dr < 0 then 8
    else 0

let extract (pos: Position) : GraphRecord =
    let stm = pos.SideToMove
    let them = flipColor stm
    let occ = pos.Occupied
    let ownPins = pos.BlockersForKing stm
    let oppPins = pos.BlockersForKing them
    let checkers = pos.Checkers

    let pieces =
        [| for sq in 0 .. 63 do
               let pc = pos.PieceOn sq

               if pc <> NoPiece then
                   let own = if pieceColor pc = stm then 1 else 0
                   yield { Square = sq; Piece = pc; Own = own; Rel = relSq stm sq } |]
        |> Array.sortBy (fun n -> (1 - n.Own, pieceType n.Piece, n.Rel))

    let nodeCount = min MaxGraphNodes pieces.Length
    let nodes = Array.zeroCreate<byte> (MaxGraphNodes * NodeFeatureCount)
    let edges = Array.zeroCreate<byte> (MaxGraphNodes * MaxGraphNodes * EdgeFeatureCount)
    let globals = Array.zeroCreate<byte> GlobalFeatureCount

    for i in 0 .. nodeCount - 1 do
        let n = pieces.[i]
        let pc = n.Piece
        let sq = n.Square
        let o = i * NodeFeatureCount
        nodes.[o] <- 1uy
        nodes.[o + 1] <- byte n.Own
        nodes.[o + 2] <- byte (pieceColor pc)
        nodes.[o + 3] <- byte (pieceType pc)
        nodes.[o + 4] <- byte sq
        nodes.[o + 5] <- byte n.Rel
        nodes.[o + 6] <- byte (fileOf sq)
        nodes.[o + 7] <- byte (rankOf sq)
        nodes.[o + 8] <- if testBit ownPins sq then 1uy else 0uy
        nodes.[o + 9] <- if testBit oppPins sq then 1uy else 0uy
        nodes.[o + 10] <- if testBit checkers sq then 1uy else 0uy
        nodes.[o + 11] <- if pieceType pc = King then 1uy else 0uy

    for i in 0 .. nodeCount - 1 do
        let a = pieces.[i]
        let aMask = attackMask a.Piece a.Square occ

        for j in 0 .. nodeCount - 1 do
            if i <> j then
                let c = pieces.[j]
                let cMask = attackMask c.Piece c.Square occ
                let eo = ((i * MaxGraphNodes) + j) * EdgeFeatureCount
                let sameRay = line a.Square c.Square <> 0UL
                let clearRay = sameRay && ((between a.Square c.Square &&& occ) = 0UL)
                let df = abs (fileOf c.Square - fileOf a.Square)
                let dr = abs (rankOf c.Square - rankOf a.Square)
                edges.[eo] <- 1uy
                edges.[eo + 1] <- if pieceColor a.Piece = pieceColor c.Piece then 1uy else 0uy
                edges.[eo + 2] <- if testBit aMask c.Square then 1uy else 0uy
                edges.[eo + 3] <- if testBit cMask a.Square then 1uy else 0uy
                edges.[eo + 4] <- if sameRay then 1uy else 0uy
                edges.[eo + 5] <- if clearRay then 1uy else 0uy
                edges.[eo + 6] <- byte (directionCode a.Square c.Square)
                edges.[eo + 7] <- byte (max df dr)

    globals.[0] <- byte stm
    let castle = pos.CastlingRights

    globals.[1] <-
        byte (
            (if (castle &&& (if stm = White then WK else BK)) <> 0 then 1 else 0)
            ||| (if (castle &&& (if stm = White then WQ else BQ)) <> 0 then 2 else 0)
            ||| (if (castle &&& (if stm = White then BK else WK)) <> 0 then 4 else 0)
            ||| (if (castle &&& (if stm = White then BQ else WQ)) <> 0 then 8 else 0)
        )
    globals.[2] <- if pos.EpSquare = NoSquare then 0uy else byte (fileOf (relSq stm pos.EpSquare) + 1)
    globals.[3] <- b pos.Rule50
    globals.[4] <- if pos.InCheck then 1uy else 0uy
    globals.[5] <- b (popCount checkers)
    globals.[6] <- b nodeCount

    let mutable ownMat = 0
    let mutable oppMat = 0

    for pt in Pawn .. King do
        let ownN = popCount (pos.PiecesCT stm pt)
        let oppN = popCount (pos.PiecesCT them pt)
        globals.[8 + pt] <- byte ownN
        globals.[14 + pt] <- byte oppN
        ownMat <- ownMat + ownN * pos.PieceValueOf pt
        oppMat <- oppMat + oppN * pos.PieceValueOf pt

    globals.[20] <- b ((ownMat + oppMat) / 100)
    globals.[21] <- b ((ownMat - oppMat + 4000) / 32)
    globals.[22] <- b (popCount (pos.ColorBB stm))
    globals.[23] <- b (popCount (pos.ColorBB them))

    GraphRecord(nodeCount, stm, pos.InCheck, popCount checkers, nodes, edges, globals)

let writeHeader (writer: BinaryWriter) (count: int) : unit =
    writer.Write(Magic)
    writer.Write(GraphVersion)
    writer.Write(int SchemaHash)
    writer.Write(RecordBytes)
    writer.Write(count)

let writeRecord (writer: BinaryWriter) (r: GraphRecord) : unit =
    writer.Write(byte r.NodeCount)
    writer.Write(byte r.SideToMove)
    writer.Write(if r.InCheck then 1uy else 0uy)
    writer.Write(byte r.CheckerCount)
    writer.Write(r.Nodes)
    writer.Write(r.Edges)
    writer.Write(r.Globals)

let toBytes (r: GraphRecord) : byte[] =
    let out = Array.zeroCreate<byte> RecordBytes
    out.[0] <- byte r.NodeCount
    out.[1] <- byte r.SideToMove
    out.[2] <- if r.InCheck then 1uy else 0uy
    out.[3] <- byte r.CheckerCount
    Buffer.BlockCopy(r.Nodes, 0, out, 4, r.Nodes.Length)
    Buffer.BlockCopy(r.Edges, 0, out, 4 + r.Nodes.Length, r.Edges.Length)
    Buffer.BlockCopy(r.Globals, 0, out, 4 + r.Nodes.Length + r.Edges.Length, r.Globals.Length)
    out
