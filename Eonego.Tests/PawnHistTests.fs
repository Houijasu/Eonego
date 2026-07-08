/// Pawn history search gates: OFF byte-identity (UsePawnHist=false must not move a node) and the
/// ON tree-changes smoke (mid-search teaching makes later reads nonzero, so the order MUST differ —
/// the policy-LMR dead-wiring lesson: prove the read actually fires).
module Eonego.Tests.PawnHistTests

open Xunit
open Eonego.Tests.TestFixtures

[<Fact>]
let ``UsePawnHist=false is byte-identical to the pre-feature tree`` () =
    match tryLoadNet () with
    | None -> ()
    | Some net ->
        let cfg = { Eonego.Search.defaultConfig with HashMb = 16 }
        let fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
        let struct (s0, n0, m0) = Eonego.Search.searchToDepthNet fen [||] 9 cfg (Some net)

        let struct (s1, n1, m1) =
            Eonego.Search.searchToDepthNet fen [||] 9 { cfg with UsePawnHist = false } (Some net)

        Assert.Equal(s0, s1)
        Assert.Equal(n0, n1)
        Assert.Equal(m0, m1)

[<Fact>]
let ``UsePawnHist=true changes the tree (read fires)`` () =
    match tryLoadNet () with
    | None -> ()
    | Some net ->
        let cfg = { Eonego.Search.defaultConfig with HashMb = 16 }
        let fen = "r1bq1rk1/pp2bppp/2n1pn2/3p4/3P4/2NBPN2/PP3PPP/R1BQ1RK1 w - - 0 9"
        let struct (_, nOff, _) = Eonego.Search.searchToDepthNet fen [||] 11 cfg (Some net)

        let struct (_, nOn, _) =
            Eonego.Search.searchToDepthNet fen [||] 11 { cfg with UsePawnHist = true } (Some net)

        Assert.NotEqual(nOff, nOn)

[<Fact>]
let ``UsePawnHist=true search is legal and deterministic`` () =
    match tryLoadNet () with
    | None -> ()
    | Some net ->
        let cfg =
            { Eonego.Search.defaultConfig with
                HashMb = 16
                UsePawnHist = true }

        let fen = "r1bq1rk1/pp2bppp/2n1pn2/3p4/3P4/2NBPN2/PP3PPP/R1BQ1RK1 w - - 0 9"
        let struct (_, n0, m0) = Eonego.Search.searchToDepthNet fen [||] 9 cfg (Some net)
        let struct (_, n1, m1) = Eonego.Search.searchToDepthNet fen [||] 9 cfg (Some net)
        Assert.Equal(n0, n1)
        Assert.Equal(m0, m1)
        Assert.NotEqual(Eonego.Move.MoveNone, m0)
        let pos = Eonego.Position.Position.OfFen fen
        let legal = collectLegal pos
        Assert.Contains(m0, legal)
