/// Eonego — df-pn (depth-first proof-number) mate oracle over the CHECKS-ONLY tree.
///
/// Solves the binary question "can the side to move force checkmate where EVERY attacker move is a
/// check?" (OR nodes = attacker, checking moves only; AND nodes = defender, all legal evasions —
/// in check by construction). Proof numbers focus expansion on the narrowest defence instead of
/// depth, which proves deep forced mates orders of magnitude faster than alpha-beta. The scope is a
/// real restriction: mates that need a quiet key move are invisible here (the main search still
/// finds those); every proof this module DOES publish is a certified real mate.
///
/// Soundness architecture: df-pn itself uses a path-blind transposition table, so graph-history
/// interaction (repetition/rule-50 facts leaking across paths) and 32-bit key collisions can fake a
/// proof. The `Verify` pass therefore replays the claim with a strict full-window proof search on
/// the same checks-only tree — NO table, path repetition = fail, rule-50 = fail — which is immune
/// to both by construction and also yields the exact mate distance and PV. Only verified proofs
/// ever leave `Solve`. Disproofs are "no checks-only mate within resources", never published as
/// game-theoretic facts.
///
/// Runs on its own thread beside the LazySMP search (driver in Search.fs), shares NOTHING but a
/// polled stop callback and the `OracleResult` slot. Own net-free Position (EnableNNUE never
/// called => Make/Unmake are pure board ops — the Retrograde.fs pattern). All buffers are
/// preallocated heap arrays: a Span over stackalloc held across a long loop mis-executes under
/// tiered-JIT OSR (see Retrograde.fs). No Printf anywhere (NativeAOT).
module Eonego.DFPN

open System
open System.Diagnostics
open System.Threading
open Eonego.Move
open Eonego.Position
open Eonego.MoveGeneration

/// Proof/disproof infinity. Saturating adds cap here — far below uint32.MaxValue, so a capped sum
/// can never wrap however many times it is re-summed. A SATURATED (not proven) pn/dn reaching this
/// value reads as dis/proof and costs completeness only, never soundness (proofs are re-verified).
[<Literal>]
let DFPNInf = 1073741824u // 1u <<< 30

/// Solver ply cap, well under Search.MaxSearchPly (246): every published mate distance fits the
/// search's mate band (MATE - plies >= MATE_IN_MAX_PLY) by construction. 128 plies of consecutive
/// checks is far beyond any practical mate.
[<Literal>]
let MaxDFPNPly = 128

/// Refuse roots with longer game histories: replay + MaxDFPNPly must stay inside Position.MaxPly
/// (1024) with margin (same crash class as the >255-ply accumulator overflow, b66568e — the
/// solver's Position is net-free so only the state stack matters, but keep the retro trigger's
/// conservatism).
[<Literal>]
let MaxRootHistory = 768

let inline private satAdd (a: uint32) (b: uint32) : uint32 =
    let s = a + b
    if s >= DFPNInf then DFPNInf else s

/// The 1+eps threshold trick (Pawlewicz-Lew), eps = 1/4, integer-only: descending with a threshold
/// just above the second-best sibling by a MULTIPLICATIVE margin prevents the O(n^2) re-expansion
/// thrash of +1 thresholds on wide frontiers.
let inline private epsUp (x: uint32) : uint32 =
    if x >= DFPNInf then DFPNInf else min DFPNInf (x + (x >>> 2) + 1u)

/// Twofold repetition against the current line AND the replayed game history — the exact scan of
/// Search.isRepetition (Position.IsRepetition is a search-only stub; the null-move bound is inert
/// here, the solver never makes null moves, but keeping it makes the copies diff-identical).
let private isRepetition (pos: Position) : bool =
    let endN = min pos.Rule50 pos.PliesFromNull

    if endN < 4 then
        false
    else
        let cur = pos.Key
        let mutable i = 4
        let mutable found = false

        while not found && i <= endN do
            if pos.KeyAt i = cur then
                found <- true

            i <- i + 2

        found

/// df-pn table entry: 16 bytes, single-slot always-replace, generation-scoped (a gen mismatch is an
/// empty slot — per-solve gen bumps clear the table without a memset).
[<Struct>]
type internal Entry =
    { Key32: uint32
      PN: uint32
      DN: uint32
      Gen: uint32 }

/// Outcome of one Solve. Cold path — a heap record is fine. `Proven` is set ONLY after the
/// verification pass certified the mate; `Disproved` means df-pn completed with dn = 0 ("no
/// checks-only mate exists", exact modulo the path-blind-table caveats — informational only).
type SolveResult =
    { Proven: bool
      Disproved: bool
      Move: Move
      MatePlies: int
      PV: Move[] }

let internal noResult =
    { Proven = false
      Disproved = false
      Move = MoveNone
      MatePlies = 0
      PV = Array.empty }

/// Lock-free single-writer publication slot for one search (`SearchControl` owns one per `go`).
/// The oracle thread writes the payload fields, then release-publishes via the Volatile flag; the
/// driver's end-of-search read happens after a Thread.Join anyway, but the Volatile pair keeps a
/// future mid-search reader correct.
[<Sealed>]
type OracleResult() =
    let mutable state = 0
    let mutable mv: Move = MoveNone
    let mutable matePlies = 0
    let mutable pv: Move[] = Array.empty

    member _.Publish(m: Move, plies: int, line: Move[]) =
        mv <- m
        matePlies <- plies
        pv <- line
        Volatile.Write(&state, 1)

    member _.TryGet() : struct (bool * Move * int) =
        if Volatile.Read(&state) = 1 then
            struct (true, mv, matePlies)
        else
            struct (false, MoveNone, 0)

    /// Read only after TryGet returned true.
    member _.PV: Move[] = pv

/// One df-pn solver: table + net-free Position + per-ply buffers. NOT thread-safe — the engine
/// serializes it (one oracle thread per `go`, joined before the next); tests construct private
/// instances so xUnit parallelism never touches the engine singleton.
[<Sealed>]
type Solver(tableMb: int) =
    // Largest power of two with entryCount * 16 bytes <= tableMb MiB.
    let entryCount =
        let target = max 1 tableMb * 65536
        let mutable c = 1

        while c * 2 <= target do
            c <- c * 2

        c

    let mask = uint64 (entryCount - 1)
    let mutable table: Entry[] = Array.empty // lazy: allocated on first solve (empty = the Retrograde unsolved-slot idiom, no nullness)
    let mutable gen = 0u

    // Net-free Position: EnableNNUE is never called, so Make/Unmake never touch accumulator state.
    let pos = Position()

    // Per-ply heap buffers (DFS: ply p's arrays are never touched while descendants run).
    // The verifier reuses plyMoves and ordKey (it runs strictly after the df-pn pass).
    let plyMoves: Move[][] = Array.init MaxDFPNPly (fun _ -> Array.zeroCreate MaxMoves)
    let childPN: uint32[][] = Array.init MaxDFPNPly (fun _ -> Array.zeroCreate MaxMoves)
    let childDN: uint32[][] = Array.init MaxDFPNPly (fun _ -> Array.zeroCreate MaxMoves)

    // Verifier triangular PV, Search.fs updatePV shape: row `ply` at ply*MaxDFPNPly, MoveNone-terminated.
    let vPV: Move[] = Array.zeroCreate (MaxDFPNPly * MaxDFPNPly)

    let mutable nodes = 0L
    let mutable vNodes = 0L
    let mutable nodeCap = 0L
    let mutable vNodeCap = 0L
    let mutable aborted = false
    let mutable stopCb: unit -> bool = fun () -> false

    /// df-pn expansions this solve (determinism gate: same position, same Solver => same count).
    member _.Nodes = nodes

    member _.VerifyNodes = vNodes

    member private _.Probe(key: uint64) : struct (bool * uint32 * uint32) =
        let e = table.[int (key &&& mask)]

        if e.Gen = gen && e.Key32 = uint32 (key >>> 32) then
            struct (true, e.PN, e.DN)
        else
            struct (false, 0u, 0u)

    member private _.Store (key: uint64) (pn: uint32) (dn: uint32) : unit =
        table.[int (key &&& mask)] <-
            { Key32 = uint32 (key >>> 32)
              PN = pn
              DN = dn
              Gen = gen }

    /// Compact buf[0..n0) to its checking moves, order-preserving. OR nodes only.
    member private _.KeepChecks (buf: Move[]) (n0: int) : int =
        let mutable n = 0

        for i = 0 to n0 - 1 do
            if pos.GivesCheck buf.[i] then
                buf.[n] <- buf.[i]
                n <- n + 1

        n

    // -----------------------------------------------------------------------------------------
    // MID (Nagai 2002). pn/dn are the ATTACKER's proof/disproof numbers at every node; the node
    // type is ply parity (even = OR/attacker, odd = AND/defender), so OR folds (min pn, satsum dn)
    // and AND folds the dual — no sign flipping. Returns the node's (pn, dn) once one crosses its
    // threshold. Terminal facts that are PATH-dependent (repetition, rule-50, ply cap) are
    // returned but never stored: the table must only hold position-intrinsic values (still not
    // GHI-proof — descendants' stored values absorb path facts — hence the verifier).
    // -----------------------------------------------------------------------------------------
    member private this.MID (ply: int) (thpn: uint32) (thdn: uint32) : struct (uint32 * uint32) =
        nodes <- nodes + 1L

        // Node cap every entry (one compare — exact, testable); the stop callback only every 2048
        // (a Volatile read through a closure is not free at millions of nodes/s).
        if not aborted then
            if nodeCap > 0L && nodes >= nodeCap then
                aborted <- true
            elif nodes &&& 2047L = 0L && stopCb () then
                aborted <- true

        if aborted then
            struct (1u, 1u) // placeholder; every caller re-checks `aborted` before trusting/storing
        elif isRepetition pos then
            struct (DFPNInf, 0u) // draw claim on this PATH => disproof here; not stored
        else
            let key = pos.Key
            // Rule-50 gate on table hits: a stored proof may need more reversible plies than this
            // path has left. Falling through to a fresh expansion is the conservative arm; the
            // verifier is the real budget check.
            let struct (hit, hpn, hdn) = this.Probe key

            if hit && (hpn >= thpn || hdn >= thdn) && pos.Rule50 < 100 then
                struct (hpn, hdn)
            else
                let isOr = (ply &&& 1) = 0
                let buf = plyMoves.[ply]
                let n0 = generateLegal pos (Span<Move>(buf))
                let n = if isOr then this.KeepChecks buf n0 else n0

                if n = 0 then
                    if isOr then
                        // No checking move (incl. attacker mated/stalemated): disproof, intrinsic.
                        this.Store key DFPNInf 0u
                        struct (DFPNInf, 0u)
                    else
                        // Defender has no move and is in check by construction: MATE. Intrinsic —
                        // and valid even at rule50 >= 100 (mate on the 100th-ply move wins; the
                        // rule-50 arm below deliberately runs AFTER this one).
                        Debug.Assert(pos.InCheck)
                        this.Store key 0u DFPNInf
                        struct (0u, DFPNInf)
                elif pos.Rule50 >= 100 then
                    struct (DFPNInf, 0u) // draw claim => disproof; path-dependent, not stored
                elif ply >= MaxDFPNPly - 1 then
                    struct (DFPNInf, 0u) // depth cap: disproof-as-unknown; not stored
                else
                    // Child init from the table (fresh transposition values), else (1, 1).
                    let cPN = childPN.[ply]
                    let cDN = childDN.[ply]

                    for i = 0 to n - 1 do
                        pos.Make buf.[i]
                        let struct (h, p, d) = this.Probe pos.Key
                        pos.Unmake buf.[i]

                        if h then
                            cPN.[i] <- p
                            cDN.[i] <- d
                        else
                            cPN.[i] <- 1u
                            cDN.[i] <- 1u

                    let mutable resPN = 0u
                    let mutable resDN = 0u
                    let mutable looping = true

                    while looping do
                        // Fold the children. OR: pn = min, dn = satsum. AND: dual.
                        let mutable mn = DFPNInf
                        let mutable sum = 0u

                        if isOr then
                            for i = 0 to n - 1 do
                                if cPN.[i] < mn then
                                    mn <- cPN.[i]

                                sum <- satAdd sum cDN.[i]
                        else
                            for i = 0 to n - 1 do
                                if cDN.[i] < mn then
                                    mn <- cDN.[i]

                                sum <- satAdd sum cPN.[i]

                        let pnN = if isOr then mn else sum
                        let dnN = if isOr then sum else mn

                        if aborted || pnN >= thpn || dnN >= thdn then
                            if not aborted then
                                this.Store key pnN dnN

                            resPN <- pnN
                            resDN <- dnN
                            looping <- false
                        else
                            // Most-proving child: OR minimizes child pn, AND minimizes child dn.
                            // Lowest index wins ties — determinism.
                            let mutable c1 = 0
                            let mutable best = DFPNInf + 1u
                            let mutable second = DFPNInf + 1u
                            let sel = if isOr then cPN else cDN

                            for i = 0 to n - 1 do
                                let v = sel.[i]

                                if v < best then
                                    second <- best
                                    best <- v
                                    c1 <- i
                                elif v < second then
                                    second <- v

                            let second = min second DFPNInf

                            // Child thresholds. The subtraction arm never underflows: the fold did
                            // not cross the threshold, so th > sum >= sum - sibling part.
                            let struct (thpnC, thdnC) =
                                if isOr then
                                    struct (
                                        min thpn (epsUp second),
                                        (if thdn >= DFPNInf then DFPNInf else thdn - dnN + cDN.[c1])
                                    )
                                else
                                    struct (
                                        (if thpn >= DFPNInf then DFPNInf else thpn - pnN + cPN.[c1]),
                                        min thdn (epsUp second)
                                    )

                            pos.Make buf.[c1]
                            let struct (p, d) = this.MID (ply + 1) thpnC thdnC
                            pos.Unmake buf.[c1]
                            cPN.[c1] <- p
                            cDN.[c1] <- d

                    struct (resPN, resDN)

    // -----------------------------------------------------------------------------------------
    // Verification pass — the soundness boundary. Strict depth-limited proof search on the same
    // checks-only tree with NO table: path repetition = fail, rule-50 = fail (after the mate arm),
    // budget exhausted = fail closed. Iterative deepening d = 1, 3, 5, ... makes the first
    // succeeding d the EXACT checks-only mate distance. The df-pn table is consulted for child
    // ORDERING only (ascending pn) — a hint can be wrong without harming soundness.
    // -----------------------------------------------------------------------------------------

    /// Length of the MoveNone-terminated PV row for `ply`.
    member private _.LineLen(ply: int) : int =
        let basep = ply * MaxDFPNPly
        let mutable i = 0

        while i < MaxDFPNPly && vPV.[basep + i] <> MoveNone do
            i <- i + 1

        i

    /// vPV row `ply` := m + row (ply+1). Same splice as Search.updatePV.
    member private this.SetPVLine (ply: int) (m: Move) : unit =
        let basep = ply * MaxDFPNPly
        vPV.[basep] <- m

        if ply + 1 < MaxDFPNPly then
            let baseC = (ply + 1) * MaxDFPNPly
            let mutable i = 0
            let mutable cont = true

            while cont && i < MaxDFPNPly - 1 do
                let c = vPV.[baseC + i]
                vPV.[basep + 1 + i] <- c

                if c = MoveNone then cont <- false else i <- i + 1

            if cont then
                vPV.[basep + MaxDFPNPly - 1] <- MoveNone
        else
            vPV.[basep + 1] <- MoveNone

    member private this.VerifyNode (ply: int) (depthLeft: int) : bool =
        vNodes <- vNodes + 1L

        if not aborted then
            if vNodes >= vNodeCap then
                aborted <- true
            elif vNodes &&& 2047L = 0L && stopCb () then
                aborted <- true

        if aborted then
            false
        elif isRepetition pos then
            false
        else
            let buf = plyMoves.[ply]
            let n0 = generateLegal pos (Span<Move>(buf))

            if (ply &&& 1) = 0 then
                // OR: some checking move proves within the budget.
                let n = this.KeepChecks buf n0

                if n = 0 || pos.Rule50 >= 100 || depthLeft <= 0 then
                    false
                else
                    // Order ascending df-pn pn (ordKey = childPN row; the df-pn pass is over).
                    let ord = childPN.[ply]

                    for i = 0 to n - 1 do
                        pos.Make buf.[i]
                        let struct (h, p, _) = this.Probe pos.Key
                        pos.Unmake buf.[i]
                        ord.[i] <- if h then p else 1u

                    // Insertion sort, stable (equal keys keep generation order — determinism).
                    for i = 1 to n - 1 do
                        let km = buf.[i]
                        let kv = ord.[i]
                        let mutable j = i - 1

                        while j >= 0 && ord.[j] > kv do
                            buf.[j + 1] <- buf.[j]
                            ord.[j + 1] <- ord.[j]
                            j <- j - 1

                        buf.[j + 1] <- km
                        ord.[j + 1] <- kv

                    let mutable ok = false
                    let mutable i = 0

                    while not ok && i < n do
                        let m = buf.[i]
                        pos.Make m
                        let r = this.VerifyNode (ply + 1) (depthLeft - 1)
                        pos.Unmake m

                        if r then
                            this.SetPVLine ply m
                            ok <- true

                        i <- i + 1

                    ok
            else
                // AND: mate now, or EVERY evasion fails within the budget.
                if n0 = 0 then
                    Debug.Assert(pos.InCheck) // defender is in check by construction
                    vPV.[ply * MaxDFPNPly] <- MoveNone // empty continuation
                    true
                elif pos.Rule50 >= 100 || depthLeft <= 0 then
                    false
                else
                    let mutable allOk = true
                    let mutable bestLen = -1
                    let mutable i = 0

                    while allOk && i < n0 do
                        let m = buf.[i]
                        pos.Make m
                        let r = this.VerifyNode (ply + 1) (depthLeft - 1)
                        pos.Unmake m

                        if not r then
                            allOk <- false
                        else
                            // PV = the LONGEST resistance, so MatePlies = |PV| holds. Copy now:
                            // the next sibling overwrites row ply+1, row ply is already safe.
                            let len = 1 + this.LineLen(ply + 1)

                            if len > bestLen then
                                this.SetPVLine ply m
                                bestLen <- len

                        i <- i + 1

                    allOk

    /// Iterative-deepening driver: first succeeding odd depth = exact checks-only mate distance.
    /// Returns (matePlies, pv) or (-1, empty). Checks-only mates always end on an attacker move,
    /// so only odd depths can succeed.
    member private this.Verify() : struct (int * Move[]) =
        let mutable d = 1
        let mutable res = struct (-1, (Array.empty: Move[]))
        let mutable go = true

        while go && d < MaxDFPNPly && not aborted do
            if this.VerifyNode 0 d then
                let len = this.LineLen 0
                Debug.Assert((len = d)) // longest resistance at the minimal depth is exactly d
                res <- struct (len, Array.sub vPV 0 len)
                go <- false
            else
                d <- d + 2

        res

    /// df-pn only, no verification — test/diagnostic entry. Returns the root (pn, dn); (0, INF) is
    /// a proof CANDIDATE, (INF, 0) a disproof, anything else an aborted/unknown run.
    member internal this.RawSolve
        (rootFen: string, rootMoves: Move[], stop: unit -> bool, nodeCapIn: int64)
        : struct (uint32 * uint32) =
        if rootMoves.Length > MaxRootHistory then
            struct (1u, 1u)
        else
            if table.Length = 0 then
                table <- Array.zeroCreate entryCount

            gen <- gen + 1u // gen 0 = the zeroed array's "empty"; first solve runs at gen 1
            nodes <- 0L
            vNodes <- 0L
            aborted <- false
            stopCb <- stop
            nodeCap <- nodeCapIn

            pos.LoadFen rootFen

            for m in rootMoves do
                pos.Make m

            this.MID 0 DFPNInf DFPNInf

    /// Full pipeline: df-pn, then (on a proof candidate) the verification pass. Only a verified
    /// mate returns Proven = true.
    member this.SolveWith
        (rootFen: string, rootMoves: Move[], stop: unit -> bool, nodeCapIn: int64, vNodeCapIn: int64)
        : SolveResult =
        let struct (pn, dn) = this.RawSolve(rootFen, rootMoves, stop, nodeCapIn)
        vNodeCap <- vNodeCapIn

        if aborted then
            noResult
        elif pn = 0u then
            match this.Verify() with
            | struct (d, pv) when d > 0 && d < MaxDFPNPly ->
                { Proven = true
                  Disproved = false
                  Move = pv.[0]
                  MatePlies = d
                  PV = pv }
            | _ -> noResult // verification failed (budget / GHI residual / collision): publish nothing
        elif dn = 0u then
            { noResult with Disproved = true }
        else
            noResult

    member this.Solve(rootFen: string, rootMoves: Move[], stop: unit -> bool) : SolveResult =
        this.SolveWith(
            rootFen,
            rootMoves,
            stop,
            int64 Tunables.DFPNNodes,
            int64 Tunables.DFPNVNodes
        )

/// Engine-wide singleton (lazy: gate-off engines never allocate the table). The UCI driver
/// serializes every `go` (stopAndJoin) and goCore joins the oracle thread, so exactly one thread
/// touches this at a time. Tests use private Solver instances instead.
let private sharedLazy = lazy (Solver(Tunables.DFPNMb))
let shared () : Solver = sharedLazy.Force()
