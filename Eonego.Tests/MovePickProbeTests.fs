/// Step-0 compile-probe verification: the [<Struct; IsByRefLike>] MovePick layout works and mutation
/// through `byref<MovePick>` in a MODULE function PERSISTS in the caller's slot (an instance method on a
/// by-ref-like struct would mutate a copy and Stage would never advance). Expanded into the full
/// MovePick suite once the stage machine lands.
module Eonego.Tests.MovePickProbeTests

#nowarn "9" // NativePtr.stackalloc

open System
open Microsoft.FSharp.NativeInterop
open Xunit
open Eonego.Move
open Eonego.Position
open Eonego.History
open Eonego.MovePick

[<Fact>]
let ``byref MovePick mutation persists across module-function calls`` () =
    let pos = Position.OfFen StartPosFen
    let tables = Tables()
    let pm = NativePtr.stackalloc<Move> 8
    let moves = Span<Move>(NativePtr.toVoidPtr pm, 8)
    let ps = NativePtr.stackalloc<int> 8
    let scores = Span<int>(NativePtr.toVoidPtr ps, 8)

    for i in 0..7 do
        moves.[i] <- i + 1

    let mutable mp = mkProbe pos tables moves scores
    let m1 = probeStep &mp
    let m2 = probeStep &mp
    Assert.Equal(2, mp.Stage) // PROVES byref mutation persists (value-copy would leave Stage=0)
    Assert.Equal(2, mp.Cur)
    Assert.Equal(moves.[1], m1)
    Assert.Equal(moves.[2], m2)
