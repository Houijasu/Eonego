/// Guardrail for the additive Position.PliesFromNull accessor (bounds the null-safe repetition walk):
/// it increments on normal moves and resets to 0 across a null move, so the search's repetition scan
/// never crosses a null boundary.
module Eonego.Tests.PliesFromNullTests

open Xunit
open Eonego.Position
open Eonego.Tests.TestFixtures

[<Fact>]
let ``PliesFromNull increments on a normal move and resets across a null move`` () =
    let p = Position.OfFen StartPosFen
    p.Make (collectLegal p).[0]
    let afterMove = p.PliesFromNull
    Assert.True(afterMove >= 1) // a real move advances the counter

    p.MakeNull()
    Assert.Equal(0, p.PliesFromNull) // null resets it -> walk window is empty right after a null

    p.Make (collectLegal p).[0]
    Assert.Equal(1, p.PliesFromNull) // and it restarts from the null boundary
