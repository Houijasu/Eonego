/// UCI_ShowWDL tests: the wdlForScore display clamp (proven mate/TB scores override the policy
/// head's root estimate) and the SearchControl.RootWdl default (ValueNone = feature off).
module Eonego.Tests.WdlTests

open Xunit
open Eonego.Search

let private rootWdl = struct (450, 400, 150)

[<Fact>]
let ``a normal score passes the root WDL through`` () =
    Assert.Equal<struct (int * int * int)>(rootWdl, wdlForScore 25 rootWdl)
    Assert.Equal<struct (int * int * int)>(rootWdl, wdlForScore (-300) rootWdl)
    Assert.Equal<struct (int * int * int)>(rootWdl, wdlForScore 0 rootWdl)

[<Fact>]
let ``a mate score clamps to a certain win or loss`` () =
    Assert.Equal<struct (int * int * int)>(struct (1000, 0, 0), wdlForScore (MATE - 5) rootWdl)
    Assert.Equal<struct (int * int * int)>(struct (0, 0, 1000), wdlForScore (-(MATE - 5)) rootWdl)

[<Fact>]
let ``a TB-band score clamps to a certain win or loss`` () =
    Assert.Equal<struct (int * int * int)>(struct (1000, 0, 0), wdlForScore (TB_WIN - 10) rootWdl)
    Assert.Equal<struct (int * int * int)>(struct (0, 0, 1000), wdlForScore (-(TB_WIN - 10)) rootWdl)

[<Fact>]
let ``the clamp boundary sits at the bottom of the TB band`` () =
    let bottom = TB_WIN - MaxSearchPly
    Assert.Equal<struct (int * int * int)>(struct (1000, 0, 0), wdlForScore bottom rootWdl)
    Assert.Equal<struct (int * int * int)>(rootWdl, wdlForScore (bottom - 1) rootWdl)
    Assert.Equal<struct (int * int * int)>(struct (0, 0, 1000), wdlForScore (-bottom) rootWdl)
    Assert.Equal<struct (int * int * int)>(rootWdl, wdlForScore (-(bottom - 1)) rootWdl)
