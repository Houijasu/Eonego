module Eonego.Program

open Eonego.Bitboard

[<EntryPoint>]
let main _ =
    // Touch the Bitboard module so all attack/geometry tables initialise (the static-init AOT smoke test);
    // the result is discarded — all output goes through Console (never printfn) once the UCI loop starts.
    init () |> ignore
    Eonego.Uci.run ()
    0
