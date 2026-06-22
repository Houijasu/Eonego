module Eonego.Program

open Eonego.Bitboard

[<EntryPoint>]
let main argv =
    init () |> ignore

    if argv.Length > 0 then
        match argv.[0] with
        | "gen" -> Eonego.Tooling.runGen argv.[1..]
        | "featuredump" -> Eonego.Tooling.runFeatureDump argv.[1..]
        | "nnforward" -> Eonego.Tooling.runNnForward argv.[1..]
        | _ ->
            Eonego.Uci.run ()
            0
    else
        Eonego.Uci.run ()
        0
