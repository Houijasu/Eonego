// Eonego — a UCI chess engine.
// Copyright (C) 2026 Houijasu
//
// This program is free software: you can redistribute it and/or modify it under
// the terms of the GNU Affero General Public License as published by the Free
// Software Foundation, either version 3 of the License, or (at your option) any
// later version.
//
// This program is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
// PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License along
// with this program. If not, see <https://www.gnu.org/licenses/>.

module Eonego.Program

open Eonego.Bitboard

[<EntryPoint>]
let main argv =
    init () |> ignore

    if argv.Length > 0 then
        match argv.[0] with
        | "gen" -> Eonego.Tooling.runGen argv.[1..]
        | "lstrace" -> Eonego.Tooling.runLsTrace argv.[1..]
        | "lsforward" -> Eonego.Tooling.runLsForward argv.[1..]
        | _ ->
            Eonego.UCI.run ()
            0
    else
        Eonego.UCI.run ()
        0
