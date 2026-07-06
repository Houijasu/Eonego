// Eonego — a UCI chess engine.
// Copyright (C) 2026 Houijasu
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

module Eonego.Program

open Eonego.Bitboard

[<EntryPoint>]
let main argv =
    init () |> ignore

    if argv.Length > 0 then
        match argv.[0] with
        | "gen" -> Eonego.Tooling.runGen argv.[1..]
        | "dumpft" -> Eonego.Tooling.runDumpFt argv.[1..]
        | "dumppolicy" -> Eonego.Tooling.runDumpPolicy argv.[1..]
        | "tbgen" -> Eonego.Tooling.runTbGen argv.[1..]
        | "tbprobe" -> Eonego.Tooling.runTbProbe argv.[1..]
        | "retro" -> Eonego.Tooling.runRetro argv.[1..]
        | _ ->
            Eonego.UCI.run ()
            0
    else
        Eonego.UCI.run ()
        0
