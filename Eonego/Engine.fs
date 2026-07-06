/// Engine identity — the single source of truth for the name/version/author shown in the UCI `id`
/// lines, the startup banner, and (mirrored by hand in Eonego.fsproj <Version>) the Windows file
/// metadata of the published exe. Compile-time literals only: reading assembly attributes via
/// reflection is exactly the kind of NativeAOT hazard this codebase avoids (see printfn history).
///
/// Versioning scheme: MAJOR.MINOR.PATCH. Bump MINOR for measured strength releases (SPRT-promoted
/// features, tuning waves), PATCH for fixes/inert changes, MAJOR for architecture-level milestones
/// (e.g. an own-trained network). Keep the fsproj <Version> in sync when bumping.
module Eonego.Engine

[<Literal>]
let Name = "Eonego"

[<Literal>]
let Version = "0.0.7"

[<Literal>]
let Author = "Houijasu"
