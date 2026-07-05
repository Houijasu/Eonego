# NNUE network (`main.nnue`)

The trained FullThreats net (~106 MB) is **not committed to git** (see root `.gitignore`).

## Before building a playable engine

```powershell
pwsh ../scripts/fetch-net.ps1
```

Then publish or build — `Eonego.fsproj` embeds `nets/main.nnue` as manifest resource `eval.nnue` when the file exists.

## Without the file

- `dotnet build` / `dotnet publish` **still succeed** (conditional `<EmbeddedResource>`).
- The exe is ~4 MB and prints `info string no NNUE net embedded; cannot search` on `go`.
- Tests that need eval **soft-skip** when the file is absent.

## Alternatives

- **Pre-built binary:** [GitHub Releases](https://github.com/Houijasu/Eonego/releases) zip already includes the embedded net.
- **Runtime override:** `EONEGO_NET=C:\path\to\net.nnue` (same architecture, version `0x6A448AFA`).
