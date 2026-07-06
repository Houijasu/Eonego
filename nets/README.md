# Trained weights (`main.nnue`, `main.policy`)

Canonical engine weights for this repo:

| File | Role | Size (approx.) |
|---|---|---|
| `main.nnue` | FullThreats NNUE eval trunk | ~106 MB (Git LFS) |
| `main.policy` | EONPOL02 policy + WDL sidecar (`ftHash`-bound to `main.nnue`) | ~450 KB |

Both are **tracked in git**. `main.nnue` is stored via **Git LFS** — clone with LFS enabled:

```powershell
git lfs install   # once per machine
git clone <repo>  # or: git lfs pull  on an existing clone
```

## Build embed

`Eonego.fsproj` embeds the files when present:

- `main.nnue` → manifest resource `eval.nnue`
- `main.policy` → manifest resource `policy.dat` (loaded only when `EONEGO_POLICY=1` or `=<path>`)

```powershell
pwsh ../publish.ps1
# -> Eonego/bin/Release/net10.0/win-x64/publish/Eonego.exe
```

## Without the files

- `dotnet build` / `dotnet publish` **still succeed** (conditional `<EmbeddedResource>`).
- Missing `main.nnue`: exe prints `info string no NNUE net embedded; cannot search` on `go`.
- Missing `main.policy`: policy stays off (default); search is unchanged.
- Tests that need eval **soft-skip** when `main.nnue` is absent.

## Runtime overrides (no rebuild)

- `EONEGO_NET=<path>` — load a compatible `.nnue` trunk from disk
- `EONEGO_POLICY=<path>` — load a `.policy` sidecar (`=1` reads the embedded `policy.dat`)

## Fallback download

If LFS was not pulled and `main.nnue` is missing:

```powershell
pwsh ../scripts/fetch-net.ps1
```

Release zips on [GitHub Releases](https://github.com/Houijasu/Eonego/releases) also ship a pre-embedded binary.
