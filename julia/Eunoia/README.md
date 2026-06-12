# Eunoia.jl

Julia bindings for [eunoia](https://github.com/jolars/eunoia), a Rust library
for area-proportional **Euler and Venn diagrams**.

The package links the `eunoia-capi` cdylib through a tiny JSON-in/JSON-out C
ABI. Native binaries are shipped as a Julia **artifact** (Option A: roll your
own JLL — no BinaryBuilder, no Yggdrasil); they are cross-compiled for all
platforms by `.github/workflows/julia-artifacts.yml` and attached to a
`julia-v*` GitHub release.

## Usage

```julia
using Eunoia

# Euler diagram from exclusive areas
layout = euler(Dict("A" => 5, "B" => 3, "A&B" => 1.5))
layout.shapes        # Vector of {type, label, x, y, radius, label_anchor}
layout.metrics.loss

# Ellipses, with a "universe" complement (adds a container frame)
euler(Dict("A" => 4, "B" => 2, "A&B" => 1); shape="ellipse", complement=3)

# Canonical 3-set Venn with ellipses
venn(["A", "B", "C"]; shape="ellipse")
```

`shape` is `"circle"`, `"ellipse"`, `"square"`, or `"rectangle"`. `euler` also
takes `input_type` (`"exclusive"`/`"inclusive"`) and `seed`.

## Plotting

Rendering lives in a **Makie package extension** that loads automatically once a
Makie backend is available, so the core package stays plot-free:

```julia
using Eunoia, CairoMakie   # or GLMakie / WGLMakie

fit = euler(Dict("A" => 5, "B" => 3, "A&B" => 1.5))
eunoiaplot(fit)                              # publication-ready figure
eunoiaplot(fit; quantities=true, legend=true)
```

`eunoiaplot(fit)` returns a `Makie.FigureAxisPlot` (equal aspect, no axis
decorations); `eunoiaplot!(ax, fit)` draws into an existing axis, and the bare
Makie `plot(fit)`/`plot!(ax, fit)` recipe forms work too. Styling keywords mirror
the `eunoia-py` `plot()` API:

- `colors` — a vector (set order) or `Dict(name => color)`; region fills blend
  the member colors perceptually (OKLab).
- `fills`, `edges` — per-region / per-set style overrides.
- `labels` — `false`/`true`, a per-set `Dict`, or a uniform style.
- `quantities` — `false`/`true`, `"original"`/`"fitted"`, `"counts"`/`"percent"`,
  or a `Dict`.
- `legend` — `false`/`true` or a `Dict` of `Legend` keywords.
- `complement` — container-box style (drawn only for complement fits).

## Local development

No artifact yet? Build the cdylib and point the package at it:

```sh
cargo build -p eunoia-capi --release
export EUNOIA_CAPI_LIB="$PWD/target/release/libeunoia_capi.so"  # .dylib on macOS
julia --project=julia/Eunoia -e 'using Eunoia; println(version())'
```

`EUNOIA_CAPI_LIB` always wins over the bundled artifact, so it's also the way
to test a local Rust change against the Julia surface.

## Releasing binaries

1. Push a `julia-v<version>` tag → the `julia-artifacts` workflow builds and
   attaches `libeunoia_capi-<triplet>.tar.gz` for each platform.
2. Regenerate the artifact hashes from those assets:

   ```sh
   julia -e 'import Pkg; Pkg.add("ArtifactUtils")'
   julia --project=julia/Eunoia julia/Eunoia/gen/generate_artifacts.jl julia-v<version>
   ```

3. Commit the updated `Artifacts.toml`.

End users then `add Eunoia` and the right binary is fetched lazily on first
use — no Rust toolchain required.
