"""
    Eunoia

Julia bindings for [eunoia](https://github.com/jolars/eunoia), a Rust library
for area-proportional Euler and Venn diagrams.

The native code is the `eunoia-capi` cdylib, which speaks a small JSON-in/
JSON-out C ABI (`eunoia_euler`, `eunoia_venn`, `eunoia_version`, `eunoia_free`).
This module dlopen's it and exposes [`euler`](@ref), [`venn`](@ref), and
[`version`](@ref).

## Locating the library

At load time the shared library is resolved in this order:

1. `ENV["EUNOIA_CAPI_LIB"]` — an explicit path to a locally built
   `libeunoia_capi`. Set this during development:
   `cargo build -p eunoia-capi --release` then point at
   `target/release/libeunoia_capi.{so,dylib}`.
2. The bundled `eunoia` artifact declared in `Artifacts.toml` (downloaded
   lazily on first use; populated by `gen/generate_artifacts.jl` from a GitHub
   release).

If neither is available a descriptive error is raised.
"""
module Eunoia

import Artifacts
import Libdl
import Pkg
using JSON3
using Printf

export euler, venn, version
export EulerFit, VennFit, Circle, Ellipse, Square, Rectangle, Point, Container

include("parse.jl")
include("types.jl")

# Resolved symbol pointers, filled in `__init__`.
const _HANDLE = Ref{Ptr{Cvoid}}(C_NULL)
const _euler = Ref{Ptr{Cvoid}}(C_NULL)
const _venn = Ref{Ptr{Cvoid}}(C_NULL)
const _version = Ref{Ptr{Cvoid}}(C_NULL)
const _free = Ref{Ptr{Cvoid}}(C_NULL)

"""Find a `libeunoia_capi.<dlext>` (or `eunoia_capi.dll`) under `dir`."""
function _find_lib(dir::AbstractString)
    ext = "." * Libdl.dlext
    for (root, _, files) in walkdir(dir)
        for f in files
            if (startswith(f, "libeunoia_capi") || startswith(f, "eunoia_capi")) &&
               endswith(f, ext)
                return joinpath(root, f)
            end
        end
    end
    error("no eunoia_capi$ext found under $dir")
end

"""Resolve the shared-library path: env override first, then the artifact."""
function _locate_library()
    override = get(ENV, "EUNOIA_CAPI_LIB", "")
    isempty(override) || return override

    toml = Artifacts.find_artifacts_toml(@__DIR__)
    if toml !== nothing
        meta = Artifacts.artifact_meta("eunoia", toml)
        if meta !== nothing
            hash = Base.SHA1(meta["git-tree-sha1"])
            Artifacts.artifact_exists(hash) ||
                Pkg.Artifacts.ensure_artifact_installed("eunoia", toml)
            return _find_lib(Artifacts.artifact_path(hash))
        end
    end

    error("""
          Could not locate the eunoia native library.

          Either set ENV["EUNOIA_CAPI_LIB"] to a locally built
          libeunoia_capi (run `cargo build -p eunoia-capi --release`), or
          populate Artifacts.toml via gen/generate_artifacts.jl.
          """)
end

function __init__()
    path = _locate_library()
    _HANDLE[] = Libdl.dlopen(path)
    _euler[] = Libdl.dlsym(_HANDLE[], :eunoia_euler)
    _venn[] = Libdl.dlsym(_HANDLE[], :eunoia_venn)
    _version[] = Libdl.dlsym(_HANDLE[], :eunoia_version)
    _free[] = Libdl.dlsym(_HANDLE[], :eunoia_free)
    return nothing
end

"""Call a `(*const c_char) -> *mut c_char` symbol, freeing the result."""
function _invoke(fnptr::Ptr{Cvoid}, json::AbstractString)
    out = ccall(fnptr, Ptr{Cchar}, (Cstring,), json)
    out == C_NULL && error("eunoia: native call returned NULL")
    try
        return unsafe_string(out)
    finally
        ccall(_free[], Cvoid, (Ptr{Cchar},), out)
    end
end

"""Invoke `fnptr` with `payload` serialized to JSON, parse, and unwrap `ok`."""
function _run(fnptr::Ptr{Cvoid}, payload)
    resp = JSON3.read(_invoke(fnptr, JSON3.write(payload)))
    resp.ok || error("eunoia: " * String(resp.error))
    return resp
end

"""
    euler(values; shape="circle", input_type="exclusive",
          complement=nothing, seed=nothing)

Fit an area-proportional Euler diagram. `values` is one of:

- a mapping from combination strings to areas, where a key is a single set
  (`"A"`) or an `&`-joined intersection (`"A&B"`):

  ```julia
  euler(Dict("A" => 5, "B" => 3, "A&B" => 1.5))
  ```

- a mapping from set names to **membership collections** (vector/set/tuple);
  each element is counted into the canonical region of the sets it belongs to,
  yielding exclusive per-region counts (so `input_type` must stay
  `"exclusive"`):

  ```julia
  euler(Dict("A" => ["x", "y"], "B" => ["y", "z"]))
  ```

Keyword arguments:

- `shape`: `"circle"` (default), `"ellipse"`, `"square"`, or `"rectangle"`.
- `input_type`: `"exclusive"` (default; per-region areas) or `"inclusive"`
  (set sizes that include overlaps; the core converts internally and the
  reported `fitted_values`/`residuals` are reconstructed in the inclusive
  scale).
- `complement`: target "universe" area outside every set (opts into container
  fitting); `nothing` to disable.
- `seed`: RNG seed for reproducible restarts; `nothing` for default.

Returns an [`EulerFit`](@ref) carrying the fitted `shapes`, the
`original_values`/`fitted_values`/`residuals` per region, the scalar fit
metrics, and—if a `complement` was given—a `container`. Per-region
`region_error` is not yet surfaced (only the scalar `diag_error`).
"""
function euler(values::AbstractDict; shape::AbstractString="circle",
               input_type::AbstractString="exclusive",
               complement::Union{Nothing,Real}=nothing,
               seed::Union{Nothing,Integer}=nothing)
    if is_membership_input(values)
        input_type == "exclusive" || error(
            "invalid_input: membership-list input is always exclusive; " *
            "do not pass input_type=\"inclusive\"")
        original_values = parse_membership_input(values)
        canonical_keys = collect(keys(original_values))
        combos = [(c, s) for (c, s) in original_values]
    else
        original_values = Dict{String,Float64}()
        canonical_keys = String[]
        combos = Tuple{String,Float64}[]
        for (k, v) in values
            ck = canonicalize(string(k))
            original_values[ck] = float(v)
            push!(canonical_keys, ck)
            push!(combos, (string(k), float(v)))
        end
    end

    payload = Dict{String,Any}(
        "sets" => [Dict("combination" => c, "size" => s) for (c, s) in combos],
        "shape" => shape,
        "input_type" => input_type,
    )
    complement === nothing || (payload["complement"] = float(complement))
    seed === nothing || (payload["seed"] = UInt64(seed))

    resp = _run(_euler[], payload)
    return _finish_euler(resp, original_values, canonical_keys, input_type)
end

"""
    venn(sets; shape="circle")

Build a canonical Venn diagram. `sets` selects the set names, as one of:

- an `Integer` `n` — `n` sets with default names `"A"`, `"B"`, …;
- a vector of set names, e.g. `["cat", "dog", "fish"]`;
- a mapping whose keys are set/combination labels (the distinct base set names
  are extracted; values are ignored, since a Venn layout is non-proportional).

The number of names selects the arrangement; `shape` is `"circle"` (n ≤ 3),
`"ellipse"` (n ≤ 5), `"square"`, or `"rectangle"` (n ≤ 3).

```julia
venn(["A", "B", "C"]; shape="ellipse")
venn(3)
```

Returns a [`VennFit`](@ref): the same structure as [`EulerFit`](@ref), but the
layout is topological, so `fitted_values` holds each region's geometric area and
`original_values` is empty.
"""
function venn(sets; shape::AbstractString="circle")
    names = _resolve_names(sets)
    payload = Dict("names" => names, "shape" => shape)
    return _build_vennfit(_run(_venn[], payload))
end

"""
    version() -> String

Return the version of the underlying `eunoia-capi` native library.
"""
function version()
    out = ccall(_version[], Ptr{Cchar}, ())
    out == C_NULL && error("eunoia: version returned NULL")
    try
        return unsafe_string(out)
    finally
        ccall(_free[], Cvoid, (Ptr{Cchar},), out)
    end
end

end # module
