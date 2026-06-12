# Typed result model mirroring the `eunoia-py` sister package (`_models.py`),
# adapted to Julia idioms. The native `eunoia-capi` returns a JSON envelope; the
# `_build_*` helpers parse it into these structs so `euler`/`venn` hand back
# typed values instead of raw `JSON3.Object`s.

"""A 2D point."""
struct Point
    x::Float64
    y::Float64
end

"""Supertype of the four fittable shapes (`Circle`, `Ellipse`, `Square`,
`Rectangle`)."""
abstract type AbstractShape end

"""A fitted circle for one input set."""
struct Circle <: AbstractShape
    set::String
    center::Point
    radius::Float64
    label_anchor::Point
end

"""A fitted ellipse for one input set."""
struct Ellipse <: AbstractShape
    set::String
    center::Point
    semi_major::Float64
    semi_minor::Float64
    rotation::Float64
    label_anchor::Point
end

"""A fitted axis-aligned square for one input set."""
struct Square <: AbstractShape
    set::String
    center::Point
    side::Float64
    label_anchor::Point
end

"""A fitted axis-aligned rectangle for one input set."""
struct Rectangle <: AbstractShape
    set::String
    center::Point
    width::Float64
    height::Float64
    label_anchor::Point
end

"""The fitted universe box drawn around a diagram fit with a `complement`. The
container's area minus the union of the shapes matches the requested complement
area."""
struct Container
    center::Point
    width::Float64
    height::Float64
end

"""Supertype of [`EulerFit`](@ref) and [`VennFit`](@ref), parametrized by the
shape type `S`."""
abstract type AbstractEulerFit{S<:AbstractShape} end

"""
    EulerFit{S}

Result of fitting an area-proportional Euler diagram with shapes of type `S`.

Fields:

- `shapes`: fitted shapes, one per input set, in input order.
- `original_values`: the requested exclusive area per region (keyed by
  combination string, e.g. `"A&B"`).
- `fitted_values`: the fitted exclusive area per region.
- `residuals`: `original_values − fitted_values` per region.
- `diag_error`: eulerAPE-style worst-case region error.
- `stress`: venneuler-style stress metric.
- `loss`: final value of the optimizer's objective.
- `iterations`: optimizer iteration count.
- `container`: the fitted universe box when fit with a `complement`, else
  `nothing`.

Per-region `region_error` is not yet surfaced (it awaits a native-side
addition); only the scalar `diag_error` is available.
"""
struct EulerFit{S} <: AbstractEulerFit{S}
    shapes::Vector{S}
    original_values::Dict{String,Float64}
    fitted_values::Dict{String,Float64}
    residuals::Dict{String,Float64}
    diag_error::Float64
    stress::Float64
    loss::Float64
    iterations::Int
    container::Union{Container,Nothing}
end

"""
    VennFit{S}

Result of laying out a (non-proportional) Venn diagram. Shares
[`EulerFit`](@ref)'s structure, but the layout is *topological*: every set
intersection is drawn regardless of area, so the proportional-error metrics are
not meaningful (left at zero) and `original_values` is empty. `fitted_values`
holds the geometric area of each region.
"""
struct VennFit{S} <: AbstractEulerFit{S}
    shapes::Vector{S}
    original_values::Dict{String,Float64}
    fitted_values::Dict{String,Float64}
    residuals::Dict{String,Float64}
    diag_error::Float64
    stress::Float64
    loss::Float64
    iterations::Int
    container::Union{Container,Nothing}
end

# ---------------------------------------------------------------------------
# Building typed structs from the capi JSON envelope
# ---------------------------------------------------------------------------

_point(o) = Point(Float64(o.x), Float64(o.y))

"""Map one tagged shape object from the JSON `shapes` array to its struct."""
function _build_shape(s)
    t = String(s.type)
    center = Point(Float64(s.x), Float64(s.y))
    anchor = _point(s.label_anchor)
    if t == "circle"
        return Circle(String(s.label), center, Float64(s.radius), anchor)
    elseif t == "ellipse"
        return Ellipse(String(s.label), center, Float64(s.semi_major),
                       Float64(s.semi_minor), Float64(s.rotation), anchor)
    elseif t == "square"
        return Square(String(s.label), center, Float64(s.side), anchor)
    elseif t == "rectangle"
        return Rectangle(String(s.label), center, Float64(s.width),
                         Float64(s.height), anchor)
    else
        error("eunoia: unknown shape type '$t'")
    end
end

"""Convert a JSON area map (symbol keys) to a `Dict{String,Float64}`. Keys are
combination strings like `"A&B"`, recovered via `String`."""
_area_dict(obj) =
    Dict{String,Float64}(String(k) => Float64(v) for (k, v) in pairs(obj))

"""Read the optional container box; absent (not null) when no complement."""
function _build_container(resp)
    haskey(resp, :container) || return nothing
    c = resp.container
    return Container(Point(Float64(c.x), Float64(c.y)),
                     Float64(c.width), Float64(c.height))
end

"""Shared field assembly for both fit kinds: shapes (narrowed to a concrete
`Vector{S}`), the three area dicts, and the scalar metrics. Returns the shape
type `S` and a tuple of constructor arguments."""
function _fit_fields(resp)
    raw = [_build_shape(s) for s in resp.shapes]
    S = isempty(raw) ? AbstractShape : typeof(raw[1])
    shapes = S === AbstractShape ? raw : Vector{S}(raw)
    m = resp.metrics
    original = _area_dict(m.target_areas)
    fitted = _area_dict(m.fitted_areas)
    residuals = Dict{String,Float64}(
        k => original[k] - get(fitted, k, 0.0) for k in keys(original))
    args = (shapes, original, fitted, residuals,
            Float64(m.diag_error), Float64(m.stress), Float64(m.loss),
            Int(m.iterations), _build_container(resp))
    return S, args
end

function _build_eulerfit(resp)
    S, args = _fit_fields(resp)
    return EulerFit{S}(args...)
end

function _build_vennfit(resp)
    S, args = _fit_fields(resp)
    return VennFit{S}(args...)
end

# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

_shape_kind(::Type{S}) where {S} = lowercase(string(nameof(S)))

function Base.show(io::IO, ::MIME"text/plain", fit::EulerFit{S}) where {S}
    n = length(fit.shapes)
    kind = n == 0 ? "shapes" : _shape_kind(S) * "s"
    @printf(io, "EulerFit (%d %s, diag_error=%.4g, stress=%.4g, loss=%.4g)",
            n, kind, fit.diag_error, fit.stress, fit.loss)

    labels = sort!(collect(keys(fit.original_values)))
    isempty(labels) && return

    label_w = maximum(length, labels)
    col_w = 12
    # `region_error` column omitted: the native library does not yet emit a
    # per-region error map (only the scalar `diag_error`).
    print(io, "\n  ", " "^label_w)
    for c in ("original", "fitted", "residual")
        print(io, lpad(c, col_w))
    end
    for k in labels
        print(io, "\n  ", rpad(k, label_w))
        print(io, lpad(@sprintf("%.4g", fit.original_values[k]), col_w))
        print(io, lpad(@sprintf("%.4g", fit.fitted_values[k]), col_w))
        print(io, lpad(@sprintf("%.4g", fit.residuals[k]), col_w))
    end
end

function Base.show(io::IO, ::MIME"text/plain", fit::VennFit{S}) where {S}
    names = [s.set for s in fit.shapes]
    kind = isempty(fit.shapes) ? "shape" : _shape_kind(S)
    print(io, "VennFit (", length(names), " sets [", kind, "]: ",
          join(names, ", "), ")")
end
