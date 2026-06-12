"""
Makie rendering for [`Eunoia`](@ref). This package extension activates when a
Makie backend is loaded (`using CairoMakie`/`GLMakie`/`WGLMakie`). It implements
[`Eunoia.eunoiaplot`](@ref)/[`Eunoia.eunoiaplot!`](@ref) plus an `EunoiaDiagram`
recipe, so `plot(fit)`/`plot!(ax, fit)` also work.

The styling API mirrors the `eunoia-py` sister package's `plot()` (colors, fills,
edges, labels, quantities, legend, complement), adapted to Makie idioms (Makie
attribute names, perceptual OKLab color blending for region fills). All geometry
is read from the native `plot_data` bundle already present on every fit.
"""
module EunoiaMakieExt

using Eunoia
using Makie
using Printf: @sprintf
using GeometryBasics: Polygon, Point2f
import Eunoia: AbstractEulerFit, eunoiaplot, eunoiaplot!

# ---------------------------------------------------------------------------
# Recipe — draws the diagram primitives into an axis. This is what powers the
# bare `plot(fit)`/`plot!(ax, fit)` forms; the publication-ready figure (equal
# aspect, hidden decorations, legend) is assembled by `eunoiaplot` below.
# ---------------------------------------------------------------------------

@recipe EunoiaDiagram (fit,) begin
    "Per-set colors: a vector (shape order), a `Dict(name => color)`, or
    `automatic` for a built-in categorical palette."
    colors = Makie.automatic
    "Per-region fill overrides: `Dict(combo => attrs)` of `poly!` keywords."
    fills = Makie.automatic
    "Set-outline style: a uniform attrs collection, a per-set `Dict`, or a
    vector (one per set, shape order) of `lines!` keywords."
    edges = Makie.automatic
    "Set labels: `false`/`true`/`nothing`, a per-set `Dict`, or a uniform style."
    labels = Makie.automatic
    "Region quantities: `false`/`true`, `\"original\"`/`\"fitted\"`,
    `\"counts\"`/`\"percent\"`, or a `Dict`."
    quantities = false
    "Legend flag — only used here to default in-diagram labels off; the Legend
    block itself is drawn by `eunoiaplot`."
    legend = false
    "Container-box style: `poly!` keywords (drawn only when the fit has one)."
    complement = Makie.automatic
    "Default region-fill transparency."
    alpha = 0.5
end

Makie.plottype(::AbstractEulerFit) = EunoiaDiagram
Makie.convert_arguments(::Type{<:EunoiaDiagram}, fit::AbstractEulerFit) = (fit,)
Makie.preferred_axis_type(::EunoiaDiagram) = Makie.Axis

function Makie.plot!(p::EunoiaDiagram)
    fit = to_value(p[1])
    pd = fit.plot_data
    names = String[s.set for s in fit.shapes]
    base = resolve_colors(p.colors[], names)

    draw_complement!(p, fit, p.complement[])
    draw_region_fills!(p, pd, base, p.fills[], p.alpha[])
    draw_outlines!(p, pd, names, base, p.edges[])

    show_labels = resolve_label_visibility(p.labels[], p.legend[])
    specs = show_labels ? label_specs(p.labels[], names) :
            Dict{String,Any}(n => nothing for n in names)

    # A set label and a region quantity can land on the exact same anchor (the
    # core derives set anchors from region anchors). When both show, stack them:
    # name above, value below. Precompute the two anchor sets to detect overlap.
    qinfo = p.quantities[] === false ? nothing : resolve_quantities(p.quantities[])
    label_points = collect_label_points(pd, names, specs, show_labels)
    quantity_points = collect_quantity_points(pd, fit, qinfo)

    show_labels && draw_set_labels!(p, pd, names, specs, quantity_points)
    qinfo === nothing || draw_quantities!(p, pd, fit, qinfo, label_points)
    return p
end

# ---------------------------------------------------------------------------
# Public entry points (methods on the `Eunoia.eunoiaplot`/`!` stubs)
# ---------------------------------------------------------------------------

function eunoiaplot(fit::AbstractEulerFit; figure = (;), axis = (;),
                    legend = false, kwargs...)
    f = Figure(; figure...)
    ax = Axis(f[1, 1]; aspect = DataAspect(), axis...)
    hidedecorations!(ax)
    hidespines!(ax)
    p = eunoiadiagram!(ax, fit; legend = legend, kwargs...)
    add_legend_if_requested!(f, fit, p, legend)
    return Makie.FigureAxisPlot(f, ax, p)
end

eunoiaplot!(ax, fit::AbstractEulerFit; kwargs...) = eunoiadiagram!(ax, fit; kwargs...)

function add_legend_if_requested!(f, fit, p, legend)
    legend === false && return
    names = String[s.set for s in fit.shapes]
    base = resolve_colors(p.colors[], names)
    a = p.alpha[]
    elements = [PolyElement(color = (base[n], a)) for n in names]
    legkw = legend isa Union{AbstractDict,NamedTuple} ? _kw(legend) : (;)
    Legend(f[1, 2], elements, names; legkw...)
    return
end

# ---------------------------------------------------------------------------
# Colors — categorical palette + perceptual (OKLab) blending of region fills,
# ported from `eunoia-py`'s `_plot.py`.
# ---------------------------------------------------------------------------

const _TAB10 = RGBf[
    parse(RGBf, "#1f77b4"), parse(RGBf, "#ff7f0e"), parse(RGBf, "#2ca02c"),
    parse(RGBf, "#d62728"), parse(RGBf, "#9467bd"), parse(RGBf, "#8c564b"),
    parse(RGBf, "#e377c2"), parse(RGBf, "#7f7f7f"), parse(RGBf, "#bcbd22"),
    parse(RGBf, "#17becf"),
]

_as_rgb(x) = (c = Makie.to_color(x); RGBf(c.r, c.g, c.b))

function resolve_colors(colors, names)
    if colors === Makie.automatic
        return Dict(n => _TAB10[mod1(i, 10)] for (i, n) in enumerate(names))
    elseif colors isa AbstractDict
        return Dict(n => (haskey(colors, n) ? _as_rgb(colors[n]) : _TAB10[mod1(i, 10)])
                    for (i, n) in enumerate(names))
    else
        length(colors) < length(names) && error(
            "eunoia: colors has $(length(colors)) entries but there are $(length(names)) sets")
        return Dict(n => _as_rgb(colors[i]) for (i, n) in enumerate(names))
    end
end

_srgb_to_linear(c) = c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055)^2.4
_linear_to_srgb(c) =
    clamp(c <= 0.0031308 ? 12.92c : 1.055 * c^(1 / 2.4) - 0.055, 0.0, 1.0)

function _srgb_to_oklab(r, g, b)
    lr = _srgb_to_linear(r); lg = _srgb_to_linear(g); lb = _srgb_to_linear(b)
    lc = 0.4122214708lr + 0.5363325363lg + 0.0514459929lb
    mc = 0.2119034982lr + 0.6806995451lg + 0.1073969566lb
    sc = 0.0883024619lr + 0.2817188376lg + 0.6299787005lb
    l_ = cbrt(lc); m_ = cbrt(mc); s_ = cbrt(sc)
    return (0.2104542553l_ + 0.7936177850m_ - 0.0040720468s_,
            1.9779984951l_ - 2.4285922050m_ + 0.4505937099s_,
            0.0259040371l_ + 0.7827717662m_ - 0.8086757660s_)
end

function _oklab_to_srgb(big_l, a, b)
    l_ = big_l + 0.3963377774a + 0.2158037573b
    m_ = big_l - 0.1055613458a - 0.0638541728b
    s_ = big_l - 0.0894841775a - 1.2914855480b
    lc = l_^3; mc = m_^3; sc = s_^3
    lr = 4.0767416621lc - 3.3077115913mc + 0.2309699292sc
    lg = -1.2684380046lc + 2.6097574011mc - 0.3413193965sc
    lb = -0.0041960863lc - 0.7034186147mc + 1.7076147010sc
    return (_linear_to_srgb(lr), _linear_to_srgb(lg), _linear_to_srgb(lb))
end

function blend_region_color(cols)
    isempty(cols) && return RGBf(0.5, 0.5, 0.5)
    n = length(cols)
    labs = [_srgb_to_oklab(c.r, c.g, c.b) for c in cols]
    big_l = sum(t -> t[1], labs) / n
    a = sum(t -> t[2], labs) / n
    b = sum(t -> t[3], labs) / n
    r, g, bb = _oklab_to_srgb(big_l, a, b)
    return RGBf(r, g, bb)
end

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

_ring(r) = Point2f[Point2f(pt[1], pt[2]) for pt in r]

function piece_to_polygon(piece)
    outer = _ring(piece.outer)
    holes = [_ring(h) for h in piece.holes]
    return isempty(holes) ? Polygon(outer) : Polygon(outer, holes)
end

# Normalize a user style collection (Dict with String/Symbol keys, or a
# NamedTuple) to a NamedTuple of `Symbol => value` kwargs.
_kw(d::AbstractDict) = (; (Symbol(k) => v for (k, v) in d)...)
_kw(nt::NamedTuple) = nt
_kw(::Nothing) = (;)

# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

function draw_complement!(p, fit, complement)
    fit.container === nothing && return
    c = fit.container
    rect = Makie.Rect2f(c.center.x - c.width / 2, c.center.y - c.height / 2,
                        c.width, c.height)
    attrs = merge((color = RGBf(0.94, 0.94, 0.94), strokecolor = RGBf(0.4, 0.4, 0.4),
                   strokewidth = 1.0), _kw(complement === Makie.automatic ? (;) : complement))
    poly!(p, rect; attrs...)
    return
end

function draw_region_fills!(p, pd, base, fills, default_alpha)
    haskey(pd, :region_pieces) || return
    for (key, pieces) in pairs(pd.region_pieces)
        combo = String(key)
        isempty(combo) && continue            # complement region — box covers it
        members = String.(split(combo, '&'))
        cols = RGBf[base[m] for m in members if haskey(base, m)]
        override = (fills isa AbstractDict && haskey(fills, combo)) ? _kw(fills[combo]) : (;)
        attrs = merge((color = blend_region_color(cols), alpha = default_alpha,
                       strokewidth = 0), override)
        for piece in pieces
            poly!(p, piece_to_polygon(piece); attrs...)
        end
    end
    return
end

function draw_outlines!(p, pd, names, base, edges)
    haskey(pd, :shape_outlines) || return
    for (i, n) in enumerate(names)
        haskey(pd.shape_outlines, Symbol(n)) || continue
        pts = _ring(pd.shape_outlines[Symbol(n)])
        length(pts) < 3 && continue
        push!(pts, pts[1])                    # close the open polyline
        attrs = merge((color = base[n], linewidth = 1.5), edge_style(n, edges, i))
        lines!(p, pts; attrs...)
    end
    return
end

function edge_style(name, edges, i)
    edges === Makie.automatic && return (;)
    if edges isa AbstractDict
        if !isempty(edges) && all(v -> v isa Union{AbstractDict,NamedTuple}, values(edges))
            return haskey(edges, name) ? _kw(edges[name]) : (;)   # per-set
        end
        return _kw(edges)                     # uniform
    elseif edges isa AbstractVector
        length(edges) < i && error(
            "eunoia: edges sequence is shorter than the number of sets")
        return _kw(edges[i])
    end
    return (;)
end

# ---- labels ----------------------------------------------------------------

function resolve_label_visibility(labels, legend)
    labels === false && return false
    labels isa AbstractDict && return true
    (labels === nothing || labels === Makie.automatic) && return legend === false
    return labels === true
end

# Per-set label resolution → Dict(name => (text, style)) or `nothing` to hide.
function label_specs(labels, names)
    base = Dict{String,Any}(n => (n, (;)) for n in names)
    labels isa AbstractDict || return base    # automatic/true/nothing → defaults
    nameset = Set(names)
    known = [String(k) for k in keys(labels) if String(k) in nameset]
    if isempty(known)                         # uniform style applied to all
        style = _kw(labels)
        return Dict{String,Any}(n => (n, style) for n in names)
    end
    for (k, v) in labels
        ks = String(k)
        ks in nameset || continue
        if v === nothing || v === false
            base[ks] = nothing
        elseif v isa AbstractString
            base[ks] = (String(v), (;))
        elseif v isa Union{AbstractDict,NamedTuple}
            d = _kw(v)
            txt = haskey(d, :text) ? String(d.text) : ks
            base[ks] = (txt, Base.structdiff(d, NamedTuple{(:text,)}))
        else
            error("eunoia: labels[$ks] must be a String, Dict, NamedTuple, nothing, or false")
        end
    end
    return base
end

function collect_label_points(pd, names, specs, show_labels)
    pts = Set{Tuple{Float64,Float64}}()
    (show_labels && haskey(pd, :set_anchors)) || return pts
    for n in names
        specs[n] === nothing && continue
        haskey(pd.set_anchors, Symbol(n)) || continue
        a = pd.set_anchors[Symbol(n)]
        push!(pts, (Float64(a[1]), Float64(a[2])))
    end
    return pts
end

function draw_set_labels!(p, pd, names, specs, quantity_points)
    haskey(pd, :set_anchors) || return
    for n in names
        spec = specs[n]
        spec === nothing && continue
        haskey(pd.set_anchors, Symbol(n)) || continue
        a = pd.set_anchors[Symbol(n)]
        text, style = spec
        valign = (Float64(a[1]), Float64(a[2])) in quantity_points ? :bottom : :center
        attrs = merge((align = (:center, valign), fontsize = 14), _kw(style))
        text!(p, Point2f(a[1], a[2]); text = text, attrs...)
    end
    return
end

# ---- quantities ------------------------------------------------------------

function resolve_quantities(q)
    source = "original"; types = ["counts"]; style = (;)
    if q === true
        # defaults
    elseif q isa AbstractString
        if q in ("original", "fitted")
            source = q
        elseif q in ("counts", "percent")
            types = [q]
        else
            error("eunoia: quantities string must be original/fitted/counts/percent; got $q")
        end
    elseif q isa Union{AbstractDict,NamedTuple}
        d = _kw(q)
        source = haskey(d, :source) ? String(d.source) : "original"
        source in ("original", "fitted") ||
            error("eunoia: quantities source must be original or fitted; got $source")
        rawtype = haskey(d, :type) ? d.type : "counts"
        types = rawtype isa AbstractString ? [String(rawtype)] : [String(t) for t in rawtype]
        all(t -> t in ("counts", "percent"), types) ||
            error("eunoia: quantities type entries must be counts or percent")
        style = Base.structdiff(d, NamedTuple{(:source, :type)})
    end
    return (source, types, style)
end

function collect_quantity_points(pd, fit, qinfo)
    pts = Set{Tuple{Float64,Float64}}()
    (qinfo !== nothing && haskey(pd, :region_anchors)) || return pts
    source, _, _ = qinfo
    vals = source == "fitted" ? fit.fitted_values : fit.original_values
    for (key, anchor) in pairs(pd.region_anchors)
        combo = String(key)
        (isempty(combo) || !haskey(vals, combo)) && continue
        push!(pts, (Float64(anchor[1]), Float64(anchor[2])))
    end
    return pts
end

function format_quantity(v, total, types)
    parts = String[]
    "counts" in types && push!(parts, @sprintf("%.3g", v))
    if "percent" in types
        pct = total > 0 ? v / total * 100 : 0.0
        s = @sprintf("%.3g%%", pct)
        push!(parts, "counts" in types ? "($s)" : s)
    end
    return join(parts, "\n")
end

function draw_quantities!(p, pd, fit, qinfo, label_points)
    haskey(pd, :region_anchors) || return
    source, types, style = qinfo
    vals = source == "fitted" ? fit.fitted_values : fit.original_values
    total = sum(values(vals); init = 0.0)
    for (key, anchor) in pairs(pd.region_anchors)
        combo = String(key)
        (isempty(combo) || !haskey(vals, combo)) && continue
        valign = (Float64(anchor[1]), Float64(anchor[2])) in label_points ? :top : :center
        attrs = merge((align = (:center, valign), fontsize = 11,
                       color = RGBf(0.41, 0.41, 0.41)), _kw(style))
        text!(p, Point2f(anchor[1], anchor[2]); text = format_quantity(vals[combo], total, types),
              attrs...)
    end
    return
end

end # module
