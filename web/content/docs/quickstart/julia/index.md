# Quickstart: Julia {#quickstart-julia}

[`Eunoia.jl`](https://github.com/jolars/Eunoia.jl) is the Julia binding. You hand
`euler` a `Dict` of set sizes, it fits an area-proportional diagram, and
`eunoiaplot` renders it through a Makie extension.

::: {.component name="eunoia" #docs-quickstart-julia-1}

The diagram fitted by the first example below.

:::

## Install {#install}

```julia
using Pkg
Pkg.add("Eunoia")
```

Or, from the Pkg REPL (press `]`):

```julia
add Eunoia
```

The native fitting engine ships as a Julia artifact, fetched lazily on first
use, so no Rust toolchain is needed.

## Fit a Diagram {#fit-a-diagram}

```julia
using Eunoia

fit = euler(Dict("A" => 5, "B" => 3, "A&B" => 1.5))
fit          # `show` prints the residual table
fit.shapes   # fitted Circle, Ellipse, Square, Rectangle, or RotatedRectangle structs
fit.loss     # fit quality (lower is better)
```

The `Dict` is keyed by **combination expression**: a single set (`"A"`) or sets
joined with `&` (`"A&B"`). Values are *exclusive* by default: the part of `A`
outside `B` is `5`, the overlap is `1.5`. Pass `input_type = "inclusive"` if your
numbers are full set sizes instead.

Membership collections are accepted too and counted into exclusive regions:

```julia
euler(Dict("A" => ["x", "y", "z"], "B" => ["y", "z", "w"]))
```

## Plot with Makie {#plot-with-makie}

Rendering lives in a Makie package extension that loads automatically once you
import a Makie backend:

```julia
using Eunoia, CairoMakie   # or GLMakie/WGLMakie

fit = euler(Dict("A" => 5, "B" => 3, "A&B" => 1.5))
eunoiaplot(fit)                                   # publication-ready figure
eunoiaplot(fit; quantities = true, legend = true)
```

`eunoiaplot(fit)` returns a `Makie.FigureAxisPlot`; `eunoiaplot!(ax, fit)` draws
into an existing axis. Styling keywords (`colors`, `fills`, `edges`, `labels`,
`quantities`, `legend`) mirror the other bindings.

## Pick a Different Shape {#pick-a-different-shape}

```julia
euler(Dict("A" => 5, "B" => 3, "A&B" => 1.5); shape = "ellipse")
```

`shape` is one of `"circle"` (default), `"ellipse"`, `"square"`,
`"rectangle"`, or `"rotated_rectangle"`. Ellipses are more flexible and often
fit three or more sets noticeably better. Rotated rectangles add an angle to
the width and height and use derivative-free fitting; see
[Shapes](../../concepts/shapes/index.md).

## Venn Diagrams {#venn-diagrams}

For a fixed, *non*-proportional Venn template (every intersection shown
regardless of size), use `venn`; it accepts an `Int`, a name vector, or a
mapping:

```julia
venn(3)                                    # three sets named A, B, C
venn(["A", "B", "C"]; shape = "ellipse")
venn(4; shape = "rotated_rectangle")
```

Ellipse Venn diagrams support up to five sets, rotated rectangles up to four,
and circles, squares, and axis-aligned rectangles up to three.

## Next Steps {#next-steps}

The full reference lives on the
[Julia API reference](https://eunoia.bz/docs/reference/julia/api/) and
[gallery](https://eunoia.bz/docs/reference/julia/gallery/). For how the fit
works across all bindings, see the
[Fitter Pipeline](../../concepts/fitter-pipeline/index.md) and
[Goodness of Fit](../../concepts/goodness-of-fit/index.md) chapters.
