# Eunoia

## See It in Action

A three-set diagram, fitted in your browser by the same WASM core the packages
ship. Drag the region sizes or switch the shape and watch it re-solve live.

::: {.component name="eunoia" #home-diagram}

Eunoia fits shapes to the sizes of sets and their intersections. Use the
[diagram builder](app/index.md) to explore your own data, or follow a
[quickstart](docs/index.md) to use Eunoia in your preferred language.

:::

## Install

::: {.panel-tabset group="language"}

### Rust

```bash
cargo add eunoia
```

Read the [Rust quickstart](docs/quickstart/rust/index.md) or the full
[Rust API reference](https://docs.rs/eunoia/).

### JavaScript

```bash
npm install @jolars/eunoia
```

Read the [JavaScript quickstart](docs/quickstart/javascript/index.md). The WASM
bindings power the web app and every interactive example on this site.

### Python

```bash
pip install eunoia
```

Read the [Python quickstart](docs/quickstart/python/index.md). The Python package
provides bindings to the Rust core and plotting with matplotlib.

### Julia

```julia
using Pkg
Pkg.add("Eunoia")
```

Read the [Julia quickstart](docs/quickstart/julia/index.md). Eunoia.jl provides
bindings to the Rust core and plotting with Makie.

### R

```r
install.packages("eulerr")
```

Read the [R quickstart](docs/quickstart/r/index.md) or the
[eulerr documentation](https://jolars.github.io/eulerr/).

:::

## Why Eunoia?

- Exact n-way intersection geometry for circles and ellipses, with analytical gradients and efficient optimizers.
- Circles, ellipses, squares, rectangles, and rotated rectangles.
- Multiple loss functions and selectable optimizers, including CMA-ES, Levenberg–Marquardt, L-BFGS, and Nelder–Mead.
- Optional bounding containers for fitting the complement of the named sets.
- Customizable labels, glyphs, colors, and exports.
- A pure Rust core with bindings for JavaScript, Python, Julia, and R.

## Citing Eunoia

If you use Eunoia or its language bindings in academic work, please cite:

> Larsson, J., & Lee, N. M. (2026). *Eunoia: Area-Proportional Euler and Venn Diagrams in Rust, Julia, Python, R, and JavaScript* (arXiv:2611.02198). arXiv. [doi:10.48550/arXiv.2611.02198](https://doi.org/10.48550/arXiv.2611.02198).

The [citation page](docs/cite/index.md) provides BibTeX, BibLaTeX, CSL, APA,
Vancouver, and Chicago formats, along with software-release citations.
