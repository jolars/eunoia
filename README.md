# Eunoia <img src='https://raw.githubusercontent.com/jolars/eunoia/refs/heads/main/images/logo.png' align="right" width="139" />

[![CI](https://github.com/jolars/eunoia/actions/workflows/build-and-test.yml/badge.svg)](https://github.com/jolars/eunoia/actions/workflows/build-and-test.yml)
[![Crates.io](https://img.shields.io/crates/v/eunoia.svg?logo=rust)](https://crates.io/crates/eunoia)
[![npm](https://img.shields.io/npm/v/@jolars/eunoia?label=npm&logo=npm)](https://badge.fury.io/js/@jolars%2Feunoia)
[![pypi](https://img.shields.io/pypi/v/eunoia?label=pypi&logo=python)](https://badge.fury.io/py/eunoia)
[![CRAN](https://img.shields.io/cran/v/eulerr?logo=r)](https://cran.r-project.org/package=eulerr)
[![codecov](https://codecov.io/gh/jolars/eunoia/graph/badge.svg?token=hpXwdIe58E)](https://codecov.io/gh/jolars/eunoia)
[![arXiv](https://img.shields.io/badge/arXiv-2611.02198-b31b1b)](https://arxiv.org/abs/2611.02198)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20665805-blue)](https://doi.org/10.5281/zenodo.20665805)

Eunoia is a Rust library for area-proportional **Euler and Venn diagrams**. Give
it a set of sizes and intersections and it fits a layout of shapes whose areas
match your data as closely as possible. It is a ground-up rewrite of the R
package [eulerr](https://github.com/jolars/eulerr): faster, more flexible, and
built to power bindings across many languages. Read the documentation at
[eunoia.bz](https://eunoia.bz/docs/).

## Highlights

- **Five shape families**: fit with circles, ellipses, squares, axis-aligned
  rectangles, or rotated rectangles; the engine is shape-agnostic until fit
  time.
- **Robust fitting**: initial initialization followed by a global/local
  optimization pipeline that mirrors and improves on eulerr.
- **Venn diagrams too**: canonical arrangements for up to three sets with
  circles, squares, or rectangles, four with rotated rectangles, and five with
  ellipses, independent of the fitter.
- **Renderable output**: region polygon extraction, clipping, and automatic
  label placement, ready to draw as SVG.
- **Many targets**: pure Rust core that ships to JavaScript via WebAssembly and
  to other languages through a small C ABI.

## Quick start (Rust)

```sh
cargo add eunoia
```

Then run the following example to fit an Euler diagram of three overlapping
sets.

```rust
use eunoia::geometry::shapes::Ellipse;
use eunoia::{DiagramSpecBuilder, Fitter, InputType};

fn main() {
    let spec = DiagramSpecBuilder::new()
        .set("Adventure", 20.0)
        .set("Comedy", 14.0)
        .set("Drama", 18.0)
        .intersection(&["Adventure", "Comedy"], 6.0)
        .intersection(&["Adventure", "Drama"], 5.0)
        .intersection(&["Comedy", "Drama"], 4.0)
        .intersection(&["Adventure", "Comedy", "Drama"], 2.0)
        .input_type(InputType::Exclusive)
        .build()
        .unwrap();

    // Swap `Ellipse` for `Circle`, `Square`, `Rectangle`, or `RotatedRectangle`.
    let layout = Fitter::<Ellipse>::new(&spec).seed(1).fit().unwrap();

    println!(
        "{} shapes, loss = {:.2e}",
        layout.shapes().len(),
        layout.loss()
    );
}
```

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jolars/eunoia/refs/heads/main/images/example-dark.svg" />
    <img src="https://raw.githubusercontent.com/jolars/eunoia/refs/heads/main/images/example.svg" width="380" alt="An area-proportional Euler diagram of three overlapping sets labelled Adventure, Comedy, and Drama." />
  </picture>
</p>

`layout.shapes()` returns the fitted geometry; the `plotting` module turns a
`Layout` into region polygons and label anchors for rendering. See the
[rustdoc](https://docs.rs/eunoia/) for the full API.

## Web app

Eunoia powers a web app for interactive diagram fitting and exploration, located
at <https://eunoia.bz/app>.

## The Eunoia ecosystem

The pure-Rust core powers bindings in several languages, all backed by the same
fitting engine:

  | Language   | Package                                                                                                               | Install                      |
  | ---------- | --------------------------------------------------------------------------------------------------------------------- | ---------------------------- |
  | R          | [`eulerr`](https://cran.r-project.org/package=eulerr) ([repo](https://github.com/jolars/eulerr))                      | `install.packages("eulerr")` |
  | Python     | [`eunoia`](https://pypi.org/project/eunoia/) ([repo](https://github.com/jolars/eunoia-py))                            | `pip install eunoia`         |
  | Julia      | [`Eunoia.jl`](https://platform.juliahub.com/ui/Packages/General/Eunoia) ([repo](https://github.com/jolars/Eunoia.jl)) | `]add Eunoia`                |
  | JavaScript | [`@jolars/eunoia`](https://www.npmjs.com/package/@jolars/eunoia) (this repo)                                          | `npm install @jolars/eunoia` |

### JavaScript/TypeScript

A TypeScript package is published as
[`@jolars/eunoia`](https://www.npmjs.com/package/@jolars/eunoia). Here's a
minimal example of fitting an Euler diagram in Node or a bundler environment:

```ts
import { euler, venn } from "@jolars/eunoia";

// Fit an Euler diagram from set sizes
const layout = euler({
  sets: { A: 5, B: 2, "A&B": 1 },
  shape: "circle", // "circle" | "ellipse" | "square" | "rectangle" | "rotatedRectangle"
  output: "shapes", // "shapes" | "polygons" | "regions"
  inputType: "exclusive", // "exclusive" | "inclusive"
  seed: 42,
});

if (layout.mode === "shapes" && layout.shape === "circle") {
  for (const c of layout.circles) {
    console.log(c.label, c.x, c.y, c.radius);
  }
}

// Or build a canonical n-set Venn diagram
const v = venn({ n: 3, output: "regions" });
```

A renderer-agnostic SVG serializer is available at `@jolars/eunoia/svg`. The
default entry is built with `wasm-pack --target bundler`, so it works with any
modern bundler (Vite, Webpack, Rollup, esbuild) and Node 20+.

### Browser/Observable (no Bundler)

The default entry imports the `.wasm` module directly, which only a bundler can
resolve. For a plain HTML page, an [Observable](https://observablehq.com)
notebook, or any environment without a build step, use the `@jolars/eunoia/web`
entry instead: a single self-contained ESM file with the WebAssembly module
inlined. Call `init()` once before fitting:

```html
<script type="module">
  import { euler, init } from "https://esm.sh/@jolars/eunoia/web";

  await init(); // instantiate the embedded WebAssembly module (once)

  const layout = euler({ sets: { A: 5, B: 2, "A&B": 1 } });
  console.log(layout.circles);
</script>
```

### Observable

```js
eunoia = import("https://esm.sh/@jolars/eunoia/web");
await eunoia.init();
layout = eunoia.euler({ sets: { A: 5, B: 2, "A&B": 1 } });
```

`@jolars/eunoia/svg` is pure JavaScript (no WebAssembly), so it already works
directly from a CDN with no `init()`. Full runnable examples (a Quarto document
and a standalone HTML page) are in [`examples/`](examples/).

## Citation

If you use Eunoia in your research, please cite the paper:

> Larsson, J., & Lee, N. M. (2026). *Eunoia: Area-Proportional Euler and Venn
> Diagrams in Rust, Julia, Python, R, and JavaScript* (arXiv:2611.02198). arXiv.
> <https://doi.org/10.48550/arXiv.2611.02198>

```bibtex
@misc{larsson2026f,
  title         = {Eunoia: {{Area}}-Proportional {{Euler}} and {{Venn}} Diagrams in {{Rust}}, {{Julia}}, {{Python}}, {{R}}, and {{JavaScript}}},
  shorttitle    = {Eunoia},
  author        = {Larsson, Johan and Lee, Nicole M.},
  year          = {2026},
  month         = nov,
  number        = {arXiv:2611.02198},
  eprint        = {2611.02198},
  primaryclass  = {stat.CO},
  publisher     = {arXiv},
  doi           = {10.48550/arXiv.2611.02198},
  archiveprefix = {arXiv}
}
```

<details>
<summary>BibLaTeX</summary>

```bibtex
@online{larsson2026f,
  title       = {Eunoia: {{Area}}-Proportional {{Euler}} and {{Venn}} Diagrams in {{Rust}}, {{Julia}}, {{Python}}, {{R}}, and {{JavaScript}}},
  shorttitle  = {Eunoia},
  author      = {Larsson, Johan and Lee, Nicole M.},
  date        = {2026-11-01},
  eprint      = {2611.02198},
  eprinttype  = {arXiv},
  eprintclass = {stat.CO},
  doi         = {10.48550/arXiv.2611.02198},
  pubstate    = {prepublished}
}
```

</details>

For reproducibility, also cite the software release you used. Select that
release in the [Zenodo archive](https://doi.org/10.5281/zenodo.20665805) and
export its citation. The concept DOI `10.5281/zenodo.20665805` identifies Eunoia
across all releases.

[CITATION.cff](CITATION.cff) contains the software metadata and preferred paper
citation. The [Eunoia citation page](https://eunoia.bz/cite/) renders the paper
in APA, BibTeX, BibLaTeX, and other formats.

## License

Eunoia is distributed under the terms of either the [MIT license](LICENSE-MIT)
or the [Apache License 2.0](LICENSE-APACHE), at your option.
