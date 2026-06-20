# Eunoia <img src='https://raw.githubusercontent.com/jolars/eunoia/refs/heads/main/images/logo.png' align="right" width="139" />

[![Build and
Test](https://github.com/jolars/eunoia/actions/workflows/build-and-test.yml/badge.svg)](https://github.com/jolars/eunoia/actions/workflows/build-and-test.yml)
[![Crates.io](https://img.shields.io/crates/v/eunoia.svg?logo=rust)](https://crates.io/crates/eunoia)
[![npm
version](https://badge.fury.io/js/@jolars%2Feunoia.svg?icon=si%3Anpm)](https://badge.fury.io/js/@jolars%2Feunoia)
[![PyPI
version](https://badge.fury.io/py/eunoia.svg?icon=si%3Apython)](https://badge.fury.io/py/eunoia)
[![CRAN
Badge](http://www.r-pkg.org/badges/version/eulerr)](https://cran.r-project.org/package=eulerr)
[![codecov](https://codecov.io/gh/jolars/eunoia/graph/badge.svg?token=hpXwdIe58E)](https://codecov.io/gh/jolars/eunoia)

Eunoia is a Rust library for area-proportional **Euler and Venn diagrams**. Give
it a set of sizes and intersections and it fits a layout of shapes whose areas
match your data as closely as possible. It is a ground-up rewrite of the R
package [eulerr](https://github.com/jolars/eulerr): faster, more flexible, and
built to power bindings across many languages.

<p align="center">
  <img src="https://raw.githubusercontent.com/jolars/eunoia/refs/heads/main/images/example.png" width="420" alt="An area-proportional Euler diagram of three overlapping sets labelled Adventure, Comedy, and Drama." />
</p>

## Highlights

- **Four shape families**: fit with circles, ellipses, squares, or rectangles;
  the engine is shape-agnostic until fit time.
- **Robust fitting**: MDS initialization followed by a global/local optimization
  pipeline (LM, CMA-ES, trust-region) that mirrors and improves on eulerr.
- **Venn diagrams too**: canonical n-set Venn arrangements (circles for n≤3,
  ellipses for n=4-5) independent of the fitter.
- **Renderable output**: region polygon extraction, clipping, and automatic
  label placement, ready to draw as SVG.
- **Many targets**: pure Rust core that ships to JavaScript via WebAssembly and
  to other languages through a small C ABI.

## Quick start (Rust)

```sh
cargo add eunoia
```

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

    // Swap `Ellipse` for `Circle`, `Square`, or `Rectangle`.
    let layout = Fitter::<Ellipse>::new(&spec).seed(1).fit().unwrap();

    println!("{} shapes, loss = {:.2e}", layout.shapes().len(), layout.loss());
}
```

`layout.shapes()` returns the fitted geometry; the `plotting` module turns a
`Layout` into region polygons and label anchors for rendering. See the
[rustdoc](https://docs.rs/eunoia/) for the full API.

## The Eunoia ecosystem

The pure-Rust core powers bindings in several languages, all backed by the same
fitting engine:

  | Language       | Package                                                                                          | Install                       |
  | -------------- | ------------------------------------------------------------------------------------------------ | ----------------------------- |
  | **Rust**       | [`eunoia`](https://crates.io/crates/eunoia)                                                      | `cargo add eunoia`            |
  | **R**          | [`eulerr`](https://CRAN.R-project.org/package=eulerr) ([repo](https://github.com/jolars/eulerr)) | `install.packages("eulerr")`  |
  | **Python**     | [`eunoia`](https://pypi.org/project/eunoia/) ([repo](https://github.com/jolars/eunoia-py))       | `pip install eunoia`          |
  | **Julia**      | [`Eunoia.jl`](https://github.com/jolars/Eunoia.jl)                                               | `Pkg.add("Eunoia")`           |
  | **JavaScript** | [`@jolars/eunoia`](https://www.npmjs.com/package/@jolars/eunoia)                                 | `npm install @jolars/eunoia`  |
  | **Web app**    | [eunoia.bz](https://eunoia.bz)                                                                   | build diagrams in the browser |

### JavaScript / TypeScript

WebAssembly bindings are published as
[`@jolars/eunoia`](https://www.npmjs.com/package/@jolars/eunoia):

```ts
import { euler, venn } from "@jolars/eunoia";

// Fit an Euler diagram from set sizes
const layout = euler({
  sets: { A: 5, B: 2, "A&B": 1 },
  shape: "circle",        // "circle" | "ellipse" | "square" | "rectangle"
  output: "shapes",       // "shapes" | "polygons" | "regions"
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

### Browser / Observable (no bundler)

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

In Observable:

```js
eunoia = import("https://esm.sh/@jolars/eunoia/web")
await eunoia.init()
layout = eunoia.euler({ sets: { A: 5, B: 2, "A&B": 1 } })
```

`@jolars/eunoia/svg` is pure JavaScript (no WebAssembly), so it already works
directly from a CDN with no `init()`.

Full runnable examples (a Quarto document and a standalone HTML page) are in
[`examples/`](examples/).

## Documentation

- Narrative docs: [eunoia.bz/docs/](https://eunoia.bz/docs/)
- Rust API reference: [docs.rs/eunoia](https://docs.rs/eunoia/)

## License

Eunoia is distributed under the terms of either the [MIT license](LICENSE-MIT)
or the [Apache License 2.0](LICENSE-APACHE), at your option.
