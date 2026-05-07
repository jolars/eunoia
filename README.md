# Eunoia <img src='https://raw.githubusercontent.com/jolars/eunoia/refs/heads/main/images/logo.png' align="right" width="139" />

[![Build and Test](https://github.com/jolars/eunoia/actions/workflows/build-and-test.yml/badge.svg)](https://github.com/jolars/eunoia/actions/workflows/build-and-test.yml)
[![codecov](https://codecov.io/gh/jolars/eunoia/graph/badge.svg?token=hpXwdIe58E)](https://codecov.io/gh/jolars/eunoia)

A Rust library for Euler and Venn Diagrams. This is a rewrite of the eulerr R
package, designed to be more flexible, faster, and support multiple language
bindings.

## JavaScript / TypeScript

WebAssembly bindings are published as
[`@jolars/eunoia`](https://www.npmjs.com/package/@jolars/eunoia):

```sh
npm install @jolars/eunoia
```

```ts
import { euler, venn } from "@jolars/eunoia";

// Fit an Euler diagram from set sizes
const layout = euler({
  sets: { A: 5, B: 2, "A&B": 1 },
  shape: "circle",        // "circle" | "ellipse" | "square"
  output: "shapes",       // "shapes" | "polygons" | "regions"
  inputType: "exclusive", // "exclusive" | "inclusive"
  seed: 42,
});

if (layout.mode === "shapes" && layout.shape === "circle") {
  for (const c of layout.circles) {
    console.log(c.label, c.x, c.y, c.radius);
  }
}
console.log(layout.metrics.loss, layout.metrics.fittedAreas);

// Or build a canonical n-set Venn diagram
const v = venn({ n: 3, output: "regions" });
```

The package is built with `wasm-pack --target bundler`, so it works with any
modern bundler (Vite, Webpack, Rollup, esbuild) and Node 20+.

The raw wasm-bindgen surface is available at `@jolars/eunoia/raw` if you need
to drop down to the underlying API.
