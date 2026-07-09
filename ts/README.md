# @jolars/eunoia

[![npm version](https://badge.fury.io/js/@jolars%2Feunoia.svg?icon=si%3Anpm)](https://badge.fury.io/js/@jolars%2Feunoia)
[![Build and Test](https://github.com/jolars/eunoia/actions/workflows/build-and-test.yml/badge.svg)](https://github.com/jolars/eunoia/actions/workflows/build-and-test.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](https://github.com/jolars/eunoia#license)

Area-proportional **Euler and Venn diagrams** for JavaScript — the WebAssembly
build of the Rust [eunoia](https://docs.rs/eunoia/) engine. Give it a set of
sizes and intersections and it fits a layout of shapes whose areas match your
data as closely as possible, then hands back plain JavaScript objects — circles,
ellipses, polygons, or per-region pieces — that you draw however you like (SVG,
Canvas, D3).

## Install

```sh
npm install @jolars/eunoia
```

The default entry is an ES module backed by WebAssembly, so it needs a bundler
that understands an `import`-ed `.wasm` module (Vite, webpack 5, Rollup,
esbuild, …) and Node 20+. It ships its own TypeScript types — no `@types`
package needed. For bundler-less environments, see [Entry points](#entry-points).

## Usage

```ts
import { euler, venn } from "@jolars/eunoia";

// Fit an Euler diagram from set sizes.
const layout = euler({
  sets: { A: 5, B: 2, "A&B": 1 },
  shape: "circle", // "circle" | "ellipse" | "square" | "rectangle"
  output: "shapes", // "shapes" | "polygons" | "regions"
  inputType: "exclusive", // "exclusive" | "inclusive"
  seed: 42,
});

if (layout.mode === "shapes" && layout.shape === "circle") {
  for (const c of layout.circles) {
    console.log(c.label, c.x, c.y, c.radius);
  }
}

// Or build a canonical n-set Venn diagram (ignores set sizes).
const v = venn({ n: 3, output: "regions" });
```

`sets` is keyed by **combination expression** — a single set (`"A"`) or sets
joined with `&` (`"A&B"`). Values are *exclusive* sizes by default; pass
`inputType: "inclusive"` if your numbers are full set unions instead.

The `output` option decides the result shape: `"shapes"` returns primitive
parameters (circles / ellipses / squares / rectangles), `"polygons"` adds one
closed outline per set, and `"regions"` returns the exclusive pieces (`A` only,
`A&B`, …) for per-region fills and label placement. Every mode also carries fit
`metrics`. See the [JavaScript quickstart][quickstart] for the full walkthrough.

## Entry points

All three ship from the one package:

| Import                | What it is                                                                            |
| --------------------- | ------------------------------------------------------------------------------------- |
| `@jolars/eunoia`      | Default. Bundler-friendly; `import`s the `.wasm` module directly. Use with a bundler.  |
| `@jolars/eunoia/web`  | Bundler-less. A single ESM file with the wasm inlined; `await init()` once before use. |
| `@jolars/eunoia/svg`  | Pure JavaScript (no wasm). Turns a `Layout` into an SVG string. Works from a CDN.       |

### Browser / Observable (no bundler)

The default entry imports the `.wasm` module directly, which only a bundler can
resolve. For a plain HTML page, an [Observable](https://observablehq.com)
notebook, or any environment without a build step, use `@jolars/eunoia/web` — a
single self-contained ESM file with the WebAssembly inlined. Call `init()` once
before fitting:

```html
<script type="module">
  import { euler, init } from "https://esm.sh/@jolars/eunoia/web";

  await init(); // instantiate the embedded wasm once; idempotent

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
directly from a CDN and during server-side rendering with no `init()`:

```ts
import { euler } from "@jolars/eunoia";
import { toSvg } from "@jolars/eunoia/svg";

const layout = euler({ sets: { A: 5, B: 2, "A&B": 1 }, output: "regions" });
document.body.innerHTML = toSvg(layout, { showLabels: true });
```

For interactivity, `toSvg` / `svgBody` can attach native hover tooltips and
`data-*` hooks to each region and set shape via `tooltip`, `interactive`, and
`regionAttrs` (each hook gets a `RegionInfo` of `{ combination, sets, area }`):

```ts
const svg = toSvg(layout, {
  interactive: true, // data-combination + data-area on each fill
  tooltip: (r) => `${r.combination}: ${r.area.toFixed(0)}`,
});
```

Eunoia sees only set/intersection sizes, so `combination -> members` is yours to
compute; format small lists in `tooltip`, or keep large ones JS-side and look
them up on hover via `data-combination` to keep the SVG small.

> Server-side rendering: call the wasm-backed `euler` / `venn` from the
> **client** (e.g. a dynamic `import("@jolars/eunoia")` inside `onMount` /
> `useEffect`) so the wasm module isn't instantiated during the server render.
> The `./svg` entry has no such restriction.

## Documentation

- [JavaScript quickstart][quickstart] — the guided getting-started.
- [JavaScript API reference](https://eunoia.bz/docs/reference/javascript/) — the
  public surface (`euler`, `venn`, label placement, the SVG serializer).
- [Narrative docs](https://eunoia.bz/docs/) — concepts: shapes, goodness of fit,
  label placement, and the fitting pipeline.

Full field-level types ship with the package; your editor's "go to definition" /
IntelliSense will surface them.

## License

Distributed under the terms of either the [MIT license](https://github.com/jolars/eunoia/blob/main/LICENSE-MIT)
or the [Apache License 2.0](https://github.com/jolars/eunoia/blob/main/LICENSE-APACHE),
at your option.

[quickstart]: https://eunoia.bz/docs/quickstart/javascript/
