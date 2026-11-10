# Introduction {#introduction}

Eunoia is a Rust library for area-proportional Euler and Venn diagrams. It is
the engine behind the
[`@jolars/eunoia`](https://www.npmjs.com/package/@jolars/eunoia) npm package,
the [`eunoia`](https://pypi.org/project/eunoia/) Python package, and the
[`Eunoia.jl`](https://github.com/jolars/Eunoia.jl) Julia package, as well as a
modern rewrite of the C++ backend behind the
[eulerr](https://github.com/jolars/eulerr) R package.

## Who This Is For {#who-this-is-for}

These docs are aimed at two audiences:

- **End users of language wrappers** (R, Python, JavaScript, Julia) who want to understand what the underlying library does, what its knobs mean, and how to read its outputs.
- **Binding authors** wiring up downstream packages. Most hosts bind the Rust core directly (wasm-bindgen, PyO3, extendr); native hosts that reach it through the C ABI have the [Bindings](bindings/c-abi/index.md) section.

For Rust API reference see [docs.rs/eunoia](https://docs.rs/eunoia/);
for the TypeScript surface see the type definitions shipped with
[`@jolars/eunoia`](https://www.npmjs.com/package/@jolars/eunoia).
This site is the narrative companion.

## Where to Start {#where-to-start}

- Getting started: the quickstart sections for [JavaScript](quickstart/javascript/index.md), [Rust](quickstart/rust/index.md), [Python](quickstart/python/index.md), [R](quickstart/r/index.md), or [Julia](quickstart/julia/index.md).
- Writing a native binding via the C ABI: [C ABI Contract](bindings/c-abi/index.md).
- Placing labels in a downstream renderer: [Label Placement](concepts/label-placement/index.md) (the only fully written chapter so far; the rest are stubs that will fill in as consumers hit the gap).
