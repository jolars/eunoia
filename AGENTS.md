# Eunoia contributor guide

Eunoia is a Rust library for area-proportional Euler and Venn diagrams. The same
core engine is exposed through WebAssembly, a TypeScript package, a C ABI, and a
statically generated SvelteKit documentation site.

Use this file for repository-wide expectations. Prefer the nearest README and
existing code when you need component-level detail.

## Repository map

  | Path                  | Purpose                                                            |
  | --------------------- | ------------------------------------------------------------------ |
  | `crates/eunoia/`      | Core Rust library and fitting algorithms; default workspace member |
  | `crates/eunoia-wasm/` | Thin `wasm-bindgen` bindings                                       |
  | `crates/eunoia-capi/` | JSON-in/JSON-out C ABI used by Eunoia.jl                           |
  | `ts/`                 | High-level TypeScript API and npm build scripts                    |
  | `npm/`                | Generated publishable `@jolars/eunoia` package; gitignored         |
  | `web/`                | SvelteKit docs and demo app, linked to `npm/`                      |
  | `paper/`              | JOSS paper and generated figures                                   |

The Cargo workspace uses Rust edition 2024 and has an MSRV of 1.88.0. Rust
modules use `module.rs` plus `module/`; do not introduce `module/mod.rs`.

## Working agreements

- Keep changes focused and preserve unrelated work in the worktree.
- Follow the surrounding code's public API, naming, error-handling, and test
  patterns before introducing a new abstraction.
- Do not edit `npm/` by hand. Regenerate it with the appropriate task.
- Do not add a production dependency unless it is necessary for the requested
  change; call out the reason and trade-off in the handoff.
- Use Conventional Commits when proposing commit messages, with scopes such as
  `fitter`, `geometry`, `ts`, and `web`.
- Public Rust APIs require rustdoc. CI treats rustdoc warnings as errors; never
  link a public item to a private item with an intra-doc link.

## Build and test commands

Tasks are defined in `Taskfile.yml` and run with [Task](https://taskfile.dev/).
Direct Cargo commands are also valid.

### Routine validation

```sh
cargo test                         # fast tests for the default core member
cargo test -p eunoia <substring>   # focused core test
cargo test --workspace             # all default workspace tests
task test-ts                       # build TS wrapper and test the pure JS SVG API
task lint                          # clippy, all targets/features, warnings denied
task doc                           # rustdoc warnings denied plus doc tests
task dev                           # full pre-PR gate
```

Run the narrowest relevant checks while iterating. Before handing off a broad or
cross-layer change, run `task dev` when practical. If a relevant check was not
run, say which one and why.

### Fitter changes

Any change that can affect fitting behavior, geometry calculations, losses,
initialization, optimizers, normalization, clustering, or packing must also run:

```sh
task test-slow
```

The ignored suite contains stochastic and regression-quality guardrails that the
default Cargo test run does not cover. `task dev` runs the smaller
`corpus_quality` guardrail but is not a substitute for `task test-slow` here.

### WASM, TypeScript, and web

```sh
task build-ts       # rebuild high-level TS output in npm/; requires existing wasm types
task build-wasm     # rebuild bundler-target wasm and then the TS package
task build-web      # build the self-contained bundler-less browser entry
task web-dev        # rebuild TS and start the Svelte dev server
cd web && pnpm check
cd web && pnpm build
```

- Run `task build-wasm` when Rust WASM bindings change.
- Run `task build-ts` for TypeScript-only changes. While `task web-dev` is
  running, rerun it manually after edits under `ts/`.
- Use `pnpm` in `web/`; the pinned package manager is declared in
  `web/package.json`.
- The web app imports the default bundler entry. The `./web` entry is for
  bundler-less consumers and is built separately by `task build-web`.

## Architecture and invariants

The core pipeline is:

```text
DiagramSpec -> preprocess -> Fitter<S> -> Layout<S> -> plotting/output
```

- `spec` describes set sizes and intersections without geometry. Preprocessing
  stores exclusive regions and converts combinations to `RegionMask`s.
- `fitter` chooses the `DiagramShape` at fit time, builds an initial layout,
  refines it across restarts, and normalizes the best result.
- `geometry` provides composable traits and fitting shapes: `Circle`, `Ellipse`,
  `Square`, and `Rectangle`. `Polygon` is for output and region extraction, not
  fitting.
- `loss` contains the optimization objectives and smooth surrogates.
- `plotting` extracts region polygons and places region labels, set labels, and
  glyphs.
- `venn` provides canonical arrangements independently of the fitter.

Preserve these invariants:

- Shape choice belongs to `Fitter<S>`, not `DiagramSpec`.

- External and FFI APIs use geometric parameters. Optimizer parameters are an
  internal representation:

  | Shape     | Geometric           | Optimizer                   |
  | --------- | ------------------- | --------------------------- |
  | Circle    | `[x, y, r]`         | identity                    |
  | Ellipse   | `[x, y, a, b, phi]` | `[x, y, ln(a), ln(b), phi]` |
  | Square    | `[x, y, side]`      | identity                    |
  | Rectangle | `[x, y, w, h]`      | `[x, y, ln(w*h), ln(w/h)]`  |

- Complement fitting appends the container's four rectangle optimizer parameters
  in the same rectangle encoding.

- The `parallel` feature is intentionally off by default and must remain
  disabled for WASM. The `corpus` feature is internal test/benchmark support,
  not public API.

- `basin` is the sole optimizer dependency. Do not introduce a second optimizer
  stack without an explicit architectural decision.

## Keep public surfaces synchronized

When a core public option changes—such as a shape, optimizer, loss, MDS solver,
builder method, or complement behavior—update every applicable surface in the
same change:

1. Core Rust API and tests in `crates/eunoia/`.
2. WASM enums/signatures and tests in `crates/eunoia-wasm/`.
3. C API input types, hand-written snake_case token parsers, implementation, and
   tests in `crates/eunoia-capi/`.
4. TypeScript wrapper and declarations under `ts/`.
5. The matching bindings documentation under `web/src/routes/docs/bindings/`.

Do not assume serde exposes a new core enum through the C API; those mappings
are deliberately manual.

The npm package publicly exports `.`, `./svg`, `./web`, and `./trajectory`. The
raw wasm-bindgen module is runtime backing and is intentionally absent from the
package `exports` map.

## Documentation site

Narrative documentation lives in `web/src/routes/docs/**` as `.svx` (mdsvex)
pages. A page's first `# H1` is its title; do not add YAML frontmatter. A page
may begin with a `<script>` block for interactive Svelte components.

Files under `web/static/` are served verbatim. Keep the two site indexes
distinct:

- `web/src/routes/sitemap.xml/+server.ts` discovers `+page.svelte` and
  `+page.svx` routes automatically. Do not hand-maintain page entries there.
- `web/static/llms.txt` is hand-curated. When adding, removing, renaming, or
  substantially re-scoping a docs page, update its link and one-line description
  and keep it in the correct section. It also contains off-site links that route
  discovery cannot supply.

## Code review rules

Prioritize behavioral and compatibility risks over formatting that automated
tools already enforce.

- Flag fitter changes that omit `task test-slow` evidence. Safe path: run the
  ignored suite and report any stochastic failure with its seed or fixture.
- Flag a changed core public option when either binding layer, TypeScript, docs,
  or tests are missing. Safe path: update all applicable surfaces together.
- Flag use of optimizer-encoded shape parameters in an external interface. Safe
  path: convert at the fitter boundary and expose geometric parameters.
- Flag manual edits to generated `npm/` artifacts. Safe path: edit `ts/` or the
  bindings and regenerate with `task build-ts`, `task build-wasm`, or
  `task build-web`.
- Flag docs route changes that leave `web/static/llms.txt` stale. Sitemap edits
  are normally unnecessary because route discovery is automatic.

## Releases

Versions are managed by `versionary.jsonc`; the Rust workspace and npm package
are versioned separately. Pushing a `v*` tag triggers the crate and npm publish
workflows. Do not hand-edit versions or publish artifacts unless the task is
explicitly a release.
