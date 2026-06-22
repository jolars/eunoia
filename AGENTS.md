# Agent instructions

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Eunoia is a Rust library for area-proportional **Euler and Venn diagrams** — a
ground-up rewrite of the R package [eulerr](https://github.com/jolars/eulerr).
The core is pure Rust; the same engine ships to JavaScript via WebAssembly and
powers a SvelteKit web app. Narrative docs: <https://eunoia.bz/docs/>. Rustdoc:
<https://docs.rs/eunoia/>.

## Repository layout

This is a **Cargo workspace** plus two JS sub-projects. Four distinct artifacts:

| Path                  | Artifact                                                                 |
| --------------------- | ------------------------------------------------------------------------ |
| `crates/eunoia/`      | Core library (pure Rust). Default workspace member. The real algorithms. |
| `crates/eunoia-wasm/` | `wasm-bindgen` surface — a thin, raw binding layer. `publish = false`.   |
| `ts/`                 | High-level TypeScript wrapper (`euler()`, `venn()`) + build script.      |
| `npm/`                | The assembled, publishable `@jolars/eunoia` package (checked in).        |
| `web/`                | SvelteKit app (Svelte 5, Tailwind 4, rolldown-vite). Links `file:../npm`.|

How the JS layers fit together: `eunoia-wasm` is compiled by `wasm-pack` into
`npm/`, then `ts/prepare-package.mjs` compiles `ts/index.ts` on top and writes
`npm/package.json` from `ts/package.json`. The npm package exposes only the
high-level API: `@jolars/eunoia` (default entry), `@jolars/eunoia/svg`, and
`@jolars/eunoia/web`. The wasm-bindgen surface still ships (it is the runtime
backing `index.js`) but is intentionally *not* exported — the `exports` map
encapsulates it, so there is no public `/raw` entry point.

The default `.` entry is the `--target bundler` build: it `import`s the `.wasm`
module directly, so it needs a bundler (this is what `web/` relies on via vite).
The `./web` entry exists for bundler-less consumers (a plain `<script
type="module">`, Observable, raw-file CDNs): `ts/build-web.mjs` runs a separate
`wasm-pack --target web` build and esbuild-inlines the wasm (base64) into a
single self-contained `npm/web.js`, exposing an explicit async `init()` that
must be awaited once before `euler()`/`venn()`. It is built by `task build-web`
(and in `publish-npm.yml`), kept off the `build-wasm` hot path since the web app
uses the `.` entry. `./svg` is pure JS (no wasm) and works from a CDN as-is.

Edition 2024, MSRV pinned to **1.88.0** (in `rust-toolchain.toml` and `devenv.nix`).
Code uses the `module.rs` + `module/` layout — never `module/mod.rs`.

## Commands

Tasks are defined in `Taskfile.yml` (run via [`task`](https://taskfile.dev),
available in the devenv shell). The cargo commands underneath work directly too.

```sh
task dev            # fmt + check + test (+ corpus guardrail) + clippy + doc — pre-PR gate
cargo test          # fast default tests (workspace, ~1s; corpus guardrail excluded)
task test-quiet     # cargo test with RUST_LOG=off
task test-debug     # cargo test with RUST_LOG=debug
task test-slow      # cargo test --workspace -- --ignored  (slow regression/stochastic)
task lint           # clippy --workspace --all-targets --all-features -- -D warnings
task doc            # RUSTDOCFLAGS=-D warnings cargo doc + cargo test --doc (mirrors CI)
task coverage-open  # llvm-cov HTML report, opened in browser
```

- **Run a single test:** `cargo test -p eunoia <test_name_substring>`. For the
  whole workspace including the wasm crate: `cargo test --workspace`.
- **Always run `task test-slow` when changing fitting behavior** — many
  regression and stochastic fit-quality tests are `#[ignore]`d out of the
  default run, including the `corpus_quality` passes. `task dev` runs the
  `corpus_quality` subset (~16s) so the pre-PR gate still catches regressions;
  `task test-slow` adds the rest (issue89/issue28, stochastic, monte-carlo).
- Tests are fast despite heavy optimization math because
  `[profile.test.package.eunoia] opt-level = 3` (in root `Cargo.toml`)
  optimizes the crate under test while keeping `debug_assert!`s live. This drops
  a `cargo test -p eunoia --lib` run from ~32s to ~2-3s. Don't remove it.
- **CI fails the build on rustdoc warnings** (`RUSTDOCFLAGS=-D warnings`) —
  broken or private intra-doc links, etc. `task dev` and `task doc` mirror this,
  so run one of them before pushing doc changes; a public item must not
  intra-doc-link (`[`...`]`) to a private item.

### WASM / web / profiling

```sh
task build-wasm     # wasm-pack build + prepare-package.mjs → regenerates npm/
task web-dev        # vite dev server in web/

task flamegraph CASE=6set   # CASE = 3circle | 4set | 6set
task perf-record CASE=...   # perf record
task samply CASE=... ITERS=200
task asm FUNC=<path::to::fn>
```

`build-wasm` deliberately bundles the `prepare-package.mjs` step (not a separate
task) so `npm/` is always self-consistent — `web/` resolves the package via a
`file:../npm` link and a half-built `npm/` silently breaks its imports. The
profiling tasks use the custom `profiling` cargo profile (release + full debug
info). The web app is **statically prerendered** (`adapter-static`).

## Architecture: the fitting pipeline

The flow is **spec → preprocess → fit → layout**, and it is **shape-agnostic
until fit time**:

1. **`spec`** — `DiagramSpecBuilder` produces a `DiagramSpec` describing *what*
   to draw (set sizes + intersections), with no geometry. Input can be
   `InputType::Exclusive` or `Inclusive`; only the exclusive view is stored, the
   inclusive view is derived on demand. A `Combination` is a named set of sets.
   `.complement(area)` opts into "universe" fitting (a bounding container whose
   leftover area matches a target). `preprocess()` drops empty sets, computes set
   areas + pairwise relations, and converts combinations to bitmask form
   (`RegionMask`), yielding the internal `PreprocessedSpec`.

2. **`fitter`** — `Fitter<'a, S: DiagramShape = Circle>` picks the shape type via
   its generic parameter at fit time, not in the spec. `fit()` runs a two-phase
   pipeline, repeated `n_restarts` times (default **10**, mirroring eulerr) in
   parallel (rayon) keeping the lowest-loss result. **Small-smooth fast path:**
   for a smooth loss (`LossType::is_smooth`) on an analytic-gradient shape with
   `n_sets ≤ 3` and no complement, the restart count is auto-reduced (the Venn
   warm-start already solves restart 0, so extra restarts add nothing — see the
   `SMALL_SMOOTH_*` consts and `examples/restart_value`); the CMA-ES escape is
   kept (self-gating, still rescues adversarial small fits). An explicit
   `n_restarts(_)` opts out.
   - **Initial layout** (`fitter/initial_layout.rs`): multidimensional scaling
     (MDS) places fixed-size shapes. Solver selectable via `MdsSolver`
     (Levenberg-Marquardt by default). Initial positions drawn per-restart
     (`InitialSampler::Uniform` like eulerr, or `LatinHypercube`).
   - **Final layout** (`fitter/final_layout.rs`): refines all shape parameters to
     minimize the loss (default `LossType::SumSquared`, the scale-invariant
     `Σ(f−t)²/Σt²`). `Optimizer` variants — `LevenbergMarquardt`, `Lbfgs`,
     `NelderMead`, `Trf` (box-constrained LM), `CmaEsLm`, `CmaEsTrf` — are cycled
     across restarts via a pool. The default is **`CmaEsTrf`**: plain LM first,
     then *only if* the loss stays above `cmaes_fallback_threshold` (1e-3) a
     bounded CMA-ES global escape followed by a box-constrained `Trf`
     (trust-region-reflective) polish, keeping the lower loss (so easy specs pay
     no extra wall time). `CmaEsLm` is the same with an unbounded LM polish.
     Box bounds for the bounded solvers come from `optimizer_bounds_for` under
     two `BoundsEnvelope`s — a tight `CMAES` cage for the global escape and a
     wider `LOCAL` cage for the TRF polish. Every optimizer here and in the MDS
     init runs on the `basin` crate (nalgebra backend) — `basin` is the sole
     optimizer dependency.
   - For small set counts a **canonical Venn warm-start** seeds restart 0 (see
     `venn.rs` and the `VENN_SEED_MAX_SETS_*` consts).
   - `fitter/clustering.rs` + `packing.rs` handle disjoint sub-diagrams;
     `fitter/normalize.rs` post-processes the final layout.
   - Returns `Layout<S>` with the fitted shapes plus fit metrics (loss, fitted
     areas). `fit_initial_only()` skips refinement (diagnostics).

3. **`geometry`** — the shape system, built on composable traits in
   `geometry/traits.rs`: `Area`, `Centroid`, `Perimeter`, `BoundingBox`,
   `Distance`, `Closed` (spatial relations), and `DiagramShape` (composes them +
   exclusive-region computation + parameter conversion). Implementing
   `DiagramShape` is what makes a type fittable. Shapes implementing it:
   **Circle, Ellipse, Square, Rectangle** (`geometry/shapes/`). `Polygon` is for
   output/region extraction, not fitting. `geometry/projective/` holds conic and
   projective-line math used for ellipse intersections; `geometry/diagram.rs`
   defines `RegionMask` (a region is identified by which sets it belongs to).

4. **`loss`** — region-error loss functions, built from C¹-smooth surrogates
   (`smooth_abs` Huber-style, `smooth_max` logsumexp) so gradient-based
   optimizers behave. Selected via `LossType`.

5. **`plotting`** — turns a `Layout` into renderable output: region polygon
   extraction, clipping (`i_overlay`), and **label placement** via poles of
   inaccessibility (`polylabel-mini`). Always available. See also
   `LABEL_PLACEMENT_PLAN.md`.

6. **`venn`** — canonical n-set Venn diagrams independent of the fitter (circles
   for n≤3, Wilkinson/Edwards ellipse arrangements for n=4..5).

### Shape parameter encodings

Each `DiagramShape` has two encodings, bridged by `to_params`/`from_params`
(geometric) and `to_optimizer_params`/`from_optimizer_params` (optimizer).
External/FFI callers want the **geometric** encoding; the **optimizer** encoding
is internal to the fitter — the log-space transforms decouple area from aspect
ratio and give the LM/CMA-ES solvers a better-conditioned Hessian.

| Shape     | Geometric         | Optimizer encoding          |
| --------- | ----------------- | --------------------------- |
| Circle    | `[x, y, r]`       | identity                    |
| Ellipse   | `[x, y, a, b, φ]` | `[x, y, ln a, ln b, φ]`     |
| Square    | `[x, y, side]`    | identity                    |
| Rectangle | `[x, y, w, h]`    | `[x, y, ln(w·h), ln(w/h)]`  |

The container used for complement fitting carries 4 trailing optimizer params in
the same `[x, y, ln(w·h), ln(w/h)]` rectangle encoding.

### Cargo features (core crate)

- `parallel` — rayon-parallel restart loop. **Not** a default and intentionally
  off for wasm (no threads).
- `corpus` — exposes the shared test fixtures (`test_utils::corpus`) to example
  binaries and benches outside `cfg(test)`. Internal, not part of the public API.

## Conventions

- **Commits:** Conventional Commits with scopes used in this repo: `(ts)`,
  `(web)`, `(geometry)`, `(fitter)`, etc. (e.g. `feat(web): …`, `fix(fitter): …`).
- **Releases** are driven by [versionary](https://github.com/jolars/versionary)
  (`versionary.jsonc`): the Rust workspace and the `ts`/npm package are versioned
  separately. Pushing a `v*` tag triggers the crates and npm publish workflows.
- All code must pass `cargo fmt -- --check` and clippy with `-D warnings`.
  Pre-commit hooks (rustfmt, clippy, biome for JS/TS) run via devenv git-hooks.
- The `corpus_quality.rs` / `synthetic_groundtruth.rs` fit-quality tests are the
  guardrail against fitter regressions; `TODO.md` tracks surfaced fitter issues.
  `corpus_quality` is `#[ignore]`d (too slow for the default suite) — run it via
  `task dev` or `task test-slow`; `synthetic_groundtruth` stays in the default run.
