# Package documentation

Diplodocus builds the entire Eunoia website. Python API sources remain in
`jolars/eunoia-py`; Julia docstrings, guides, gallery, and changelog remain in
`jolars/Eunoia.jl`. The site configuration declares both packages alongside the
project's authored Markdown and browser components.

`docs-sources.lock.json` pins source revisions, package versions, Diplodocus, and
Julia. Update a package's revision and version together. The toolchain needs
`uv`, Python 3.11 or newer, Julia, and Rust 1.98.0 for Diplodocus. The core and
WASM builds continue to use Rust 1.88.0. The generator builder can enter the
Diplodocus checkout's devenv shell when the active Cargo is older.

## Build from sources

From the repository's devenv shell:

```sh
task build-wasm
task docs-prepare
task docs-packages
cd web
pnpm test:docs
pnpm test:markdown
```

`pnpm check`, `pnpm build`, and `pnpm dev` prepare package environments as needed.
The first run fetches pinned sources, builds the Python extension in an isolated
environment, and prepares Julia with CairoMakie. An absolute `EUNOIA_CAPI_LIB`
selects a local native library instead of Julia's matching downloaded artifact.
CI uses `.github/actions/prepare-docs` to prepare these environments explicitly.

Checkouts and Julia's prepared source copy live under `.docs-sources/`.
Snapshots, environments, and extraction and rendering caches live under
`.diplodocus/`. Cache keys include source contents, generator bytes,
configuration, browser bundles, and environment metadata. Cached outputs are
verified before reuse.

## Local sources

From the repository root:

```sh
task docs-packages -- --python-source ../eunoia-py --julia-source ../Eunoia.jl --diplodocus-source ../diplodocus
```

Paths resolve from the current working directory. From `web/`, sibling paths
start with `../../`. `EUNOIA_PYTHON_SOURCE`, `EUNOIA_JULIA_SOURCE`, and
`DIPLODOCUS_SOURCE` provide the same overrides. Prefer absolute environment
variable paths when running commands from several directories.

`DIPLODOCUS_BIN` selects an existing executable and skips its checkout and build.
`DIPLODOCUS_CARGO` selects another Cargo executable. `EUNOIA_DOCS_PYTHON` and
`EUNOIA_DOCS_JULIA` select package runtimes. Restart the development command after
changing overrides or environment dependencies. It watches the prepared Julia
sources; edits to the original Julia checkout require restarting preparation.

## Portable snapshots

The site now uses one snapshot containing all authored pages, both packages,
executed figures, branding, and browser bundles:

```sh
task docs-extract
EUNOIA_DOCS_SNAPSHOT="$PWD/.diplodocus/site/documentation.sqlite" \
  DIPLODOCUS_BIN=/path/to/diplodocus task docs-packages
```

`--snapshot /path/to/documentation.sqlite` supplies the same override. This mode
skips package checkouts, environment preparation, extraction, and the Vite build.
It requires neither Python nor Julia. The generator receives
`--allow-components` because this is a trusted snapshot of our own browser code.
The Eunoia checkout still supplies its publication metadata and public assets.

The former `EUNOIA_PYTHON_DOCS_SNAPSHOT` and `EUNOIA_JULIA_DOCS_SNAPSHOT` settings
belonged to the retired fragment importer. Re-extract those package-only
snapshots as a complete site snapshot. `task docs-python` and `task docs-julia`
remain compatibility aliases for the complete site build.

## Compatibility

Native package pages live under `/packages/python/` and `/packages/julia/`.
Diplodocus generates permanent redirects from the old `/docs/reference/`
routes. The publication step retains the old Julia heading and dispatch
fragments and the `/_diplodocus/julia/redirects.json` contract used by the
Eunoia.jl redirect workflow. The site checks these routes and fragments before
publishing local build output.
