# Package documentation

Diplodocus builds the entire Eunoia website. Python API sources remain in
`jolars/eunoia-py`; Julia docstrings, guides, gallery, and changelog remain in
`jolars/Eunoia.jl`. The site configuration declares both packages alongside the
project's authored Markdown and browser components.

`docs-sources.lock.json` records the Python and Julia source revisions, package
versions, and Julia runtime used in CI. Update a package's revision and version
together. Site commands use `diplodocus` on `PATH`. The devenv shell supplies it
from the upstream development branch, and CI uses the default release from
`jolars/setup-diplodocus`. Run `devenv update diplodocus` to refresh the local
input. Source builds also need `uv`, Python 3.11 or newer, and Julia.

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
The first run fetches pinned sources, builds the Python extension and installs
Sphinx in an isolated environment, and prepares Julia with CairoMakie. An
absolute `EUNOIA_CAPI_LIB` selects a local native library instead of Julia's
matching downloaded artifact.
CI uses `.github/actions/prepare-docs` to prepare these environments explicitly.
Julia preparation always instantiates and precompiles the documentation
environment and renders a small PNG, including when the environment marker is
cached. The Julia depot and site caches can restore independently, so compilation
and font setup must finish before Diplodocus starts its execution deadlines.

Checkouts and Julia's prepared source copy live under `.docs-sources/`.
Snapshots, environments, and extraction and rendering caches live under
`.diplodocus/`. Cache keys include source contents, generator bytes,
configuration, browser bundles, and environment metadata. Cached outputs are
verified before reuse.

## Local sources

From the repository root:

```sh
DIPLODOCUS_BIN=../diplodocus/target/debug/diplodocus \
  task docs-packages -- --python-source ../eunoia-py --julia-source ../Eunoia.jl
```

Paths resolve from the current working directory. From `web/`, sibling paths
start with `../../`. `EUNOIA_PYTHON_SOURCE` and `EUNOIA_JULIA_SOURCE`
provide the same source overrides. Prefer absolute environment
variable paths when running commands from several directories.

`DIPLODOCUS_BIN` or `--diplodocus-bin` selects an existing executable.
`EUNOIA_DOCS_PYTHON` and `EUNOIA_DOCS_JULIA` select package runtimes.
Restart the development command after changing overrides or environment
dependencies. It watches the prepared Julia
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
