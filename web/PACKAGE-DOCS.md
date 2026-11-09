# Package documentation

Diplodocus imports the pinned Python API and the Julia guides, gallery, and API
into this SvelteKit site. Python sources remain in `jolars/eunoia-py`. Julia
docstrings, `docs/src/`, and `CHANGELOG.md` remain in `jolars/Eunoia.jl`.
The website supplies navigation, search, metadata, and Markdown negotiation.

The source revisions and Julia runtime are pinned in `docs-sources.lock.json`.
Update the full revision and package version together when advancing a package.
The Diplodocus pin includes Documenter disclosure support and uses snapshot IR
version 13. Older snapshots must be extracted again.

## Building

Enter the project's devenv shell on NixOS. The build needs `uv`, Python 3.11 or
newer, Julia, and the separate Rust 1.98.0 toolchain used by Diplodocus. Eunoia's
core and WASM builds continue to use Rust 1.88.0. The source builder can enter
the Diplodocus checkout's devenv shell when the active Cargo is older.

```sh
task docs-prepare
task docs-packages
cd web
pnpm check
pnpm build
```

`pnpm dev`, `pnpm check`, and `pnpm build` also generate the documentation
before starting SvelteKit. The first run fetches the sources, builds the Python
extension in an isolated environment, and prepares Julia and CairoMakie.
Julia's matching native artifact is downloaded if needed. An absolute
`EUNOIA_CAPI_LIB` path selects a locally built native library instead.

Source checkouts live in `.docs-sources/`. Julia uses a prepared source copy
there, including the generated changelog page, so dependency preparation does
not alter the original guides. Snapshots, virtual environments, and caches live
in `.diplodocus/`. The generator checks input contents and generated output
before reusing its cache. Each language's owned routes and assets are replaced
together, with rollback on failure.

CI uses `.github/actions/prepare-docs` to install the language runtimes and
prepare the package environments. Diplodocus runs after preparation; the
generator itself does not install package dependencies.

## Local sources

From the repository root:

```sh
task docs-packages -- --python-source ../eunoia-py --julia-source ../Eunoia.jl --diplodocus-source ../diplodocus
task docs-julia -- --julia-source ../Eunoia.jl
```

Paths resolve from the current working directory. From `web/`, sibling paths
start with `../../`. `EUNOIA_PYTHON_SOURCE`, `EUNOIA_JULIA_SOURCE`, and
`DIPLODOCUS_SOURCE` provide the same overrides. Prefer absolute environment
variable paths when running commands from several directories.

`DIPLODOCUS_BIN` selects an existing executable and skips its source checkout
and build. `DIPLODOCUS_CARGO` selects another Cargo executable.
`EUNOIA_DOCS_PYTHON` and `EUNOIA_DOCS_JULIA` select the package runtimes. After
editing sibling sources while the dev server runs, regenerate or restart the
server; startup hooks do not watch sibling repositories.

## Portable snapshots

```sh
task docs-python -- --snapshot /path/to/python.sqlite --diplodocus-bin /path/to/diplodocus
task docs-julia -- --snapshot /path/to/julia.sqlite --diplodocus-bin /path/to/diplodocus
```

For subsequent website commands, set `EUNOIA_PYTHON_DOCS_SNAPSHOT` and
`EUNOIA_JULIA_DOCS_SNAPSHOT` together with `DIPLODOCUS_BIN`. This mode needs
neither package checkout nor Python or Julia. Figures and source links come
from the snapshots.

## Compatibility and validation

Julia documentation lives under `/docs/reference/julia/`. The importer retains
Documenter heading and method-signature anchors recorded in
`julia-docs-legacy.json`. It includes the supplied Makie extension and groups
fields with their types. Private items needed by public links remain accessible
but are excluded from navigation and search.

Run `pnpm test:docs`, `pnpm test:markdown`, `pnpm check`, and `pnpm build` after
changing the integration. Checks cover Python overloads and inherited members,
all 23 Julia exports, result fields, Makie methods, the 30 Julia figures,
disclosures, legacy anchors, links, the sitemap, and Markdown negotiation.
Generated pages export `reference.md`, which the Markdown build uses
automatically. Sitemap discovery also remains automatic.

The Julia site emits `/_diplodocus/julia/redirects.json`. After deploying this
site, run Eunoia.jl's manual **Redirect current documentation** workflow. It
verifies all destinations before changing the old root, `stable`, and `dev`
entry points. Redirects preserve queries and fragments. Historical versions and
existing HTML and figure assets remain available; the old CI no longer
publishes over those redirects.
