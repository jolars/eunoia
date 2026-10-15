# Python API documentation

Diplodocus extracts the Python API into a portable SQLite snapshot and exports
HTML and Markdown fragments. The website adds its existing layout, navigation,
theme, and metadata. The original Python documentation remains available for
longer guides.

`pnpm dev`, `pnpm check`, and `pnpm build` generate the reference first. The
initial run fetches the source revisions in `python-docs.lock.json` into
`.docs-sources/` and builds Diplodocus with its separate Rust 1.98.0 toolchain.
Eunoia's core and WASM builds keep Rust 1.88.0. On NixOS, the generator can use
the Diplodocus checkout's `devenv shell` when the active Cargo is older.

From the repository root, use local checkouts while developing:

```sh
task docs-python -- --python-source ../eunoia-py --diplodocus-source ../diplodocus
```

Paths resolve from the current working directory. From `web/`, use:

```sh
pnpm docs:python --python-source ../../eunoia-py --diplodocus-source ../../diplodocus
```

The environment variables `EUNOIA_PYTHON_SOURCE` and `DIPLODOCUS_SOURCE` select
the same overrides for development, checks, and builds. Use absolute paths when
setting them for several commands. `DIPLODOCUS_BIN` or `--diplodocus-bin` selects
an already built executable, and `DIPLODOCUS_CARGO` selects a Cargo executable.
An explicit executable skips the Diplodocus source checkout and build.
After editing sibling Python or Diplodocus sources while the dev server runs,
rerun `task docs-python` with the same overrides or restart `pnpm dev`; the
startup hooks do not watch sibling repositories.

To generate the website from an existing snapshot without a Python checkout:

```sh
pnpm docs:python --snapshot /path/to/documentation.sqlite --diplodocus-bin /path/to/diplodocus
```

Set `EUNOIA_PYTHON_DOCS_SNAPSHOT` to use that snapshot in subsequent website
commands, together with `DIPLODOCUS_BIN`. This mode needs neither source checkout
nor a Python runtime. Exported snapshots retain
the API documentation and source links, so generation does not import the
Python package or require its native extension.

Generated snapshots and caches live in `.diplodocus/python/`. Generated route
and asset directories carry ownership markers; the generator refuses to replace
author-written directories. Input contents determine cache reuse, including
uncommitted changes in explicit source overrides. Regeneration replaces owned
output so removed API pages do not survive as stale routes. None of these
artifacts are committed.

Run `pnpm test:python-docs`, `pnpm test:markdown`, `pnpm check`, and `pnpm build`
after changing the integration. The generator validates that local links in
published API bodies resolve to hosted pages or retained assets. The sitemap
discovers the generated concrete routes, and the Markdown build includes their
exported reference text.

The checks require the 15 public objects from the Python API index, both
five-overload fitting functions, result fields, plotting methods, inherited
member links, and matching HTML and Markdown anchors. The build also checks
the rendered pages, sitemap entries, and Markdown artifacts.
