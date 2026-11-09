# Python API documentation

The Python API uses the shared package documentation importer described in
[PACKAGE-DOCS.md](PACKAGE-DOCS.md). The source pin is now in
`docs-sources.lock.json`, alongside Julia and Diplodocus.

Existing `task docs-python`, `pnpm docs:python`, source overrides, and
`EUNOIA_PYTHON_DOCS_SNAPSHOT` commands remain available. Run `pnpm test:docs`
for the combined Python and Julia checks.
