# Eunoia website

Diplodocus generates the homepage, documentation, Python and Julia references,
citation page, search, and navigation. Svelte provides the browser-only diagram
builder and examples. Vite builds those components into one portable bundle
with their worker, WASM, and fonts.

## Development

Enter the repository's devenv shell and build the npm package first:

```sh
task build-wasm
cd web
pnpm install --frozen-lockfile
pnpm dev
```

The development command builds the site, watches Markdown and browser sources,
and serves the local Cloudflare Worker with live reload on port 5173. Use
`pnpm dev --port 8000` to choose another port. Restart after changing site
configuration or source pins. On NixOS, devenv runs the lockfile's Workerd
with the Nix dynamic loader. Local package overrides are described in
[PACKAGE-DOCS.md](PACKAGE-DOCS.md).

```sh
pnpm check
pnpm build
pnpm test:docs
pnpm test:markdown
pnpm test:members
pnpm preview --port 8787
```

Build and test commands should run inside `compute run --` when using the
shared development machine. `pnpm preview` serves the current `dist/` without
rebuilding. See [Cloudflare hosting](cloudflare/README.md) for deployment.

## Sources

- `content/` contains Markdown pages at their published directory paths.
- `python/` supplies the Python package's API index.
- `site.config.mjs` declares Diplodocus presentation, navigation, packages,
  browser components, and redirects.
- `src/lib/` contains the existing diagram rendering, controls, export, and
  fitting worker code.
- `src/widgets/index.ts` implements Diplodocus's mount and cleanup interface.
  Component block IDs select configurations from `src/widgets/examples.json`.
- `static/` contains public assets and the hand-curated `llms.txt`.
- `legacy-routes.json` records published routes and anchors from the old site.

Keep existing explicit heading IDs. For a new interactive example, add a
configuration to `examples.json` and use its key as the block ID:

```markdown
::: {.component name="eunoia" #example-id}

A useful static description or image remains available without JavaScript.

:::
```

All examples use one declared bundle so navigation reuses cached WASM and fonts.
The mount adapter reads the authored block's ID from its containing element.
Widget CSS is scoped during the Vite build, and the theme adapter follows
Diplodocus's theme without changing its styles.

Use an explicit `{#heading-id}` on headings targeted by local fragment links.
Keep list items and quoted paragraphs on one source line; the pinned Markdown
adapter does not yet handle every continuation prefix. Ordinary paragraphs and
code blocks can retain their normal wrapping. The site build rejects unsupported
rendered content and broken links.

## Generated output

`scripts/site.mjs` prepares package sources, builds browser components, and
extracts a unified SQLite snapshot. Diplodocus renders that snapshot into a
cached native site. `scripts/site-output.mjs` assembles public assets, version
tokens, page descriptions, icons, and legacy anchor aliases in a staging
directory. Content, links, references, and compatibility checks must pass before
that directory replaces `dist/`; a failed build preserves the previous site.

Do not edit `.widgets/`, `dist/`, `.docs-sources/`, or `.diplodocus/` by hand.
Diplodocus owns the sitemap, robots file, Markdown exports, and Markdown Worker.
