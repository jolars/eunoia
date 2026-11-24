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

Wait for Wrangler's `Ready on ...` message, then press `b` to open a browser or
`x` to stop the server. If Workerd reports a dynamically linked executable error
on NixOS, refresh the environment by running `devenv shell -- task web-dev` from
the repository root. The dev command exits if Wrangler fails, closing its file
watchers.

```sh
pnpm check
pnpm build
pnpm test:docs
pnpm test:markdown
pnpm test:members
pnpm preview --port 8787
```

`pnpm preview` serves the current `dist/` without
rebuilding. See [Cloudflare hosting](cloudflare/README.md) for deployment.

## Sources

- `content/` contains Markdown pages at their published directory paths.
- `python/` supplies the Python package's API index.
- `site.config.mjs` declares Diplodocus presentation, navigation, packages,
  browser components, and redirects.
- `src/lib/` contains the existing diagram rendering, controls, export, and
  fitting worker code.
- `src/widgets/index.ts` implements Diplodocus's mount and cleanup interface.
  Component blocks pass their settings through Diplodocus's `props` attribute.
- `static/` contains public assets and the hand-curated `llms.txt`.
- `legacy-routes.json` records published routes and anchors from the old site.

Keep existing explicit heading IDs. Give each interactive example a unique ID
and pass its settings as a single-line JSON object in the block:

```markdown
::: {.component name="eunoia" #example-id props='{"component":"diagram","props":{"sets":{"A":5,"B":3,"A&B":1}}}'}

A useful static description or image remains available without JavaScript.

:::
```

All examples use one declared bundle so navigation reuses cached WASM and fonts.
The mount adapter uses `component` to select a widget and passes the nested
`props` to it. Escape apostrophes in the JSON as `\u0027`.
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
