---
paths:
  - "web/**"
---

# Docs site (`web/`)

Rules for the SvelteKit docs/marketing site under `web/`. The app is statically
prerendered (`adapter-static`); `static/` files are served verbatim.

## Page format

Narrative docs are `.svx` (mdsvex) pages under `web/src/routes/docs/**`. A
page's title is its first `# H1` — there is no YAML frontmatter. Pages may open
with a `<script>` block for interactive Svelte components.

## Site-root indexes — maintained differently, don't conflate them

- **`web/src/routes/sitemap.xml/+server.ts` is auto-generated.** It globs every
  `+page.{svelte,svx}`, so new pages appear with no edit. Any future
  `llms-full.txt` should be generated the same way, not hand-maintained.

- **`web/static/llms.txt` is hand-curated** ([llmstxt.org](https://llmstxt.org)
  format) and served verbatim. It does **not** update automatically. When you
  **add, remove, rename, or re-scope a docs page**, update `llms.txt` by hand:
  fix the affected link and its one-line description, and keep it grouped under
  the right section (Getting started / Concepts / Reference / Optional). It also
  carries off-site links (docs.rs, npm, PyPI, Eunoia.jl, eulerr, GitHub) that no
  glob can discover. `web/static/robots.txt` points at it.
