# Bundled fonts

These woff2 files (Latin subset) are vendored so diagram labels render with
the same glyphs in the browser and in every export (SVG / PNG / PDF).

| Family | Files | Role |
| ------ | ----- | ---- |
| **Arimo** | `arimo-{400,700}{,-italic}.woff2` | Sans-serif. Metric-compatible with Arial / Helvetica. |
| **Tinos** | `tinos-{400,700}{,-italic}.woff2` | Serif. Metric-compatible with Times New Roman / Times. |

Both are by Steve Matteson / Google ("Croscore" fonts), licensed under the
**Apache License 2.0**. Because they are metric-compatible with the PDF
base-14 fonts (Helvetica / Times), PDF export can fall back to those built-in
fonts without changing the layout — so we don't need to embed the TTFs.

Subsets pulled from Fontsource (`@fontsource/arimo`, `@fontsource/tinos`,
`files/*-latin-*.woff2`). To refresh, re-download the same files; the
`@font-face` registration lives in `src/lib/fonts.ts`.
