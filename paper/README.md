# Paper

Edit `paper.md` and `paper.bib`. The figures under `images/` are checked in;
regenerate them with `task paper-figures` and `task paper-features` when needed.
Use `task paper-build` for the JOSS PDF.

## arXiv submission

Run `task arxiv` from the repository root. It requires Node.js, Docker with a
running daemon, and `tar`. Like `task paper-build`, it uses the
`openjournals/inara` image, which Docker downloads if needed.

The task exports fresh preprint LaTeX from the Markdown, includes only the
referenced figures, and runs PDFLaTeX without shell escape until cross-references
stabilize (up to five passes). Inara embeds the formatted bibliography in the
LaTeX, so the bundle does not need a separate bibliography file. Generated files
stay in a temporary directory until the build succeeds.

- Upload `paper/arxiv.tar.gz` to arXiv. It contains `main.tex` and the figures.
- Review `paper/arxiv.pdf` locally. This preview is separate from the archive.
- Select PDFLaTeX during submission and check arXiv's generated PDF before
  completing the submission.

See [arXiv's TeX submission instructions](https://info.arxiv.org/help/submit_tex.html)
for details. Both generated artifacts are ignored by Git.
