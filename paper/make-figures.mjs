// Generates the figures for the JOSS paper from the in-repo npm package.
//
// Usage: node paper/make-figures.mjs   (from the repository root, after
// `task build-wasm`). Writes SVGs to paper/images/ and converts them to PDF
// with inkscape (text converted to paths).
//
// Sizing model. `toSvg` emits an SVG with only a `viewBox` (in layout units)
// and no physical width/height, and each fitted diagram lives at a different
// coordinate scale. Exported that way, the PDFs come out at wildly different,
// mostly tiny intrinsic sizes, which is why the document then has to override
// every image width by hand. Instead we bake a physical size (mm) into each
// figure and a fixed physical label height (mm) into every plot, so the figures
// render at sensible, consistent sizes with identical fonts and need no
// `width=` in paper.md. Change PLOT_MM/LABEL_MM here to rescale everything.

import { execFileSync } from "node:child_process";
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const { euler, venn } = await import("../npm/index.js");
const { toSvg } = await import("../npm/svg.js");

const outDir = join(dirname(fileURLToPath(import.meta.url)), "images");
mkdirSync(outDir, { recursive: true });

// Physical widths (mm): a standalone plot, one cell of the shape grid, and the
// label height. Label height is fixed and independent of the plot width, so the
// grid cells can be smaller than a standalone plot while the labels stay the
// same physical size everywhere.
const PLOT_MM = 70;
const CELL_MM = 55;
const LABEL_MM = 3;

// Headline example: a real, named four-set specification (eulerr's `gene_sets`
// clinical markers), which shows off fitting beyond three sets.
const geneSets = {
  sets: {
    SE: 13,
    Treat: 28,
    "Anti-CCP": 101,
    DAS28: 91,
    "SE&Treat": 1,
    "SE&DAS28": 14,
    "Treat&Anti-CCP": 6,
    "SE&Anti-CCP&DAS28": 1,
  },
  order: ["SE", "Treat", "Anti-CCP", "DAS28"],
};

// Shape comparison: a varied three-set with moderate overlaps that renders
// cleanly across all four shape families. Placeholder to revisit with a real
// combination once one that fits the non-ellipse shapes well is chosen.
const shapeExample = {
  sets: {
    A: 20,
    B: 14,
    C: 18,
    "A&B": 6,
    "A&C": 5,
    "B&C": 4,
    "A&B&C": 2,
  },
  order: ["A", "B", "C"],
};

const svgOptions = {
  fontFamily: "DejaVu Sans, sans-serif",
};

function fit(sets, shape) {
  return euler({
    sets,
    shape,
    output: "regions",
    inputType: "exclusive",
    seed: 1,
  });
}

// viewBox width/height of a serialized SVG (layout units).
function viewBox(svg) {
  const m = svg.match(/viewBox="([^"]+)"/);
  const [, , w, h] = m[1].trim().split(/\s+/).map(Number);
  return { w, h };
}

// Inject a physical size onto the SVG root so the PDF exports at that size.
function withSize(svg, widthMm, heightMm) {
  return svg.replace(
    '<svg xmlns="http://www.w3.org/2000/svg"',
    `<svg xmlns="http://www.w3.org/2000/svg" width="${widthMm}mm" height="${heightMm}mm"`,
  );
}

// Serialize a layout whose label renders at LABEL_MM once the plot is shown
// `plotWmm` wide. `toSvg` puts font sizes in layout units, so the physical font
// is `labelSize * plotWmm / viewBoxWidth`; solving for LABEL_MM fixes it.
function serialize(layout, plotWmm, extra = {}) {
  const opts = { ...svgOptions, ...extra };
  const labelSize = (LABEL_MM * viewBox(toSvg(layout, opts)).w) / plotWmm;
  return toSvg(layout, { ...opts, labelSize });
}

// A standalone plot: PLOT_MM wide, physical size baked in.
function plotFigure(layout, extra = {}) {
  const svg = serialize(layout, PLOT_MM, extra);
  const { w, h } = viewBox(svg);
  return withSize(svg, PLOT_MM, (PLOT_MM * h) / w);
}

// Nest a cell SVG at (x, y) sized `w` x `h` in grid units, preserving its root
// viewBox so it scales into that box.
function nest(svg, x, y, w, h) {
  return svg.replace(
    '<svg xmlns="http://www.w3.org/2000/svg"',
    `<svg x="${x}" y="${y}" width="${w}" height="${h}"`,
  );
}

// Lay cells out in a `cols`-wide grid. Cells are width-fit (each nested box
// matches its plot's aspect ratio) so the horizontal scale, and thus the label
// font, is identical in every cell. The grid is given a physical size of
// `cols * CELL_MM` wide, so labels render at LABEL_MM just like standalone
// plots. `cell` is the cell width in grid units; mm-per-unit ties them together.
function grid(cells, cols, { cell = 100, band = 12 } = {}) {
  const perMm = cell / CELL_MM; // grid units per mm
  const aspects = cells.map((c) => {
    const { w, h } = viewBox(c.svg);
    return h / w;
  });
  const cellH = cell * Math.max(...aspects);
  const rowH = cellH + band;
  const captionFont = LABEL_MM * perMm;
  const rows = Math.ceil(cells.length / cols);
  const body = cells
    .map(({ svg, label }, i) => {
      const x = (i % cols) * cell;
      const y = Math.floor(i / cols) * rowH;
      const h = cell * aspects[i];
      // Title band sits above the plot; the plot is nested below it.
      const caption = label
        ? `<text x="${x + cell / 2}" y="${y + band * 0.7}" ` +
          `text-anchor="middle" font-family="DejaVu Sans, sans-serif" ` +
          `font-weight="bold" font-size="${captionFont}">${label}</text>`
        : "";
      return caption + nest(svg, x, y + band, cell, h);
    })
    .join("\n");
  const vbW = cols * cell;
  const vbH = rows * rowH;
  return withSize(
    `<svg xmlns="http://www.w3.org/2000/svg" ` +
      `viewBox="0 0 ${vbW} ${vbH}">\n${body}\n</svg>`,
    vbW / perMm,
    vbH / perMm,
  );
}

function write(name, svg) {
  const svgPath = join(outDir, `${name}.svg`);
  writeFileSync(svgPath, svg);
  execFileSync("inkscape", [
    svgPath,
    "--export-type=pdf",
    "--export-text-to-path",
    "-o",
    join(outDir, `${name}.pdf`),
  ]);
  console.log(`wrote ${name}.svg + ${name}.pdf`);
}

// Figure 1: the real four-set example, ellipse fit with quantities shown.
const ellipseFit = fit(geneSets.sets, "ellipse");
write(
  "euler_4set",
  plotFigure(ellipseFit, {
    setOrder: geneSets.order,
    showCounts: true,
    formatCount: (v) => String(Math.round(v)),
  }),
);
console.log(
  `  gene_sets ellipse: diagError = ${ellipseFit.metrics.diagError.toExponential(2)}`,
);

// Figure 2: the symmetric three-set under all four shape families, in a 2x2 grid.
const shapes = ["circle", "ellipse", "square", "rectangle"];
write(
  "shape_families",
  grid(
    shapes.map((shape) => {
      const layout = fit(shapeExample.sets, shape);
      console.log(
        `  shapeExample ${shape}: diagError = ${layout.metrics.diagError.toExponential(2)}`,
      );
      return {
        svg: serialize(layout, CELL_MM, { setOrder: shapeExample.order }),
        label: shape,
      };
    }),
    2,
  ),
);

// Figure 3: canonical five-set Venn diagram (ellipses).
write("venn5", plotFigure(venn({ n: 5, shape: "ellipse", output: "regions" })));
