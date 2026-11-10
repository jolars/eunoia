export const juliaPublicApi = [
  "euler",
  "venn",
  "version",
  "eunoiaplot",
  "eunoiaplot!",
  "place_labels",
  "label_boxes",
  "place_set_labels",
  "place_glyphs",
  "place_glyph_boxes",
  "EulerFit",
  "VennFit",
  "Circle",
  "Ellipse",
  "Square",
  "Rectangle",
  "RotatedRectangle",
  "Point",
  "Container",
  "BoundingBox",
  "LabelPlacement",
  "GlyphPlacements",
  "GlyphBoxPlacements",
];

export function validateJuliaCoverage(pages) {
  const byPath = new Map(pages.map((page) => [page.path, page]));
  const requireContent = (condition, message) => {
    if (!condition)
      throw new Error(`Julia documentation coverage failed: ${message}`);
  };
  for (const name of ["", "api", "gallery", "changelog", ...juliaPublicApi]) {
    requireContent(byPath.has(name), `Missing page: ${name || "overview"}`);
  }
  for (const page of pages) {
    for (const format of ["html", "markdown"]) {
      requireContent(
        !page[format].includes("Unsupported content"),
        `${page.path}: unsupported ${format}`,
      );
      requireContent(
        !page[format].includes("@raw html"),
        `${page.path}: raw disclosure fences in ${format}`,
      );
      requireContent(
        !/```@(?:example|docs|autodocs|index)\b/.test(page[format]),
        `${page.path}: unprocessed Documenter directive`,
      );
    }
  }
  for (const name of ["EulerFit", "VennFit"]) {
    for (const field of [
      "shapes",
      "original_values",
      "fitted_values",
      "residuals",
      "region_error",
      "diag_error",
      "stress",
      "loss",
      "iterations",
      "container",
      "plot_data",
    ]) {
      requireContent(
        byPath.get(name).html.includes(`id="field-${field}"`),
        `${name}.${field} has no field anchor`,
      );
      requireContent(
        byPath.get(name).markdown.includes(`id="field-${field}"`),
        `${name}.${field} has no Markdown anchor`,
      );
    }
  }
  const overview = byPath.get("").html;
  requireContent(
    /<details\b[^>]*><summary>BibLaTeX<\/summary>/.test(overview),
    "Missing citation disclosure",
  );
  requireContent(
    (overview.match(/<img\b/g) ?? []).length === 1,
    "Missing homepage figure",
  );
  requireContent(
    (byPath.get("gallery").html.match(/<img\b/g) ?? []).length === 29,
    "The gallery must retain 29 figures",
  );
  requireContent(
    byPath.get("eunoiaplot").html.includes("GridPosition"),
    "Missing Makie plotting implementation",
  );
  requireContent(
    byPath.get("eunoiaplot!").html.includes("label_placement"),
    "Missing Makie mutating implementation",
  );
}
