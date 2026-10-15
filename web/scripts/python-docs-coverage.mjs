const mount = "/docs/reference/python/";

export const publicApi = [
  "Circle",
  "Container",
  "Ellipse",
  "EulerFit",
  "EunoiaError",
  "Point",
  "Rectangle",
  "RotatedRectangle",
  "Square",
  "VennFit",
  "euler",
  "get_options",
  "options",
  "reset_options",
  "venn",
];
const fields = [
  "shapes",
  "original_values",
  "fitted_values",
  "residuals",
  "region_error",
  "diag_error",
  "stress",
  "loss",
  "container",
  "members",
  "plot_data",
];

function requireContent(condition, message) {
  if (!condition) throw new Error(`Python API coverage failed: ${message}`);
}

function anchors(content, markdown) {
  if (markdown) content = content.replace(/```[\s\S]*?```/g, "");
  const found = new Set(
    [...content.matchAll(/\bid="([^"]+)"/g)].map((match) => match[1]),
  );
  if (markdown) {
    for (const match of content.matchAll(/^#{1,6} (.+)$/gm)) {
      found.add(
        match[1]
          .replaceAll("`", "")
          .toLowerCase()
          .replace(/[^\w -]/g, "")
          .replaceAll(" ", "-"),
      );
    }
  }
  return found;
}

function links(content, markdown) {
  if (markdown) {
    content = content.replace(/```[\s\S]*?```/g, "");
    return [...content.matchAll(/\]\((?:<([^>]+)>|([^\s)]+))\)/g)].map(
      (match) => match[1] ?? match[2],
    );
  }
  return [...content.matchAll(/\bhref="([^"]+)"/g)].map((match) =>
    match[1].replaceAll("&amp;", "&"),
  );
}

export function validateApiCoverage(pages) {
  const names = pages
    .filter((page) => page.path)
    .map((page) => page.path)
    .sort();
  requireContent(
    JSON.stringify(names) === JSON.stringify([...publicApi].sort()),
    "published objects differ from the 15 APIs in eunoia-py's docs/api.md",
  );
  const byName = new Map(pages.map((page) => [page.path, page]));
  for (const name of ["euler", "venn"]) {
    const page = byName.get(name);
    for (const format of ["html", "markdown"]) {
      requireContent(
        [...anchors(page[format], format === "markdown")].filter((anchor) =>
          anchor.startsWith("overload-"),
        ).length === 5,
        `${name} must retain five overloads in ${format}`,
      );
      requireContent(
        page[format].includes("Parameters") && page[format].includes("Returns"),
        `${name} must retain parameter and return documentation in ${format}`,
      );
    }
  }
  for (const format of ["html", "markdown"]) {
    const result = byName.get("EulerFit")[format];
    const ids = anchors(result, format === "markdown");
    for (const member of [...fields, "plot", "plot_plotly"]) {
      requireContent(
        ids.has(member),
        `EulerFit.${member} is missing from ${format}`,
      );
      requireContent(
        links(byName.get("VennFit")[format], format === "markdown").includes(
          `${mount}EulerFit/#${member}`,
        ),
        `VennFit.${member} must link to its inherited documentation in ${format}`,
      );
    }
    const targets = new Map(
      pages.map((page) => [
        mount + (page.path ? `${page.path}/` : ""),
        anchors(page[format], format === "markdown"),
      ]),
    );
    for (const page of pages) {
      const current = mount + (page.path ? `${page.path}/` : "");
      for (const link of links(page[format], format === "markdown")) {
        const url = new URL(link, `https://eunoia.bz${current}`);
        if (
          url.origin !== "https://eunoia.bz" ||
          !url.pathname.startsWith(mount)
        )
          continue;
        const destination = targets.get(url.pathname);
        requireContent(
          destination,
          `${current} links to an unpublished ${format} page: ${link}`,
        );
        requireContent(
          !url.hash || destination.has(decodeURIComponent(url.hash.slice(1))),
          `${current} links to a missing ${format} anchor: ${link}`,
        );
      }
    }
  }
}
