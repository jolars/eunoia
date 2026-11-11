import { readFileSync } from "node:fs";
import { VERSION } from "./version.config.js";

export const legacyRoutes = JSON.parse(
  readFileSync(new URL("./legacy-routes.json", import.meta.url), "utf8"),
);

export const pageTitles = {
  "/": "Eunoia",
  "/packages/julia/": "Eunoia for Julia · Eunoia",
  "/packages/python/license": "Python license · Eunoia",
  "/packages/julia/license": "Julia license · Eunoia",
};

export function canonicalPath(artifact) {
  return `/${artifact.replace(/index\.html$/, "").replace(/\.html$/, "")}`;
}

const chapter = (text, path) => ({ text, link: `docs/${path}index.html` });

export function configuration(root, sources, revisions) {
  const redirects = {};
  for (const [route, { artifact }] of Object.entries(legacyRoutes)) {
    if (route === canonicalPath(artifact)) continue;
    redirects[route] = artifact;
    redirects[route.replace(/\/$/, "")] = artifact;
  }
  return {
    project: { name: "Eunoia" },
    presentation: {
      title: "Eunoia",
      description:
        "Area-proportional Euler and Venn diagrams, with bindings for Rust, JavaScript, Python, Julia, and R, and an interactive diagram builder.",
      "site-url": "https://eunoia.bz/",
      "canonical-url-style": "clean",
      redirects,
      logo: { repository: "eunoia", path: "web/static/logo.png" },
      "open-graph": {
        repository: "eunoia",
        path: "web/static/og.png",
        alt: "Eunoia: area-proportional Euler and Venn diagrams.",
      },
      home: {
        tagline: "Area-Proportional Euler and Venn Diagrams",
        description:
          "Fit set sizes and intersections with circles, ellipses, squares, and rectangles. A Rust engine shared by JavaScript, Python, Julia, and R.",
        guide: "docs/index.html",
        actions: [
          { label: "Try it in the browser", path: "app/index.html" },
          {
            label: "Read the docs",
            path: "docs/index.html",
            icon: "book-open",
          },
        ],
      },
      sections: [{ label: "App", path: "app/index.html" }],
      sidebar: {
        "/docs/": [
          chapter("Introduction", ""),
          {
            text: "Quickstart",
            items: ["Rust", "JavaScript", "Python", "Julia", "R"].map((name) =>
              chapter(name, `quickstart/${name.toLowerCase()}/`),
            ),
          },
          {
            text: "Concepts",
            items: [
              ["Fitter pipeline", "fitter-pipeline"],
              ["Goodness of fit", "goodness-of-fit"],
              ["Shapes", "shapes"],
              ["Label placement", "label-placement"],
              ["Glyphs", "glyphs"],
              ["Complement", "complement"],
            ].map(([name, slug]) => chapter(name, `concepts/${slug}/`)),
          },
          {
            text: "Reference",
            items: [
              chapter("Overview", "reference/"),
              chapter("JavaScript API", "reference/javascript/"),
              chapter("C ABI contract", "bindings/c-abi/"),
              { text: "Python API", link: "packages/python/index.html" },
              { text: "Julia API", link: "packages/julia/api.html" },
            ],
          },
          chapter("Citation", "cite/"),
        ],
      },
      footer: {
        "docs-version": `v${VERSION}`,
        copyright: "© 2025–2026 Johan Larsson",
        licenses: [
          {
            label: "MIT",
            url: "https://github.com/jolars/eunoia/blob/main/LICENSE-MIT",
          },
          {
            label: "Apache-2.0",
            url: "https://github.com/jolars/eunoia/blob/main/LICENSE-APACHE",
          },
        ],
        links: [
          { label: "GitHub", url: "https://github.com/jolars/eunoia" },
          { label: "crates.io", url: "https://crates.io/crates/eunoia" },
          { label: "docs.rs", url: "https://docs.rs/eunoia" },
          { label: "npm", url: "https://www.npmjs.com/package/@jolars/eunoia" },
        ],
        "built-with": true,
      },
    },
    repositories: [
      { id: "eunoia", path: root, url: "https://github.com/jolars/eunoia" },
      ...["python", "julia"].map((id) => ({
        id,
        path: sources[id],
        url: `https://github.com/jolars/${id === "python" ? "eunoia-py" : "Eunoia.jl"}`,
        revision: revisions[id],
      })),
    ],
    packages: ["python", "julia"].map((id) => ({
      id,
      name: id === "python" ? "Python" : "Julia",
      slug: id,
      ecosystem: id,
      repository: id,
      path: ".",
      "metadata-path": id === "python" ? "pyproject.toml" : "Project.toml",
      targets: [
        {
          id: "api",
          extractor: id,
          path: id === "python" ? "python/eunoia" : "src/Eunoia.jl",
          role: "public-api",
          ...(id === "julia"
            ? { julia: { extensions: ["ext/EunoiaMakieExt.jl"] } }
            : {}),
        },
      ],
    })),
    content: [
      {
        id: "site",
        owner: "project",
        repository: "eunoia",
        path: "web/content",
        mount: "",
        format: "gfm",
      },
      {
        id: "python-overview",
        owner: "python",
        repository: "eunoia",
        path: "web/python",
        mount: "",
        format: "gfm",
      },
      {
        id: "julia-manual",
        owner: "julia",
        repository: "julia",
        path: "docs/src",
        mount: "",
        format: "documenter",
        execution: {
          mode: "execute",
          engine: "julia",
          "julia-project": "docs/Project.toml",
        },
      },
    ],
    components: [
      {
        id: "eunoia",
        repository: "eunoia",
        path: "web/.widgets",
        module: "index.js",
        stylesheets: ["style.css"],
      },
    ],
  };
}
