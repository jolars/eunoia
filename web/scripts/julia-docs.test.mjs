import assert from "node:assert/strict";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { test } from "node:test";
import {
  apiDocuments,
  configuration,
  documentMap,
  juliaAliases,
  publishFragments,
} from "./package-docs.mjs";

const mount = "/docs/reference/julia/";
function document(path, name = null, visible = true) {
  return {
    path: `packages/julia/${path}.html`,
    markdown_path: `packages/julia/${path}.md`,
    public_name: name,
    title: name ?? path,
    visible,
  };
}
function manifest() {
  return {
    format_version: 2,
    documents: [
      { path: "index.html", public_name: null, visible: true },
      ...["index", "api", "gallery", "CHANGELOG"].map((path) => document(path)),
      document("reference/plot", "Eunoia.eunoiaplot!"),
      document("reference/fit", "Eunoia.EulerFit"),
      document("reference/module", "Eunoia"),
      document("reference/abstract", "Eunoia.AbstractEulerFit", false),
      document("reference/private", "Eunoia._unused", false),
    ],
  };
}

test("Julia maps guides, mutating functions, modules, and grouped types to stable routes", () => {
  const input = manifest();
  assert.deepEqual(
    apiDocuments(input, "julia").map((document) => document.public_name),
    ["Eunoia.EulerFit", "Eunoia.eunoiaplot!"],
  );
  const urls = documentMap(input, "julia");
  assert.equal(urls["packages/julia/index.html"], mount);
  assert.equal(urls["packages/julia/CHANGELOG.html"], `${mount}changelog/`);
  assert.equal(
    urls["packages/julia/reference/plot.html"],
    `${mount}eunoiaplot!/`,
  );
  assert.equal(urls["packages/julia/reference/fit.html"], `${mount}EulerFit/`);
  assert.equal(urls["packages/julia/reference/module.html"], `${mount}module/`);
  assert.equal(
    urls["packages/julia/reference/abstract.html"],
    `${mount}internal/AbstractEulerFit/`,
  );
  input.documents.push(document("other", "Eunoia.EulerFit"));
  assert.throws(
    () => apiDocuments(input, "julia"),
    /Duplicate public Julia name/,
  );
  input.documents = [document("unsafe", "Eunoia.../escape")];
  assert.throws(() => documentMap(input, "julia"));
});

test("Documenter aliases preserve heading positions and method signature fragments", () => {
  const legacy = {
    pages: { gallery: { anchors: ["A-Gallery", "An-Old-Heading"] } },
    api: { "Eunoia.euler-Tuple{AbstractDict}": "euler" },
  };
  const gallery = juliaAliases(
    {
      path: "gallery",
      html: '<h1 id="a-gallery">A Gallery</h1><p>Introduction.</p><h2 id="an-old-heading">An Old Heading</h2>',
      markdown:
        '<a id="a-gallery"></a>\n# A Gallery\n\nIntroduction.\n\n<a id="an-old-heading"></a>\n## An Old Heading\n',
    },
    legacy,
  );
  assert.match(
    gallery.html,
    /Introduction\.<\/p><a id="An-Old-Heading"><\/a><h2/,
  );
  assert.match(gallery.markdown, /Introduction\.\n\n<a id="An-Old-Heading">/);
  juliaAliases(gallery, legacy);
  assert.equal(gallery.html.match(/id="An-Old-Heading"/g).length, 1);
  const api = juliaAliases(
    {
      path: "euler",
      document: { public_name: "Eunoia.euler" },
      html: "<h1>Eunoia.euler</h1>",
      markdown: "# Eunoia.euler\n",
    },
    legacy,
  );
  for (const content of [api.html, api.markdown])
    assert.ok(content.includes('id="Eunoia.euler-Tuple{AbstractDict}"'));
});

test("Julia publication retains needed private types, figures, and separate Python output", async (context) => {
  const root = await mkdtemp(join(tmpdir(), "eunoia-julia-docs-"));
  context.after(() => rm(root, { recursive: true, force: true }));
  const fragments = join(root, "fragments");
  await mkdir(join(fragments, "assets"), { recursive: true });
  const python = join(root, "src/routes/docs/reference/python/note.md");
  await mkdir(dirname(python), { recursive: true });
  await writeFile(python, "An existing Python document.\n");
  const input = manifest();
  for (const entry of input.documents.filter((entry) => entry.markdown_path)) {
    const path = join(fragments, entry.path);
    await mkdir(dirname(path), { recursive: true });
    let html = `<h1>${entry.title}</h1>`;
    if (entry.public_name === "Eunoia.eunoiaplot!")
      html += `<a href="${mount}internal/AbstractEulerFit/#type">Fit</a><img src="/_diplodocus/julia/figure.png">`;
    if (entry.public_name === "Eunoia.AbstractEulerFit")
      html += '<h2 id="type">Type</h2>';
    await writeFile(path, html);
    await writeFile(join(fragments, entry.markdown_path), `# ${entry.title}\n`);
  }
  const bytes = new Uint8Array([137, 80, 78, 71]);
  await writeFile(join(fragments, "assets/figure.png"), bytes);
  await writeFile(
    join(fragments, "assets/search.json"),
    JSON.stringify([
      { path: `${mount}eunoiaplot!/` },
      { path: `${mount}internal/AbstractEulerFit/` },
      { path: `${mount}internal/_unused/` },
    ]),
  );
  assert.equal(
    await publishFragments({
      root,
      fragments,
      manifest: input,
      version: "1.0.1",
      language: "julia",
    }),
    8,
  );
  const routes = join(root, "src/routes/docs/reference/julia");
  assert.match(
    await readFile(join(routes, "eunoiaplot!/+page.svelte"), "utf8"),
    /Eunoia.eunoiaplot!/,
  );
  assert.match(
    await readFile(
      join(routes, "internal/AbstractEulerFit/+page.svelte"),
      "utf8",
    ),
    /name="robots" content="noindex"/,
  );
  await assert.rejects(
    readFile(join(routes, "internal/_unused/+page.svelte")),
    { code: "ENOENT" },
  );
  assert.equal(
    await readFile(python, "utf8"),
    "An existing Python document.\n",
  );
  assert.deepEqual(
    new Uint8Array(
      await readFile(join(root, "static/_diplodocus/julia/figure.png")),
    ),
    bytes,
  );
  assert.deepEqual(
    JSON.parse(
      await readFile(
        join(root, "static/_diplodocus/julia/search.json"),
        "utf8",
      ),
    ),
    [{ path: `${mount}eunoiaplot!/` }],
  );
  const redirects = JSON.parse(
    await readFile(
      join(root, "static/_diplodocus/julia/redirects.json"),
      "utf8",
    ),
  );
  assert.equal(redirects.base_url, "https://eunoia.bz/docs/reference/julia/");
  assert.equal(redirects.api["Eunoia.euler-Tuple{AbstractDict}"], "euler");
});

test("configuration declares prepared Julia examples and the explicit Makie extension", () => {
  const julia = configuration("julia", "/tmp/Julia source", "revision");
  assert.match(julia, /repositories:\n/);
  assert.match(julia, /path: "\/tmp\/Julia source"/);
  assert.match(julia, /extensions: \["ext\/EunoiaMakieExt.jl"\]/);
  assert.match(julia, /format: documenter/);
  assert.match(julia, /engine: julia\n {6}julia-project: docs\/Project.toml/);
  const python = configuration("python", "/tmp/python", "revision");
  assert.match(python, /packages:\n/);
  assert.match(python, /path: python\/eunoia/);
  assert.ok(!python.includes("julia-project"));
});
