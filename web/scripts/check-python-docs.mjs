import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import worker from "../cloudflare/markdown-worker.js";
import { publicApi, validateApiCoverage } from "./python-docs-coverage.mjs";

const root = fileURLToPath(new URL("../", import.meta.url));
const pages = [];
for (const name of ["", ...publicApi]) {
  const directory = join(root, "src/routes/docs/reference/python", name);
  const source = await readFile(join(directory, "+page.svelte"), "utf8");
  const body = source.match(/const content = (".*");/);
  assert.ok(body, `Missing prerendered API content for ${name || "index"}`);
  pages.push({
    path: name,
    html: JSON.parse(body[1]),
    markdown: await readFile(join(directory, "reference.md"), "utf8"),
  });
}
validateApiCoverage(pages);
if (process.argv.includes("--built")) {
  const sitemap = await readFile(join(root, "dist/sitemap.xml"), "utf8");
  const env = {
    ASSETS: {
      async fetch(request) {
        const path = new URL(request.url).pathname;
        try {
          const body = await readFile(
            join(root, "dist", path, path.endsWith("/") ? "index.html" : ""),
          );
          return new Response(body, {
            headers: {
              "Content-Type": path.endsWith(".txt")
                ? "text/plain"
                : "text/html",
            },
          });
        } catch {
          return new Response("Not found", { status: 404 });
        }
      },
    },
  };
  for (const page of pages) {
    const route = `docs/reference/python/${page.path ? `${page.path}/` : ""}`;
    const html = await readFile(
      join(root, "dist", route, "index.html"),
      "utf8",
    );
    assert.equal(
      [...html.matchAll(/<h1\b/g)].length,
      1,
      `${route} must have one H1`,
    );
    assert.ok(
      sitemap.includes(`<loc>https://eunoia.bz/${route}</loc>`),
      `${route} is missing from the sitemap`,
    );
    const markdown = await readFile(
      join(root, "dist/_agent_markdown", route, "index.txt"),
      "utf8",
    );
    assert.equal(markdown, page.markdown, `${route} has stale Markdown`);
    const url = `https://eunoia.bz/${route}`;
    const negotiated = await worker.fetch(
      new Request(url, { headers: { Accept: "text/markdown" } }),
      env,
    );
    assert.equal(negotiated.status, 200);
    assert.equal(
      negotiated.headers.get("Content-Type"),
      "text/markdown; charset=utf-8",
    );
    assert.equal(await negotiated.text(), page.markdown);
    const head = await worker.fetch(
      new Request(url, {
        method: "HEAD",
        headers: { Accept: "text/markdown" },
      }),
      env,
    );
    assert.equal(head.status, 200);
    assert.equal(await head.text(), "");
  }
  assert.equal(
    [
      ...sitemap.matchAll(
        /<loc>https:\/\/eunoia.bz\/docs\/reference\/python\//g,
      ),
    ].length,
    16,
  );
}
console.log(
  "Python API coverage, five-overload families, class members, inherited links, and HTML/Markdown anchors passed.",
);
