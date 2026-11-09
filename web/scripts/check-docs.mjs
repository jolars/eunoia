import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import worker from "../cloudflare/markdown-worker.js";
import { json, webRoot } from "./docs-common.mjs";
import { profiles } from "./package-docs.mjs";
import { anchors } from "./python-docs-coverage.mjs";

const built = process.argv.includes("--built");
const languages = process.argv.includes("--julia")
  ? ["julia"]
  : process.argv.includes("--python")
    ? ["python"]
    : ["python", "julia"];
const env = {
  ASSETS: {
    async fetch(request) {
      const path = new URL(request.url).pathname;
      try {
        return new Response(
          await readFile(
            join(webRoot, "dist", path, path.endsWith("/") ? "index.html" : ""),
          ),
          {
            headers: {
              "Content-Type": path.endsWith(".txt")
                ? "text/plain"
                : "text/html",
            },
          },
        );
      } catch {
        return new Response("Not found", { status: 404 });
      }
    },
  },
};
for (const language of languages) {
  const profile = profiles[language];
  const route = join(webRoot, "src/routes/docs/reference", language);
  const ownership = await json(join(route, ".diplodocus-owned.json"));
  assert.equal(ownership.generator, `eunoia-${language}-docs`);
  const pages = [];
  for (const path of ownership.pages) {
    const directory = join(route, path);
    const source = await readFile(join(directory, "+page.svelte"), "utf8");
    const content = source.match(/const content = (".*");/);
    assert.ok(content, `Missing generated content: ${language}/${path}`);
    pages.push({
      path,
      html: JSON.parse(content[1]),
      markdown: await readFile(join(directory, "reference.md"), "utf8"),
    });
  }
  profile.validate(pages);
  if (language === "julia") {
    const legacy = await json(join(webRoot, "julia-docs-legacy.json"));
    const byPath = new Map(pages.map((page) => [page.path, page]));
    for (const [path, { anchors: expected }] of Object.entries(legacy.pages)) {
      for (const format of ["html", "markdown"]) {
        const ids = anchors(byPath.get(path)[format], format === "markdown");
        for (const id of expected)
          assert.ok(
            ids.has(id),
            `Missing legacy ${format} anchor: ${path}#${id}`,
          );
      }
    }
    for (const [id, path] of Object.entries(legacy.api)) {
      for (const format of ["html", "markdown"])
        assert.ok(
          anchors(byPath.get(path)[format], format === "markdown").has(id),
          `Missing legacy API anchor: ${path}#${id}`,
        );
    }
  }
  if (built) {
    const sitemap = await readFile(join(webRoot, "dist/sitemap.xml"), "utf8");
    for (const page of pages) {
      const path = profile.mount + (page.path ? `${page.path}/` : "");
      const html = await readFile(
        join(webRoot, "dist", path, "index.html"),
        "utf8",
      );
      assert.equal(
        [...html.matchAll(/<h1\b/g)].length,
        1,
        `${path} must have one H1`,
      );
      assert.ok(
        sitemap.includes(`<loc>https://eunoia.bz${path}</loc>`),
        `${path} is missing from the sitemap`,
      );
      assert.equal(
        await readFile(
          join(webRoot, "dist/_agent_markdown", path, "index.txt"),
          "utf8",
        ),
        page.markdown,
      );
      for (const method of ["GET", "HEAD"]) {
        const response = await worker.fetch(
          new Request(`https://eunoia.bz${path}`, {
            method,
            headers: { Accept: "text/markdown" },
          }),
          env,
        );
        assert.equal(response.status, 200, `${method} ${path}`);
        assert.equal(
          response.headers.get("Content-Type"),
          "text/markdown; charset=utf-8",
        );
        assert.equal(
          await response.text(),
          method === "GET" ? page.markdown : "",
        );
      }
    }
  }
  console.log(
    `${profile.title}: ${pages.length} pages passed content, links, and compatibility checks${built ? ", including built routes and Markdown negotiation" : ""}.`,
  );
}
