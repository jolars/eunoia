import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { join, relative } from "node:path";
import { fileURLToPath } from "node:url";
import { canonicalPath, legacyRoutes } from "../site.config.mjs";
import { exists, files, webRoot } from "./docs-common.mjs";
import { validateJuliaCoverage } from "./julia-docs-coverage.mjs";
import { anchors, validateApiCoverage } from "./python-docs-coverage.mjs";

export async function validateSite(root = join(webRoot, "dist")) {
  const paths = await files(root);
  const knownFiles = new Set(
    paths.map((path) => relative(root, path).replaceAll("\\", "/")),
  );
  const pages = new Map();
  const titles = new Set();
  const sitemap = await readFile(join(root, "sitemap.xml"), "utf8");
  for (const artifact of [...knownFiles].filter((path) =>
    path.endsWith(".html"),
  )) {
    const html = await readFile(join(root, artifact), "utf8");
    const markdown = await readFile(
      join(root, artifact.replace(/\.html$/, ".md")),
      "utf8",
    );
    const url = canonicalPath(artifact);
    const title = html.match(/<title>([^<]*)<\/title>/)?.[1];
    assert.ok(
      title && !titles.has(title),
      `Missing or repeated title: ${url}: ${title}`,
    );
    titles.add(title);
    const main = html.match(/<main\b[^>]*>([\s\S]*?)<\/main>/)?.[1];
    assert.ok(main, `Missing main content: ${url}`);
    assert.equal(
      [...main.matchAll(/<h1\b/g)].length,
      1,
      `Expected one H1: ${url}`,
    );
    assert.ok(
      html.includes(`rel="canonical" href="https://eunoia.bz${url}"`),
      `Missing canonical: ${url}`,
    );
    assert.doesNotMatch(
      main,
      /class="unsupported|Unsupported content|%EUNOIA_VERSION/,
      `Unrendered content: ${url}`,
    );
    if (!html.includes('content="noindex"'))
      assert.ok(
        sitemap.includes(`<loc>https://eunoia.bz${url}</loc>`),
        `Missing sitemap entry: ${url}`,
      );
    pages.set(url, { artifact, html, markdown, body: main });
  }
  const redirects = await readFile(join(root, "_redirects"), "utf8");
  for (const [route, entry] of Object.entries(legacyRoutes)) {
    const url = canonicalPath(entry.artifact);
    const page = pages.get(url);
    assert.ok(page, `Lost route ${route}: ${entry.artifact}`);
    if (route !== url)
      assert.ok(
        redirects.includes(`${route} ${url} 301`),
        `Missing redirect: ${route}`,
      );
    for (const format of ["html", "markdown"]) {
      const ids = anchors(page[format], format === "markdown");
      for (const id of Object.keys({ ...entry.headings, ...entry.aliases }))
        assert.ok(ids.has(id), `Lost ${format} anchor: ${route}#${id}`);
    }
  }
  for (const [current, page] of pages) {
    const references = [
      ...page.html.matchAll(/\b(?:href|src|poster)="([^"]*)"/g),
    ].map((match) => match[1]);
    for (const [, candidates] of page.html.matchAll(/\bsrcset="([^"]*)"/g)) {
      for (const candidate of candidates.split(",")) {
        const source = candidate.trim().split(/\s+/)[0];
        assert.ok(
          source.startsWith("/") || /^https?:\/\//.test(source),
          `Image candidate depends on the current route: ${current}: ${source}`,
        );
        references.push(source);
      }
    }
    for (const raw of references) {
      const url = new URL(
        raw.replaceAll("&amp;", "&"),
        `https://eunoia.bz${current}`,
      );
      if (url.origin !== "https://eunoia.bz") continue;
      const path = decodeURIComponent(url.pathname);
      const target =
        pages.get(path) ??
        pages.get(
          canonicalPath(legacyRoutes[path]?.artifact ?? "__missing__.html"),
        );
      if (target) {
        assert.ok(
          !url.hash ||
            anchors(target.html).has(decodeURIComponent(url.hash.slice(1))),
          `Broken anchor in ${current}: ${raw}`,
        );
      } else {
        assert.ok(
          knownFiles.has(path.slice(1)),
          `Broken link in ${current}: ${raw}`,
        );
      }
    }
  }
  for (const language of ["python", "julia"]) {
    const prefix = `/docs/reference/${language}/`;
    const selected = Object.entries(legacyRoutes)
      .filter(([route]) => route.startsWith(prefix))
      .map(([route, entry]) => {
        const page = pages.get(canonicalPath(entry.artifact));
        return {
          path: route.slice(prefix.length).replace(/\/$/, ""),
          url: canonicalPath(entry.artifact),
          html: page.body,
          markdown: page.markdown,
        };
      });
    (language === "python" ? validateApiCoverage : validateJuliaCoverage)(
      selected,
    );
  }
  assert.ok(
    await exists(join(root, "__diplodocus/worker.js")),
    "Missing generated Markdown Worker.",
  );
  console.log(
    `${pages.size} pages passed content, link, anchor, sitemap, and API coverage checks.`,
  );
}

if (process.argv[1] === fileURLToPath(import.meta.url)) await validateSite();
