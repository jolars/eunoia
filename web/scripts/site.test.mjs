import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { canonicalPath, legacyRoutes } from "../site.config.mjs";
import { checkout, files, json, webRoot } from "./docs-common.mjs";
import { assetUrl, preserveAnchors, rewriteUrl } from "./site-output.mjs";

test("old heading and method anchors retain their intended positions", () => {
  const entry = {
    headings: { "Old-title": "A title" },
    aliases: { "Eunoia.euler-Tuple{AbstractDict}": "Eunoia.euler" },
  };
  const result = preserveAnchors(
    '<main><h1>A title</h1><p>Introduction.</p><h2 id="euler"><code>Eunoia.euler</code></h2><p>Details.</p></main>',
    '# A title\n\nIntroduction.\n\n<a id="euler"></a>\n\n## `Eunoia.euler`\n\nDetails.\n',
    entry,
  );
  assert.match(result.html, /id="Old-title"><\/a><h1>/);
  assert.match(
    result.html,
    /id="Eunoia.euler-Tuple\{AbstractDict\}"><\/a><h2 id="euler">/,
  );
  assert.ok(
    result.markdown.indexOf('id="Eunoia.euler-Tuple{AbstractDict}"') >
      result.markdown.indexOf("Introduction."),
  );
  assert.throws(
    () => preserveAnchors("<main><h1>Other</h1></main>", "# Other\n", entry),
    /Cannot preserve anchor/,
  );
});

test("API links preserve queries and fragments when switching to native routes", () => {
  const old = "/docs/reference/python/euler/";
  const destination = canonicalPath(legacyRoutes[old].artifact);
  assert.equal(
    rewriteUrl(`${old}?example=one#overload`, "/docs/"),
    `${destination}?example=one#overload`,
  );
  assert.equal(
    rewriteUrl("https://example.org/docs/", "/docs/"),
    "https://example.org/docs/",
  );
});

test("responsive image candidates remain valid after client navigation", () => {
  assert.equal(assetUrl("assets/logo.png", "/"), "/assets/logo.png");
  assert.equal(
    assetUrl("../../assets/logo.png", "/docs/concepts/"),
    "/assets/logo.png",
  );
  assert.equal(
    assetUrl("https://example.org/badge.svg", "/docs/"),
    "https://example.org/badge.svg",
  );
});

test("every authored component has a unique, configured widget ID", async () => {
  const examples = await json(join(webRoot, "src/widgets/examples.json"));
  const expected = new Set([
    "home-diagram",
    "diagram-builder",
    ...Object.keys(examples),
  ]);
  const found = new Set();
  for (const path of await files(join(webRoot, "content"))) {
    const source = await readFile(path, "utf8");
    for (const [, id] of source.matchAll(
      /\.component name="eunoia" #([\w-]+)/g,
    )) {
      assert.ok(expected.has(id), `Unknown widget: ${id}`);
      assert.ok(!found.has(id), `Repeated widget: ${id}`);
      found.add(id);
    }
  }
  assert.deepEqual(found, expected);
});

test("an interrupted source fetch can recover its initialized checkout", async (context) => {
  const directory = await mkdtemp(join(tmpdir(), "eunoia-docs-fetch-"));
  context.after(() => rm(directory, { recursive: true, force: true }));
  const source = join(directory, "origin");
  await mkdir(source);
  const git = (args, cwd = source) =>
    execFileSync("git", args, { cwd, encoding: "utf8" }).trim();
  git(["init", "--quiet"]);
  await writeFile(join(source, "api.py"), "def euler(): pass\n");
  git(["add", "api.py"]);
  git([
    "-c",
    "core.hooksPath=/dev/null",
    "-c",
    "user.name=Docs Test",
    "-c",
    "user.email=docs@example.org",
    "commit",
    "--quiet",
    "-m",
    "test",
  ]);
  const revision = git(["rev-parse", "HEAD"]);
  const destination = join(directory, ".docs-sources", `test-${revision}`);
  await mkdir(destination, { recursive: true });
  git(["init", "--quiet"], destination);
  git(
    ["remote", "add", "origin", join(directory, "interrupted-origin")],
    destination,
  );
  assert.equal(
    await checkout(
      "test",
      { repository: source, revision },
      undefined,
      directory,
    ),
    destination,
  );
  assert.equal(
    await readFile(join(destination, "api.py"), "utf8"),
    "def euler(): pass\n",
  );
});
