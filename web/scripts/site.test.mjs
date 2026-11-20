import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { canonicalPath, legacyRoutes } from "../site.config.mjs";
import { checkout, files, webRoot } from "./docs-common.mjs";
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

test("authored widgets have unique IDs and inline props", async () => {
  const found = new Set();
  for (const path of await files(join(webRoot, "content"))) {
    const source = await readFile(path, "utf8");
    for (const [block, attrs] of source.matchAll(
      /^::: \{\.component name="eunoia"([^\n]*)\}$/gm,
    )) {
      const id = attrs.match(/ #([\w-]+)/)?.[1];
      const props = attrs.match(/ props='([^']+)'/)?.[1];
      assert.ok(id && props, `Missing ID or props: ${block}`);
      assert.ok(!found.has(id), `Repeated widget: ${id}`);
      found.add(id);
      const config = JSON.parse(props);
      assert.ok(
        ["hero", "builder", "diagram", "pipeline"].includes(config.component),
        `Unknown component: ${id}`,
      );
      if (["diagram", "pipeline"].includes(config.component))
        assert.ok(
          Object.keys(config.props.sets).length > 0,
          `Missing sets: ${id}`,
        );
    }
  }
  assert.ok(found.has("home-diagram"));
  assert.ok(found.has("diagram-builder"));
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
