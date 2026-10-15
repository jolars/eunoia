import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import {
  mkdir,
  mkdtemp,
  readFile,
  rename,
  rm,
  writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import {
  apiDocuments,
  checkout,
  documentMap,
  publishFragments,
  validateLinks,
} from "./python-docs.mjs";

function manifest(names = ["euler", "EulerFit"]) {
  return {
    format_version: 2,
    documents: [
      {
        path: "index.html",
        qualified_name: null,
        public_name: null,
        visible: true,
      },
      {
        path: "packages/python/index.html",
        qualified_name: null,
        public_name: null,
        visible: true,
      },
      {
        path: "internal.html",
        qualified_name: "eunoia._fit",
        public_name: "eunoia._fit",
        visible: false,
      },
      {
        path: "version.html",
        qualified_name: "eunoia.__version__",
        public_name: "eunoia.__version__",
        visible: true,
      },
      ...names.map((name) => ({
        path: `api/${name}.html`,
        markdown_path: `api/${name}.md`,
        qualified_name: `eunoia._implementation.${name}`,
        public_name: `eunoia.${name}`,
        visible: true,
      })),
    ],
  };
}

test("public names supply stable API routes and hide implementation modules", () => {
  assert.deepEqual(
    apiDocuments(manifest()).map((document) => document.public_name),
    ["eunoia.euler", "eunoia.EulerFit"],
  );
  const urls = documentMap(manifest());
  assert.equal(urls["api/euler.html"], "/docs/reference/python/euler/");
  assert.equal(urls["version.html"], undefined);
  assert.equal(urls["internal.html"], undefined);
  assert.throws(
    () => apiDocuments(manifest(["euler", "euler"])),
    /Duplicate public Python name/,
  );
  assert.throws(
    () => apiDocuments({ ...manifest(), format_version: 1 }),
    /expected version 2/,
  );
});

test("links must resolve to published pages or retained assets", () => {
  const pages = new Set(["/docs/reference/python/euler/"]);
  const assets = new Set(["/_diplodocus/python/figure.png"]);
  validateLinks(
    '<a href="#overload-1">Overload</a><a href="https://github.com/jolars/eunoia-py">Source</a><img src="/_diplodocus/python/figure.png">',
    "/docs/reference/python/euler/",
    pages,
    assets,
  );
  assert.throws(
    () =>
      validateLinks(
        '<a href="../private.html">Private</a>',
        "/docs/reference/python/euler/",
        pages,
        assets,
      ),
    /Unmapped local URL/,
  );
});

test("publication retains Markdown and assets and removes stale owned pages", async (context) => {
  const directory = await mkdtemp(join(tmpdir(), "eunoia-python-docs-"));
  context.after(() => rm(directory, { recursive: true, force: true }));
  const root = join(directory, "web");
  const fragments = join(directory, "fragments");
  await mkdir(join(fragments, "assets"), { recursive: true });
  await mkdir(join(fragments, "api"));
  const asset = new Uint8Array([0, 255, 10, 42]);
  await writeFile(join(fragments, "assets/figure.png"), asset);
  await writeFile(
    join(fragments, "assets/content.css"),
    ".syntax-code { color: inherit; }\n",
  );
  await writeFile(
    join(fragments, "assets/search.json"),
    JSON.stringify([
      { path: "/docs/reference/python/euler/" },
      { path: "internal.html" },
    ]),
  );
  for (const name of ["euler", "EulerFit"]) {
    await writeFile(
      join(fragments, `api/${name}.html`),
      `<h1>${name}</h1><h2 id="plot">plot</h2><pre><code>{value}</code></pre><img src="/_diplodocus/python/figure.png">`,
    );
    await writeFile(
      join(fragments, `api/${name}.md`),
      `# ${name}\n\n## plot\n\n\`{value}\`\n`,
    );
  }
  assert.equal(
    await publishFragments({
      root,
      fragments,
      manifest: manifest(),
      version: "0.6.0",
    }),
    3,
  );
  const routes = join(root, "src/routes/docs/reference/python");
  assert.match(
    await readFile(join(routes, "+page.svelte"), "utf8"),
    /Python API/,
  );
  assert.deepEqual(
    JSON.parse(
      await readFile(
        join(root, "static/_diplodocus/python/search.json"),
        "utf8",
      ),
    ),
    [{ path: "/docs/reference/python/euler/" }],
  );
  assert.match(
    await readFile(join(routes, "EulerFit/+page.svelte"), "utf8"),
    /id=\\"plot\\"/,
  );
  assert.equal(
    await readFile(join(routes, "euler/reference.md"), "utf8"),
    "# euler\n\n## plot\n\n`{value}`\n",
  );
  assert.deepEqual(
    new Uint8Array(
      await readFile(join(root, "static/_diplodocus/python/figure.png")),
    ),
    asset,
  );
  const previous = await readFile(join(routes, "euler/+page.svelte"), "utf8");
  await writeFile(
    join(fragments, "api/euler.html"),
    '<a href="private.html">Missing target</a>',
  );
  await assert.rejects(
    publishFragments({
      root,
      fragments,
      manifest: manifest(),
      version: "0.6.0",
    }),
    /Unmapped local URL/,
  );
  assert.equal(
    await readFile(join(routes, "euler/+page.svelte"), "utf8"),
    previous,
  );
  await writeFile(join(fragments, "api/euler.html"), "<h1>Replacement</h1>");
  await writeFile(
    join(fragments, "assets/figure.png"),
    new Uint8Array([10, 20]),
  );
  await assert.rejects(
    publishFragments({
      root,
      fragments,
      manifest: manifest(),
      version: "0.6.0",
      async moveDirectory(from, to) {
        if (
          from.includes(".staging-") &&
          to === join(root, "static/_diplodocus/python")
        )
          throw new Error("Injected asset installation failure");
        await rename(from, to);
      },
    }),
    /Injected asset installation failure/,
  );
  assert.equal(
    await readFile(join(routes, "euler/+page.svelte"), "utf8"),
    previous,
  );
  assert.deepEqual(
    new Uint8Array(
      await readFile(join(root, "static/_diplodocus/python/figure.png")),
    ),
    asset,
  );
  await writeFile(join(fragments, "api/euler.html"), "<h1>euler</h1>");
  assert.equal(
    await publishFragments({
      root,
      fragments,
      manifest: manifest(["euler"]),
      version: "0.6.0",
    }),
    2,
  );
  await assert.rejects(readFile(join(routes, "EulerFit/+page.svelte")), {
    code: "ENOENT",
  });
  await writeFile(join(routes, "notes.md"), "An author-written note.\n");
  await assert.rejects(
    publishFragments({
      root,
      fragments,
      manifest: manifest(),
      version: "0.6.0",
    }),
    /unowned documentation file/,
  );
  assert.equal(
    await readFile(join(routes, "notes.md"), "utf8"),
    "An author-written note.\n",
  );
});

test("a source fetch retries an initialized cache without a checked-out HEAD", async (context) => {
  const directory = await mkdtemp(join(tmpdir(), "eunoia-python-docs-fetch-"));
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

test("publication preserves author files and validates before replacing output", async (context) => {
  const directory = await mkdtemp(join(tmpdir(), "eunoia-python-docs-owned-"));
  context.after(() => rm(directory, { recursive: true, force: true }));
  const route = join(directory, "src/routes/docs/reference/python");
  await mkdir(route, { recursive: true });
  await writeFile(join(route, "+page.svx"), "# Written by hand\n");
  await assert.rejects(
    publishFragments({
      root: directory,
      fragments: "unused",
      manifest: manifest(),
      version: "0.6.0",
    }),
    /unowned documentation directory/,
  );
  assert.equal(
    await readFile(join(route, "+page.svx"), "utf8"),
    "# Written by hand\n",
  );
});
