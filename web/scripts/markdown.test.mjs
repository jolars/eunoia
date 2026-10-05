import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { test } from "node:test";
import worker, { wantsMarkdown } from "../cloudflare/markdown-worker.js";
import { cleanDocsSource } from "./build-markdown.mjs";

test("docs source keeps prose and code but removes Svelte internals", async () => {
  const source = await readFile(
    new URL(
      "../src/routes/docs/quickstart/javascript/+page.svx",
      import.meta.url,
    ),
    "utf8",
  );
  const markdown = cleanDocsSource(source);
  assert.match(markdown, /^# Quickstart: JavaScript/m);
  assert.match(markdown, /```html\n<script type="module">/);
  assert.doesNotMatch(markdown, /import DiagramExample from/);
  assert.doesNotMatch(markdown, /<DiagramExample/);
});

test("Markdown negotiation respects disabled media ranges", () => {
  assert.equal(wantsMarkdown("text/html, text/markdown;q=0.8"), true);
  assert.equal(wantsMarkdown("text/markdown;q=0, text/html"), false);
  assert.equal(wantsMarkdown("*/*"), false);
});

test("Worker serves Markdown for an explicit request and HTML by default", async () => {
  const requested = [];
  const env = {
    ASSETS: {
      async fetch(request) {
        requested.push(request.url);
        return request.url.includes("_agent_markdown")
          ? new Response("# Shapes\n", {
              headers: { "Content-Type": "text/plain" },
            })
          : new Response("<h1>Shapes</h1>", {
              headers: { "Content-Type": "text/html", Vary: "Accept-Encoding" },
            });
      },
    },
  };
  const markdown = await worker.fetch(
    new Request("https://eunoia.bz/docs/concepts/shapes/", {
      headers: { Accept: "text/markdown" },
    }),
    env,
  );
  assert.equal(
    markdown.headers.get("Content-Type"),
    "text/markdown; charset=utf-8",
  );
  assert.equal(markdown.headers.get("Vary"), "Accept");
  assert.equal(await markdown.text(), "# Shapes\n");
  assert.equal(
    requested[0],
    "https://eunoia.bz/_agent_markdown/docs/concepts/shapes/index.txt",
  );

  const html = await worker.fetch(
    new Request("https://eunoia.bz/docs/concepts/shapes/"),
    env,
  );
  assert.equal(html.headers.get("Content-Type"), "text/html");
  assert.equal(html.headers.get("Vary"), "Accept-Encoding, Accept");
  assert.equal(await html.text(), "<h1>Shapes</h1>");
});

test("Worker serves non-page assets from its asset binding", async () => {
  const requested = [];
  const env = {
    ASSETS: {
      async fetch(request) {
        requested.push(request.url);
        return new Response("asset", { status: 200 });
      },
    },
  };
  const response = await worker.fetch(
    new Request("https://eunoia.bz/_app/example.js"),
    env,
  );
  assert.equal(await response.text(), "asset");
  assert.deepEqual(requested, ["https://eunoia.bz/_app/example.js"]);
});
