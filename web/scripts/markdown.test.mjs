import assert from "node:assert/strict";
import { test } from "node:test";
import worker from "../cloudflare/markdown-worker.js";

function assetBinding({ missingMarkdown = false } = {}) {
  const requests = [];
  return {
    requests,
    ASSETS: {
      async fetch(request) {
        requests.push(request);
        const path = new URL(request.url).pathname;
        if (path === "/missing/")
          return new Response("Missing", { status: 404 });
        if (path === "/old/")
          return new Response(null, {
            status: 301,
            headers: { Location: "/docs/" },
          });
        const markdown = path.endsWith(".md");
        if (markdown && missingMarkdown)
          return new Response("Missing", { status: 404 });
        const asset = path.endsWith(".js");
        return new Response(
          request.method === "HEAD"
            ? null
            : markdown
              ? "# Shapes\n"
              : asset
                ? "asset"
                : "<h1>Shapes</h1>",
          {
            headers: {
              "Content-Type": markdown
                ? "text/plain"
                : asset
                  ? "text/javascript"
                  : "text/html",
              Vary: "Accept-Encoding",
              ETag: markdown ? '"markdown"' : '"html"',
            },
          },
        );
      },
    },
  };
}

test("the generated Worker negotiates adjacent Markdown for directory and clean API URLs", async () => {
  for (const [path, expected] of [
    ["/docs/concepts/shapes/", "/docs/concepts/shapes/index.md"],
    [
      "/packages/python/reference/example",
      "/packages/python/reference/example.md",
    ],
  ]) {
    const env = assetBinding();
    const response = await worker.fetch(
      new Request(`https://eunoia.bz${path}`, {
        headers: { Accept: "text/markdown" },
      }),
      env,
    );
    assert.equal(response.status, 200);
    assert.equal(
      response.headers.get("Content-Type"),
      "text/markdown; charset=utf-8",
    );
    assert.equal(response.headers.get("Vary"), "Accept-Encoding, Accept");
    assert.equal(await response.text(), "# Shapes\n");
    assert.equal(new URL(env.requests[1].url).pathname, expected);
  }
});

test("media quality, HEAD, and representation validators remain correct", async () => {
  for (const [accept, markdown] of [
    ["text/markdown", true],
    ["text/html, text/markdown;q=0.8", false],
    ["text/markdown;q=0", false],
    ["text/markdown;q=1.1", false],
    ["*/*", false],
  ]) {
    const env = assetBinding();
    const response = await worker.fetch(
      new Request("https://eunoia.bz/docs/", {
        method: "HEAD",
        headers: { Accept: accept, "If-None-Match": '"html"' },
      }),
      env,
    );
    assert.equal(
      response.headers.get("Content-Type")?.startsWith("text/markdown"),
      markdown,
    );
    assert.equal(await response.text(), "");
    if (markdown) {
      assert.equal(env.requests[0].headers.get("If-None-Match"), null);
      assert.equal(response.headers.get("ETag"), '"markdown"');
    }
  }
});

test("missing Markdown falls back to HTML and assets, redirects, and 404s pass through", async () => {
  const fallback = await worker.fetch(
    new Request("https://eunoia.bz/docs/", {
      headers: { Accept: "text/markdown" },
    }),
    assetBinding({ missingMarkdown: true }),
  );
  assert.equal(await fallback.text(), "<h1>Shapes</h1>");
  assert.match(fallback.headers.get("Vary"), /Accept/);
  for (const [path, status] of [
    ["/assets/app.js", 200],
    ["/old/", 301],
    ["/missing/", 404],
  ]) {
    const env = assetBinding();
    const response = await worker.fetch(
      new Request(`https://eunoia.bz${path}`, {
        headers: { Accept: "text/markdown" },
      }),
      env,
    );
    assert.equal(response.status, status);
    assert.equal(env.requests.length, 1);
  }
});
