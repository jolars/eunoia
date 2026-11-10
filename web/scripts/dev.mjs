import { spawn } from "node:child_process";
import { watch } from "node:fs";
import { join } from "node:path";
import { build } from "vite";
import { webRoot } from "./docs-common.mjs";
import { buildSite, cliOptions, prepareSources } from "./site.mjs";

const options = cliOptions(process.argv.slice(2));
await buildSite(options);
const preview = spawn(
  "pnpm",
  [
    "exec",
    "wrangler",
    "dev",
    "--local",
    "--infer-origin-from-routes=false",
    "--live-reload",
    "--config",
    "cloudflare/wrangler.jsonc",
    "--port",
    options.port ?? "5173",
    "--ip",
    options.host ?? "127.0.0.1",
  ],
  { cwd: webRoot, stdio: "inherit" },
);
const watchers = [];
let bundleWatcher;
let timer;
let pending = false;
let running = false;
let bundling = false;

async function rebuild() {
  if (running || bundling || !pending) return;
  pending = false;
  running = true;
  try {
    await buildSite(options, { widgets: false });
  } catch (error) {
    console.error(error);
  } finally {
    running = false;
    if (pending) schedule();
  }
}

function schedule() {
  pending = true;
  clearTimeout(timer);
  timer = setTimeout(rebuild, 200);
}

if (!options.snapshot && !process.env.EUNOIA_DOCS_SNAPSHOT) {
  const { sources } = await prepareSources(options);
  for (const path of [
    join(webRoot, "content"),
    join(webRoot, "python"),
    join(webRoot, "static"),
    join(sources.python, "python/eunoia"),
    join(sources.julia, "src"),
    join(sources.julia, "ext"),
    join(sources.julia, "docs/src"),
  ])
    watchers.push(watch(path, { recursive: true }, schedule));
  bundleWatcher = await build({
    root: webRoot,
    configFile: join(webRoot, "vite.config.ts"),
    build: { watch: {}, emptyOutDir: true },
  });
  let first = true;
  bundleWatcher.on("event", (event) => {
    if (event.code === "START") bundling = true;
    if (event.code === "END") {
      bundling = false;
      if (first) first = false;
      else schedule();
    }
  });
}

const stop = () => preview.kill("SIGTERM");
process.once("SIGINT", stop);
process.once("SIGTERM", stop);
try {
  await new Promise((resolve, reject) => {
    preview.once("error", reject);
    preview.once("exit", (code) =>
      code ? reject(new Error(`Preview exited with ${code}.`)) : resolve(),
    );
  });
} finally {
  clearTimeout(timer);
  for (const watcher of watchers) watcher.close();
  await bundleWatcher?.close();
  preview.kill("SIGTERM");
}
