#!/usr/bin/env node

// Post-processes the wasm-pack-generated `npm/` directory so it's ready to
// publish as `@jolars/eunoia`:
//
// 1. Compiles the hand-written TypeScript surface at `ts/index.ts` into
//    `npm/index.{js,d.ts}`.
// 2. Patches `npm/package.json`: name + scope, version (synced from cargo
//    metadata), `main`/`types` → wrapper, `files` list, publish metadata.
// 3. Removes the inner `.gitignore` wasm-pack writes (which would shadow the
//    repo-level rules).

import { execSync } from "node:child_process";
import { existsSync } from "node:fs";
import { access, copyFile, readFile, rm, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, "..");
const npmDir = resolve(repoRoot, "npm");
const tsDir = here;
const pkgPath = resolve(npmDir, "package.json");

// ---------------------------------------------------------------------------
// 1. Compile the TS wrapper into npm/
// ---------------------------------------------------------------------------

async function fileExists(p) {
  try {
    await access(p);
    return true;
  } catch {
    return false;
  }
}

const wrapperNodeModules = resolve(tsDir, "node_modules");
if (!existsSync(wrapperNodeModules)) {
  console.log("Installing TypeScript for the wrapper build…");
  execSync("pnpm install --silent", {
    cwd: tsDir,
    stdio: "inherit",
  });
}

// tsc resolves `import * as wasm from "./eunoia_wasm.js"` relative to the
// source file. wasm-pack writes the .d.ts to npm/ — copy it next to the
// wrapper source so tsc can find it. The copy is gitignored.
const wasmDtsSrc = resolve(npmDir, "eunoia_wasm.d.ts");
const wasmDtsDst = resolve(tsDir, "eunoia_wasm.d.ts");
if (!(await fileExists(wasmDtsSrc))) {
  throw new Error(
    `prepare-package: ${wasmDtsSrc} not found — run \`task build-wasm\` first`,
  );
}
await copyFile(wasmDtsSrc, wasmDtsDst);

execSync("pnpm exec tsc --project tsconfig.json", {
  cwd: tsDir,
  stdio: "inherit",
});

// Leave the copied .d.ts in place so IDEs can resolve types when editing
// the wrapper source. It's gitignored.

// ---------------------------------------------------------------------------
// 2. Patch npm/package.json
// ---------------------------------------------------------------------------

const cargoMetadata = JSON.parse(
  execSync("cargo metadata --format-version 1 --no-deps", {
    cwd: repoRoot,
    encoding: "utf8",
  }),
);
const workspaceVersion = cargoMetadata.packages.find(
  (p) => p.name === "eunoia",
).version;

const pkg = JSON.parse(await readFile(pkgPath, "utf8"));

pkg.name = "@jolars/eunoia";
pkg.version = workspaceVersion;
pkg.description =
  "Area-proportional Euler and Venn diagrams — WebAssembly bindings for the eunoia Rust library";
pkg.repository = {
  type: "git",
  url: "git+https://github.com/jolars/eunoia.git",
};
pkg.homepage = "https://github.com/jolars/eunoia";
pkg.bugs = { url: "https://github.com/jolars/eunoia/issues" };
pkg.keywords = [
  "euler",
  "venn",
  "diagram",
  "visualization",
  "wasm",
  "webassembly",
];
pkg.publishConfig = { access: "public" };

// The wrapper is the public entry point.
pkg.main = "index.js";
pkg.module = "index.js";
pkg.types = "index.d.ts";
pkg.exports = {
  ".": {
    types: "./index.d.ts",
    default: "./index.js",
  },
  // Power users can deep-import the raw wasm-bindgen surface.
  "./raw": {
    types: "./eunoia_wasm.d.ts",
    default: "./eunoia_wasm.js",
  },
};

pkg.files = [
  "index.js",
  "index.d.ts",
  "eunoia_wasm.js",
  "eunoia_wasm.d.ts",
  "eunoia_wasm_bg.js",
  "eunoia_wasm_bg.wasm",
  "eunoia_wasm_bg.wasm.d.ts",
];

await writeFile(pkgPath, JSON.stringify(pkg, null, 2) + "\n");

// ---------------------------------------------------------------------------
// 3. Drop the inner .gitignore wasm-pack writes
// ---------------------------------------------------------------------------

await rm(resolve(npmDir, ".gitignore"), { force: true });

console.log(`Prepared ${pkg.name}@${pkg.version} for publish`);
