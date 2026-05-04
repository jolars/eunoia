#!/usr/bin/env node

// Builds the publishable `npm/` directory from:
//
//   - the wasm-pack output already in `npm/` (eunoia_wasm.{js,d.ts}, *_bg.*)
//   - the TypeScript wrapper in `ts/index.ts` (compiled with tsc)
//   - `ts/package.json`, the canonical source of truth for the npm package's
//     metadata — versionary updates the version field here on each release.
//
// Steps:
//   1. Compile the TS wrapper into npm/index.{js,d.ts}.
//   2. Replace npm/package.json with ts/package.json minus build-only fields
//      (`private`, `devDependencies`, `packageManager`).
//   3. Remove the inner `.gitignore` wasm-pack writes (would shadow repo rules).

import { execSync } from "node:child_process";
import { existsSync } from "node:fs";
import { access, copyFile, readFile, rm, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, "..");
const npmDir = resolve(repoRoot, "npm");
const tsDir = here;

async function fileExists(p) {
  try {
    await access(p);
    return true;
  } catch {
    return false;
  }
}

// ---------------------------------------------------------------------------
// 1. Compile the TS wrapper into npm/
// ---------------------------------------------------------------------------

if (!existsSync(resolve(tsDir, "node_modules"))) {
  console.log("Installing TypeScript for the wrapper build…");
  execSync("pnpm install --silent", { cwd: tsDir, stdio: "inherit" });
}

// tsc resolves `import * as wasm from "./eunoia_wasm.js"` relative to the
// source file — copy the wasm-pack-generated .d.ts next to the wrapper
// source so resolution succeeds. The copy is gitignored.
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
// `ts/index.ts`. It's gitignored.

// ---------------------------------------------------------------------------
// 2. Write npm/package.json from ts/package.json
// ---------------------------------------------------------------------------

const tsPkg = JSON.parse(
  await readFile(resolve(tsDir, "package.json"), "utf8"),
);

// Strip fields that exist only to support the local build, not the publish.
const {
  private: _private,
  devDependencies,
  packageManager,
  ...published
} = tsPkg;
void _private;
void devDependencies;
void packageManager;

await writeFile(
  resolve(npmDir, "package.json"),
  JSON.stringify(published, null, 2) + "\n",
);

// ---------------------------------------------------------------------------
// 3. Drop the inner .gitignore wasm-pack writes
// ---------------------------------------------------------------------------

await rm(resolve(npmDir, ".gitignore"), { force: true });

console.log(`Prepared ${published.name}@${published.version} for publish`);
