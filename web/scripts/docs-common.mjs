import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { access, mkdir, readdir, readFile, stat } from "node:fs/promises";
import { join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

export const webRoot = fileURLToPath(new URL("../", import.meta.url));
export const projectRoot = resolve(webRoot, "..");

export function run(command, args, cwd, capture = false, env = process.env) {
  const result = spawnSync(command, args, {
    cwd,
    encoding: "utf8",
    stdio: capture ? "pipe" : "inherit",
    env,
  });
  if (result.error || result.status !== 0) {
    throw new Error(
      `${command} ${args.join(" ")} failed: ${result.error?.message ?? result.stderr ?? result.status}`,
    );
  }
  return result.stdout?.trim();
}

export async function exists(path) {
  try {
    await access(path);
    return true;
  } catch {
    return false;
  }
}

export async function json(path) {
  return JSON.parse(await readFile(path, "utf8"));
}

export function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

export async function files(root) {
  const found = [];
  for (const entry of await readdir(root, { withFileTypes: true })) {
    if (entry.name === "__pycache__" || entry.name.endsWith(".pyc")) continue;
    const path = join(root, entry.name);
    if (entry.isDirectory()) found.push(...(await files(path)));
    else if (entry.isFile()) found.push(path);
  }
  return found.sort();
}

export async function digest(paths) {
  const hash = createHash("sha256");
  for (const [label, path] of paths) {
    hash.update(label);
    hash.update(await readFile(path));
  }
  return hash.digest("hex");
}

export async function sourceInputs(root, names, prefix) {
  const inputs = [];
  for (const name of names) {
    const path = join(root, name);
    if (!(await exists(path))) continue;
    const paths = (await stat(path)).isDirectory() ? await files(path) : [path];
    inputs.push(
      ...paths.map((file) => [`${prefix}/${relative(root, file)}`, file]),
    );
  }
  return inputs;
}

export async function generatedDigest(routes, assets) {
  return digest([
    ...(await sourceInputs(routes, ["."], "routes")),
    ...(await sourceInputs(assets, ["."], "assets")),
  ]);
}

export async function checkout(name, pin, override, cacheRoot = projectRoot) {
  if (override) return resolve(override);
  const destination = join(
    cacheRoot,
    ".docs-sources",
    `${name}-${pin.revision}`,
  );
  if (!(await exists(join(destination, ".git")))) {
    await mkdir(destination, { recursive: true });
    run("git", ["init", "--quiet"], destination);
  }
  if (
    spawnSync("git", ["rev-parse", "--verify", "HEAD"], {
      cwd: destination,
      stdio: "ignore",
    }).status !== 0
  ) {
    const remotes = run("git", ["remote"], destination, true).split("\n");
    run(
      "git",
      [
        "remote",
        remotes.includes("origin") ? "set-url" : "add",
        "origin",
        pin.repository,
      ],
      destination,
    );
    run(
      "git",
      ["fetch", "--quiet", "--depth=1", "origin", pin.revision],
      destination,
    );
    run("git", ["checkout", "--quiet", "--detach", "FETCH_HEAD"], destination);
  }
  const head = run("git", ["rev-parse", "HEAD"], destination, true);
  if (
    head !== pin.revision ||
    run("git", ["status", "--porcelain"], destination, true)
  ) {
    throw new Error(
      `Cached ${name} checkout differs from its pin; remove ${destination} and retry.`,
    );
  }
  return destination;
}

export async function buildDiplodocus(source, pin, binary) {
  if (binary) return resolve(binary);
  const args = [
    "build",
    "--locked",
    "--manifest-path",
    join(source, "Cargo.toml"),
    "--bin",
    "diplodocus",
  ];
  const cargo = process.env.DIPLODOCUS_CARGO ?? "cargo";
  if (run(cargo, ["--version"], source, true).startsWith(`cargo ${pin.rust} `))
    run(cargo, args, source, false, {
      ...process.env,
      CARGO_BUILD_JOBS: process.env.CARGO_BUILD_JOBS ?? "2",
    });
  else if (spawnSync("devenv", ["--version"], { stdio: "ignore" }).status === 0)
    run("devenv", ["shell", "--", "cargo", ...args], source, false, {
      ...process.env,
      CARGO_BUILD_JOBS: process.env.CARGO_BUILD_JOBS ?? "2",
    });
  else
    throw new Error(
      `Diplodocus needs Rust ${pin.rust}; provide DIPLODOCUS_CARGO or --diplodocus-bin.`,
    );
  return join(source, "target/debug/diplodocus");
}
