import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import {
  access,
  cp,
  mkdir,
  readdir,
  readFile,
  rename,
  rm,
  stat,
  writeFile,
} from "node:fs/promises";
import { dirname, join, relative, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { validateApiCoverage } from "./python-docs-coverage.mjs";

const webRoot = fileURLToPath(new URL("../", import.meta.url));
const projectRoot = resolve(webRoot, "..");
const mount = "/docs/reference/python/";
const assetMount = "/_diplodocus/python/";
const ownership = ".diplodocus-owned.json";

function run(command, args, cwd, capture = false) {
  const result = spawnSync(command, args, {
    cwd,
    encoding: "utf8",
    stdio: capture ? "pipe" : "inherit",
  });
  if (result.error || result.status !== 0) {
    throw new Error(
      `${command} ${args.join(" ")} failed: ${result.error?.message ?? result.stderr ?? result.status}`,
    );
  }
  return result.stdout?.trim();
}

async function exists(path) {
  try {
    await access(path);
    return true;
  } catch {
    return false;
  }
}

async function json(path) {
  return JSON.parse(await readFile(path, "utf8"));
}

function escapeHtml(value) {
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

async function digest(paths) {
  const hash = createHash("sha256");
  for (const [label, path] of paths) {
    hash.update(label);
    hash.update(await readFile(path));
  }
  return hash.digest("hex");
}

async function sourceInputs(root, names, prefix) {
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

async function generatedDigest(routes, assets) {
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

export function apiDocuments(manifest) {
  if (manifest.format_version !== 2)
    throw new Error(
      "Unsupported Diplodocus fragment format; expected version 2.",
    );
  const documents = manifest.documents.filter(
    (document) =>
      document.visible &&
      /^eunoia\.[A-Za-z][A-Za-z0-9_]*$/.test(document.public_name ?? ""),
  );
  const names = new Set();
  for (const document of documents) {
    if (names.has(document.public_name))
      throw new Error(`Duplicate public Python name: ${document.public_name}`);
    names.add(document.public_name);
  }
  if (!documents.length)
    throw new Error(
      "The snapshot contains no public eunoia API documentation.",
    );
  return documents.sort((a, b) =>
    a.public_name.localeCompare(b.public_name, "en"),
  );
}

export function documentMap(manifest) {
  const documents = {};
  for (const document of apiDocuments(manifest)) {
    documents[document.path] =
      `${mount}${document.public_name.slice("eunoia.".length)}/`;
  }
  return documents;
}

export function validateLinks(html, currentUrl, urls, assets) {
  for (const match of html.matchAll(/\b(?:href|src|poster)="([^"]*)"/g)) {
    const raw = match[1].replaceAll("&amp;", "&");
    const url = new URL(raw, `https://eunoia.bz${currentUrl}`);
    if (url.origin !== "https://eunoia.bz") continue;
    if (!urls.has(url.pathname) && !assets.has(url.pathname)) {
      throw new Error(`Unmapped local URL in ${currentUrl}: ${raw}`);
    }
  }
}

function pageSource(body, math) {
  const content = JSON.stringify(body).replaceAll("</", "<\\/");
  const mathStyles = math
    ? `\n  <link rel="stylesheet" href="${assetMount}katex/katex.min.css" />`
    : "";
  return `<script>\n  const content = ${content};\n</script>\n\n<svelte:head>\n  <link rel="stylesheet" href="${assetMount}content.css" />${mathStyles}\n</svelte:head>\n\n<div class="diplodocus-content">{@html content}</div>\n`;
}

function indexContent(documents, version) {
  const groups = [
    ["Fitting and diagrams", ["euler", "venn"]],
    ["Results", ["EulerFit", "VennFit"]],
    [
      "Geometry",
      [
        "Point",
        "Circle",
        "Ellipse",
        "Square",
        "Rectangle",
        "RotatedRectangle",
        "Container",
      ],
    ],
    ["Plot options", ["options", "get_options", "reset_options"]],
    ["Errors", ["EunoiaError"]],
  ];
  const byName = new Map(
    documents.map((document) => [document.public_name.slice(7), document]),
  );
  const lines = [
    "# Python API",
    "",
    `Reference for eunoia ${version}.`,
    "",
    "The [Python quickstart](/docs/quickstart/python/) introduces fitting and plotting. The [Python package documentation](https://jolars.github.io/eunoia-py/) retains the longer guides.",
  ];
  for (const [title, names] of groups) {
    const selected = names.filter((name) => byName.has(name));
    if (!selected.length) continue;
    lines.push("", `## ${title}`, "");
    for (const name of selected) {
      lines.push(`- [\`eunoia.${name}\`](${mount}${name}/)`);
      byName.delete(name);
    }
  }
  if (byName.size) {
    lines.push("", "## Other APIs", "");
    for (const [name] of byName)
      lines.push(`- [\`eunoia.${name}\`](${mount}${name}/)`);
  }
  lines.push(
    "",
    "## Version",
    "",
    `\`eunoia.__version__\` is \`${version}\`.`,
    "",
  );
  const markdown = lines.join("\n");
  const html = markdown
    .split("\n")
    .map((line) => {
      if (line.startsWith("# ")) return `<h1>${escapeHtml(line.slice(2))}</h1>`;
      if (line.startsWith("## "))
        return `<h2${line === "## Version" ? ' id="version"' : ""}>${escapeHtml(line.slice(3))}</h2>`;
      if (line.startsWith("- ")) return `<li>${inline(line.slice(2))}</li>`;
      return line ? `<p>${inline(line)}</p>` : "";
    })
    .join("\n")
    .replace(/(?:<li>[^\n]+<\/li>\n?)+/g, (list) => `<ul>${list}</ul>`);
  return { html, markdown };
}

function inline(text) {
  return escapeHtml(text)
    .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2">$1</a>')
    .replace(/`([^`]+)`/g, "<code>$1</code>");
}

async function assertOwned(directory) {
  if (!(await exists(directory))) return;
  if (!(await exists(join(directory, ownership)))) {
    throw new Error(
      `Refusing to replace unowned documentation directory: ${directory}`,
    );
  }
  const manifest = await json(join(directory, ownership));
  if (
    manifest.generator !== "eunoia-python-docs" ||
    !Array.isArray(manifest.files)
  ) {
    throw new Error(`Invalid documentation ownership marker: ${directory}`);
  }
  const owned = new Set([...manifest.files, ownership]);
  for (const path of await files(directory)) {
    if (!owned.has(relative(directory, path).split(sep).join("/"))) {
      throw new Error(
        `Refusing to remove an unowned documentation file: ${path}`,
      );
    }
  }
}

export async function publishFragments({
  root,
  fragments,
  manifest,
  version,
  checkCoverage = false,
  moveDirectory = rename,
}) {
  const routes = join(root, "src/routes/docs/reference/python");
  const assets = join(root, "static/_diplodocus/python");
  await assertOwned(routes);
  await assertOwned(assets);
  const documents = apiDocuments(manifest);
  const urlMap = documentMap(manifest);
  const urls = new Set([
    mount,
    ...documents.map((document) => urlMap[document.path]),
  ]);
  const assetPaths = await files(join(fragments, "assets"));
  const assetUrls = new Set(
    assetPaths.map(
      (path) =>
        assetMount +
        relative(join(fragments, "assets"), path).split(sep).join("/"),
    ),
  );
  const stagedRoutes = `${routes}.staging-${process.pid}`;
  const stagedAssets = `${assets}.staging-${process.pid}`;
  await rm(stagedRoutes, { recursive: true, force: true });
  await rm(stagedAssets, { recursive: true, force: true });
  try {
    await mkdir(stagedRoutes, { recursive: true });
    await cp(join(fragments, "assets"), stagedAssets, { recursive: true });
    const searchPath = join(stagedAssets, "search.json");
    if (await exists(searchPath)) {
      const search = await json(searchPath);
      await writeFile(
        searchPath,
        `${JSON.stringify(search.filter((entry) => urls.has(entry.path)))}\n`,
      );
    }
    const index = indexContent(documents, version);
    const pages = [
      {
        path: "",
        title: "Python API",
        html: index.html,
        markdown: index.markdown,
      },
    ];
    for (const document of documents) {
      const html = await readFile(join(fragments, document.path), "utf8");
      validateLinks(html, urlMap[document.path], urls, assetUrls);
      pages.push({
        path: document.public_name.slice(7),
        title: document.public_name,
        html,
        markdown: await readFile(
          join(fragments, document.markdown_path),
          "utf8",
        ),
      });
    }
    if (checkCoverage) validateApiCoverage(pages);
    for (const page of pages) {
      const directory = join(stagedRoutes, page.path);
      await mkdir(directory, { recursive: true });
      await writeFile(
        join(directory, "+page.svelte"),
        pageSource(
          page.html,
          assetUrls.has(`${assetMount}katex/katex.min.css`),
        ),
      );
      await writeFile(
        join(directory, "+page.ts"),
        `export const load = () => (${JSON.stringify({ title: page.title, description: `${page.title}: eunoia ${version} Python API reference.` })});\n`,
      );
      await writeFile(join(directory, "reference.md"), page.markdown);
    }
    const metadata = {
      generator: "eunoia-python-docs",
      pages: pages.map((page) => page.path),
      version,
    };
    for (const directory of [stagedRoutes, stagedAssets]) {
      await writeFile(
        join(directory, ownership),
        `${JSON.stringify({ ...metadata, files: (await files(directory)).map((path) => relative(directory, path).split(sep).join("/")) })}\n`,
      );
    }
    const replacements = [];
    try {
      for (const [destination, staging] of [
        [routes, stagedRoutes],
        [assets, stagedAssets],
      ]) {
        await mkdir(dirname(destination), { recursive: true });
        const replacement = {
          destination,
          backup: `${destination}.previous-${process.pid}`,
          previous: await exists(destination),
          installed: false,
        };
        if (replacement.previous)
          await moveDirectory(destination, replacement.backup);
        replacements.push(replacement);
        await moveDirectory(staging, destination);
        replacement.installed = true;
      }
    } catch (error) {
      for (const replacement of replacements.reverse()) {
        if (replacement.installed)
          await rm(replacement.destination, { recursive: true, force: true });
        if (replacement.previous)
          await moveDirectory(replacement.backup, replacement.destination);
      }
      throw error;
    }
    for (const replacement of replacements)
      await rm(replacement.backup, { recursive: true, force: true });
    return pages.length;
  } finally {
    await rm(stagedRoutes, { recursive: true, force: true });
    await rm(stagedAssets, { recursive: true, force: true });
  }
}

async function buildDiplodocus(source, pin, binary) {
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
    run(cargo, args, source);
  else if (spawnSync("devenv", ["--version"], { stdio: "ignore" }).status === 0)
    run("devenv", ["shell", "--", "cargo", ...args], source);
  else
    throw new Error(
      `Diplodocus needs Rust ${pin.rust}; provide DIPLODOCUS_CARGO or --diplodocus-bin.`,
    );
  return join(source, "target/debug/diplodocus");
}

export async function buildPythonDocs(options = {}) {
  const pin = await json(join(webRoot, "python-docs.lock.json"));
  const workspace = join(projectRoot, ".diplodocus/python");
  await mkdir(workspace, { recursive: true });
  const suppliedBinary = options.diplodocusBin ?? process.env.DIPLODOCUS_BIN;
  const source = suppliedBinary
    ? null
    : await checkout(
        "diplodocus",
        pin.diplodocus,
        options.diplodocusSource ?? process.env.DIPLODOCUS_SOURCE,
      );
  const snapshot = options.snapshot ?? process.env.EUNOIA_PYTHON_DOCS_SNAPSHOT;
  const python = snapshot
    ? null
    : await checkout(
        "eunoia-py",
        pin.python,
        options.pythonSource ?? process.env.EUNOIA_PYTHON_SOURCE,
      );
  const inputs = [
    ...(source
      ? await sourceInputs(
          source,
          [
            "src",
            "Cargo.toml",
            "Cargo.lock",
            "build.rs",
            "rust-toolchain.toml",
            "docs/cli-intro.md",
          ],
          "diplodocus",
        )
      : [["diplodocus-binary", resolve(suppliedBinary)]]),
    ...(await sourceInputs(
      webRoot,
      [
        "scripts/python-docs.mjs",
        "scripts/python-docs-coverage.mjs",
        "python-docs.lock.json",
      ],
      "host",
    )),
  ];
  if (snapshot) inputs.push(["snapshot", resolve(snapshot)]);
  else
    inputs.push(
      ...(await sourceInputs(
        python,
        ["python/eunoia", "pyproject.toml", "Cargo.toml", "src"],
        "python",
      )),
    );
  const revisions = {
    diplodocus: source
      ? run("git", ["rev-parse", "HEAD"], source, true)
      : run(resolve(suppliedBinary), ["--version"], projectRoot, true),
    python: python ? run("git", ["rev-parse", "HEAD"], python, true) : null,
  };
  const key = createHash("sha256")
    .update(await digest(inputs))
    .update(JSON.stringify(revisions))
    .digest("hex");
  const cachePath = join(workspace, "cache.json");
  const routeRoot = join(webRoot, "src/routes/docs/reference/python");
  const assetRoot = join(webRoot, "static/_diplodocus/python");
  if (
    (await exists(cachePath)) &&
    (await exists(join(routeRoot, ownership))) &&
    (await exists(join(assetRoot, ownership)))
  ) {
    const cached = await json(cachePath);
    if (
      cached.key === key &&
      cached.output === (await generatedDigest(routeRoot, assetRoot))
    )
      return;
  }
  const binary = await buildDiplodocus(source, pin.diplodocus, suppliedBinary);
  const database = snapshot
    ? resolve(snapshot)
    : join(workspace, "documentation.sqlite");
  if (!snapshot) {
    const revision = revisions.python;
    const config = `[project]\nname = "Eunoia"\n\n[[repository]]\nid = "python"\npath = ${JSON.stringify(python)}\nurl = "https://github.com/jolars/eunoia-py"\nrevision = ${JSON.stringify(revision)}\n\n[[package]]\nid = "python"\nname = "Eunoia for Python"\nslug = "python"\necosystem = "python"\nrepository = "python"\npath = "."\nmetadata-path = "pyproject.toml"\ntargets = [{ id = "api", extractor = "python", path = "python/eunoia", role = "public-api" }]\n`;
    const configPath = join(workspace, "diplodocus.toml");
    await writeFile(configPath, config);
    run(
      binary,
      ["extract", "--config", configPath, "--output", database],
      projectRoot,
    );
  }
  const fragments = join(workspace, "fragments");
  run(
    binary,
    [
      "generate",
      "--input",
      database,
      "--format",
      "fragments",
      "--output",
      fragments,
    ],
    projectRoot,
  );
  const manifest = await json(join(fragments, "documents.json"));
  const mappings = {
    format_version: 1,
    documents: documentMap(manifest),
    assets: {},
  };
  for (const path of await files(join(fragments, "assets"))) {
    const suffix = relative(join(fragments, "assets"), path)
      .split(sep)
      .join("/");
    mappings.assets[`assets/${suffix}`] = assetMount + suffix;
  }
  const mapPath = join(workspace, "urls.json");
  await writeFile(mapPath, JSON.stringify(mappings));
  run(
    binary,
    [
      "generate",
      "--input",
      database,
      "--format",
      "fragments",
      "--url-map",
      mapPath,
      "--output",
      fragments,
    ],
    projectRoot,
  );
  const finalManifest = await json(join(fragments, "documents.json"));
  const version =
    apiDocuments(finalManifest)[0].package_version ?? pin.python.version;
  const count = await publishFragments({
    root: webRoot,
    fragments,
    manifest: finalManifest,
    version,
    checkCoverage: true,
  });
  await writeFile(
    cachePath,
    `${JSON.stringify({
      key,
      output: await generatedDigest(routeRoot, assetRoot),
      pages: count,
    })}\n`,
  );
  console.log(`Generated ${count} Python API pages with Diplodocus.`);
}

if (
  process.argv[1] &&
  fileURLToPath(import.meta.url) === resolve(process.argv[1])
) {
  const options = {};
  const names = {
    "--python-source": "pythonSource",
    "--diplodocus-source": "diplodocusSource",
    "--snapshot": "snapshot",
    "--diplodocus-bin": "diplodocusBin",
  };
  for (let index = 2; index < process.argv.length; index += 2) {
    const name = names[process.argv[index]];
    if (!name || !process.argv[index + 1])
      throw new Error(`Unknown or incomplete option: ${process.argv[index]}`);
    options[name] = process.argv[index + 1];
  }
  await buildPythonDocs(options);
}
