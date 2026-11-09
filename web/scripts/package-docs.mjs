import { createHash } from "node:crypto";
import { cp, mkdir, readFile, rename, rm, writeFile } from "node:fs/promises";
import { dirname, join, relative, resolve, sep } from "node:path";
import {
  buildDiplodocus,
  checkout,
  digest,
  escapeHtml,
  exists,
  files,
  generatedDigest,
  json,
  projectRoot,
  run,
  sourceInputs,
  webRoot,
} from "./docs-common.mjs";
import { prepareEnvironment } from "./docs-environment.mjs";
import { validateJuliaCoverage } from "./julia-docs-coverage.mjs";
import { validateApiCoverage } from "./python-docs-coverage.mjs";

export { checkout, files } from "./docs-common.mjs";
export const profiles = {
  python: {
    prefix: "eunoia.",
    mount: "/docs/reference/python/",
    assets: "/_diplodocus/python/",
    title: "Python",
    validate: validateApiCoverage,
  },
  julia: {
    prefix: "Eunoia.",
    mount: "/docs/reference/julia/",
    assets: "/_diplodocus/julia/",
    title: "Julia",
    validate: validateJuliaCoverage,
  },
};
const { mount } = profiles.python;
const ownership = ".diplodocus-owned.json";
export function apiDocuments(manifest, language = "python") {
  if (manifest.format_version !== 2)
    throw new Error(
      "Unsupported Diplodocus fragment format; expected version 2.",
    );
  const documents = manifest.documents.filter(
    (document) =>
      document.visible &&
      (language === "python"
        ? /^eunoia\.[A-Za-z][A-Za-z0-9_]*$/
        : /^Eunoia\.[\p{ID_Start}][\p{ID_Continue}]*!?$/u
      ).test(document.public_name ?? ""),
  );
  const names = new Set();
  for (const document of documents) {
    if (names.has(document.public_name))
      throw new Error(
        `Duplicate public ${profiles[language].title} name: ${document.public_name}`,
      );
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

function hostDocuments(manifest, language) {
  return language === "python"
    ? apiDocuments(manifest)
    : manifest.documents.filter((document) =>
        document.path.startsWith("packages/julia/"),
      );
}

export function documentMap(manifest, language = "python") {
  const documents = {};
  if (language === "julia") {
    apiDocuments(manifest, language);
    for (const document of hostDocuments(manifest, language)) {
      let path;
      if (document.public_name === "Eunoia") path = "module";
      else if (document.public_name) {
        path = document.public_name.replace(/^Eunoia\./, "");
        if (!/^[_\p{ID_Start}][\p{ID_Continue}]*!?$/u.test(path)) {
          throw new Error(
            `Unsupported Julia route name: ${document.public_name}`,
          );
        }
        if (!document.visible) path = `internal/${path}`;
      } else {
        path = document.path
          .slice("packages/julia/".length)
          .replace(/\.html$/, "");
        path = path === "index" ? "" : path.replace(/^CHANGELOG$/, "changelog");
      }
      documents[document.path] =
        profiles.julia.mount + (path ? `${path}/` : "");
    }
    documents["index.html"] = "/docs/";
    return documents;
  }
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

function pageSource(body, math, language = "python", visible = true) {
  const assetMount = profiles[language].assets;
  const content = JSON.stringify(body).replaceAll("</", "<\\/");
  const mathStyles = math
    ? `\n  <link rel="stylesheet" href="${assetMount}katex/katex.min.css" />`
    : "";
  return `<script>\n  const content = ${content};\n</script>\n\n<svelte:head>\n  <link rel="stylesheet" href="${assetMount}content.css" />${mathStyles}${visible ? "" : '\n  <meta name="robots" content="noindex" />'}\n</svelte:head>\n\n<div class="diplodocus-content">{@html content}</div>\n`;
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

export function juliaAliases(page, legacy) {
  const known = new Set(
    [...page.html.matchAll(/\bid="([^"]+)"/g)].map((match) => match[1]),
  );
  const marker = (name) => {
    if (known.has(name)) return "";
    known.add(name);
    return `<a id="${escapeHtml(name)}"></a>`;
  };
  const first = [
    ...(page.document?.public_name ? [page.document.public_name] : []),
    ...Object.entries(legacy.api)
      .filter(([, path]) => path === page.path)
      .map(([id]) => id),
  ]
    .map(marker)
    .join("");
  page.html = page.html.replace(/<h1\b/, `${first}<h1`);
  page.markdown = first + (first ? "\n" : "") + page.markdown;
  const candidates = new Set(legacy.pages[page.path]?.anchors ?? []);
  page.html = page.html.replace(
    /<h([1-6])\b[^>]*\bid="([^"]+)"[^>]*>([\s\S]*?)<\/h\1>/g,
    (heading, _level, id, content) => {
      const label = content
        .replace(/<[^>]*>/g, "")
        .replaceAll("&amp;", "&")
        .replaceAll("&lt;", "<")
        .replaceAll("&gt;", ">")
        .trim();
      const names = [label.replace(/\s+/g, "-")].filter((name) =>
        candidates.has(name),
      );
      if (page.path === "api") {
        names.push(
          ...Object.keys(legacy.api).filter(
            (name) => name.split("-Tuple")[0] === label,
          ),
        );
      }
      const aliases = names.map(marker).join("");
      if (aliases) {
        const anchor = `<a id="${id}"></a>`;
        if (page.markdown.includes(anchor))
          page.markdown = page.markdown.replace(
            anchor,
            `${aliases}\n${anchor}`,
          );
        else page.markdown = `${aliases}\n${page.markdown}`;
      }
      return aliases + heading;
    },
  );
  return page;
}

async function siteRoutes(root) {
  const directory = join(root, "src/routes");
  if (!(await exists(directory))) return [];
  return (await files(directory))
    .filter((path) => /\+page\.(svx|svelte)$/.test(path))
    .map((path) => {
      const route = relative(directory, dirname(path))
        .split(sep)
        .filter(Boolean)
        .join("/");
      return route ? `/${route}/` : "/";
    });
}

async function selectedDocuments(manifest, fragments, language) {
  const candidates = hostDocuments(manifest, language);
  if (language === "python") return candidates;
  const urls = documentMap(manifest, language);
  const byUrl = new Map(
    candidates.map((document) => [urls[document.path], document]),
  );
  const selected = new Set(candidates.filter((document) => document.visible));
  for (const document of selected) {
    const html = await readFile(join(fragments, document.path), "utf8");
    for (const [, target] of html.matchAll(/\bhref="([^"]+)"/g)) {
      const url = new URL(
        target.replaceAll("&amp;", "&"),
        `https://eunoia.bz${urls[document.path]}`,
      );
      if (url.origin === "https://eunoia.bz" && byUrl.has(url.pathname))
        selected.add(byUrl.get(url.pathname));
    }
  }
  return [...selected];
}

async function assertOwned(directory, language) {
  if (!(await exists(directory))) return;
  if (!(await exists(join(directory, ownership)))) {
    throw new Error(
      `Refusing to replace unowned documentation directory: ${directory}`,
    );
  }
  const manifest = await json(join(directory, ownership));
  if (
    manifest.generator !== `eunoia-${language}-docs` ||
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
  language = "python",
  sourceRevision = null,
}) {
  const {
    mount,
    assets: assetMount,
    title: languageTitle,
  } = profiles[language];
  const routes = join(root, "src/routes/docs/reference", language);
  const assets = join(root, "static/_diplodocus", language);
  await assertOwned(routes, language);
  await assertOwned(assets, language);
  const documents = await selectedDocuments(manifest, fragments, language);
  const urlMap = documentMap(manifest, language);
  const urls = new Set([
    mount,
    ...documents.map((document) => urlMap[document.path]),
  ]);
  const allowedUrls = new Set([...urls, ...(await siteRoutes(root))]);
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
        `${JSON.stringify(search.filter((entry) => documents.some((document) => document.visible && urlMap[document.path] === entry.path)))}\n`,
      );
    }
    const index =
      language === "python" ? indexContent(documents, version) : null;
    const pages = index
      ? [
          {
            path: "",
            title: "Python API",
            html: index.html,
            markdown: index.markdown,
          },
        ]
      : [];
    const legacy =
      language === "julia"
        ? await json(join(webRoot, "julia-docs-legacy.json"))
        : null;
    if (legacy) {
      await writeFile(
        join(stagedAssets, "redirects.json"),
        `${JSON.stringify({
          format_version: 1,
          generator: "eunoia-julia-docs",
          base_url: "https://eunoia.bz/docs/reference/julia/",
          package_version: version,
          source_revision: sourceRevision,
          pages: {
            "": "",
            api: "api",
            gallery: "gallery",
            CHANGELOG: "changelog",
            changelog: "changelog",
          },
          api: legacy.api,
        })}\n`,
      );
    }
    for (const document of documents) {
      const html = await readFile(join(fragments, document.path), "utf8");
      validateLinks(html, urlMap[document.path], allowedUrls, assetUrls);
      const page = {
        path: urlMap[document.path].slice(mount.length).replace(/\/$/, ""),
        title:
          document.public_name ??
          (language === "julia" && urlMap[document.path] === mount
            ? "Eunoia for Julia"
            : document.title),
        document,
        visible: document.visible,
        html,
        markdown: await readFile(
          join(fragments, document.markdown_path),
          "utf8",
        ),
      };
      pages.push(legacy ? juliaAliases(page, legacy) : page);
    }
    if (checkCoverage) profiles[language].validate(pages);
    for (const page of pages) {
      const directory = join(stagedRoutes, page.path);
      await mkdir(directory, { recursive: true });
      await writeFile(
        join(directory, "+page.svelte"),
        pageSource(
          page.html,
          assetUrls.has(`${assetMount}katex/katex.min.css`),
          language,
          page.visible !== false,
        ),
      );
      await writeFile(
        join(directory, "+page.ts"),
        `export const load = () => (${JSON.stringify({ title: page.title, description: `${page.title}: eunoia ${version} ${languageTitle} documentation.` })});\n`,
      );
      await writeFile(join(directory, "reference.md"), page.markdown);
    }
    const metadata = {
      generator: `eunoia-${language}-docs`,
      pages: pages.map((page) => page.path),
      visible: pages
        .filter((page) => page.visible !== false)
        .map((page) => page.path),
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

export function configuration(language, source, revision) {
  const common = `project:
  name: "Eunoia"
repositories:
  - id: ${language}
    path: ${JSON.stringify(source)}
    url: "https://github.com/jolars/${language === "python" ? "eunoia-py" : "Eunoia.jl"}"
    revision: ${JSON.stringify(revision)}
packages:
  - id: ${language}
    name: "Eunoia for ${profiles[language].title}"
    slug: ${language}
    ecosystem: ${language}
    repository: ${language}
    path: "."
    metadata-path: ${language === "python" ? "pyproject.toml" : "Project.toml"}
    targets:
      - id: api
        extractor: ${language}
        path: ${language === "python" ? "python/eunoia" : "src/Eunoia.jl"}
        role: public-api
`;
  if (language === "python") return common;
  return (
    common +
    `        julia:
          extensions: ["ext/EunoiaMakieExt.jl"]
content:
  - id: julia-manual
    owner: julia
    repository: julia
    path: docs/src
    mount: ""
    format: documenter
    execution:
      mode: execute
      engine: julia
      julia-project: docs/Project.toml
`
  );
}

export async function buildPackageDocs(language, options = {}) {
  const pin = await json(join(webRoot, "docs-sources.lock.json"));
  const workspace = join(projectRoot, ".diplodocus", language);
  await mkdir(workspace, { recursive: true });
  const suppliedBinary = options.diplodocusBin ?? process.env.DIPLODOCUS_BIN;
  const generator = suppliedBinary
    ? null
    : await checkout(
        "diplodocus",
        pin.diplodocus,
        options.diplodocusSource ?? process.env.DIPLODOCUS_SOURCE,
      );
  const snapshot =
    options.snapshot ??
    process.env[`EUNOIA_${language.toUpperCase()}_DOCS_SNAPSHOT`];
  const source = snapshot
    ? null
    : await checkout(
        language === "python" ? "eunoia-py" : "eunoia-jl",
        pin[language],
        options[`${language}Source`] ??
          process.env[`EUNOIA_${language.toUpperCase()}_SOURCE`],
      );
  const prepared = source
    ? await prepareEnvironment(language, source, workspace, options)
    : { env: {} };
  const extractionSource = prepared.source;
  const inputs = [
    ...(generator
      ? await sourceInputs(
          generator,
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
        "scripts/package-docs.mjs",
        "scripts/docs-common.mjs",
        "scripts/docs-environment.mjs",
        "scripts/python-docs-coverage.mjs",
        "scripts/julia-docs-coverage.mjs",
        "docs-sources.lock.json",
        "julia-docs-legacy.json",
      ],
      "host",
    )),
  ];
  if (snapshot) inputs.push(["snapshot", resolve(snapshot)]);
  else
    inputs.push(
      ...(await sourceInputs(
        extractionSource,
        language === "python"
          ? [
              "python/eunoia",
              "pyproject.toml",
              "Cargo.toml",
              "Cargo.lock",
              "src",
            ]
          : [
              "src",
              "ext",
              "docs/src",
              "docs/Project.toml",
              "docs/Manifest.toml",
              "Project.toml",
              "Manifest.toml",
              "Artifacts.toml",
              "CHANGELOG.md",
              "LICENSE",
            ],
        language,
      )),
    );
  if (source) inputs.push(["environment", join(workspace, "environment.json")]);
  const revisions = {
    diplodocus: generator
      ? run("git", ["rev-parse", "HEAD"], generator, true)
      : null,
    source: source ? run("git", ["rev-parse", "HEAD"], source, true) : null,
  };
  const key = createHash("sha256")
    .update(await digest(inputs))
    .update(JSON.stringify(revisions))
    .digest("hex");
  const cachePath = join(workspace, "cache.json");
  const routeRoot = join(webRoot, "src/routes/docs/reference", language);
  const assetRoot = join(webRoot, "static/_diplodocus", language);
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
  const binary = await buildDiplodocus(
    generator,
    pin.diplodocus,
    suppliedBinary,
  );
  const database = snapshot
    ? resolve(snapshot)
    : join(workspace, "documentation.sqlite");
  if (source) {
    const configPath = join(workspace, "diplodocus.yaml");
    await writeFile(
      configPath,
      configuration(language, extractionSource, revisions.source),
    );
    run(
      binary,
      ["extract", "--config", configPath, "--output", database],
      projectRoot,
      false,
      { ...process.env, ...prepared.env },
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
    documents: documentMap(manifest, language),
    assets: {},
  };
  for (const path of await files(join(fragments, "assets"))) {
    const suffix = relative(join(fragments, "assets"), path)
      .split(sep)
      .join("/");
    mappings.assets[`assets/${suffix}`] = profiles[language].assets + suffix;
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
    apiDocuments(finalManifest, language)[0].package_version ??
    pin[language].version;
  const count = await publishFragments({
    root: webRoot,
    fragments,
    manifest: finalManifest,
    version,
    language,
    sourceRevision: revisions.source,
    checkCoverage: true,
  });
  await writeFile(
    cachePath,
    `${JSON.stringify({ key, output: await generatedDigest(routeRoot, assetRoot), pages: count })}\n`,
  );
  console.log(
    `Generated ${count} ${profiles[language].title} documentation pages with Diplodocus.`,
  );
}

export function cliOptions(arguments_ = process.argv.slice(2)) {
  const names = {
    "--python-source": "pythonSource",
    "--julia-source": "juliaSource",
    "--diplodocus-source": "diplodocusSource",
    "--diplodocus-bin": "diplodocusBin",
    "--snapshot": "snapshot",
    "--python": "python",
    "--julia": "julia",
  };
  const options = {};
  for (let index = 0; index < arguments_.length; index += 2) {
    const name = names[arguments_[index]];
    if (!name || !arguments_[index + 1])
      throw new Error(`Unknown or incomplete option: ${arguments_[index]}`);
    options[name] = arguments_[index + 1];
  }
  return options;
}
