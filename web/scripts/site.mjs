import { createHash } from "node:crypto";
import { cp, mkdir, mkdtemp, rename, rm, writeFile } from "node:fs/promises";
import { delimiter, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { configuration } from "../site.config.mjs";
import { validateSite } from "./check-site.mjs";
import {
  buildDiplodocus,
  checkout,
  digest,
  exists,
  json,
  projectRoot,
  run,
  sourceInputs,
  webRoot,
} from "./docs-common.mjs";
import { prepareEnvironment } from "./docs-environment.mjs";
import { finishSite } from "./site-output.mjs";

export function cliOptions(args = process.argv.slice(3)) {
  const names = {
    "--python-source": "pythonSource",
    "--julia-source": "juliaSource",
    "--diplodocus-source": "diplodocusSource",
    "--diplodocus-bin": "diplodocusBin",
    "--snapshot": "snapshot",
    "--python": "python",
    "--julia": "julia",
    "--port": "port",
    "--host": "host",
  };
  const options = {};
  for (let i = 0; i < args.length; i += 2) {
    const name = names[args[i]];
    if (!name || !args[i + 1])
      throw new Error(`Unknown or incomplete option: ${args[i]}`);
    options[name] = args[i + 1];
  }
  return options;
}

export async function prepareSources(options = {}) {
  const pins = await json(join(webRoot, "docs-sources.lock.json"));
  const sources = {},
    revisions = {},
    environments = [];
  for (const language of ["python", "julia"]) {
    const source = await checkout(
      language === "python" ? "eunoia-py" : "eunoia-jl",
      pins[language],
      options[`${language}Source`] ??
        process.env[`EUNOIA_${language.toUpperCase()}_SOURCE`],
    );
    revisions[language] = run("git", ["rev-parse", "HEAD"], source, true);
    const prepared = await prepareEnvironment(
      language,
      source,
      join(projectRoot, ".diplodocus", language),
      options,
    );
    sources[language] = prepared.source;
    environments.push(prepared.env);
  }
  const paths = [
    ...environments.map((env) => env.PATH?.split(delimiter)[0]).filter(Boolean),
    process.env.PATH,
  ];
  const env = {
    ...process.env,
    ...Object.assign({}, ...environments),
    PATH: [...new Set(paths)].join(delimiter),
  };
  return { pins, sources, revisions, env };
}

export async function generator(options = {}) {
  const pins = await json(join(webRoot, "docs-sources.lock.json"));
  const supplied = options.diplodocusBin ?? process.env.DIPLODOCUS_BIN;
  const source = supplied
    ? null
    : await checkout(
        "diplodocus",
        pins.diplodocus,
        options.diplodocusSource ?? process.env.DIPLODOCUS_SOURCE,
      );
  return buildDiplodocus(source, pins.diplodocus, supplied);
}

export async function extractSite(
  binary,
  options,
  { widgets = true, check = false } = {},
) {
  const prepared = await prepareSources(options);
  if (widgets) run("pnpm", ["exec", "vite", "build"], webRoot);
  const cache = join(projectRoot, ".diplodocus/site");
  await mkdir(cache, { recursive: true });
  const configPath = join(cache, "diplodocus.yaml");
  await writeFile(
    configPath,
    `${JSON.stringify(configuration(projectRoot, prepared.sources, prepared.revisions), null, 2)}\n`,
  );
  if (check) {
    run(
      binary,
      ["check", "--config", configPath],
      projectRoot,
      false,
      prepared.env,
    );
    return;
  }
  const inputs = [
    ["generator", binary],
    ...(await sourceInputs(
      webRoot,
      [
        "content",
        "python",
        ".widgets",
        "static",
        "site.config.mjs",
        "legacy-routes.json",
        "docs-sources.lock.json",
        "scripts/site.mjs",
        "scripts/docs-common.mjs",
        "scripts/docs-environment.mjs",
      ],
      "site",
    )),
    ...(await sourceInputs(
      prepared.sources.python,
      ["python/eunoia", "pyproject.toml", "Cargo.toml", "Cargo.lock", "src"],
      "python",
    )),
    ...(await sourceInputs(
      prepared.sources.julia,
      [
        "src",
        "ext",
        "docs/src",
        "docs/Project.toml",
        "docs/Manifest.toml",
        "Project.toml",
        "Artifacts.toml",
        "README.md",
        "LICENSE",
      ],
      "julia",
    )),
    ...(await sourceInputs(
      projectRoot,
      [
        ".diplodocus/python/environment.json",
        ".diplodocus/julia/environment.json",
      ],
      "environments",
    )),
    ["configuration", configPath],
  ];
  const key = createHash("sha256")
    .update(await digest(inputs))
    .update(JSON.stringify(prepared.revisions))
    .digest("hex");
  const database = join(cache, "documentation.sqlite");
  const marker = join(cache, "cache.json");
  const databaseDigest = () => digest([["snapshot", database]]);
  const cached =
    (await exists(marker)) && (await exists(database))
      ? await json(marker)
      : null;
  if (cached?.key !== key || cached.snapshot !== (await databaseDigest())) {
    run(
      binary,
      ["extract", "--config", configPath, "--output", database],
      projectRoot,
      false,
      prepared.env,
    );
    await writeFile(
      marker,
      `${JSON.stringify({ key, snapshot: await databaseDigest() })}\n`,
    );
  }
  return database;
}

export async function generateSite(binary, database) {
  const cache = join(projectRoot, ".diplodocus/site");
  await mkdir(cache, { recursive: true });
  const workspace = await mkdtemp(join(cache, "build-"));
  const staging = join(workspace, "site");
  const output = join(webRoot, "dist");
  const previous = join(workspace, "previous");
  try {
    const native = join(cache, "rendered");
    const marker = join(cache, "render-cache.json");
    const key = await digest([
      ["snapshot", database],
      ["generator", binary],
    ]);
    const outputDigest = async () =>
      digest(await sourceInputs(native, ["."], "rendered"));
    const cached =
      (await exists(marker)) && (await exists(native))
        ? await json(marker)
        : null;
    if (cached?.key !== key || cached.output !== (await outputDigest())) {
      run(
        binary,
        [
          "generate",
          "--input",
          database,
          "--allow-components",
          "--output",
          native,
        ],
        projectRoot,
      );
      await writeFile(
        marker,
        `${JSON.stringify({ key, output: await outputDigest() })}\n`,
      );
    }
    await cp(native, staging, { recursive: true });
    await finishSite(staging);
    await validateSite(staging);
    const hadPrevious = await exists(output);
    if (hadPrevious) await rename(output, previous);
    try {
      await rename(staging, output);
    } catch (error) {
      if (hadPrevious) await rename(previous, output);
      throw error;
    }
    console.log(`Built the Diplodocus site in ${output}.`);
  } finally {
    await rm(workspace, { recursive: true, force: true });
  }
}

export async function buildSite(options = {}, settings = {}) {
  const binary = await generator(options);
  const snapshot = options.snapshot ?? process.env.EUNOIA_DOCS_SNAPSHOT;
  const database = snapshot
    ? resolve(snapshot)
    : await extractSite(binary, options, settings);
  if (!settings.check) await generateSite(binary, database);
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const command = process.argv[2] ?? "build";
  const options = cliOptions();
  if (command === "build") await buildSite(options);
  else if (command === "check") await buildSite(options, { check: true });
  else if (command === "extract")
    await extractSite(await generator(options), options);
  else throw new Error(`Unknown site command: ${command}`);
}
