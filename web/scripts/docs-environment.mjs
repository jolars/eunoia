import { createHash } from "node:crypto";
import { cp, mkdir, rm, writeFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import {
  digest,
  exists,
  json,
  projectRoot,
  run,
  sourceInputs,
} from "./docs-common.mjs";

async function stageJulia(source, workspace) {
  const paths = [
    "src",
    "ext",
    "docs/src",
    "docs/Project.toml",
    "docs/Manifest.toml",
    "Project.toml",
    "Artifacts.toml",
    "CHANGELOG.md",
    "LICENSE",
    "README.md",
    "CITATION.cff",
  ];
  const key = await digest(await sourceInputs(source, paths, "julia"));
  const directory = join(projectRoot, ".docs-sources", "eunoia-jl-prepared");
  const marker = join(workspace, "source.json");
  if (
    (await exists(marker)) &&
    (await exists(join(directory, "Project.toml"))) &&
    (await json(marker)).key === key
  )
    return directory;
  await rm(directory, { recursive: true, force: true });
  for (const path of paths) {
    if (!(await exists(join(source, path)))) continue;
    await mkdir(dirname(join(directory, path)), { recursive: true });
    await cp(join(source, path), join(directory, path), { recursive: true });
  }
  if (await exists(join(source, "CHANGELOG.md"))) {
    await cp(
      join(source, "CHANGELOG.md"),
      join(directory, "docs/src/CHANGELOG.md"),
    );
  }
  await writeFile(marker, `${JSON.stringify({ key })}\n`);
  return directory;
}

export async function prepareEnvironment(
  language,
  originalSource,
  workspace,
  options = {},
) {
  await mkdir(workspace, { recursive: true });
  const source =
    language === "julia"
      ? await stageJulia(originalSource, workspace)
      : originalSource;
  const marker = join(workspace, "environment.json");
  const selectedRuntime =
    language === "python"
      ? (options.python ??
        process.env.EUNOIA_DOCS_PYTHON ??
        process.env.UV_PYTHON ??
        "python3")
      : (options.julia ?? process.env.EUNOIA_DOCS_JULIA ?? "julia");
  const runtime =
    language === "python"
      ? run("uv", ["python", "find", selectedRuntime], source, true)
      : selectedRuntime;
  const observed = run(runtime, ["--version"], source, true);
  const native = process.env.EUNOIA_CAPI_LIB;
  const inputs =
    language === "python"
      ? ["pyproject.toml", "Cargo.toml", "Cargo.lock", "src"]
      : [
          "Project.toml",
          "Manifest.toml",
          "Artifacts.toml",
          "docs/Project.toml",
          "docs/Manifest.toml",
        ];
  const key = async () =>
    createHash("sha256")
      .update(
        await digest([
          ...(await sourceInputs(source, inputs, language)),
          ...(language === "julia" && native
            ? [["native-library", resolve(native)]]
            : []),
        ]),
      )
      .update(
        JSON.stringify({
          source,
          runtime,
          observed,
          depot: process.env.JULIA_DEPOT_PATH,
        }),
      )
      .digest("hex");
  const interpreter =
    language === "python"
      ? join(
          workspace,
          "venv",
          process.platform === "win32" ? "Scripts/python.exe" : "bin/python",
        )
      : runtime;
  if (await exists(marker)) {
    const cached = await json(marker);
    if (
      cached.key === (await key()) &&
      (language !== "python" || (await exists(interpreter)))
    ) {
      return { source, env: cached.env };
    }
  }
  await mkdir(workspace, { recursive: true });
  if (language === "python") {
    run(
      "uv",
      [
        "venv",
        "--allow-existing",
        "--python",
        runtime,
        join(workspace, "venv"),
      ],
      source,
    );
    run(
      "uv",
      [
        "pip",
        "install",
        "--python",
        interpreter,
        "--editable",
        source,
        "--config-setting=build-args=--locked",
        "--reinstall-package",
        "eunoia",
        "--no-cache",
      ],
      source,
    );
    run(
      interpreter,
      ["-c", "import eunoia; print('Prepared eunoia', eunoia.__version__)"],
      source,
    );
  } else {
    const env = { ...process.env, JULIA_PKG_PRECOMPILE_AUTO: "0" };
    run(
      runtime,
      [
        "--startup-file=no",
        `--project=${source}`,
        "-e",
        "using Pkg; Pkg.instantiate()",
      ],
      source,
      false,
      env,
    );
    run(
      runtime,
      [
        "--startup-file=no",
        `--project=${join(source, "docs")}`,
        "-e",
        "using Pkg; Pkg.develop(PackageSpec(path=ARGS[1])); Pkg.instantiate()",
        source,
      ],
      source,
      false,
      env,
    );
    run(
      runtime,
      [
        "--startup-file=no",
        `--project=${join(source, "docs")}`,
        "-e",
        'using Eunoia, CairoMakie; CairoMakie.activate!(type="png"); println("Prepared native Eunoia ", Eunoia.version())',
      ],
      source,
      false,
      env,
    );
  }
  const env =
    language === "python"
      ? {
          PATH: `${dirname(interpreter)}${process.platform === "win32" ? ";" : ":"}${process.env.PATH}`,
          MPLBACKEND: "Agg",
        }
      : {
          PATH: `${dirname(run(runtime, ["--startup-file=no", "-e", "print(joinpath(Sys.BINDIR, Base.julia_exename()))"], source, true))}${process.platform === "win32" ? ";" : ":"}${process.env.PATH}`,
        };
  await writeFile(marker, `${JSON.stringify({ key: await key(), env })}\n`);
  return { source, env };
}
