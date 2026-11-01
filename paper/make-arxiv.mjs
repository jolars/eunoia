import { execFileSync } from "node:child_process";
import {
  copyFileSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { dirname, isAbsolute, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const paper = dirname(fileURLToPath(import.meta.url));
const work = mkdtempSync(join(tmpdir(), "eunoia-arxiv-"));
const source = join(work, "source");
const build = join(work, "build");
const image = "openjournals/inara";
const rerunWarning = /Rerun to get|Please (?:re)?run/;
const dockerArgs = [
  "run",
  "--rm",
  "--user",
  `${process.getuid()}:${process.getgid()}`,
  "--volume",
  `${work}:/data`,
  "--env",
  "JOURNAL=joss",
];

function run(command, args) {
  execFileSync(command, args, { stdio: "inherit" });
}

try {
  mkdirSync(source);
  mkdirSync(build);
  copyFileSync(join(paper, "paper.md"), join(work, "paper.md"));
  copyFileSync(join(paper, "paper.bib"), join(work, "paper.bib"));
  run("docker", [...dockerArgs, image, "-o", "preprint", "paper.md"]);

  // Inara's citeproc output embeds the bibliography in the LaTeX source.
  const tex = readFileSync(join(work, "paper.preprint.tex"), "utf8");
  writeFileSync(join(source, "main.tex"), tex);
  const figures = new Set();
  for (const match of tex.matchAll(
    /\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}/g,
  )) {
    const figure = match[1];
    const local = relative(paper, resolve(paper, figure));
    if (isAbsolute(figure) || local.startsWith("..")) {
      throw new Error(`Figure must be inside paper/: ${figure}`);
    }
    const destination = join(source, local);
    mkdirSync(dirname(destination), { recursive: true });
    copyFileSync(join(paper, local), destination);
    figures.add(local);
  }

  // Compile only the files that will be uploaded, without shell escape.
  let log;
  for (let pass = 0; pass < 5; pass++) {
    run("docker", [
      ...dockerArgs,
      "--workdir",
      "/data/source",
      "--entrypoint",
      "pdflatex",
      image,
      "-no-shell-escape",
      "-interaction=nonstopmode",
      "-halt-on-error",
      "-output-directory=/data/build",
      "main.tex",
    ]);
    log = readFileSync(join(build, "main.log"), "utf8").replace(/\r?\n/g, " ");
    if (!rerunWarning.test(log) && !/undefined references/.test(log)) break;
  }
  if (
    /undefined references|undefined citations|Citation .* undefined|Reference .* undefined/.test(
      log,
    )
  ) {
    throw new Error("The arXiv build has unresolved references or citations.");
  }
  if (rerunWarning.test(log)) {
    throw new Error(
      "The arXiv build did not stabilize after five LaTeX passes.",
    );
  }

  run("tar", [
    "-czf",
    join(work, "arxiv.tar.gz"),
    "-C",
    source,
    "main.tex",
    ...figures,
  ]);
  copyFileSync(join(work, "arxiv.tar.gz"), join(paper, "arxiv.tar.gz"));
  copyFileSync(join(build, "main.pdf"), join(paper, "arxiv.pdf"));
  console.log(`Upload: ${join(paper, "arxiv.tar.gz")}`);
  console.log(`Preview: ${join(paper, "arxiv.pdf")}`);
} finally {
  rmSync(work, { recursive: true, force: true });
}
