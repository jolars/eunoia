import { join } from "node:path";
import { checkout, json, projectRoot, webRoot } from "./docs-common.mjs";
import { prepareEnvironment } from "./docs-environment.mjs";
import { cliOptions } from "./package-docs.mjs";

const options = cliOptions();
const pins = await json(join(webRoot, "docs-sources.lock.json"));
for (const language of ["python", "julia"]) {
  if (
    options.snapshot ||
    process.env[`EUNOIA_${language.toUpperCase()}_DOCS_SNAPSHOT`]
  )
    continue;
  const source = await checkout(
    language === "python" ? "eunoia-py" : "eunoia-jl",
    pins[language],
    options[`${language}Source`] ??
      process.env[`EUNOIA_${language.toUpperCase()}_SOURCE`],
  );
  await prepareEnvironment(
    language,
    source,
    join(projectRoot, ".diplodocus", language),
    options,
  );
}
