import { buildPackageDocs, cliOptions } from "./package-docs.mjs";

const options = cliOptions();
for (const language of ["python", "julia"])
  await buildPackageDocs(language, options);
