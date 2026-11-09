import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { buildPackageDocs, cliOptions } from "./package-docs.mjs";

export const buildJuliaDocs = (options = {}) =>
  buildPackageDocs("julia", options);
if (
  process.argv[1] &&
  fileURLToPath(import.meta.url) === resolve(process.argv[1])
) {
  await buildJuliaDocs(cliOptions());
}
