import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { buildPackageDocs, cliOptions } from "./package-docs.mjs";

export {
  apiDocuments,
  checkout,
  documentMap,
  publishFragments,
  validateLinks,
} from "./package-docs.mjs";
export const buildPythonDocs = (options = {}) =>
  buildPackageDocs("python", options);
if (
  process.argv[1] &&
  fileURLToPath(import.meta.url) === resolve(process.argv[1])
) {
  await buildPythonDocs(cliOptions());
}
