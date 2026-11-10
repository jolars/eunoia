import { readFileSync } from "node:fs";

// Single source of truth for the version shown in the web app and docs.
// `ts/package.json` is kept in lockstep with the Rust crate by versionary
// (the `ts` package `follows` `.` in versionary.jsonc), so this is the
// published crate/npm version. Vite injects `__EUNOIA_VERSION__` into widgets,
// and the site publisher resolves version tokens in generated documentation.
const pkg = JSON.parse(
  readFileSync(new URL("../ts/package.json", import.meta.url), "utf8"),
);

/** Full version, e.g. "1.5.0". */
export const VERSION = pkg.version;

/** Major.minor only, e.g. "1.5" — for the `eunoia = "1.5"` install snippet. */
export const VERSION_MINOR = VERSION.split(".").slice(0, 2).join(".");
