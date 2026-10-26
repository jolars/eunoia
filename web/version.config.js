import { readFileSync } from "node:fs";

// Single source of truth for the version shown in the web app and docs.
// `ts/package.json` is kept in lockstep with the Rust crate by versionary
// (the `ts` package `follows` `.` in versionary.jsonc), so this is the
// published crate/npm version. Read once at build time by `vite.config.ts`
// for the `__EUNOIA_VERSION__` define and `%EUNOIA_VERSION%` markdown tokens.
const pkg = JSON.parse(
  readFileSync(new URL("../ts/package.json", import.meta.url), "utf8"),
);

/** Full version, e.g. "1.5.0". */
export const VERSION = pkg.version;

/** Major.minor only, e.g. "1.5" — for the `eunoia = "1.5"` install snippet. */
export const VERSION_MINOR = VERSION.split(".").slice(0, 2).join(".");
