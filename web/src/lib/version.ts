// Published version, injected at build time from `ts/package.json`
// (see `vite.config.ts` and `version.config.js`). Updates automatically with
// each release; nothing here needs manual bumping.

/** Full version, e.g. "1.5.0". */
export const VERSION = __EUNOIA_VERSION__;

/** GitHub release matching the current build. */
export const RELEASE_URL = `https://github.com/jolars/eunoia/releases/tag/v${VERSION}`;
