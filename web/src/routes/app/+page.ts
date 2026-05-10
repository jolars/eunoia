// The diagram tool runs entirely in the browser (WASM worker, large
// transitive WASM bundle from @jolars/eunoia via DiagramSvg). Prerender
// the empty HTML shell, but don't try to SSR the page module.
export const ssr = false;
