import { copyFileSync, existsSync } from "node:fs";
import { resolve } from "node:path";
import { svelte } from "@sveltejs/vite-plugin-svelte";
import Icons from "unplugin-icons/vite";
import { defineConfig, type Plugin } from "vite";
import viteCompression from "vite-plugin-compression";
import Sitemap from "vite-plugin-sitemap";
import topLevelAwait from "vite-plugin-top-level-await";
import wasm from "vite-plugin-wasm";

// GitHub Pages serves static files only — there is no SPA rewrite rule. The
// site is a single `index.html` shell that route-decides client-side, so any
// path other than `/` (e.g. `/app/`) would return GitHub's default 404. The
// well-known fix: emit `dist/404.html` as a byte-for-byte copy of
// `dist/index.html`. GH Pages uses 404.html as a fallback, the same shell
// boots, and our pathname-based router takes over.
function spa404Fallback(): Plugin {
  return {
    name: "spa-404-fallback",
    apply: "build",
    closeBundle() {
      const outDir = resolve(__dirname, "dist");
      const src = resolve(outDir, "index.html");
      const dst = resolve(outDir, "404.html");
      if (existsSync(src)) copyFileSync(src, dst);
    },
  };
}

// https://vite.dev/config/
export default defineConfig({
  base: "/", // Use root path for custom domain (change if using GitHub Pages subdirectory)
  plugins: [
    wasm(),
    topLevelAwait(),
    svelte(),
    Icons({ compiler: "svelte" }),
    Sitemap({
      hostname: "https://eunoia.bz",
      dynamicRoutes: ["/", "/app/", "/about/", "/cite/"],
    }),
    viteCompression({ algorithm: "brotliCompress" }),
    spa404Fallback(),
  ],
  worker: {
    format: "es",
    plugins: () => [wasm(), topLevelAwait()],
  },
  build: {
    target: "esnext",
    minify: "esbuild",
    sourcemap: false,
    cssCodeSplit: true,
  },
});
