import { sveltekit } from "@sveltejs/kit/vite";
import tailwindcss from "@tailwindcss/vite";
import Icons from "unplugin-icons/vite";
import { defineConfig } from "vite";
import viteCompression from "vite-plugin-compression";
import topLevelAwait from "vite-plugin-top-level-await";
import wasm from "vite-plugin-wasm";
import { VERSION } from "./version.config.js";

// https://vite.dev/config/
export default defineConfig({
  // Build-time version, surfaced to the app as `__EUNOIA_VERSION__`
  // (see `src/lib/version.ts` and `src/app.d.ts`).
  define: {
    __EUNOIA_VERSION__: JSON.stringify(VERSION),
  },
  plugins: [
    wasm(),
    topLevelAwait(),
    tailwindcss(),
    sveltekit(),
    Icons({ compiler: "svelte" }),
    viteCompression({ algorithm: "brotliCompress" }),
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
