import { svelte } from "@sveltejs/vite-plugin-svelte";
import { defineConfig } from "vite";
import viteCompression from "vite-plugin-compression";
import Sitemap from "vite-plugin-sitemap";
import topLevelAwait from "vite-plugin-top-level-await";
import wasm from "vite-plugin-wasm";

// https://vite.dev/config/
export default defineConfig({
  base: "/", // Use root path for custom domain (change if using GitHub Pages subdirectory)
  plugins: [
    wasm(),
    topLevelAwait(),
    svelte(),
    Sitemap({ hostname: "https://eunoia.fit" }),
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
