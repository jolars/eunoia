import { defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";
import wasm from "vite-plugin-wasm";
import Sitemap from "vite-plugin-sitemap";
import topLevelAwait from "vite-plugin-top-level-await";
import viteCompression from "vite-plugin-compression";

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
  build: {
    target: "esnext",
    minify: "esbuild",
    sourcemap: false,
    cssCodeSplit: true,
  },
});
