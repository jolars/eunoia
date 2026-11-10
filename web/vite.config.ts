import { svelte } from "@sveltejs/vite-plugin-svelte";
import tailwindcss from "@tailwindcss/vite";
import { defineConfig } from "vite";
import wasm from "vite-plugin-wasm";
import { VERSION } from "./version.config.js";

export default defineConfig({
  define: { __EUNOIA_VERSION__: JSON.stringify(VERSION) },
  base: "./",
  publicDir: false,
  plugins: [wasm(), svelte({ configFile: false }), tailwindcss()],
  worker: { format: "es", plugins: () => [wasm()] },
  css: {
    postcss: {
      plugins: [
        {
          postcssPlugin: "eunoia-widget-scope",
          OnceExit(root) {
            root.walkRules((rule) => {
              let parent: typeof rule.parent | typeof root.parent = rule.parent;
              while (parent) {
                if (
                  parent.type === "rule" ||
                  (parent.type === "atrule" && /keyframes$/.test(parent.name))
                )
                  return;
                parent = parent.parent;
              }
              // Tailwind's reset and utilities belong to the interactive island.
              // Prefixing them keeps the surrounding Diplodocus theme independent.
              rule.selectors = rule.selectors.map((selector: string) => {
                if (
                  selector === ":root" ||
                  selector === ":host" ||
                  selector === "body"
                )
                  return ".eunoia-widget";
                if (selector === ".dark") return ".eunoia-widget.dark";
                return `.eunoia-widget ${selector}`;
              });
            });
          },
        },
      ],
    },
  },
  build: {
    target: "esnext",
    outDir: ".widgets",
    emptyOutDir: true,
    cssCodeSplit: false,
    rollupOptions: {
      input: "src/widgets/index.ts",
      preserveEntrySignatures: "strict",
      output: {
        entryFileNames: "index.js",
        chunkFileNames: "chunks/[name]-[hash].js",
        assetFileNames: (asset) =>
          asset.names.some((name) => name.endsWith(".css"))
            ? "style.css"
            : "assets/[name]-[hash][extname]",
      },
    },
  },
});
