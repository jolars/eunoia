import adapter from "@sveltejs/adapter-static";
import { sveltekit } from "@sveltejs/kit/vite";
import { vitePreprocess } from "@sveltejs/vite-plugin-svelte";
import tailwindcss from "@tailwindcss/vite";
import { escapeSvelte, mdsvex } from "mdsvex";
import rehypeSlug from "rehype-slug";
import { createHighlighter } from "shiki";
import Icons from "unplugin-icons/vite";
import { defineConfig } from "vite";
import viteCompression from "vite-plugin-compression";
import topLevelAwait from "vite-plugin-top-level-await";
import wasm from "vite-plugin-wasm";
import { VERSION, VERSION_MINOR } from "./version.config.js";

// Replace version tokens in markup before mdsvex/shiki run, so the real
// version is baked into highlighted code fences (where Svelte expressions and
// Vite's `define` can't reach). `%EUNOIA_VERSION%` → the full version,
// `%EUNOIA_VERSION_MINOR%` → major.minor. Must run before mdsvex below.
const versionTokens = {
  name: "eunoia-version-tokens",
  markup({ content }: { content: string }) {
    if (!content.includes("%EUNOIA_VERSION")) return;
    const code = content
      .replaceAll("%EUNOIA_VERSION_MINOR%", VERSION_MINOR)
      .replaceAll("%EUNOIA_VERSION%", VERSION);
    return { code };
  },
};

// Build-time syntax highlighter. Shiki runs only during preprocess/prerender
// (Node), so none of it ships to the client. Dual-theme output emits
// `--shiki-light`/`--shiki-dark` CSS variables (defaultColor: false); app.css
// maps them to the active theme via the `.dark` class toggle.
const LANGS = [
  "rust",
  "bash",
  "toml",
  "r",
  "python",
  "julia",
  "js",
  "ts",
  "html",
  "c",
  "json",
];
const highlighter = await createHighlighter({
  themes: ["github-light", "github-dark"],
  langs: LANGS,
});

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
    sveltekit({
      extensions: [".svelte", ".svx"],
      preprocess: [
        versionTokens,
        vitePreprocess(),
        mdsvex({
          extensions: [".svx"],
          // Give headings stable ids so in-page anchor links (and
          // prose-headings:scroll-mt-20) resolve.
          rehypePlugins: [rehypeSlug],
          highlight: {
            highlighter(code, lang) {
              const language = lang && LANGS.includes(lang) ? lang : "text";
              const html = highlighter.codeToHtml(code, {
                lang: language,
                themes: { light: "github-light", dark: "github-dark" },
                defaultColor: false,
              });
              // Escape so Svelte doesn't parse `{`, backticks, etc. in the code.
              return `{@html \`${escapeSvelte(html)}\`}`;
            },
          },
        }),
      ],
      adapter: adapter({
        pages: "dist",
        assets: "dist",
        fallback: undefined,
        strict: true,
      }),
      prerender: {
        entries: ["*"],
      },
    }),
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
