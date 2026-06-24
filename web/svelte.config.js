import adapter from "@sveltejs/adapter-static";
import { vitePreprocess } from "@sveltejs/vite-plugin-svelte";
import { escapeSvelte, mdsvex } from "mdsvex";
import rehypeSlug from "rehype-slug";
import { createHighlighter } from "shiki";

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
];
const highlighter = await createHighlighter({
  themes: ["github-light", "github-dark"],
  langs: LANGS,
});

/** @type {import("@sveltejs/kit").Config} */
export default {
  extensions: [".svelte", ".svx"],
  preprocess: [
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
  kit: {
    adapter: adapter({
      pages: "dist",
      assets: "dist",
      fallback: undefined,
      strict: true,
    }),
    prerender: {
      entries: ["*"],
    },
  },
};
