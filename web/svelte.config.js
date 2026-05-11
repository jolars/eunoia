import adapter from "@sveltejs/adapter-static";
import { vitePreprocess } from "@sveltejs/vite-plugin-svelte";
import { mdsvex } from "mdsvex";

/** @type {import("@sveltejs/kit").Config} */
export default {
  extensions: [".svelte", ".svx"],
  preprocess: [
    vitePreprocess(),
    // No syntax highlighter wired up yet — fenced code renders as plain
    // <pre><code>. Disabling the default Prism highlighter silences the
    // "failed to load language *" warnings on chapters with code blocks.
    mdsvex({ extensions: [".svx"], highlight: false }),
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
