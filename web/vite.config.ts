import { defineConfig } from 'vite'
import { svelte } from '@sveltejs/vite-plugin-svelte'
import wasm from 'vite-plugin-wasm'
import topLevelAwait from 'vite-plugin-top-level-await'

// https://vite.dev/config/
export default defineConfig({
  base: '/', // Use root path for custom domain (change if using GitHub Pages subdirectory)
  plugins: [
    wasm(),
    topLevelAwait(),
    svelte()
  ],
  build: {
    target: 'esnext'
  }
})
