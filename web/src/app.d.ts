/// <reference types="unplugin-icons/types/svelte" />
/// <reference types="vite/client" />

// See https://svelte.dev/docs/kit/types#app
declare global {
  // Injected at build time by Vite's `define` (see vite.config.ts).
  const __EUNOIA_VERSION__: string;

  namespace App {
    // interface Error {}
    // interface Locals {}
    // interface PageData {}
    // interface PageState {}
    // interface Platform {}
  }
}

export {};
