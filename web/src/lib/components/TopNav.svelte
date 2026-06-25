<script lang="ts">
  import { page } from "$app/state";
  import ThemeToggle from "$lib/components/ThemeToggle.svelte";
  import IconMenu from "~icons/lucide/menu";
  import IconX from "~icons/lucide/x";
  import IconGithub from "~icons/simple-icons/github";

  // Source repository — shown as a persistent icon link that never collapses
  // into the mobile menu.
  const repoUrl = "https://github.com/jolars/eunoia";

  // Shared nav for every route. Active route is highlighted by matching
  // the current pathname against each entry's `match` predicate.
  const currentPath = $derived(page.url.pathname);

  type NavItem = {
    label: string;
    href: string;
    match: (path: string) => boolean;
  };

  // Order is left→right. Home gets the wordmark slot, not a nav button.
  const items: NavItem[] = [
    { label: "Home", href: "/", match: (p) => p === "/" },
    { label: "App", href: "/app/", match: (p) => p.startsWith("/app") },
    { label: "Docs", href: "/docs/", match: (p) => p.startsWith("/docs") },
    { label: "Cite", href: "/cite/", match: (p) => p.startsWith("/cite") },
  ];

  // Below the `sm` breakpoint the links collapse into a hamburger toggle.
  // Close the panel whenever the route changes so it never lingers.
  let menuOpen = $state(false);
  $effect(() => {
    void currentPath;
    menuOpen = false;
  });
</script>

<nav class="border-b border-line bg-surface">
  <div class="max-w-7xl mx-auto px-6 flex items-center justify-between">
    <a href="/" class="text-2xl font-bold py-4 text-ink">Eunoia</a>

    <!-- Right cluster: nav links (sm+), then the always-visible GitHub link
         and the mobile-only hamburger. -->
    <div class="flex items-center gap-1">
      <!-- Desktop nav: horizontal list at `sm` and up. -->
      <ul class="hidden sm:flex items-center gap-1">
        {#each items as item}
          {@const active = item.match(currentPath)}
          <li>
            <a
              href={item.href}
              class="px-4 py-4 text-sm font-medium border-b-2 transition-colors inline-block"
              class:border-accent={active}
              class:text-accent={active}
              class:border-transparent={!active}
              class:text-muted={!active}
              class:hover:text-ink={!active}>{item.label}</a
            >
          </li>
        {/each}
      </ul>

      <!-- Theme cycle: system → light → dark. -->
      <ThemeToggle />

      <!-- GitHub: persistent at every breakpoint, outside the hamburger. -->
      <a
        href={repoUrl}
        class="inline-flex items-center justify-center p-2 text-faint hover:text-ink transition-colors"
        aria-label="Source on GitHub"
      >
        <IconGithub class="w-5 h-5" />
      </a>

      <!-- Mobile toggle: shown only below `sm`. -->
      <button
        type="button"
        class="sm:hidden inline-flex items-center justify-center p-2 -mr-2 text-faint hover:text-ink transition-colors"
        aria-label="Toggle navigation menu"
        aria-controls="mobile-nav"
        aria-expanded={menuOpen}
        onclick={() => (menuOpen = !menuOpen)}
      >
        {#if menuOpen}
          <IconX class="w-6 h-6" />
        {:else}
          <IconMenu class="w-6 h-6" />
        {/if}
      </button>
    </div>
  </div>

  <!-- Mobile panel: stacked links below `sm`, shown when toggled. Kept in
       the DOM (toggled via `hidden`) so `aria-controls` stays valid. -->
  <ul
    id="mobile-nav"
    class="sm:hidden border-t border-line px-6 py-2"
    class:hidden={!menuOpen}
  >
    {#each items as item}
      {@const active = item.match(currentPath)}
      <li>
        <a
          href={item.href}
          class="block px-2 py-3 text-sm font-medium border-l-2 transition-colors"
          class:border-accent={active}
          class:text-accent={active}
          class:border-transparent={!active}
          class:text-muted={!active}
          class:hover:text-ink={!active}
          onclick={() => (menuOpen = false)}>{item.label}</a
        >
      </li>
    {/each}
  </ul>
</nav>
