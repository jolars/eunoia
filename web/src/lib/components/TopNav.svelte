<script lang="ts">
  // Shared nav for every route. Active route is highlighted by matching
  // `currentPath` against each entry's `match` predicate.
  interface Props {
    currentPath: string;
  }
  let { currentPath }: Props = $props();

  type NavItem = {
    label: string;
    href: string;
    match: (path: string) => boolean;
  };

  // Order is left→right. Home gets the wordmark slot, not a nav button.
  const items: NavItem[] = [
    { label: "App", href: "/app/", match: (p) => p.startsWith("/app") },
    { label: "About", href: "/about/", match: (p) => p.startsWith("/about") },
    { label: "Cite", href: "/cite/", match: (p) => p.startsWith("/cite") },
  ];

  const isHome = $derived(currentPath === "/" || currentPath === "");
</script>

<nav class="border-b border-gray-200 bg-white">
  <div class="max-w-7xl mx-auto px-6 flex items-center justify-between">
    <a
      href="/"
      class="text-2xl font-bold py-4 transition-colors"
      class:text-blue-600={isHome}
      class:text-gray-900={!isHome}
    >Eunoia</a>
    <ul class="flex items-center gap-1">
      {#each items as item}
        {@const active = item.match(currentPath)}
        <li>
          <a
            href={item.href}
            class="px-4 py-4 text-sm font-medium border-b-2 transition-colors inline-block"
            class:border-blue-500={active}
            class:text-blue-600={active}
            class:border-transparent={!active}
            class:text-gray-500={!active}
            class:hover:text-gray-900={!active}
          >{item.label}</a>
        </li>
      {/each}
      <li class="ml-2">
        <a
          href="https://github.com/jolars/eunoia"
          class="px-3 py-1.5 text-sm text-gray-500 hover:text-gray-900"
          target="_blank"
          rel="noreferrer"
        >GitHub →</a>
      </li>
    </ul>
  </div>
</nav>
