<script lang="ts">
  import { page } from "$app/state";
  import { SUMMARY } from "#lib/docs/SUMMARY.ts";

  const currentPath = $derived(page.url.pathname);
</script>

<nav class="space-y-6 text-sm">
  {#each SUMMARY as section}
    <div>
      {#if section.title}
        <p
          class="text-xs font-semibold text-muted uppercase tracking-wider mb-2"
        >
          {section.title}
        </p>
      {/if}
      <ul class="space-y-1">
        {#each section.chapters as chapter}
          {@const active =
            currentPath === chapter.slug ||
            (chapter.slug === "/docs/reference/python/" &&
              currentPath.startsWith(chapter.slug))}
          <li>
            <a
              href={chapter.slug}
              class="block px-3 py-1.5 rounded transition-colors"
              class:bg-accent-soft={active}
              class:text-accent={active}
              class:font-medium={active}
              class:text-ink={!active}
              class:hover:bg-inset={!active}>{chapter.title}</a
            >
          </li>
        {/each}
      </ul>
    </div>
  {/each}
</nav>
