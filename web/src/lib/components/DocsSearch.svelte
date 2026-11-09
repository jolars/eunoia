<script lang="ts">
  import { goto } from "$app/navigation";
  import { SUMMARY } from "#lib/docs/SUMMARY.ts";

  interface SearchEntry {
    title: string;
    path: string;
    text?: string;
    ecosystem?: string;
  }

  let query = $state("");
  let loading = $state(false);
  let loaded = false;
  let entries = $state<SearchEntry[]>(
    SUMMARY.flatMap((section) =>
      section.chapters.map((chapter) => ({
        title: `${section.title ? `${section.title}: ` : ""}${chapter.title}`,
        path: chapter.slug,
      })),
    ),
  );

  async function load() {
    if (loaded) return;
    loaded = true;
    loading = true;
    const indexes = await Promise.allSettled(
      ["python", "julia"].map(async (language) => {
        const response = await fetch(`/_diplodocus/${language}/search.json`);
        if (!response.ok) throw new Error("Search index unavailable");
        return (await response.json()) as SearchEntry[];
      }),
    );
    const unique = new Map(entries.map((entry) => [entry.path, entry]));
    for (const result of indexes) {
      if (result.status === "fulfilled") {
        for (const entry of result.value) unique.set(entry.path, entry);
      }
    }
    entries = [...unique.values()];
    loading = false;
  }

  const results = $derived.by(() => {
    const terms = query.trim().toLowerCase().split(/\s+/).filter(Boolean);
    if (!terms.length) return [];
    return entries
      .map((entry) => {
        const title = entry.title.toLowerCase();
        const text =
          `${title} ${entry.text ?? ""} ${entry.ecosystem ?? ""}`.toLowerCase();
        return {
          entry,
          score: terms.every((term) => text.includes(term))
            ? terms.reduce(
                (score, term) => score + (title.includes(term) ? 10 : 1),
                0,
              )
            : 0,
        };
      })
      .filter((result) => result.score > 0)
      .sort(
        (a, b) =>
          b.score - a.score || a.entry.title.localeCompare(b.entry.title),
      )
      .slice(0, 8)
      .map((result) => result.entry);
  });
</script>

<form
  role="search"
  class="mb-6"
  onsubmit={(event) => {
    event.preventDefault();
    if (results[0]) void goto(results[0].path);
  }}
>
  <label
    for="documentation-search"
    class="block text-xs font-semibold text-muted mb-2"
    >Search documentation</label
  >
  <input
    id="documentation-search"
    type="search"
    bind:value={query}
    onfocus={load}
    placeholder="Guides and APIs"
    autocomplete="off"
    class="w-full min-w-0 rounded border border-line bg-surface px-3 py-2 text-sm text-ink focus:outline-2 focus:outline-accent"
  />
  <noscript
    ><p class="text-xs text-muted mt-2">
      Use your browser’s find command to search this page.
    </p></noscript
  >
  {#if query.trim()}
    <div aria-live="polite">
      {#if loading}
        <p class="text-xs text-muted mt-2">Loading search…</p>
      {:else if !results.length}
        <p class="text-xs text-muted mt-2">No matching documentation.</p>
      {/if}
    </div>
    <ul
      class="mt-2 space-y-1 text-sm"
      aria-label="Documentation search results"
    >
      {#each results as result}
        <li>
          <a
            href={result.path}
            class="block rounded px-3 py-2 text-accent hover:bg-inset break-words"
          >
            {result.title}
            {#if result.ecosystem}<span
                class="block text-xs text-muted capitalize"
                >{result.ecosystem}</span
              >{/if}
          </a>
        </li>
      {/each}
    </ul>
  {/if}
</form>
