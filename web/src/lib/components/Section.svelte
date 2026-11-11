<script lang="ts">
  interface Props {
    title: string;
    open?: boolean;
    children: import("svelte").Snippet;
  }

  let { title, open: initialOpen = true, children }: Props = $props();
  // svelte-ignore state_referenced_locally
  let isOpen = $state(initialOpen);
</script>

<div class="bg-surface border border-line rounded-lg">
  <button
    type="button"
    onclick={() => (isOpen = !isOpen)}
    class="w-full flex items-center justify-between px-2 sm:px-4 py-3 text-left text-sm font-semibold text-ink hover:bg-inset rounded-lg"
    aria-expanded={isOpen}
  >
    <span>{title}</span>
    <span class="text-xs text-faint">{isOpen ? "▾" : "▸"}</span>
  </button>
  {#if isOpen}
    <div class="px-2 pb-3 sm:px-4 sm:pb-4">
      {@render children()}
    </div>
  {/if}
</div>
