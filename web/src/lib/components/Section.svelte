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

<div class="bg-white rounded-lg shadow">
  <button
    type="button"
    onclick={() => (isOpen = !isOpen)}
    class="w-full flex items-center justify-between px-5 py-3 text-left text-sm font-semibold text-gray-700 hover:bg-gray-50 rounded-lg"
    aria-expanded={isOpen}
  >
    <span>{title}</span>
    <span class="text-xs text-gray-400">{isOpen ? "▾" : "▸"}</span>
  </button>
  {#if isOpen}
    <div class="px-5 pb-5">
      {@render children()}
    </div>
  {/if}
</div>
