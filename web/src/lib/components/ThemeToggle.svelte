<script lang="ts">
  import { theme } from "$lib/theme.svelte";
  import IconSun from "~icons/lucide/sun";
  import IconMoon from "~icons/lucide/moon";

  // Two-state toggle: shows the current theme, flips to the other on click.
  // First visit follows the OS; clicking commits an explicit light/dark.
  const isDark = $derived(theme.resolved === "dark");
  const label = $derived(
    isDark ? "Switch to light theme" : "Switch to dark theme",
  );
</script>

<button
  type="button"
  onclick={() => theme.toggle()}
  class="inline-flex items-center justify-center p-2 text-faint hover:text-ink transition-colors"
  aria-label={label}
  title={label}
>
  {#if isDark}
    <IconMoon class="w-5 h-5" />
  {:else}
    <IconSun class="w-5 h-5" />
  {/if}
</button>
