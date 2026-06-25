<script lang="ts">
  import { RELEASE_URL, VERSION } from "$lib/version";
</script>

<footer
  class="mt-12 border-t border-line pt-6 text-xs text-muted flex flex-wrap items-center gap-x-4 gap-y-1"
>
  <span>
    Documentation for Eunoia
    <a href={RELEASE_URL} class="text-accent hover:underline">v{VERSION}</a>
  </span>
</footer>
