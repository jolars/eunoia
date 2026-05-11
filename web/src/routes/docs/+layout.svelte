<script lang="ts">
import DocsSidebar from "$lib/components/DocsSidebar.svelte";

let { children } = $props();
</script>

<div class="max-w-7xl mx-auto px-6 py-8">
  <div class="grid grid-cols-1 lg:grid-cols-[14rem_minmax(0,1fr)] gap-10">
    <aside class="lg:sticky lg:top-6 lg:self-start">
      <DocsSidebar />
    </aside>
    <article
      class="prose prose-sm sm:prose-base max-w-none prose-headings:scroll-mt-20 prose-pre:bg-gray-50 prose-pre:border prose-pre:border-gray-200 prose-pre:text-gray-800 prose-code:before:content-none prose-code:after:content-none prose-code:bg-gray-100 prose-code:px-1 prose-code:py-0.5 prose-code:rounded prose-code:font-normal prose-a:text-blue-600 prose-a:no-underline hover:prose-a:underline"
    >
      {@render children()}
    </article>
  </div>
</div>
