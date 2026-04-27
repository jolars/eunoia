<script lang="ts">
  const bib = `@Manual{eunoia,
  title  = {{Eunoia}: Area-Proportional Euler and {Venn} Diagrams},
  author = {Johan Larsson},
  year   = {2026},
  url    = {https://eunoia.fit},
}

@inproceedings{larsson2018,
  author    = {Larsson, Johan and Gustafsson, Peter},
  title     = {A Case Study in Fitting Area-Proportional {E}uler Diagrams with Ellipses using eulerr},
  booktitle = {Proceedings of International Workshop on Set Visualization and Reasoning},
  volume    = {2116},
  pages     = {84--91},
  year      = {2018},
}`;

  let copied = $state(false);

  async function copy() {
    try {
      await navigator.clipboard.writeText(bib);
      copied = true;
      setTimeout(() => (copied = false), 1500);
    } catch {
      copied = false;
    }
  }
</script>

<div class="max-w-3xl mx-auto p-6">
  <article class="bg-white rounded-lg shadow p-8 space-y-4 text-sm text-gray-800">
    <header>
      <h1 class="text-2xl font-bold mb-1">Citation</h1>
      <p class="text-gray-600">
        If you use Eunoia in academic work, please cite the package and, where
        appropriate, the methodology paper for ellipse-based fitting.
      </p>
    </header>

    <div class="relative">
      <button
        type="button"
        onclick={copy}
        class="absolute top-2 right-2 px-2 py-1 text-xs bg-gray-100 text-gray-700 rounded hover:bg-gray-200"
      >
        {copied ? "Copied" : "Copy"}
      </button>
      <pre class="bg-gray-50 border border-gray-200 rounded p-4 font-mono text-xs overflow-x-auto whitespace-pre">{bib}</pre>
    </div>
  </article>
</div>
