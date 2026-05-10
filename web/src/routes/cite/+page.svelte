<script lang="ts">
  // Hand-written citation formats. Single source of truth: ../../../../CITATION.cff
  // at the repo root. If that file changes, update each format below.
  type Format = {
    key: string;
    label: string;
    value: string;
    /** Render as a pre/monospace code block when true; soft-wrapped prose otherwise. */
    code: boolean;
  };

  const formats: Format[] = [
    {
      key: "bibtex",
      label: "BibTeX",
      code: true,
      value: `@inproceedings{larsson2018,
  author    = {Larsson, Johan and Gustafsson, Peter},
  title     = {A Case Study in Fitting Area-Proportional {Euler} Diagrams with Ellipses Using eulerr},
  booktitle = {Proceedings of International Workshop on Set Visualization and Reasoning 2018},
  year      = {2018},
  address   = {Edinburgh, UK},
  url       = {https://ceur-ws.org/Vol-2116/paper7.pdf},
}`,
    },
    {
      key: "biblatex",
      label: "BibLaTeX",
      code: true,
      value: `@inproceedings{larsson2018,
  author       = {Larsson, Johan and Gustafsson, Peter},
  title        = {A Case Study in Fitting Area-Proportional {Euler} Diagrams with Ellipses Using eulerr},
  date         = {2018-06-18},
  eventtitle   = {Set Visualization and Reasoning 2018},
  venue        = {Edinburgh, UK},
  booktitle    = {Proceedings of International Workshop on Set Visualization and Reasoning 2018},
  url          = {https://ceur-ws.org/Vol-2116/paper7.pdf},
}`,
    },
    {
      key: "csl",
      label: "CSL YAML",
      code: true,
      value: `---
references:
  - id: larsson2018
    type: paper-conference
    title: "A Case Study in Fitting Area-Proportional Euler Diagrams with Ellipses Using eulerr"
    author:
      - family: Larsson
        given: Johan
      - family: Gustafsson
        given: Peter
    issued:
      - year: 2018
    container-title: "Proceedings of International Workshop on Set Visualization and Reasoning 2018"
    event: "Set Visualization and Reasoning 2018"
    event-place: "Edinburgh, UK"
    URL: "https://ceur-ws.org/Vol-2116/paper7.pdf"
---`,
    },
    {
      key: "apa",
      label: "APA",
      code: false,
      value:
        "Larsson, J., & Gustafsson, P. (2018). A case study in fitting area-proportional Euler diagrams with ellipses using eulerr. In Proceedings of International Workshop on Set Visualization and Reasoning 2018. https://ceur-ws.org/Vol-2116/paper7.pdf",
    },
    {
      key: "vancouver",
      label: "Vancouver",
      code: false,
      value:
        "Larsson J, Gustafsson P. A case study in fitting area-proportional Euler diagrams with ellipses using eulerr. In: Proceedings of International Workshop on Set Visualization and Reasoning 2018; 2018 Jun 18; Edinburgh, UK. Available from: https://ceur-ws.org/Vol-2116/paper7.pdf",
    },
    {
      key: "chicago",
      label: "Chicago",
      code: false,
      value:
        "Larsson, Johan, and Peter Gustafsson. 2018. “A Case Study in Fitting Area-Proportional Euler Diagrams with Ellipses Using eulerr.” In Proceedings of International Workshop on Set Visualization and Reasoning 2018. https://ceur-ws.org/Vol-2116/paper7.pdf.",
    },
  ];

  let activeKey = $state(formats[0].key);
  let active = $derived(formats.find((f) => f.key === activeKey) ?? formats[0]);

  let copied = $state(false);
  let copyTimer: ReturnType<typeof setTimeout> | null = null;

  async function copy() {
    try {
      await navigator.clipboard.writeText(active.value);
      copied = true;
      if (copyTimer) clearTimeout(copyTimer);
      copyTimer = setTimeout(() => (copied = false), 1500);
    } catch {
      copied = false;
    }
  }

  // Reset the "Copied" badge when the user switches tabs so it doesn't claim
  // the previous tab's payload is on the clipboard.
  $effect(() => {
    void activeKey;
    copied = false;
  });
</script>

<div class="max-w-3xl mx-auto p-6">
  <article class="bg-white rounded-lg shadow p-8 space-y-4 text-sm text-gray-800">
    <header>
      <h1 class="text-2xl font-bold mb-1">Citation</h1>
      <p class="text-gray-600">
        If you use Eunoia (or the R package eulerr) in academic work, please
        cite the paper below. The repository's
        <a href="https://github.com/jolars/eunoia/blob/main/CITATION.cff" class="text-blue-600 hover:underline">CITATION.cff</a>
        is the source of truth.
      </p>
    </header>

    <div class="flex flex-wrap gap-1.5" role="tablist" aria-label="Citation format">
      {#each formats as f}
        <button
          type="button"
          role="tab"
          aria-selected={f.key === activeKey}
          onclick={() => (activeKey = f.key)}
          class={`px-3 py-1.5 text-xs rounded border transition-colors ${
            f.key === activeKey
              ? "bg-blue-600 text-white border-blue-600"
              : "bg-white text-gray-700 border-gray-300 hover:bg-gray-50"
          }`}
        >{f.label}</button>
      {/each}
    </div>

    <div class="relative">
      <button
        type="button"
        onclick={copy}
        class="absolute top-2 right-2 px-2 py-1 text-xs bg-gray-100 text-gray-700 rounded hover:bg-gray-200"
      >
        {copied ? "Copied" : "Copy"}
      </button>
      {#if active.code}
        <pre class="bg-gray-50 border border-gray-200 rounded p-4 pr-16 font-mono text-xs overflow-x-auto whitespace-pre">{active.value}</pre>
      {:else}
        <p class="bg-gray-50 border border-gray-200 rounded p-4 pr-16 text-sm leading-relaxed whitespace-pre-wrap">{active.value}</p>
      {/if}
    </div>
  </article>
</div>
