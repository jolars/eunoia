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
      value: `@misc{larsson2026f,
  title = {Eunoia: {Area}-Proportional {Euler} and {Venn} Diagrams in {Rust}, {Julia}, {Python}, {R}, and {JavaScript}},
  shorttitle = {Eunoia},
  author = {Larsson, Johan and Lee, Nicole M.},
  year = 2026,
  month = nov,
  number = {arXiv:2611.02198},
  eprint = {2611.02198},
  primaryclass = {stat.CO},
  publisher = {arXiv},
  doi = {10.48550/arXiv.2611.02198},
  archiveprefix = {arXiv}
}`,
    },
    {
      key: "biblatex",
      label: "BibLaTeX",
      code: true,
      value: `@online{larsson2026f,
  title = {Eunoia: {Area}-Proportional {Euler} and {Venn} Diagrams in {Rust}, {Julia}, {Python}, {R}, and {JavaScript}},
  shorttitle = {Eunoia},
  author = {Larsson, Johan and Lee, Nicole M.},
  date = {2026-11-01},
  eprint = {2611.02198},
  eprinttype = {arXiv},
  eprintclass = {stat.CO},
  doi = {10.48550/arXiv.2611.02198},
  pubstate = {prepublished}
}`,
    },
    {
      key: "csl",
      label: "CSL YAML",
      code: true,
      value: `---
references:
  - id: larsson2026f
    type: article
    title: "Eunoia: Area-Proportional Euler and Venn Diagrams in Rust, Julia, Python, R, and JavaScript"
    author:
      - family: Larsson
        given: Johan
      - family: Lee
        given: Nicole M.
    issued:
      - year: 2026
        month: 11
        day: 1
    publisher: arXiv
    number: "arXiv:2611.02198"
    DOI: "10.48550/arXiv.2611.02198"
    URL: "https://arxiv.org/abs/2611.02198"
---`,
    },
    {
      key: "apa",
      label: "APA",
      code: false,
      value:
        "Larsson, J., & Lee, N. M. (2026). Eunoia: Area-proportional Euler and Venn diagrams in Rust, Julia, Python, R, and JavaScript. arXiv. https://doi.org/10.48550/arXiv.2611.02198",
    },
    {
      key: "vancouver",
      label: "Vancouver",
      code: false,
      value:
        "Larsson J, Lee NM. Eunoia: Area-Proportional Euler and Venn Diagrams in Rust, Julia, Python, R, and JavaScript [preprint]. arXiv; 2026 Nov 1. arXiv:2611.02198. Available from: https://doi.org/10.48550/arXiv.2611.02198",
    },
    {
      key: "chicago",
      label: "Chicago",
      code: false,
      value:
        "Larsson, Johan, and Nicole M. Lee. 2026. “Eunoia: Area-Proportional Euler and Venn Diagrams in Rust, Julia, Python, R, and JavaScript.” arXiv preprint arXiv:2611.02198. https://doi.org/10.48550/arXiv.2611.02198.",
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

<div class="text-ink">
  <main class="max-w-5xl mx-auto px-6 py-16 space-y-4">
    <header>
      <h1 class="text-3xl font-bold mb-2">Citation</h1>
      <p class="text-muted">
        If you use Eunoia or any of its derived packages (eulerr in R, eunoia in
        Python, Eunoia.jl in Julia, or `@jolars/eunoia` in npm) in academic
        work, please cite the paper below.
      </p>
    </header>

    <div
      class="flex flex-wrap gap-1.5"
      role="tablist"
      aria-label="Citation format"
    >
      {#each formats as f}
        <button
          type="button"
          role="tab"
          aria-selected={f.key === activeKey}
          onclick={() => (activeKey = f.key)}
          class={`px-3 py-1.5 text-xs rounded border transition-colors ${
            f.key === activeKey
              ? "bg-blue-600 text-white border-blue-600"
              : "bg-surface text-ink border-line hover:bg-inset"
          }`}>{f.label}</button
        >
      {/each}
    </div>

    <div class="relative">
      <button
        type="button"
        onclick={copy}
        class="absolute top-2 right-2 px-2 py-1 text-xs bg-inset text-ink rounded hover:bg-line"
      >
        {copied ? "Copied" : "Copy"}
      </button>
      {#if active.code}
        <pre
          class="bg-inset border border-line rounded p-4 pr-16 font-mono text-xs overflow-x-auto whitespace-pre">{active.value}</pre>
      {:else}
        <p
          class="bg-inset border border-line rounded p-4 pr-16 text-sm leading-relaxed whitespace-pre-wrap"
        >
          {active.value}
        </p>
      {/if}
    </div>
  </main>
</div>
