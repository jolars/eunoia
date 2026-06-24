<script lang="ts">
import IconGithub from "~icons/simple-icons/github";
import IconRust from "~icons/simple-icons/rust";
import IconNpm from "~icons/simple-icons/npm";
import IconR from "~icons/simple-icons/r";
import IconPython from "~icons/simple-icons/python";
import IconJulia from "~icons/simple-icons/julia";
import IconBook from "~icons/lucide/book-open";
import HeroWidget from "$lib/components/HeroWidget.svelte";

// Repo links — kept here so the landing has a single declarative source of
// truth that's easy to scan and edit.
const repoUrl = "https://github.com/jolars/eunoia";
const cratesUrl = "https://crates.io/crates/eunoia";
const docsRsUrl = "https://docs.rs/eunoia";
const npmUrl = "https://www.npmjs.com/package/@jolars/eunoia";
const pypiUrl = "https://pypi.org/project/eunoia/";
const pyDocsUrl = "https://jolars.github.io/eunoia-py";
const juliaRepoUrl = "https://github.com/jolars/Eunoia.jl";
const juliaDocsUrl = "https://jolars.github.io/Eunoia.jl/stable";
const eulerrUrl = "https://cran.r-project.org/package=eulerr";
const eulerrRepoUrl = "https://github.com/jolars/eulerr";
const eulerrDocsUrl = "https://jolars.github.io/eulerr/";
const johanUrl = "https://jolars.co";
</script>

<div class="text-ink">
  <main class="max-w-5xl mx-auto px-6 py-16 space-y-16">
    <section>
      <div
        class="flex flex-col items-center gap-6 sm:flex-row-reverse sm:items-start sm:gap-8"
      >
        <img
          src="/logo.png"
          alt="Eunoia logo"
          class="w-32 shrink-0 sm:w-40 md:w-48 h-auto"
        />
        <div>
          <h2 class="text-3xl font-bold leading-tight mb-4">
            Area-Proportional Euler and Venn Diagrams
          </h2>
          <p class="text-lg text-muted mb-4">
            A Rust library for fitting set-visualization diagrams with circles,
            ellipses, squares, and rectangles. Accurate, fast, and shipped with
            WASM bindings for the web. Powers the R package
            <a href={eulerrRepoUrl} class="text-accent hover:underline">eulerr</a>,
            npm package <a href={npmUrl} class="text-accent hover:underline">@jolars/eunoia</a>,
            Julia package <a href={juliaRepoUrl} class="text-accent hover:underline">Eunoia.jl</a>,
            and Python package <a href={pypiUrl} class="text-accent hover:underline">eunoia</a>.
          </p>
          <div class="flex flex-wrap gap-3 pt-2">
            <a
              href="/app/"
              class="px-5 py-2.5 rounded bg-blue-600 text-white text-sm font-medium hover:bg-blue-700"
            >Try it in the browser →</a>
            <a
              href={repoUrl}
              class="px-5 py-2.5 rounded border border-line bg-surface text-sm font-medium hover:bg-inset inline-flex items-center gap-2"
            ><IconGithub class="w-4 h-4" /> Source code</a>
          </div>
        </div>
      </div>
    </section>

    <section class="space-y-4">
      <h3 class="text-xl font-semibold">See It in Action</h3>
      <p class="text-sm text-muted max-w-2xl">
        A three-set diagram, fitted in your browser by the same WASM core the
        packages ship. Drag the region sizes or switch the shape and watch it
        re-solve live.
      </p>
      <HeroWidget />
    </section>

    <section class="space-y-4">
      <h3 class="text-xl font-semibold">Install</h3>
      <div class="grid sm:grid-cols-2 gap-4">
        <div class="bg-surface border border-line rounded p-4 space-y-2">
          <div class="flex items-center justify-between">
            <span class="font-medium inline-flex items-center gap-2"><IconRust class="w-4 h-4" /> Rust</span>
            <a href={cratesUrl} class="text-xs text-accent hover:underline">crates.io</a>
          </div>
          <pre class="bg-inset border border-line rounded p-2 font-mono text-xs overflow-x-auto">cargo add eunoia</pre>
          <p class="text-xs text-muted">
            Docs at
            <a href={docsRsUrl} class="text-accent hover:underline">docs.rs/eunoia</a>.
          </p>
        </div>
        <div class="bg-surface border border-line rounded p-4 space-y-2">
          <div class="flex items-center justify-between">
            <span class="font-medium inline-flex items-center gap-2"><IconNpm class="w-4 h-4" /> JavaScript / TypeScript</span>
            <a href={npmUrl} class="text-xs text-accent hover:underline">npm</a>
          </div>
          <pre class="bg-inset border border-line rounded p-2 font-mono text-xs overflow-x-auto">npm install @jolars/eunoia</pre>
          <p class="text-xs text-muted">
            WASM bindings; powers
            <a href="/app/" class="text-accent hover:underline">the web app</a> and all the interactive demos on this site.
          </p>
        </div>
        <div class="bg-surface border border-line rounded p-4 space-y-2">
          <div class="flex items-center justify-between">
            <span class="font-medium inline-flex items-center gap-2"><IconR class="w-4 h-4" /> R</span>
            <a href={eulerrUrl} class="text-xs text-accent hover:underline">CRAN</a>
          </div>
          <pre class="bg-inset border border-line rounded p-2 font-mono text-xs overflow-x-auto">install.packages("eulerr")</pre>
          <p class="text-xs text-muted">
            R users can use
            <a href={eulerrRepoUrl} class="text-accent hover:underline">eulerr</a>,
            which is now backed by the same Rust core as Eunoia; docs at
            <a href={eulerrDocsUrl} class="text-accent hover:underline">jolars.github.io/eulerr</a>.
          </p>
        </div>
        <div class="bg-surface border border-line rounded p-4 space-y-2">
          <div class="flex items-center justify-between">
            <span class="font-medium inline-flex items-center gap-2"><IconPython class="w-4 h-4" /> Python</span>
            <a href={pypiUrl} class="text-xs text-accent hover:underline">PyPI</a>
          </div>
          <pre class="bg-inset border border-line rounded p-2 font-mono text-xs overflow-x-auto">pip install eunoia</pre>
          <p class="text-xs text-muted">
            Bindings to the Rust core with matplotlib plotting; docs at
            <a href={pyDocsUrl} class="text-accent hover:underline">jolars.github.io/eunoia-py</a>.
          </p>
        </div>
        <div class="bg-surface border border-line rounded p-4 space-y-2">
          <div class="flex items-center justify-between">
            <span class="font-medium inline-flex items-center gap-2"><IconJulia class="w-4 h-4" /> Julia</span>
            <a href={juliaRepoUrl} class="text-xs text-accent hover:underline">GitHub</a>
          </div>
          <pre class="bg-inset border border-line rounded p-2 font-mono text-xs overflow-x-auto">Pkg.add("Eunoia")</pre>
          <p class="text-xs text-muted">
            Bindings to the Rust core with Makie plotting; docs at
            <a href={juliaDocsUrl} class="text-accent hover:underline">jolars.github.io/Eunoia.jl</a>.
          </p>
        </div>
      </div>
    </section>

    <section class="space-y-4">
      <h3 class="text-xl font-semibold">Why Eunoia?</h3>
      <ul class="grid sm:grid-cols-2 gap-x-8 gap-y-2 text-sm text-ink list-disc pl-5">
        <li>Accurate and fast: uses exact n-way intersection geometry, analytical gradients, and efficient optimizers.</li>
        <li>Exact n-way intersection geometry for circles and ellipses.</li>
        <li>Multiple shapes: circles, ellipses, squares, and rectangles.</li>
        <li>Multiple loss functions to prioritize different aspects of fit quality.</li>
        <li>Customizable optimizers (CMA-ES, Levendberg-Marquardt, L-BFGS, Nelder-Mead).</li>
        <li>Optional bounding-rectangle "complement" for universe sets.</li>
        <li>Extensive customization of plot aesthetics.</li>
        <li>Pure Rust core, no platform-specific deps.</li>
      </ul>
    </section>

    <section class="space-y-3">
      <h3 class="text-xl font-semibold">Citing Eunoia</h3>
      <p class="text-sm text-muted">
        If you use Eunoia or any of its derived packages (eulerr in R and
        `@jolars/eunoia` in npm) in academic work, please cite the paper:
      </p>
      <p class="text-sm bg-surface border border-line rounded p-4 leading-relaxed text-ink">
        Larsson, J., &amp; Gustafsson, P. (2018). A case study in fitting
        area-proportional Euler diagrams with ellipses using eulerr. In
        Proceedings of International Workshop on Set Visualization and
        Reasoning 2018.
        <a
          href="https://ceur-ws.org/Vol-2116/paper7.pdf"
          class="text-accent hover:underline">https://ceur-ws.org/Vol-2116/paper7.pdf</a>
      </p>
      <p class="text-sm text-muted">
        Other formats (BibTeX, BibLaTeX, CSL, Vancouver, Chicago) are
        available on the
        <a href="/cite/" class="text-accent hover:underline">citation page</a>.
      </p>
    </section>

    <footer class="border-t border-line pt-6 text-xs text-muted flex flex-wrap items-center gap-x-4 gap-y-1">
      <span>© 2025–{new Date().getFullYear()} <a href={johanUrl} class="text-accent inline-flex items-center hover:underline">Johan Larsson</a></span>
      <a href={repoUrl} class="hover:text-ink inline-flex items-center gap-1.5"><IconGithub class="w-3.5 h-3.5" /> GitHub</a>
      <a href={cratesUrl} class="hover:text-ink inline-flex items-center gap-1.5"><IconRust class="w-3.5 h-3.5" /> crates.io</a>
      <a href={docsRsUrl} class="hover:text-ink inline-flex items-center gap-1.5"><IconBook class="w-3.5 h-3.5" /> docs.rs</a>
      <a href={npmUrl} class="hover:text-ink inline-flex items-center gap-1.5"><IconNpm class="w-3.5 h-3.5" /> npm</a>
    </footer>
  </main>
</div>
