<script lang="ts">
import IconGithub from "~icons/simple-icons/github";
import IconRust from "~icons/simple-icons/rust";
import IconNpm from "~icons/simple-icons/npm";
import IconR from "~icons/simple-icons/r";
import IconPython from "~icons/simple-icons/python";
import IconJulia from "~icons/simple-icons/julia";
import IconBook from "~icons/lucide/book-open";

// Repo links — kept here so the landing has a single declarative source of
// truth that's easy to scan and edit.
const repoUrl = "https://github.com/jolars/eunoia";
const cratesUrl = "https://crates.io/crates/eunoia";
const docsRsUrl = "https://docs.rs/eunoia";
const npmUrl = "https://www.npmjs.com/package/@jolars/eunoia";
const pypiUrl = "https://pypi.org/project/eunoia/";
const pyDocsUrl = "https://jolars.github.io/eunoia-py";
const eulerrUrl = "https://cran.r-project.org/package=eulerr";
const eulerrRepoUrl = "https://github.com/jolars/eulerr";
const eulerrDocsUrl = "https://jolars.github.io/eulerr/";
const johanUrl = "https://jolars.co";
</script>

<div class="text-gray-800">
  <main class="max-w-5xl mx-auto px-6 py-16 space-y-16">
    <section>
      <!-- Stacked + logo-centred below `sm`; logo to the right alongside the
           text at `sm` and up. Avoids the heading wrapping around a float. -->
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
          <p class="text-lg text-gray-600 mb-4">
            A Rust library for fitting set-visualization diagrams with circles,
            ellipses, squares, and rectangles — accurate, fast, and shipped with
            WASM bindings for the web. Powers the R package
            <a href={eulerrRepoUrl} class="text-blue-600 hover:underline">eulerr</a>,
            npm package <a href={npmUrl} class="text-blue-600 hover:underline">@jolars/eunoia</a>,
            and the <a href={pypiUrl} class="text-blue-600 hover:underline">Python package</a>,
            with a Julia wrapper planned.
          </p>
          <div class="flex flex-wrap gap-3 pt-2">
            <a
              href="/app/"
              class="px-5 py-2.5 rounded bg-blue-600 text-white text-sm font-medium hover:bg-blue-700"
            >Try it in the browser →</a>
            <a
              href={repoUrl}
              class="px-5 py-2.5 rounded border border-gray-300 bg-white text-sm font-medium hover:bg-gray-50 inline-flex items-center gap-2"
            ><IconGithub class="w-4 h-4" /> Source code</a>
          </div>
        </div>
      </div>
    </section>

    <section class="space-y-4">
      <h3 class="text-xl font-semibold">Install</h3>
      <div class="grid sm:grid-cols-2 gap-4">
        <div class="bg-white border border-gray-200 rounded p-4 space-y-2">
          <div class="flex items-center justify-between">
            <span class="font-medium inline-flex items-center gap-2"><IconRust class="w-4 h-4" /> Rust</span>
            <a href={cratesUrl} class="text-xs text-blue-600 hover:underline">crates.io</a>
          </div>
          <pre class="bg-gray-50 border border-gray-200 rounded p-2 font-mono text-xs overflow-x-auto">cargo add eunoia</pre>
          <p class="text-xs text-gray-500">
            Docs at
            <a href={docsRsUrl} class="text-blue-600 hover:underline">docs.rs/eunoia</a>.
          </p>
        </div>
        <div class="bg-white border border-gray-200 rounded p-4 space-y-2">
          <div class="flex items-center justify-between">
            <span class="font-medium inline-flex items-center gap-2"><IconNpm class="w-4 h-4" /> JavaScript / TypeScript</span>
            <a href={npmUrl} class="text-xs text-blue-600 hover:underline">npm</a>
          </div>
          <pre class="bg-gray-50 border border-gray-200 rounded p-2 font-mono text-xs overflow-x-auto">npm install @jolars/eunoia</pre>
          <p class="text-xs text-gray-500">
            WASM bindings; powers
            <a href="/app/" class="text-blue-600 hover:underline">this site</a>.
          </p>
        </div>
        <div class="bg-white border border-gray-200 rounded p-4 space-y-2">
          <div class="flex items-center justify-between">
            <span class="font-medium inline-flex items-center gap-2"><IconR class="w-4 h-4" /> R</span>
            <a href={eulerrUrl} class="text-xs text-blue-600 hover:underline">CRAN</a>
          </div>
          <pre class="bg-gray-50 border border-gray-200 rounded p-2 font-mono text-xs overflow-x-auto">install.packages("eulerr")</pre>
          <p class="text-xs text-gray-500">
            R users can use
            <a href={eulerrRepoUrl} class="text-blue-600 hover:underline">eulerr</a>,
            which is now backed by the same Rust core as Eunoia; docs at
            <a href={eulerrDocsUrl} class="text-blue-600 hover:underline">jolars.github.io/eulerr</a>.
          </p>
        </div>
        <div class="bg-white border border-gray-200 rounded p-4 space-y-2">
          <div class="flex items-center justify-between">
            <span class="font-medium inline-flex items-center gap-2"><IconPython class="w-4 h-4" /> Python</span>
            <a href={pypiUrl} class="text-xs text-blue-600 hover:underline">PyPI</a>
          </div>
          <pre class="bg-gray-50 border border-gray-200 rounded p-2 font-mono text-xs overflow-x-auto">pip install eunoia</pre>
          <p class="text-xs text-gray-500">
            Bindings to the Rust core with matplotlib plotting; docs at
            <a href={pyDocsUrl} class="text-blue-600 hover:underline">jolars.github.io/eunoia-py</a>.
          </p>
        </div>
        <div class="bg-white border border-gray-200 rounded p-4 space-y-2">
          <span class="font-medium inline-flex items-center gap-2"><IconJulia class="w-4 h-4" /> Julia</span>
          <p class="text-xs text-gray-500">
            Planned thin wrapper around the Rust core, in its own
            repository. Track progress on the
            <a href={repoUrl} class="text-blue-600 hover:underline">main repo</a>.
          </p>
        </div>
      </div>
    </section>

    <section class="space-y-4">
      <h3 class="text-xl font-semibold">Why Eunoia?</h3>
      <ul class="grid sm:grid-cols-2 gap-x-8 gap-y-2 text-sm text-gray-700 list-disc pl-5">
        <li>Accurate and fast: uses exact n-way intersection geometry, analytical gradients, state-of-the-art optimizers, and is written in Rust.</li>
        <li>Exact n-way intersection geometry for circles and ellipses.</li>
        <li>Multiple shapes: circles, ellipses, squares, and rectangles.</li>
        <li>Multiple loss functions to prioritize different aspects of fit quality.</li>
        <li>Pluggable optimizers (CMA-ES, Levendberg-Marquardt, L-BFGS, Nelder-Mead).</li>
        <li>Optional bounding-rectangle "complement" for universe sets.</li>
        <li>Extensive customization of plot aesthetics.</li>
        <li>Pure Rust core, no platform-specific deps.</li>
      </ul>
    </section>

    <section class="space-y-3">
      <h3 class="text-xl font-semibold">Citing Eunoia</h3>
      <p class="text-sm text-gray-600">
        If you use Eunoia or any of its derived packages (eulerr in R and
        `@jolars/eunoia` in npm) in academic work, please cite the paper:
      </p>
      <p class="text-sm bg-white border border-gray-200 rounded p-4 leading-relaxed text-gray-800">
        Larsson, J., &amp; Gustafsson, P. (2018). A case study in fitting
        area-proportional Euler diagrams with ellipses using eulerr. In
        Proceedings of International Workshop on Set Visualization and
        Reasoning 2018.
        <a
          href="https://ceur-ws.org/Vol-2116/paper7.pdf"
          class="text-blue-600 hover:underline">https://ceur-ws.org/Vol-2116/paper7.pdf</a>
      </p>
      <p class="text-sm text-gray-600">
        Other formats (BibTeX, BibLaTeX, CSL, Vancouver, Chicago) are
        available on the
        <a href="/cite/" class="text-blue-600 hover:underline">citation page</a>.
      </p>
    </section>

    <footer class="border-t border-gray-200 pt-6 text-xs text-gray-500 flex flex-wrap items-center gap-x-4 gap-y-1">
      <span>© 2025–{new Date().getFullYear()} <a href={johanUrl} class="text-blue-600 inline-flex items-center hover:underline">Johan Larsson</a></span>
      <a href={repoUrl} class="hover:text-gray-700 inline-flex items-center gap-1.5"><IconGithub class="w-3.5 h-3.5" /> GitHub</a>
      <a href={cratesUrl} class="hover:text-gray-700 inline-flex items-center gap-1.5"><IconRust class="w-3.5 h-3.5" /> crates.io</a>
      <a href={docsRsUrl} class="hover:text-gray-700 inline-flex items-center gap-1.5"><IconBook class="w-3.5 h-3.5" /> docs.rs</a>
      <a href={npmUrl} class="hover:text-gray-700 inline-flex items-center gap-1.5"><IconNpm class="w-3.5 h-3.5" /> npm</a>
    </footer>
  </main>
</div>
