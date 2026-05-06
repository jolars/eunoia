<script lang="ts">
  // Repo links — kept here so the landing has a single declarative source of
  // truth that's easy to scan and edit.
  const repoUrl = "https://github.com/jolars/eunoia";
  const cratesUrl = "https://crates.io/crates/eunoia";
  const docsRsUrl = "https://docs.rs/eunoia";
  const npmUrl = "https://www.npmjs.com/package/@jolars/eunoia";
</script>

<div class="text-gray-800">
  <main class="max-w-5xl mx-auto px-6 py-16 space-y-16">
    <section class="space-y-4">
      <h2 class="text-3xl font-bold leading-tight">
        Area-proportional Euler and Venn diagrams
      </h2>
      <p class="text-lg text-gray-600 max-w-2xl">
        A Rust library for fitting set-visualization diagrams with circles,
        ellipses, squares, and rectangles — accurate, fast, and shipped with
        bindings for the web (WASM) and, soon, R, Python, and Julia.
      </p>
      <div class="flex flex-wrap gap-3 pt-2">
        <a
          href="/app/"
          class="px-5 py-2.5 rounded bg-blue-600 text-white text-sm font-medium hover:bg-blue-700"
        >Try it in the browser →</a>
        <a
          href={repoUrl}
          class="px-5 py-2.5 rounded border border-gray-300 bg-white text-sm font-medium hover:bg-gray-50"
        >Source code</a>
      </div>
    </section>

    <section class="space-y-4">
      <h3 class="text-xl font-semibold">Install</h3>
      <div class="grid sm:grid-cols-2 gap-4">
        <div class="bg-white border border-gray-200 rounded p-4 space-y-2">
          <div class="flex items-center justify-between">
            <span class="font-medium">Rust</span>
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
            <span class="font-medium">JavaScript / TypeScript</span>
            <a href={npmUrl} class="text-xs text-blue-600 hover:underline">npm</a>
          </div>
          <pre class="bg-gray-50 border border-gray-200 rounded p-2 font-mono text-xs overflow-x-auto">npm install @jolars/eunoia</pre>
          <p class="text-xs text-gray-500">
            WASM bindings; powers
            <a href="/app/" class="text-blue-600 hover:underline">this site</a>.
          </p>
        </div>
        <div class="bg-white border border-gray-200 rounded p-4 space-y-2 sm:col-span-2">
          <span class="font-medium">R, Python, Julia</span>
          <p class="text-xs text-gray-500">
            Planned thin wrappers around the Rust core, each in its own
            repository. Track progress on the
            <a href={repoUrl} class="text-blue-600 hover:underline">main repo</a>.
          </p>
        </div>
      </div>
    </section>

    <section class="space-y-4">
      <h3 class="text-xl font-semibold">Why eunoia?</h3>
      <ul class="grid sm:grid-cols-2 gap-x-8 gap-y-2 text-sm text-gray-700 list-disc pl-5">
        <li>Exact n-way intersection geometry for circles and ellipses.</li>
        <li>Multiple shapes: circles, ellipses, axis-aligned squares and rectangles.</li>
        <li>Pluggable optimizers (CMA-ES → LM, L-BFGS, Nelder-Mead).</li>
        <li>Canonical Venn warm-start for n ≤ 5.</li>
        <li>Optional bounding-rectangle "complement" for universe sets.</li>
        <li>Pure Rust core, no platform-specific deps.</li>
      </ul>
    </section>

    <section class="space-y-3">
      <h3 class="text-xl font-semibold">Cite</h3>
      <p class="text-sm text-gray-600">
        If you use eunoia (or its predecessor, the R package eulerr) in
        academic work, please cite the paper. Multiple formats are available
        on the
        <a href="/cite/" class="text-blue-600 hover:underline">citation page</a>.
      </p>
    </section>

    <footer class="border-t border-gray-200 pt-6 text-xs text-gray-500 flex flex-wrap gap-x-4 gap-y-1">
      <span>MIT-licensed</span>
      <a href={repoUrl} class="hover:text-gray-700">GitHub</a>
      <a href={cratesUrl} class="hover:text-gray-700">crates.io</a>
      <a href={docsRsUrl} class="hover:text-gray-700">docs.rs</a>
      <a href={npmUrl} class="hover:text-gray-700">npm</a>
    </footer>
  </main>
</div>
