<script lang="ts">
</script>

<div class="max-w-3xl mx-auto p-6 prose">
  <article class="bg-white rounded-lg shadow p-8 space-y-6 text-sm leading-relaxed text-gray-800">
    <header>
      <h1 class="text-2xl font-bold mb-2">About Eunoia</h1>
      <p class="text-gray-600">
        Eunoia draws area-proportional Euler and Venn diagrams in the browser.
        It is a WebAssembly port of the algorithms behind the
        <a class="text-blue-600 underline" href="https://github.com/jolars/eulerr" target="_blank" rel="noreferrer">eulerr</a>
        R package, with a deeper dive into ellipses and configurable losses.
      </p>
    </header>

    <section>
      <h2 class="text-lg font-semibold mb-2">How a diagram is fitted</h2>
      <ol class="list-decimal list-inside space-y-1">
        <li>An initial layout is computed with multidimensional scaling on the requested set sizes and overlaps.</li>
        <li>The shape parameters (circle or ellipse) are then refined by minimising a loss function via numerical optimisation.</li>
        <li>The fitted layout is normalised and rotated for a stable, consistent appearance.</li>
      </ol>
    </section>

    <section>
      <h2 class="text-lg font-semibold mb-2">Diagnostics</h2>
      <p>
        Several scalars summarise how well the geometry reproduces the
        requested set relationships. They appear under <em>Goodness of fit</em>
        on the App tab.
      </p>
      <ul class="list-disc list-inside space-y-1 mt-2">
        <li>
          <strong>stress</strong>: a scale-invariant measure due to
          <em>Wilkinson (2012)</em>, popularised by <code>venneuler</code>. Lower is better.
        </li>
        <li>
          <strong>diagError</strong>: the largest absolute discrepancy between
          a fitted region's share of the total area and the requested share.
          Defined by <em>Micallef &amp; Rodgers (2014)</em>; values &lt; 0.01
          generally indicate a faithful diagram.
        </li>
        <li>
          <strong>regionError</strong> (per-row in the Fit table): the same
          quantity, but reported per region.
        </li>
      </ul>
    </section>

    <section>
      <h2 class="text-lg font-semibold mb-2">Limitations</h2>
      <p>
        Not every set of region areas can be drawn faithfully with circles or
        even ellipses. A classic example:
      </p>
      <pre class="bg-gray-50 border border-gray-200 rounded p-3 font-mono text-xs">A=5, B=3, C=1, A&amp;B=2, A&amp;B&amp;C=2</pre>
      <p>
        No two-dimensional Euler diagram with circles can satisfy these
        constraints exactly. In such cases <em>diagError</em> will be large,
        and the geometry will be the best compromise the optimiser found.
      </p>
    </section>

    <section>
      <h2 class="text-lg font-semibold mb-2">References</h2>
      <ul class="list-disc list-inside space-y-1">
        <li>Wilkinson, L. (2012). Exact and approximate area-proportional circular Venn and Euler diagrams. <em>IEEE Transactions on Visualization and Computer Graphics</em>, 18(2), 321–331.</li>
        <li>Micallef, L., &amp; Rodgers, P. (2014). eulerAPE: drawing area-proportional 3-Venn diagrams using ellipses. <em>PLOS ONE</em>, 9(7), e101717.</li>
        <li>Larsson, J., &amp; Gustafsson, P. (2018). A case study in fitting area-proportional Euler diagrams with ellipses using eulerr. <em>CEUR Workshop Proceedings</em>, 2116, 84–91.</li>
      </ul>
    </section>
  </article>
</div>
