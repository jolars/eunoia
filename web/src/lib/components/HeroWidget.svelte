<script lang="ts">
  import type { LabelPlacement } from "@jolars/eunoia";
  import { leaderPath } from "@jolars/eunoia/svg";
  import { defaultColorFor } from "$lib/colors";
  import type { FitInputs } from "$lib/fit";
  import { DEFAULT_FONT_FAMILY } from "$lib/fonts";
  import { theme } from "$lib/theme.svelte";
  import type { FitResult, ShapeType } from "$lib/types/diagram";

  // A small, self-contained "play with it" widget for the landing page: seven
  // sliders (the full exclusive-region spec of a 3-set diagram) plus a shape
  // toggle, refitting live as you drag. The heavy *fit* runs in the app's
  // worker (`fit.worker.ts`) so dragging never blocks the main thread; the
  // (cheap) label placement runs here on the main thread so we can showcase the
  // real `placeLabelsForRegions` feature — poles of inaccessibility for interior
  // labels, with leader lines to exterior slots when a region is too tight.

  type Point = { x: number; y: number };
  type Piece = { outer: { vertices: Point[] }; holes: { vertices: Point[] }[] };
  type Region = { combination: string; pieces: Piece[]; labelAnchor: Point };

  type WorkerResponse =
    | { id: number; ready: true }
    | { id: number; result: FitResult | null }
    | { id: number; error: string };

  const SHAPES: { value: ShapeType; label: string }[] = [
    { value: "circle", label: "Circle" },
    { value: "ellipse", label: "Ellipse" },
    { value: "square", label: "Square" },
    { value: "rectangle", label: "Rectangle" },
  ];

  // Slider definitions. `key` doubles as the spec input string (exclusive
  // region: "A" = items in A only, "A&B" = in A and B but not C, …), so every
  // non-negative combination is a realizable Euler target.
  const SLIDERS: { key: string; label: string }[] = [
    { key: "A", label: "A only" },
    { key: "B", label: "B only" },
    { key: "C", label: "C only" },
    { key: "A&B", label: "A ∩ B" },
    { key: "A&C", label: "A ∩ C" },
    { key: "B&C", label: "B ∩ C" },
    { key: "A&B&C", label: "A ∩ B ∩ C" },
  ];

  const SLIDER_MAX = 12;

  // Palette swapped by theme: the brand "default" set (light pastels tuned for a
  // white artboard, matching the docs examples) in light mode; the saturated
  // "tableau10" in dark mode, where those pastels would wash out. `theme.resolved`
  // is `$state`-backed, so this — and the `COLOR` map below — recompute when the
  // theme flips (toggle or OS change in `system` mode).
  const PALETTE = $derived(theme.resolved === "dark" ? "tableau10" : "default");

  // Three-set colour map. The widget is always A/B/C, so we skip DiagramExample's
  // name-parsing and assign palette slots directly.
  const COLOR = $derived({
    A: defaultColorFor(0, PALETTE),
    B: defaultColorFor(1, PALETTE),
    C: defaultColorFor(2, PALETTE),
  });

  // A pleasant, clearly-overlapping default spec.
  let sizes = $state<Record<string, number>>({
    A: 5,
    B: 5,
    C: 5,
    "A&B": 2,
    "A&C": 2,
    "B&C": 2,
    "A&B&C": 1,
  });
  let shape = $state<ShapeType>("circle");

  let regions = $state<Region[]>([]);
  let error = $state<string | null>(null);
  let booting = $state(true);
  let fitting = $state(false);

  // Lazily-loaded high-level module, held so the measurement effect can call
  // `placeLabelsForRegions` synchronously once regions and text sizes are in.
  let eunoia = $state<typeof import("@jolars/eunoia") | null>(null);
  // Hidden `<text>` host used to measure each label's rendered box via getBBox.
  let measureContainer: SVGGElement | null = $state(null);
  let measuredSizes = $state<Record<string, { w: number; h: number }>>({});

  const LABEL_SIZE = 5;

  let container: HTMLDivElement | null = $state(null);

  let worker: Worker | null = null;
  let nextId = 0;
  let pendingId = -1;
  let debounce: ReturnType<typeof setTimeout> | null = null;
  // Flips once the worker reports `ready`; gates the reactive fit effect.
  let ready = $state(false);

  function buildInputs(): FitInputs {
    const rows = SLIDERS.map((s) => ({ input: s.key, size: sizes[s.key] }));
    return {
      rows,
      inputType: "exclusive",
      shapeType: shape,
      diagramType: "euler",
      vennN: 3,
      advanced: {
        // Single-stage Levenberg–Marquardt (no CMA-ES fallback) keeps each fit
        // a few milliseconds, so dragging feels live. A fixed seed pins the
        // layout so it doesn't flip to a mirrored-but-equivalent arrangement
        // between frames.
        optimizer: "LevenbergMarquardt",
        lossType: "SumSquared",
        // Few restarts (vs. the core default of 10): for a fixed-seed 3-set
        // preview, restart 0 is the canonical warm-start and one extra random
        // attempt is plenty — this is the main speed lever for live dragging.
        restarts: 2,
        showRegions: true,
        seed: 1,
        useSeed: true,
        tolerance: 0,
        useComplement: false,
        complement: null,
      },
    };
  }

  function postFit() {
    if (!worker) return;
    pendingId = ++nextId;
    fitting = true;
    worker.postMessage({ id: pendingId, type: "fit", inputs: buildInputs() });
  }

  function boot() {
    if (worker) return;
    // Load the high-level module on the main thread for label placement (cheap;
    // the heavy fit stays in the worker).
    import("@jolars/eunoia").then((m) => {
      eunoia = m;
    });
    const w = new Worker(new URL("../fit.worker.ts", import.meta.url), {
      type: "module",
    });
    worker = w;
    w.onmessage = (e: MessageEvent<WorkerResponse>) => {
      const msg = e.data;
      if ("ready" in msg) {
        booting = false;
        ready = true;
        return;
      }
      if ("error" in msg) {
        booting = false;
        fitting = false;
        error = msg.error;
        return;
      }
      if (msg.id !== pendingId) return;
      fitting = false;
      error = null;
      const layout = msg.result?.layout;
      regions =
        layout && layout.mode === "regions" ? (layout.regions as Region[]) : [];
    };
    w.onerror = (e) => {
      booting = false;
      fitting = false;
      error = `Worker error: ${e.message}`;
    };
    w.postMessage({ id: ++nextId, type: "init" });
  }

  // Boot lazily the first time the widget scrolls into view, so the worker +
  // wasm never weigh on the landing page's initial load (mirrors basin's
  // playground). Falls back to an immediate boot where IntersectionObserver is
  // unavailable.
  $effect(() => {
    if (!container) return;
    if (typeof IntersectionObserver === "undefined") {
      boot();
      return;
    }
    const io = new IntersectionObserver(
      (entries) => {
        if (entries.some((en) => en.isIntersecting)) {
          io.disconnect();
          boot();
        }
      },
      { rootMargin: "200px" },
    );
    io.observe(container);
    return () => io.disconnect();
  });

  // Refit (debounced) whenever the spec or shape changes — once the worker is
  // ready. Reading `sizes`/`shape` here registers the reactive dependency; the
  // first run after `ready` flips fires the initial fit.
  $effect(() => {
    void sizes.A;
    void sizes.B;
    void sizes.C;
    void sizes["A&B"];
    void sizes["A&C"];
    void sizes["B&C"];
    void sizes["A&B&C"];
    void shape;
    if (!ready) return;
    if (debounce) clearTimeout(debounce);
    debounce = setTimeout(postFit, 120);
  });

  $effect(() => () => {
    if (debounce) clearTimeout(debounce);
    worker?.terminate();
    worker = null;
  });

  function regionFill(combination: string): string {
    const parts = combination.split("&").filter(Boolean);
    if (parts.length === 0) return "#9ca3af";
    let r = 0;
    let g = 0;
    let b = 0;
    let n = 0;
    for (const p of parts) {
      const hex = COLOR[p as keyof typeof COLOR];
      const m = hex?.match(/^#([0-9a-f]{6})$/i);
      if (!m) continue;
      r += parseInt(m[1].slice(0, 2), 16);
      g += parseInt(m[1].slice(2, 4), 16);
      b += parseInt(m[1].slice(4, 6), 16);
      n++;
    }
    if (n === 0) return "#9ca3af";
    return `rgb(${Math.round(r / n)},${Math.round(g / n)},${Math.round(b / n)})`;
  }

  function ringPath(verts: Point[]): string {
    if (verts.length === 0) return "";
    let d = `M ${verts[0].x},${verts[0].y}`;
    for (let i = 1; i < verts.length; i++)
      d += ` L ${verts[i].x},${verts[i].y}`;
    return `${d} Z`;
  }

  function piecePath(piece: Piece): string {
    let d = ringPath(piece.outer.vertices);
    for (const h of piece.holes) d += ` ${ringPath(h.vertices)}`;
    return d;
  }

  // Measure labels via the hidden `<text>` host + getBBox. Runs after the DOM
  // patch that adds the measurement nodes for the current `regions`, so getBBox
  // returns real dimensions (same chain DiagramExample uses).
  $effect(() => {
    void regions;
    if (!measureContainer || regions.length === 0) {
      measuredSizes = {};
      return;
    }
    const sizes: Record<string, { w: number; h: number }> = {};
    const nodes =
      measureContainer.querySelectorAll<SVGGraphicsElement>(
        "text[data-measure]",
      );
    for (const t of Array.from(nodes)) {
      const combo = t.getAttribute("data-measure");
      if (combo === null) continue;
      const bb = t.getBBox();
      sizes[combo] = { w: bb.width, h: bb.height };
    }
    measuredSizes = sizes;
  });

  // Pad each box so the placer guarantees a visible gap between neighbouring
  // labels (it enforces non-overlap on the padded box; we draw at the original
  // size). See the label-placement padding note in eulerr/eunoia docs.
  const paddedSizes = $derived.by(() => {
    const out: Record<string, { w: number; h: number }> = {};
    for (const [k, v] of Object.entries(measuredSizes)) {
      out[k] = { w: v.w + 2, h: v.h + 2 };
    }
    return out;
  });

  const placements = $derived.by<Record<string, LabelPlacement>>(() => {
    if (!eunoia || regions.length === 0) return {};
    if (Object.keys(paddedSizes).length === 0) return {};
    try {
      return eunoia.placeLabelsForRegions({
        regions,
        sizes: paddedSizes,
        strategy: {
          leader: { type: "straight", placement: "raycast", margin: 2 },
          precision: 0.05,
        },
      });
    } catch (err) {
      console.warn("[HeroWidget] placement failed", err);
      return {};
    }
  });

  const viewBox = $derived.by(() => {
    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    const consume = (p: Point) => {
      if (p.x < minX) minX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.x > maxX) maxX = p.x;
      if (p.y > maxY) maxY = p.y;
    };
    for (const region of regions) {
      for (const piece of region.pieces) {
        for (const v of piece.outer.vertices) consume(v);
      }
    }
    // Grow to include exterior label boxes so leader-lined labels aren't clipped.
    for (const [combo, p] of Object.entries(placements)) {
      const s = measuredSizes[combo];
      if (!s) continue;
      consume({ x: p.anchor.x - s.w / 2, y: p.anchor.y - s.h / 2 });
      consume({ x: p.anchor.x + s.w / 2, y: p.anchor.y + s.h / 2 });
    }
    if (!Number.isFinite(minX)) return { x: 0, y: 0, w: 100, h: 100 };
    const w = maxX - minX;
    const h = maxY - minY;
    const pad = Math.max(w, h) * 0.06;
    return { x: minX - pad, y: minY - pad, w: w + 2 * pad, h: h + 2 * pad };
  });

  const viewBoxAttr = $derived(
    `${viewBox.x} ${viewBox.y} ${viewBox.w} ${viewBox.h}`,
  );
  const aspectRatio = $derived(viewBox.w / viewBox.h || 1);
</script>

<div
  bind:this={container}
  class="grid gap-8 md:grid-cols-[minmax(0,1fr)_18rem] items-start"
>
  <!-- Diagram. No `paper` island here (unlike the docs examples): this is a
       for-show widget, so the canvas follows the theme and the strokes/labels
       use `currentColor` to stay legible in both light and dark. -->
  <div
    class="relative rounded-lg border border-line bg-surface p-4 min-h-[18rem] grid place-items-center"
  >
    {#if error}
      <p class="text-sm text-red-600 text-center px-4">{error}</p>
    {:else if booting}
      <div class="text-center text-sm text-muted">
        <div
          class="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto mb-3"
        ></div>
        Loading the fitter…
      </div>
    {:else}
      <svg
        viewBox={viewBoxAttr}
        class="w-full text-ink"
        style="aspect-ratio: {aspectRatio}; max-height: 360px;"
        preserveAspectRatio="xMidYMid meet"
        font-family={DEFAULT_FONT_FAMILY}
        xmlns="http://www.w3.org/2000/svg"
      >
        <!-- Hidden measurement host: each label rendered once so getBBox can
             report its real size for the placer. -->
        <g bind:this={measureContainer} visibility="hidden" aria-hidden="true">
          {#each regions as region}
            {#if region.combination !== ""}
              <text data-measure={region.combination} font-size={LABEL_SIZE}
                >{region.combination}</text
              >
            {/if}
          {/each}
        </g>
        {#each regions as region}
          {#each region.pieces as piece}
            <path
              d={piecePath(piece)}
              fill={regionFill(region.combination)}
              fill-opacity="0.75"
            />
          {/each}
        {/each}
        {#each regions as region}
          {#each region.pieces as piece}
            <path
              d={piecePath(piece)}
              fill="none"
              stroke="currentColor"
              stroke-opacity="1"
              stroke-width="0.5"
            />
          {/each}
        {/each}
        <!-- Labels: poles of inaccessibility inside roomy regions; leader lines
             out to an exterior slot when a region is too tight to hold its
             label. This is the real `placeLabelsForRegions` output. -->
        {#each regions as region}
          {#if region.combination !== ""}
            {@const p = placements[region.combination]}
            {@const anchor = p?.anchor ?? region.labelAnchor}
            {@const isExterior =
              p?.kind === "exteriorRaycast" ||
              p?.kind === "exteriorForceDirected"}
            {#if isExterior && p?.tether}
              <path
                d={leaderPath(
                  p.tether,
                  p.leaderEnd ?? anchor,
                  p.leaderWaypoints,
                )}
                fill="none"
                stroke="currentColor"
                stroke-opacity="0.45"
                stroke-width="0.4"
              />
            {/if}
            <text
              x={anchor.x}
              y={anchor.y}
              text-anchor="middle"
              dominant-baseline="central"
              font-size={LABEL_SIZE}
              fill="currentColor">{region.combination}</text
            >
          {/if}
        {/each}
      </svg>
      {#if fitting}
        <span
          class="absolute top-3 right-3 inline-block h-3 w-3 rounded-full border-2 border-blue-500 border-t-transparent animate-spin"
          aria-label="Fitting"
        ></span>
      {/if}
    {/if}
  </div>

  <!-- Controls -->
  <div class="space-y-4">
    <div class="flex flex-wrap gap-1.5">
      {#each SHAPES as s}
        <button
          type="button"
          onclick={() => (shape = s.value)}
          class="px-2.5 py-1 rounded text-xs font-medium border transition-colors {shape ===
          s.value
            ? 'bg-blue-600 text-white border-blue-600'
            : 'bg-surface border-line text-muted hover:bg-inset'}"
          >{s.label}</button
        >
      {/each}
    </div>

    <div class="space-y-2.5">
      {#each SLIDERS as s}
        <label class="block">
          <span class="flex items-center justify-between text-xs mb-0.5">
            <span class="text-ink">{s.label}</span>
            <span class="font-mono text-muted tabular-nums">{sizes[s.key]}</span
            >
          </span>
          <input
            type="range"
            min="0"
            max={SLIDER_MAX}
            step="1"
            value={sizes[s.key]}
            oninput={(e) =>
              (sizes[s.key] = Number(
                (e.currentTarget as HTMLInputElement).value,
              ))}
            class="w-full accent-blue-600"
          />
        </label>
      {/each}
    </div>

    <p class="text-xs text-muted">
      Values are exclusive region sizes. Drag any of them and the layout refits
      live. Try switching the shape; circles can't always represent every
      overlap, but ellipses usually can.
    </p>
  </div>
</div>
