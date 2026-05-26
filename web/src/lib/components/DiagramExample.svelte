<script lang="ts">
import { onMount } from "svelte";
import { browser } from "$app/environment";
import type { LabelPlacement } from "@jolars/eunoia";
import { defaultColorFor } from "$lib/colors";
import { DEFAULT_FONT_FAMILY } from "$lib/fonts";
import { leaderPath } from "$lib/leader";

interface Props {
  /** Set sizes keyed by combination, e.g. `{ A: 5, B: 3, "A&B": 1.5 }`. */
  sets: Record<string, number>;
  /** Default `"exclusive"`. */
  inputType?: "exclusive" | "inclusive";
  /** Shape primitive to fit. Default `"ellipse"` since most chapter examples use n>=3. */
  shape?: "circle" | "ellipse" | "square" | "rectangle";
  /** Exterior placement strategy. Default `"raycast"`. */
  strategy?: "raycast" | "forceDirected";
  /** RNG seed (lock examples so the chapter renders the same diagram for every reader). */
  seed?: number;
  /** Caption rendered below the diagram. */
  caption?: string;
  /** Label font-size in diagram coordinates. Default `5`. */
  labelSize?: number;
  /**
   * Per-side padding (user units) added to the size we hand the placer.
   * The algorithm separates labels to non-overlap; without padding they
   * end up edge-to-edge. With `padding = p`, neighbouring labels are
   * guaranteed at least `2 * p` of visual gap. Default `1`.
   */
  padding?: number;
  /**
   * Exterior margin (user units) — gap between the diagram bbox and
   * exterior labels. Default `2`. The library's own default is
   * `0.5 * max(label_w, label_h)` *per label*, which pushes long
   * labels (`A&B&C&D` and friends) far away from the diagram; a
   * fixed small value keeps them all close in.
   */
  margin?: number;
}

let {
  sets,
  inputType = "exclusive",
  shape = "ellipse",
  strategy = "raycast",
  seed = 42,
  caption,
  labelSize = 5,
  padding = 1,
  margin = 2,
}: Props = $props();

type Point = { x: number; y: number };
type Region = {
  combination: string;
  pieces: { outer: { vertices: Point[] }; holes: { vertices: Point[] }[] }[];
  labelAnchor: Point;
};

// Lazy-loaded eunoia module — held here so the measurement effect can call
// `placeLabelsForRegions` synchronously once regions are in place.
let eunoia: typeof import("@jolars/eunoia") | null = $state(null);

let regions: Region[] = $state([]);
let error: string | null = $state(null);
let measureContainer: SVGGElement | null = $state(null);
let measuredSizes: Record<string, { w: number; h: number }> = $state({});

// Set names for color assignment.
const setNames = $derived(
  Array.from(
    new Set(
      Object.keys(sets).flatMap((k) =>
        k.split("&").map((s) => s.trim()).filter(Boolean),
      ),
    ),
  ),
);

const colorMap = $derived.by(() => {
  const m = new Map<string, string>();
  setNames.forEach((n, i) => m.set(n, defaultColorFor(i)));
  return m;
});

function regionFill(combination: string): string {
  const parts = combination
    .split("&")
    .map((s) => s.trim())
    .filter(Boolean);
  if (parts.length === 0) return "#9ca3af";
  if (parts.length === 1) return colorMap.get(parts[0]) ?? defaultColorFor(0);
  let r = 0,
    g = 0,
    b = 0,
    n = 0;
  for (const p of parts) {
    const hex = colorMap.get(p);
    if (!hex) continue;
    const m = hex.match(/^#([0-9a-f]{6})$/i);
    if (!m) continue;
    r += parseInt(m[1].slice(0, 2), 16);
    g += parseInt(m[1].slice(2, 4), 16);
    b += parseInt(m[1].slice(4, 6), 16);
    n++;
  }
  if (n === 0) return "#9ca3af";
  return `rgb(${Math.round(r / n)},${Math.round(g / n)},${Math.round(b / n)})`;
}

function piecePath(piece: {
  outer: { vertices: Point[] };
  holes: { vertices: Point[] }[];
}): string {
  function ringPath(verts: Point[]): string {
    if (verts.length === 0) return "";
    let d = `M ${verts[0].x},${verts[0].y}`;
    for (let i = 1; i < verts.length; i++) {
      d += ` L ${verts[i].x},${verts[i].y}`;
    }
    return d + " Z";
  }
  let d = ringPath(piece.outer.vertices);
  for (const h of piece.holes) d += " " + ringPath(h.vertices);
  return d;
}

function normaliseRegions(input: Region[]): Region[] {
  let minX = Infinity,
    minY = Infinity,
    maxX = -Infinity,
    maxY = -Infinity;
  for (const r of input) {
    for (const piece of r.pieces) {
      for (const v of piece.outer.vertices) {
        if (v.x < minX) minX = v.x;
        if (v.y < minY) minY = v.y;
        if (v.x > maxX) maxX = v.x;
        if (v.y > maxY) maxY = v.y;
      }
    }
  }
  if (!isFinite(minX)) return input;
  const span = Math.max(maxX - minX, maxY - minY) || 1;
  const k = 100 / span;
  const np = (p: Point): Point => ({
    x: (p.x - minX) * k,
    y: (p.y - minY) * k,
  });
  return input.map((r) => ({
    combination: r.combination,
    labelAnchor: np(r.labelAnchor),
    pieces: r.pieces.map((piece) => ({
      outer: { vertices: piece.outer.vertices.map(np) },
      holes: piece.holes.map((h) => ({ vertices: h.vertices.map(np) })),
    })),
  }));
}

// Fit on mount. After this completes the SVG renders (its `{#if regions}`
// gate flips), `measureContainer` populates, the measurement effect fires,
// and the placement `$derived` runs — same chain DiagramSvg.svelte uses.
onMount(async () => {
  if (!browser) return;
  try {
    const mod = await import("@jolars/eunoia");
    eunoia = mod;
    const layout = mod.euler({
      sets,
      inputType,
      shape,
      output: "regions",
      seed,
      polygonVertices: 128,
    });
    if (layout.mode !== "regions") {
      throw new Error(`expected regions mode, got ${layout.mode}`);
    }
    regions = normaliseRegions(layout.regions as Region[]);
  } catch (err) {
    error = err instanceof Error ? err.message : String(err);
  }
});

// Measure labels via hidden SVG `<text>` + getBBox. Effect runs after the
// DOM patch that adds the measurement nodes for `regions`, so getBBox
// returns real (non-zero) dimensions.
$effect(() => {
  void regions;
  void labelSize;
  if (!measureContainer || regions.length === 0) {
    measuredSizes = {};
    return;
  }
  const sizes: Record<string, { w: number; h: number }> = {};
  const nodes = measureContainer.querySelectorAll<SVGGraphicsElement>(
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

// Pad each measured size by `padding` on every side before handing the
// placer the box. The placer guarantees non-overlap on the padded box,
// which translates to a `2 * padding` minimum gap between rendered
// labels. We render text at the original (un-padded) size below.
const paddedSizes = $derived.by(() => {
  const out: Record<string, { w: number; h: number }> = {};
  for (const [k, v] of Object.entries(measuredSizes)) {
    out[k] = { w: v.w + 2 * padding, h: v.h + 2 * padding };
  }
  return out;
});

const placements: Record<string, LabelPlacement> = $derived.by(() => {
  if (!eunoia || regions.length === 0) return {};
  if (Object.keys(paddedSizes).length === 0) return {};
  try {
    return eunoia.placeLabelsForRegions({
      regions,
      sizes: paddedSizes,
      strategy: {
        leader: { type: "straight", placement: strategy, margin },
        precision: 0.05,
      },
    });
  } catch (err) {
    console.warn("[DiagramExample] placement failed", err);
    return {};
  }
});

const viewBox = $derived.by(() => {
  let minX = Infinity,
    minY = Infinity,
    maxX = -Infinity,
    maxY = -Infinity;
  function consume(p: Point) {
    if (p.x < minX) minX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.x > maxX) maxX = p.x;
    if (p.y > maxY) maxY = p.y;
  }
  for (const r of regions) {
    for (const piece of r.pieces) {
      for (const v of piece.outer.vertices) consume(v);
    }
  }
  for (const [combo, p] of Object.entries(placements)) {
    const s = measuredSizes[combo];
    if (!s) continue;
    consume({ x: p.anchor.x - s.w / 2, y: p.anchor.y - s.h / 2 });
    consume({ x: p.anchor.x + s.w / 2, y: p.anchor.y + s.h / 2 });
  }
  if (!isFinite(minX)) {
    return { x: 0, y: 0, w: 100, h: 100 };
  }
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

<!-- `paper`: keep example diagrams on white in dark mode so their
     hardcoded-black labels/strokes stay readable (see app.css). -->
<figure class="paper not-prose my-6 rounded-lg border border-line bg-surface p-4">
  <div class="relative">
    {#if error}
      <div
        class="aspect-square w-full flex items-center justify-center text-sm text-red-600"
      >{error}</div>
    {:else}
      <svg
        viewBox={viewBoxAttr}
        class="w-full"
        style="aspect-ratio: {aspectRatio}; max-height: 400px;"
        preserveAspectRatio="xMidYMid meet"
        font-family={DEFAULT_FONT_FAMILY}
        xmlns="http://www.w3.org/2000/svg"
      >
        <g
          bind:this={measureContainer}
          visibility="hidden"
          aria-hidden="true"
        >
          {#each regions as region}
            {#if region.combination !== ""}
              <text
                data-measure={region.combination}
                font-size={labelSize}
              >{region.combination}</text>
            {/if}
          {/each}
        </g>
        {#each regions as region}
          <path
            d={piecePath(region.pieces[0] ?? { outer: { vertices: [] }, holes: [] })}
            fill={regionFill(region.combination)}
            fill-opacity="0.55"
          />
          {#each region.pieces.slice(1) as piece}
            <path
              d={piecePath(piece)}
              fill={regionFill(region.combination)}
              fill-opacity="0.55"
            />
          {/each}
        {/each}
        <!-- Borders drawn as a separate pass after all fills so they sit on
             top of neighbouring regions (mirrors DiagramSvg region mode). -->
        {#each regions as region}
          {#each region.pieces as piece}
            <path
              d={piecePath(piece)}
              fill="none"
              stroke="#374151"
              stroke-width="0.5"
            />
          {/each}
        {/each}
        {#each regions as region}
          {#if region.combination !== ""}
            {@const p = placements[region.combination]}
            {@const isExterior =
              p?.kind === "exteriorRaycast" ||
              p?.kind === "exteriorForceDirected"}
            {@const anchor = p?.anchor ?? region.labelAnchor}
            {#if isExterior && p?.tether}
              <path
                d={leaderPath(
                  p.tether,
                  p.leaderEnd ?? anchor,
                  p.leaderWaypoints,
                )}
                fill="none"
                stroke="#6b7280"
                stroke-width="0.4"
                stroke-opacity="0.7"
              />
            {/if}
            <text
              x={anchor.x}
              y={anchor.y}
              text-anchor="middle"
              dominant-baseline="central"
              font-size={labelSize}
              fill="#111827"
            >{region.combination}</text>
          {/if}
        {/each}
      </svg>
    {/if}
  </div>
  {#if caption}
    <figcaption class="mt-3 text-sm text-muted text-center">
      {caption}
    </figcaption>
  {/if}
</figure>
