<script lang="ts">
import type { LabelPlacement, LabelSize, Region } from "@jolars/eunoia";
import { placeLabelsForRegions } from "@jolars/eunoia";
import {
  nestedSets,
  regionTitleLines,
  svgBody,
  type ToSvgOptions,
  viewBox,
} from "@jolars/eunoia/svg";
import { appState } from "../state.svelte";
import type { DiagramStyle, FitResult } from "../types/diagram";

interface Props {
  result: FitResult | null;
  style: DiagramStyle;
  bind?: (svg: SVGSVGElement | null) => void;
}

let { result, style, bind: bindFn }: Props = $props();

let svgEl: SVGSVGElement | null = $state(null);

$effect(() => {
  if (bindFn) bindFn(svgEl);
});

// Label sizing uses getBBox(), which depends on the bundled webfonts being
// loaded. They register at app start but may still be loading on first paint,
// so re-measure once they're ready (Arimo/Tinos are Arial/Times-metric, so
// the pre-load fallback is close, but this makes it exact).
let fontsReady = $state(false);
$effect(() => {
  if (typeof document !== "undefined" && "fonts" in document) {
    document.fonts.ready.then(() => {
      fontsReady = true;
    });
  } else {
    fontsReady = true;
  }
});

// Padding around the diagram in user units. Coordinates from runFit are
// normalized so the largest axis spans ~100 units, so this is ~10 units.
const PADDING = 10;

let fontWeight = $derived(style.fontBold ? 700 : 400);
let fontItalic = $derived(style.fontItalic ? "italic" : "normal");

// Region list + hole-aware nested-set folding, in the shape the serializer
// expects. Both the hidden measurement pass below and `svgBody` derive labels
// from these, so they can't drift.
let regions: Region[] = $derived(
  result && result.layout.mode === "regions" ? result.layout.regions : [],
);
let isRegion = $derived(result?.layout.mode === "regions");
let nested = $derived(
  result && result.layout.mode === "regions" ? nestedSets(result.layout) : {},
);

function fmt(v: number): string {
  if (Math.abs(v) >= 100) return v.toFixed(0);
  if (Math.abs(v) >= 10) return v.toFixed(1);
  return v.toFixed(2);
}

// Per-region label-fit map. Measure each region's combined label via hidden
// `<text>` + `getBBox()` (actual rendered dimensions, not a char-width
// heuristic); the measured sizes feed both `placeLabelsForRegions` (does the
// box inscribe inside the region polygon?) and the serializer's label-aware
// viewBox. Keyed by `region.combination` off the `data-fit-region` nodes
// rendered in the hidden `<g>` below.
let measureContainer: SVGGElement | null = $state(null);
let measuredSizes: Record<string, LabelSize> = $state({});

$effect(() => {
  // Re-measure when result changes, font scales, or showCounts toggles.
  // Reading these explicitly so Svelte tracks them as dependencies.
  void result;
  void style.labelSize;
  void style.fontBold;
  void style.fontItalic;
  void style.fontFamily;
  void style.showCounts;
  void nested;
  void fontsReady;
  if (!measureContainer || !isRegion) {
    measuredSizes = {};
    return;
  }
  const sizes: Record<string, LabelSize> = {};
  const nodes = measureContainer.querySelectorAll<SVGGraphicsElement>(
    "text[data-fit-region]",
  );
  for (const t of Array.from(nodes)) {
    const combo = t.getAttribute("data-fit-region");
    if (combo === null) continue;
    const bb = t.getBBox();
    const cur = sizes[combo];
    if (cur) {
      sizes[combo] = {
        w: Math.max(cur.w, bb.width),
        h: cur.h + bb.height + style.labelSize * 0.1,
      };
    } else {
      sizes[combo] = { w: bb.width, h: bb.height };
    }
  }
  measuredSizes = sizes;
});

// Per-region placement from the eunoia core (wasm). Defaults to
// `Strict + Raycast`: each region's label sits at its POI when the box fits
// inside the polygon, otherwise the anchor is raycast outside the diagram
// bbox (or container, when complement is set) with a leader back to the POI.
// Empty until the first DOM measurement runs; the serializer then falls back
// to each region's own POI for unplaced regions.
let regionPlacements: Record<string, LabelPlacement> = $derived.by(() => {
  if (!result || result.layout.mode !== "regions") return {};
  const sizes = measuredSizes;
  if (Object.keys(sizes).length === 0) return {};
  try {
    return placeLabelsForRegions({
      regions: result.layout.regions,
      container: result.layout.container,
      sizes,
      strategy: {
        leader:
          style.labelPlacement === "elbow"
            ? { type: "elbow" }
            : { type: "straight", placement: style.labelPlacement },
        precision: Math.max(0.05, style.labelSize * 0.05),
        tether: style.labelTether,
        // Stop the leader a fraction of a glyph-height short of the text box
        // edge so the line doesn't kiss the glyph contours.
        leaderGap: style.labelSize * 0.25,
      },
    });
  } catch (err) {
    console.warn("[place] failed, falling back to region POIs:", err);
    return {};
  }
});

// The single options object handed to the serializer — the adapter from the
// app's `DiagramStyle` + computed set order/placements to `@jolars/eunoia/svg`.
let svgOptions: ToSvgOptions = $derived({
  palette: style.palette,
  colors: style.colors,
  setOrder: appState.setNames,
  alpha: style.alpha,
  strokeWidth: style.strokeWidth,
  labelSize: style.labelSize,
  fontWeight,
  fontStyle: fontItalic,
  showCounts: style.showCounts,
  legend: { show: style.showLegend, position: style.legendPosition },
  padding: PADDING,
  placements: regionPlacements,
  labelSizes: measuredSizes,
  complement: result?.complement,
});

let vb = $derived(
  result ? viewBox(result.layout, svgOptions) : { x: 0, y: 0, w: 100, h: 100 },
);
let vbAttr = $derived(`${vb.x} ${vb.y} ${vb.w} ${vb.h}`);
let aspectRatio = $derived(vb.w / vb.h);

// Inner SVG markup from the shared serializer. Rendered via `{@html}` inside
// our own bound `<svg>` so `svgEl` stays a live handle for the export
// toolbar, and the hidden measurement `<g>` can live alongside it.
let body = $derived(result ? svgBody(result.layout, svgOptions) : "");
</script>

<svg
  bind:this={svgEl}
  viewBox={vbAttr}
  class="w-full"
  style="aspect-ratio: {aspectRatio}; max-height: 80vh;"
  preserveAspectRatio="xMidYMid meet"
  font-family={style.fontFamily}
  xmlns="http://www.w3.org/2000/svg"
>
  {#if result && isRegion}
    <g
      bind:this={measureContainer}
      visibility="hidden"
      aria-hidden="true"
      data-fit-measure
    >
      {#each regions as region}
        {#each regionTitleLines(region.combination, nested) as title}
          <text
            data-fit-region={region.combination}
            font-size={style.labelSize}
            font-weight={fontWeight}
            font-style={fontItalic}
          >
            {title}
          </text>
        {/each}
        {#if style.showCounts}
          <text
            data-fit-region={region.combination}
            font-size={style.labelSize * 0.75}
          >
            {fmt(region.totalArea)}
          </text>
        {/if}
      {/each}
    </g>
  {/if}
  {#if result}
    <!-- eslint-disable-next-line svelte/no-at-html-tags -->
    {@html body}
  {/if}
</svg>
