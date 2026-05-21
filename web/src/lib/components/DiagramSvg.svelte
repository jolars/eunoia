<script lang="ts">
  import { placeLabelsForRegions, placementsBbox } from "@jolars/eunoia";
  import type { LabelPlacement } from "@jolars/eunoia";
  import type {
    DiagramStyle,
    FitResult,
    Polygon,
    RegionPiece,
  } from "../types/diagram";
  import { defaultColorFor } from "../colors";
  import { leaderPath } from "../leader";
  import { appState } from "../state.svelte";

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

  // Padding around the diagram in user units. Coordinates from runFit are
  // normalized so the largest axis spans ~100 units, so this is ~10 units.
  const PADDING = 10;

  type Bounds = { minX: number; minY: number; maxX: number; maxY: number };

  let bounds: Bounds = $derived.by(() => {
    let minX = Infinity,
      minY = Infinity,
      maxX = -Infinity,
      maxY = -Infinity;

    function consume(p: { x: number; y: number }) {
      if (p.x < minX) minX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.x > maxX) maxX = p.x;
      if (p.y > maxY) maxY = p.y;
    }

    if (!result) {
      return { minX: 0, minY: 0, maxX: 100, maxY: 100 };
    }

    if (result.shapeMode === "region") {
      for (const r of result.regions) {
        for (const piece of r.pieces) {
          for (const v of piece.outer.vertices) consume(v);
          for (const h of piece.holes) {
            for (const v of h.vertices) consume(v);
          }
        }
      }
    } else {
      for (const p of result.polygons) {
        for (const v of p.vertices) consume(v);
      }
      for (const c of result.circles) {
        consume({ x: c.x - c.radius, y: c.y - c.radius });
        consume({ x: c.x + c.radius, y: c.y + c.radius });
      }
      for (const e of result.ellipses) {
        const cos = Math.cos(e.rotation);
        const sin = Math.sin(e.rotation);
        const dx = Math.sqrt(
          e.semi_major * e.semi_major * cos * cos +
            e.semi_minor * e.semi_minor * sin * sin,
        );
        const dy = Math.sqrt(
          e.semi_major * e.semi_major * sin * sin +
            e.semi_minor * e.semi_minor * cos * cos,
        );
        consume({ x: e.x - dx, y: e.y - dy });
        consume({ x: e.x + dx, y: e.y + dy });
      }
      for (const s of result.squares) {
        const h = s.side / 2;
        consume({ x: s.x - h, y: s.y - h });
        consume({ x: s.x + h, y: s.y + h });
      }
      for (const r of result.rectangles) {
        const hw = r.width / 2;
        const hh = r.height / 2;
        consume({ x: r.x - hw, y: r.y - hh });
        consume({ x: r.x + hw, y: r.y + hh });
      }
    }

    if (result.container) {
      const c = result.container;
      consume({ x: c.x - c.width / 2, y: c.y - c.height / 2 });
      consume({ x: c.x + c.width / 2, y: c.y + c.height / 2 });
    }

    // Extend bounds to cover label boxes — both exterior strategies place
    // anchors well outside the diagram bbox; without this the viewBox
    // clips them off-screen. `placementsBbox` returns the union of every
    // (anchor ± half-label) so we just consume its corners.
    if (result.shapeMode === "region") {
      const labelBbox = placementsBbox({
        placements: regionPlacements,
        sizes: measuredSizes,
      });
      if (labelBbox) {
        const halfW = labelBbox.width / 2;
        const halfH = labelBbox.height / 2;
        consume({ x: labelBbox.x - halfW, y: labelBbox.y - halfH });
        consume({ x: labelBbox.x + halfW, y: labelBbox.y + halfH });
      }
    }

    if (!isFinite(minX)) {
      return { minX: 0, minY: 0, maxX: 100, maxY: 100 };
    }
    return { minX, minY, maxX, maxY };
  });

  // Result-driven label list (drives lookups for circles/ellipses/etc. and
  // hole-aware nested labels). Order follows whatever the wasm result
  // returned; the legend uses `legendLabels` instead so its order stays
  // stable across reseeds.
  let setLabels: string[] = $derived.by(() => {
    if (!result) return [];
    if (result.shapeMode === "region") {
      const seen = new Set<string>();
      for (const r of result.regions) {
        for (const ch of r.combination.split("&")) {
          const trimmed = ch.trim();
          if (trimmed) seen.add(trimmed);
        }
      }
      return Array.from(seen);
    }
    if (result.circles.length > 0) return result.circles.map((c) => c.label);
    if (result.ellipses.length > 0) return result.ellipses.map((e) => e.label);
    if (result.squares.length > 0) return result.squares.map((s) => s.label);
    if (result.rectangles.length > 0) return result.rectangles.map((r) => r.label);
    return result.polygons.map((p) => p.label);
  });

  // Legend lists the user-input set order, restricted to labels that
  // actually appear in the fit result (so deleted/empty rows don't show up).
  let legendLabels: string[] = $derived.by(() => {
    const present = new Set(setLabels);
    return appState.setNames.filter((n) => present.has(n));
  });

  // Color index keyed off the stable, input-order set list rather than the
  // fit output, so the same set keeps its color across reseeds and reshuffled
  // result arrays. Any label that appears in the fit but isn't in the spec
  // (shouldn't normally happen) falls back to its result-order index.
  let setColorMap: Map<string, string> = $derived.by(() => {
    const m = new Map<string, string>();
    const order = appState.setNames;
    const indexOf = new Map<string, number>();
    order.forEach((name, i) => indexOf.set(name, i));
    setLabels.forEach((l, i) => {
      const idx = indexOf.get(l) ?? i;
      m.set(l, style.colors[l] ?? defaultColorFor(idx, style.palette));
    });
    return m;
  });

  function regionFill(combination: string): string {
    const sets = combination
      .split("&")
      .map((s) => s.trim())
      .filter(Boolean);
    const fallback = defaultColorFor(0, style.palette);
    if (sets.length === 1) {
      return setColorMap.get(sets[0]) || fallback;
    }
    const colors = sets
      .map((s) => setColorMap.get(s))
      .filter((c): c is string => !!c);
    if (colors.length === 0) return fallback;
    return mixColors(colors);
  }

  function mixColors(colors: string[]): string {
    let r = 0,
      g = 0,
      b = 0,
      n = 0;
    const ctx = document.createElement("canvas").getContext("2d");
    if (!ctx) return colors[0];
    for (const c of colors) {
      ctx.fillStyle = c;
      const computed = ctx.fillStyle as string;
      const m = computed.match(
        /rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)/,
      );
      if (m) {
        r += +m[1];
        g += +m[2];
        b += +m[3];
        n++;
      } else if (computed.startsWith("#")) {
        const hex = computed.slice(1);
        const full =
          hex.length === 3 ? hex.split("").map((c) => c + c).join("") : hex;
        r += parseInt(full.slice(0, 2), 16);
        g += parseInt(full.slice(2, 4), 16);
        b += parseInt(full.slice(4, 6), 16);
        n++;
      }
    }
    if (n === 0) return colors[0];
    return `rgb(${Math.round(r / n)},${Math.round(g / n)},${Math.round(b / n)})`;
  }

  /**
   * Sets that have no exclusive single-set region (e.g. a fully-nested B
   * inside A) get their set label folded into the largest region that
   * contains them — mirroring the core's `largest_containing_region_anchor`
   * fallback. Mapping is region combination → ordered list of nested set
   * names hosted there.
   *
   * Folding (rather than dropping the name at `setAnchors`) routes the
   * nested set name through the same measurement + `placeLabelsForRegions`
   * pipeline as every other label, so it (a) stacks above the host region's
   * quantity instead of overlapping it, and (b) is raycast outside with a
   * leader when the host region is too small to hold it.
   *
   * Every nested set is guaranteed a host: `setLabels` only contains sets
   * that appear in some region, and a set with no *exclusive* region must
   * therefore appear in at least one intersection region.
   */
  let regionNestedSets: Record<string, string[]> = $derived.by(() => {
    const map: Record<string, string[]> = {};
    if (!result || result.shapeMode !== "region") return map;
    const hasExclusive = new Set<string>();
    for (const r of result.regions) {
      if (!r.combination.includes("&")) hasExclusive.add(r.combination.trim());
    }
    for (const name of setLabels) {
      if (hasExclusive.has(name)) continue;
      let best: { combo: string; area: number } | null = null;
      for (const r of result.regions) {
        const sets = r.combination.split("&").map((s) => s.trim());
        if (!sets.includes(name)) continue;
        if (!best || r.totalArea > best.area) {
          best = { combo: r.combination, area: r.totalArea };
        }
      }
      if (best) (map[best.combo] ??= []).push(name);
    }
    return map;
  });

  /**
   * Title lines (set names) shown inside a region's label box, stacked above
   * its quantity: the set's own name for an exclusive single-set region, or
   * the names of any sets nested into this region otherwise.
   */
  function regionTitleLines(combination: string): string[] {
    if (!combination.includes("&")) return [combination];
    return regionNestedSets[combination] ?? [];
  }

  let fontWeight = $derived(style.fontBold ? 700 : 400);
  let fontItalic = $derived(style.fontItalic ? "italic" : "normal");

  /**
   * Per-region label-fit map. Demo wiring: measure each region's combined
   * label via a hidden `<text>` element + `getBBox()` (so we use actual
   * rendered dimensions instead of a char-width heuristic), then call into
   * the eunoia core's `fit_label_in_region` (via
   * `placeRegionLabelsForRegions`) to check whether an axis-aligned `(w, h)`
   * rectangle inscribes inside the region polygon (with holes). Visible
   * labels are gated on the result.
   *
   * The measurement is keyed by `region.combination` and uses a
   * `<text data-fit-region="…">` element rendered in a hidden `<g>` below
   * (see the `fit-measure` block in the SVG body).
   */
  let measureContainer: SVGGElement | null = $state(null);
  let measuredSizes: Record<string, { w: number; h: number }> = $state({});

  $effect(() => {
    // Re-measure when result changes, font scales, or showCounts toggles.
    // Reading these explicitly so Svelte tracks them as dependencies.
    void result;
    void style.labelSize;
    void style.fontBold;
    void style.fontItalic;
    void style.showCounts;
    void regionNestedSets;
    console.debug("[fit-measure] effect run", {
      hasContainer: !!measureContainer,
      mode: result?.shapeMode,
      labelSize: style.labelSize,
    });
    if (!measureContainer || !result || result.shapeMode !== "region") {
      measuredSizes = {};
      return;
    }
    const sizes: Record<string, { w: number; h: number }> = {};
    const nodes = measureContainer.querySelectorAll<SVGGraphicsElement>(
      "text[data-fit-region]",
    );
    console.debug("[fit-measure] nodes found", nodes.length);
    for (const t of Array.from(nodes)) {
      const combo = t.getAttribute("data-fit-region");
      if (combo === null) continue;
      const bb = t.getBBox();
      console.debug("[fit-measure] node", combo, {
        text: t.textContent,
        bb: { x: bb.x, y: bb.y, w: bb.width, h: bb.height },
      });
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

  /**
   * Per-region placement decision from the eunoia core. Defaults to the
   * `Strict + Raycast` strategy: each region's label sits at its POI when
   * the box fits inside the polygon, otherwise the anchor is raycast
   * outside the diagram bbox (or container, when complement is set) with
   * a leader line drawn back to the region's POI.
   *
   * Until the first DOM measurement runs (or for regions we didn't
   * measure — typically intersection regions with showCounts off), we
   * fall back to the region's own POI from the fit result.
   */
  let regionPlacements: Record<string, LabelPlacement> = $derived.by(() => {
    if (!result || result.shapeMode !== "region") return {};
    const sizes = measuredSizes;
    if (Object.keys(sizes).length === 0) {
      return {};
    }
    try {
      const placements = placeLabelsForRegions({
        regions: result.regions,
        container: result.container,
        sizes,
        strategy: {
          exterior: style.labelPlacement,
          precision: Math.max(0.05, style.labelSize * 0.05),
          tether: style.labelTether,
          // Stop the leader a fraction of a glyph-height short of the
          // text box edge so the line doesn't kiss the glyph contours.
          // `sizes` here are raw measured bboxes (no pre-padding), so
          // without this the leader would terminate exactly at the
          // outermost glyph edge.
          leaderGap: style.labelSize * 0.25,
        },
      });
      console.debug("[place]", {
        labelSize: style.labelSize,
        mode: style.labelPlacement,
        sizes,
        placements: Object.fromEntries(
          Object.entries(placements).map(([k, p]) => [k, p.kind]),
        ),
      });
      return placements;
    } catch (err) {
      console.warn("[place] failed, falling back to region POIs:", err);
      return {};
    }
  });

  function regionAnchor(
    combo: string,
    fallbackX: number,
    fallbackY: number,
  ): { x: number; y: number } {
    const p = regionPlacements[combo];
    if (p) return p.anchor;
    return { x: fallbackX, y: fallbackY };
  }

  let viewBox = $derived.by(() => {
    let { minX, minY, maxX, maxY } = bounds;
    let w = maxX - minX;
    let h = maxY - minY;
    let lx = minX - PADDING;
    let ly = minY - PADDING;
    let lw = w + 2 * PADDING;
    let lh = h + 2 * PADDING;


    // Reserve space for the legend.
    if (result && style.showLegend && legendLabels.length > 0) {
      const legendW = Math.max(20, style.labelSize * 2);
      const legendH =
        Math.max(8, style.labelSize * 1.4) * legendLabels.length + 8;
      switch (style.legendPosition) {
        case "right":
          lw += legendW;
          break;
        case "left":
          lw += legendW;
          lx -= legendW;
          break;
        case "top":
          lh += legendH;
          ly -= legendH;
          break;
        case "bottom":
          lh += legendH;
          break;
      }
    }
    return { x: lx, y: ly, w: lw, h: lh };
  });

  let viewBoxAttr = $derived(
    `${viewBox.x} ${viewBox.y} ${viewBox.w} ${viewBox.h}`,
  );
  let aspectRatio = $derived(viewBox.w / viewBox.h);

  function polygonPath(p: Polygon): string {
    if (p.vertices.length === 0) return "";
    const v0 = p.vertices[0];
    let d = `M ${v0.x},${v0.y}`;
    for (let i = 1; i < p.vertices.length; i++) {
      d += ` L ${p.vertices[i].x},${p.vertices[i].y}`;
    }
    d += " Z";
    return d;
  }

  /**
   * One SVG `d` string for a region piece — its outer ring plus any hole
   * rings concatenated. The core library normalises orientations (CCW
   * outer, CW holes), so the SVG default `fill-rule: nonzero` fills only
   * the donut/cookie shape correctly without further bookkeeping.
   */
  function piecePath(piece: RegionPiece): string {
    let d = polygonPath(piece.outer);
    for (const h of piece.holes) {
      d += " " + polygonPath(h);
    }
    return d;
  }


  function fmt(v: number): string {
    if (Math.abs(v) >= 100) return v.toFixed(0);
    if (Math.abs(v) >= 10) return v.toFixed(1);
    return v.toFixed(2);
  }

  let strokeW = $derived(style.strokeWidth);
  let showStroke = $derived(style.strokeWidth > 0);

  // Legend layout
  let legendBox = $derived.by(() => {
    if (!result || !style.showLegend || legendLabels.length === 0) {
      return null;
    }
    const swatch = style.labelSize * 0.9;
    const gap = swatch * 0.4;
    const lineH = swatch + gap;
    const totalH = lineH * legendLabels.length;
    const padding = swatch * 0.5;
    let x = 0,
      y = 0;
    switch (style.legendPosition) {
      case "right":
        x = bounds.maxX + PADDING * 0.5;
        y = (bounds.minY + bounds.maxY) / 2 - totalH / 2;
        break;
      case "left":
        x = bounds.minX - PADDING * 0.5 - swatch * 6;
        y = (bounds.minY + bounds.maxY) / 2 - totalH / 2;
        break;
      case "top":
        x = (bounds.minX + bounds.maxX) / 2 - swatch * 3;
        y = bounds.minY - PADDING * 0.5 - totalH;
        break;
      case "bottom":
        x = (bounds.minX + bounds.maxX) / 2 - swatch * 3;
        y = bounds.maxY + PADDING * 0.5;
        break;
    }
    return { x, y, swatch, gap, lineH, padding };
  });
</script>

<svg
  bind:this={svgEl}
  viewBox={viewBoxAttr}
  class="w-full"
  style="aspect-ratio: {aspectRatio}; max-height: 80vh;"
  preserveAspectRatio="xMidYMid meet"
  xmlns="http://www.w3.org/2000/svg"
>
  {#if result && result.shapeMode === "region"}
    <g
      bind:this={measureContainer}
      visibility="hidden"
      aria-hidden="true"
      data-fit-measure
    >
      {#each result.regions as region}
        {#each regionTitleLines(region.combination) as title}
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
    {#if result.container}
      {@const c = result.container}
      <rect
        x={c.x - c.width / 2}
        y={c.y - c.height / 2}
        width={c.width}
        height={c.height}
        fill="none"
        stroke="#9ca3af"
        stroke-width={Math.max(strokeW, 0.5)}
        stroke-dasharray="2 2"
      />
      {#if result.complement !== undefined && style.showCounts}
        <text
          x={c.x + c.width / 2 - 1.5}
          y={c.y - c.height / 2 + 2}
          text-anchor="end"
          dominant-baseline="hanging"
          font-size={style.labelSize * 0.75}
          fill="#6b7280"
        >
          {fmt(result.complement)}
        </text>
      {/if}
    {/if}
    {#if result.shapeMode === "region"}
      {#each result.regions as region}
        {@const fill = regionFill(region.combination)}
        {#each region.pieces as piece}
          <path
            d={piecePath(piece)}
            fill={fill}
            fill-opacity={style.alpha}
            stroke="none"
          />
        {/each}
      {/each}
      {#if showStroke}
        {#each result.regions as region}
          {#each region.pieces as piece}
            <path
              d={piecePath(piece)}
              fill="none"
              stroke="black"
              stroke-width={strokeW}
            />
          {/each}
        {/each}
      {/if}
      {#each result.regions as region}
        {@const placement = regionPlacements[region.combination]}
        {@const isExterior =
          placement?.kind === "exteriorRaycast" ||
          placement?.kind === "exteriorForceDirected"}
        {@const anchor = regionAnchor(
          region.combination,
          region.labelX,
          region.labelY,
        )}
        {@const titleLines = regionTitleLines(region.combination)}
        {#if isExterior && placement?.tether}
          <path
            d={leaderPath(
              placement.tether,
              placement.leaderEnd ?? anchor,
              placement.leaderControl1,
              placement.leaderControl2,
            )}
            fill="none"
            stroke="#6b7280"
            stroke-width={Math.max(strokeW * 0.5, 0.3)}
            stroke-opacity="0.6"
          />
        {/if}
        {#each titleLines as title, i}
          <text
            x={anchor.x}
            y={anchor.y + i * style.labelSize}
            text-anchor="middle"
            dominant-baseline="central"
            font-size={style.labelSize}
            font-weight={fontWeight}
            font-style={fontItalic}
            fill="black"
          >
            {title}
          </text>
        {/each}
        {#if style.showCounts}
          <text
            x={anchor.x}
            y={anchor.y + titleLines.length * style.labelSize}
            text-anchor="middle"
            dominant-baseline="central"
            font-size={style.labelSize * 0.75}
            fill="#374151"
          >
            {fmt(region.totalArea)}
          </text>
        {/if}
      {/each}
    {:else}
      {#each result.circles as circle, i}
        {@const color = setColorMap.get(circle.label) || defaultColorFor(i, style.palette)}
        <circle
          cx={circle.x}
          cy={circle.y}
          r={circle.radius}
          fill={color}
          fill-opacity={style.alpha}
          stroke="none"
        />
      {/each}
      {#each result.ellipses as ellipse, i}
        {@const color = setColorMap.get(ellipse.label) || defaultColorFor(i, style.palette)}
        <ellipse
          cx={ellipse.x}
          cy={ellipse.y}
          rx={ellipse.semi_major}
          ry={ellipse.semi_minor}
          transform={`rotate(${(ellipse.rotation * 180) / Math.PI} ${ellipse.x} ${ellipse.y})`}
          fill={color}
          fill-opacity={style.alpha}
          stroke="none"
        />
      {/each}
      {#each result.squares as square, i}
        {@const color = setColorMap.get(square.label) || defaultColorFor(i, style.palette)}
        <rect
          x={square.x - square.side / 2}
          y={square.y - square.side / 2}
          width={square.side}
          height={square.side}
          fill={color}
          fill-opacity={style.alpha}
          stroke="none"
        />
      {/each}
      {#each result.rectangles as rect, i}
        {@const color = setColorMap.get(rect.label) || defaultColorFor(i, style.palette)}
        <rect
          x={rect.x - rect.width / 2}
          y={rect.y - rect.height / 2}
          width={rect.width}
          height={rect.height}
          fill={color}
          fill-opacity={style.alpha}
          stroke="none"
        />
      {/each}
      {#if showStroke}
        {#each result.circles as circle}
          <circle
            cx={circle.x}
            cy={circle.y}
            r={circle.radius}
            fill="none"
            stroke="black"
            stroke-width={strokeW}
          />
        {/each}
        {#each result.ellipses as ellipse}
          <ellipse
            cx={ellipse.x}
            cy={ellipse.y}
            rx={ellipse.semi_major}
            ry={ellipse.semi_minor}
            transform={`rotate(${(ellipse.rotation * 180) / Math.PI} ${ellipse.x} ${ellipse.y})`}
            fill="none"
            stroke="black"
            stroke-width={strokeW}
          />
        {/each}
        {#each result.squares as square}
          <rect
            x={square.x - square.side / 2}
            y={square.y - square.side / 2}
            width={square.side}
            height={square.side}
            fill="none"
            stroke="black"
            stroke-width={strokeW}
          />
        {/each}
        {#each result.rectangles as rect}
          <rect
            x={rect.x - rect.width / 2}
            y={rect.y - rect.height / 2}
            width={rect.width}
            height={rect.height}
            fill="none"
            stroke="black"
            stroke-width={strokeW}
          />
        {/each}
      {/if}
      {#each result.circles as circle}
        <text
          x={circle.labelX ?? circle.x}
          y={circle.labelY ?? circle.y}
          text-anchor="middle"
          dominant-baseline="central"
          font-size={style.labelSize}
          font-weight={fontWeight}
          font-style={fontItalic}
        >
          {circle.label}
        </text>
      {/each}
      {#each result.ellipses as ellipse}
        <text
          x={ellipse.labelX ?? ellipse.x}
          y={ellipse.labelY ?? ellipse.y}
          text-anchor="middle"
          dominant-baseline="central"
          font-size={style.labelSize}
          font-weight={fontWeight}
          font-style={fontItalic}
        >
          {ellipse.label}
        </text>
      {/each}
      {#each result.squares as square}
        <text
          x={square.labelX ?? square.x}
          y={square.labelY ?? square.y}
          text-anchor="middle"
          dominant-baseline="central"
          font-size={style.labelSize}
          font-weight={fontWeight}
          font-style={fontItalic}
        >
          {square.label}
        </text>
      {/each}
      {#each result.rectangles as rect}
        <text
          x={rect.labelX ?? rect.x}
          y={rect.labelY ?? rect.y}
          text-anchor="middle"
          dominant-baseline="central"
          font-size={style.labelSize}
          font-weight={fontWeight}
          font-style={fontItalic}
        >
          {rect.label}
        </text>
      {/each}
      {#if style.showCounts && result.shapeMode === "outline"}
        {#each Object.entries(result.metrics?.fitted ?? {}) as [combo, area]}
          {#if combo.indexOf("&") < 0}
            {@const c = result.circles.find((c) => c.label === combo)}
            {@const e = result.ellipses.find((e) => e.label === combo)}
            {@const sq = result.squares.find((s) => s.label === combo)}
            {@const rc = result.rectangles.find((r) => r.label === combo)}
            {#if c}
              <text
                x={c.labelX ?? c.x}
                y={(c.labelY ?? c.y) + style.labelSize}
                text-anchor="middle"
                dominant-baseline="central"
                font-size={style.labelSize * 0.75}
                fill="#374151"
              >
                {fmt(area)}
              </text>
            {:else if e}
              <text
                x={e.labelX ?? e.x}
                y={(e.labelY ?? e.y) + style.labelSize}
                text-anchor="middle"
                dominant-baseline="central"
                font-size={style.labelSize * 0.75}
                fill="#374151"
              >
                {fmt(area)}
              </text>
            {:else if sq}
              <text
                x={sq.labelX ?? sq.x}
                y={(sq.labelY ?? sq.y) + style.labelSize}
                text-anchor="middle"
                dominant-baseline="central"
                font-size={style.labelSize * 0.75}
                fill="#374151"
              >
                {fmt(area)}
              </text>
            {:else if rc}
              <text
                x={rc.labelX ?? rc.x}
                y={(rc.labelY ?? rc.y) + style.labelSize}
                text-anchor="middle"
                dominant-baseline="central"
                font-size={style.labelSize * 0.75}
                fill="#374151"
              >
                {fmt(area)}
              </text>
            {/if}
          {/if}
        {/each}
      {/if}
    {/if}

    {#if legendBox}
      <g>
        {#each legendLabels as label, i}
          {@const yi = legendBox.y + i * legendBox.lineH}
          {@const color = setColorMap.get(label) || defaultColorFor(i, style.palette)}
          <rect
            x={legendBox.x}
            y={yi}
            width={legendBox.swatch}
            height={legendBox.swatch}
            fill={color}
            fill-opacity={style.alpha}
            stroke={showStroke ? "black" : "none"}
            stroke-width={Math.max(0.5, strokeW * 0.75)}
          />
          <text
            x={legendBox.x + legendBox.swatch + legendBox.gap}
            y={yi + legendBox.swatch / 2}
            dominant-baseline="central"
            font-size={style.labelSize}
            font-weight={fontWeight}
            font-style={fontItalic}
          >
            {label}
          </text>
        {/each}
      </g>
    {/if}
  {/if}
</svg>
