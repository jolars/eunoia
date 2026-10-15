<script lang="ts">
  import type {
    GlyphBoxPlacements,
    GlyphPlacements,
    LabelPlacement,
    LabelSize,
    Region,
    Rect,
    SetOutline,
  } from "@jolars/eunoia";
  import {
    labelObstacles,
    placeGlyphsForRegions,
    placeLabelsForRegions,
    placeSetLabels,
  } from "@jolars/eunoia";
  import {
    nestedSets,
    regionTitleLines,
    svgBody,
    type ToSvgOptions,
    viewBox,
  } from "@jolars/eunoia/svg";
  import { glyphStatus, memberListsForRegions } from "../members.svelte";
  import { placeMemberLabels, type MeasuredMemberLine } from "../member-layout";
  import { appState } from "../state.svelte";
  import type { DiagramStyle, FitResult } from "../types/diagram";
  import { vennAnnotationsForRegions, vennCombinations } from "../venn";

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
  // Set names drawn outside their shapes rather than inside the diagram. The
  // serializer drops the interior copies when `setLabelPlacements` is set, so
  // the region label boxes must be measured without their title lines too —
  // otherwise the region placer sizes each box for text that is not there.
  let setLabelsOutside = $derived(style.setLabelMode === "outside");
  // Per-set silhouettes for the set-label placer. Region output carries them
  // as `shapeOutlines` (decomposition clips them away, so they ride along
  // separately); polygon output — Advanced → Show regions off — *is* a list of
  // set outlines already. Both satisfy `SetOutline`, so the placer takes
  // either and the mode stays available with regions turned off.
  let setOutlines: SetOutline[] = $derived.by(() => {
    if (!result) return [];
    if (result.layout.mode === "regions") return result.layout.shapeOutlines;
    if (result.layout.mode === "polygons") return result.layout.polygons;
    return [];
  });
  let nested = $derived(
    result && result.layout.mode === "regions" ? nestedSets(result.layout) : {},
  );

  function fmt(v: number): string {
    if (Math.abs(v) >= 100) return v.toFixed(0);
    if (Math.abs(v) >= 10) return v.toFixed(1);
    return v.toFixed(2);
  }

  // Venn geometry is topological, so its region areas are not quantities: the
  // numbers and rosters come from the per-region annotations instead, matched
  // against whatever is being drawn. An Euler fit has none of this — its areas
  // *are* the quantities, so both stay undefined and the renderer uses the
  // layout's own numbers and the spec rows.
  //
  // With region output that is the fitted region list; with polygon output
  // (Advanced → Show regions off) only the set outlines exist, and the
  // serializer draws per-set counts, so the current n's keys stand in.
  let vennAnnotations = $derived.by(() => {
    if (appState.diagramType !== "venn" || !result) return undefined;
    const combinations =
      result.layout.mode === "regions"
        ? [
            ...vennCombinations(appState.vennN),
            ...result.layout.regions.map((r) => r.combination),
          ]
        : vennCombinations(appState.vennN);
    return vennAnnotationsForRegions(appState.vennRegions, combinations);
  });

  // Authoritative when present: a Venn region with no quantity entered draws no
  // count at all (see `ToSvgOptions.counts`).
  let regionCounts: Record<string, number> | undefined = $derived(
    vennAnnotations?.counts,
  );

  function countFor(region: Region): number | undefined {
    if (!regionCounts) return region.totalArea;
    return regionCounts[region.combination];
  }

  // Per-region label-fit map. Measure each region's combined label via hidden
  // `<text>` + `getBBox()` (actual rendered dimensions, not a char-width
  // heuristic); the measured sizes feed both `placeLabelsForRegions` (does the
  // box inscribe inside the region polygon?) and the serializer's label-aware
  // viewBox. Keyed by `region.combination` off the `data-fit-region` nodes
  // rendered in the hidden `<g>` below.
  let measureContainer: SVGGElement | null = $state(null);
  let measuredSizes: Record<string, LabelSize> = $state({});
  let measuredSetSizes: Record<string, LabelSize> = $state({});
  let measuredHeaderLines: Record<string, MeasuredMemberLine[]> = $state({});
  let measuredRegionLines: Record<string, Rect[]> = $state({});

  $effect(() => {
    // Re-measure when result changes, font scales, or showCounts toggles.
    // Reading these explicitly so Svelte tracks them as dependencies.
    void result;
    void style.labelSize;
    void style.fontBold;
    void style.fontItalic;
    void style.fontFamily;
    void style.showCounts;
    void regionCounts;
    void nested;
    void fontsReady;
    void setLabelsOutside;
    // Region label boxes only exist in region mode, but set names are
    // measured in either — so bail only when neither pass has anything to do.
    if (!measureContainer || (!isRegion && !setLabelsOutside)) {
      measuredSizes = {};
      measuredSetSizes = {};
      return;
    }
    const sizes: Record<string, LabelSize> = {};
    const lineBoxes: Record<string, Rect[]> = {};
    const context = document.createElement("canvas").getContext("2d");
    const headerLines: Record<string, MeasuredMemberLine[]> = {};
    const nodes = measureContainer.querySelectorAll<SVGGraphicsElement>(
      "text[data-fit-region]",
    );
    for (const t of Array.from(nodes)) {
      const combo = t.getAttribute("data-fit-region");
      if (combo === null) continue;
      const bb = t.getBBox();
      // SVG boxes include font ascent and descent even when a line has no
      // descenders. Use the visible ink for glyph clearance instead.
      let ink = { x: bb.x, y: bb.y, width: bb.width, height: bb.height };
      if (context) {
        const font = getComputedStyle(t);
        context.font = `${font.fontStyle} ${font.fontWeight} ${font.fontSize} ${font.fontFamily}`;
        const metrics = context.measureText(t.textContent?.trim() ?? "");
        const height =
          metrics.actualBoundingBoxAscent + metrics.actualBoundingBoxDescent;
        if (Number.isFinite(metrics.fontBoundingBoxAscent) && height > 0) {
          ink = {
            x: -metrics.width / 2 - metrics.actualBoundingBoxLeft,
            y:
              bb.y +
              metrics.fontBoundingBoxAscent -
              metrics.actualBoundingBoxAscent,
            width:
              metrics.actualBoundingBoxLeft + metrics.actualBoundingBoxRight,
            height,
          };
        }
      }
      (lineBoxes[combo] ??= []).push({
        x: ink.x + ink.width / 2,
        y: ink.y + ink.height / 2,
        width: ink.width,
        height: ink.height,
      });
      (headerLines[combo] ??= []).push({
        text: t.textContent?.trim() ?? "",
        w: bb.width,
        h: bb.height,
        fontSize: Number(t.getAttribute("font-size")),
        fontWeight: t.getAttribute("font-weight") ?? 400,
        fontStyle: t.getAttribute("font-style") ?? "normal",
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
    measuredRegionLines = lineBoxes;
    measuredHeaderLines = headerLines;

    const setBoxes: Record<string, LabelSize> = {};
    const setNodes =
      measureContainer.querySelectorAll<SVGGraphicsElement>(
        "text[data-fit-set]",
      );
    for (const t of Array.from(setNodes)) {
      const name = t.getAttribute("data-fit-set");
      if (name === null) continue;
      const bb = t.getBBox();
      setBoxes[name] = { w: bb.width, h: bb.height };
    }
    measuredSetSizes = setBoxes;
  });

  // Per-region placement from the eunoia core (wasm). Defaults to
  // `Strict + Raycast`: each region's label sits at its POI when the box fits
  // inside the polygon, otherwise the anchor is raycast outside the diagram
  // bbox (or container, when complement is set) with a leader back to the POI.
  // Empty until the first DOM measurement runs; the serializer then falls back
  // to each region's own POI for unplaced regions.
  let baseRegionPlacements: Record<string, LabelPlacement> = $derived.by(() => {
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

  // Region-label keep-outs, also handed to the set-label placer so a set name
  // never lands on a region's quantity.
  let regionLabelObstacles = $derived(
    labelObstacles({
      placements: baseRegionPlacements,
      sizes: measuredSizes,
      padding: style.labelSize * 0.15,
    }),
  );

  // Exterior set names: each hugs its own shape's outline at the angle with
  // the most free space, no leader line.
  let baseSetLabelPlacements: Record<string, LabelPlacement> | undefined =
    $derived.by(() => {
      if (!setLabelsOutside || !result || setOutlines.length === 0) {
        return undefined;
      }
      if (Object.keys(measuredSetSizes).length === 0) return undefined;
      try {
        return placeSetLabels({
          outlines: setOutlines,
          container: result.layout.container,
          sizes: measuredSetSizes,
          strategy: {
            obstacles: regionLabelObstacles,
            precision: Math.max(0.05, style.labelSize * 0.05),
          },
        });
      } catch (err) {
        console.warn("[place sets] failed, falling back to interior:", err);
        return undefined;
      }
    });

  // Keep-out boxes for both packers: labels are drawn over the marks, so the
  // marks steer clear of the boxes we just measured for them. Set names are
  // outside their own shape but can still overhang a neighbouring region, so
  // they join the list.
  let glyphObstacles = $derived([
    ...Object.entries(measuredRegionLines).flatMap(([key, lines]) => {
      const anchor = baseRegionPlacements[key]?.anchor;
      if (!anchor) return [];
      const padding = style.labelSize * 0.1;
      return lines.map((line) => ({
        x: anchor.x + line.x,
        y: anchor.y + line.y,
        width: line.width + 2 * padding,
        height: line.height + 2 * padding,
      }));
    }),
    ...(baseSetLabelPlacements
      ? labelObstacles({
          placements: baseSetLabelPlacements,
          sizes: measuredSetSizes,
          padding: style.labelSize * 0.15,
        })
      : []),
  ]);

  // Interactive glyph budget: deriving counts from the spec quantities means a
  // user typing large numbers (say population-scale frequencies) would ask the
  // packer for that many dots. Beyond this the diagram is unreadable anyway,
  // so skip rendering instead of stalling the UI.
  const MAX_GLYPHS = 2000;

  // eulerGlyphs-style unit glyphs: counts are the spec's exclusive region
  // quantities rounded to integers (plus the complement inside the container,
  // when fitted), packed by the eunoia core with a shared auto-sized radius.
  let glyphPlacements: GlyphPlacements | undefined = $derived.by(() => {
    if (
      style.glyphMode !== "dots" ||
      !result ||
      result.layout.mode !== "regions"
    )
      return undefined;
    const counts: Record<string, number> = {};
    let total = 0;
    // A Venn's target areas are all 1.0 (the canonical spec is synthetic), so
    // the entered quantities are the only meaningful counts there.
    for (const [combo, quantity] of Object.entries(
      regionCounts ?? result.metrics.target,
    )) {
      const n = Math.round(quantity);
      if (n > 0) {
        counts[combo] = n;
        total += n;
      }
    }
    if (result.complement !== undefined) {
      const n = Math.round(result.complement);
      if (n > 0) {
        counts[""] = n;
        total += n;
      }
    }
    if (total === 0) return undefined;
    if (total > MAX_GLYPHS) {
      console.warn(
        `[glyphs] skipped: ${total} glyphs exceed the interactive budget of ${MAX_GLYPHS}`,
      );
      return undefined;
    }
    try {
      return placeGlyphsForRegions({
        regions: result.layout.regions,
        counts,
        options: {
          arrangement: style.glyphArrangement,
          gap: style.glyphGap,
          boundaryGap: style.glyphBoundaryGap,
          seed: style.glyphSeed,
          obstacles: glyphObstacles,
        },
      });
    } catch (err) {
      console.warn("[glyphs] placement failed:", err);
      return undefined;
    }
  });

  // Member names per region, keyed by the region's canonical combination. Rows
  // are matched by set membership, not by their raw text, since the core
  // canonicalizes what `buildSets` passes through.
  let memberLabels: Record<string, string[]> = $derived.by(() => {
    if (style.glyphMode !== "members") return {};
    if (!result || result.layout.mode !== "regions") return {};
    // Venn is driven by `vennN`, not by the rows, so it carries its own
    // per-region rosters — Euler row names must not leak into it.
    if (vennAnnotations) return vennAnnotations.members;
    return memberListsForRegions(appState.rows, result.layout.regions);
  });

  // Measuring costs a `<text>` node and a `getBBox()` each, and the packer is
  // O(n) rows per region — the same argument as MAX_GLYPHS, at the scale text
  // stops being legible.
  const MAX_MEMBER_LABELS = 500;

  let memberBudgetExceeded = $derived(
    Object.values(memberLabels).reduce((n, l) => n + l.length, 0) >
      MAX_MEMBER_LABELS,
  );

  // Second measurement pass, mirroring the label one: hidden `<text>` per member
  // name, read back with getBBox() in the packer's input order. An incomplete
  // prefix switches the entire roster to a callout rather than omitting names.
  let memberSizes: Record<string, LabelSize[]> = $state({});
  let missingHeaders: Record<string, MeasuredMemberLine> = $state({});

  $effect(() => {
    void result;
    void memberLabels;
    void memberBudgetExceeded;
    void style.memberLabelSize;
    void style.fontFamily;
    void fontsReady;
    if (!measureContainer || !isRegion || memberBudgetExceeded) {
      memberSizes = {};
      return;
    }
    const sizes: Record<string, LabelSize[]> = {};
    const nodes = measureContainer.querySelectorAll<SVGGraphicsElement>(
      "text[data-fit-member]",
    );
    for (const t of Array.from(nodes)) {
      const combo = t.getAttribute("data-fit-member");
      if (combo === null) continue;
      const bb = t.getBBox();
      const cur = sizes[combo] ?? [];
      cur.push({ w: bb.width, h: bb.height });
      sizes[combo] = cur;
    }
    memberSizes = sizes;
    const headings: Record<string, MeasuredMemberLine> = {};
    for (const t of Array.from(
      measureContainer.querySelectorAll<SVGGraphicsElement>(
        "text[data-fit-missing]",
      ),
    )) {
      const combo = t.getAttribute("data-fit-missing");
      if (combo === null) continue;
      const bb = t.getBBox();
      headings[combo] = {
        text: t.textContent?.trim() ?? "",
        w: bb.width,
        h: bb.height,
        fontSize: style.memberLabelSize,
        fontWeight: 400,
      };
    }
    missingHeaders = headings;
  });

  // Member text boxes, packed at a single diagram-wide scale. Shrink-only, so
  // `style.memberLabelSize` is the ceiling and the rendered size is
  // `memberLabelSize * placements.scale`.
  let memberLayout = $derived.by(() => {
    if (style.glyphMode !== "members" || !result) return undefined;
    if (result.layout.mode !== "regions") return undefined;
    if (memberBudgetExceeded) return undefined;
    if (Object.keys(memberSizes).length === 0) return undefined;
    if (
      Object.entries(memberLabels).some(
        ([key, names]) =>
          memberSizes[key]?.length !== names.length || !missingHeaders[key],
      )
    )
      return undefined;
    try {
      return placeMemberLabels({
        layout: result.layout,
        labels: memberLabels,
        sizes: memberSizes,
        headerSizes: measuredSizes,
        headerLines: measuredHeaderLines,
        missingHeaders,
        fontSize: style.memberLabelSize,
        boxOptions: {
          arrangement: style.glyphArrangement,
          gap: style.glyphGap,
          seed: style.glyphSeed,
        },
        strategy: {
          leader:
            style.labelPlacement === "elbow"
              ? { type: "elbow" }
              : { type: "straight", placement: style.labelPlacement },
          precision: Math.max(0.05, style.labelSize * 0.05),
          tether: style.labelTether,
          leaderGap: style.labelSize * 0.25,
        },
        setSizes: setLabelsOutside ? measuredSetSizes : undefined,
        padding: style.labelSize * 0.15,
      });
    } catch (err) {
      console.warn("[glyph boxes] placement failed:", err);
      return undefined;
    }
  });

  let glyphBoxPlacements: GlyphBoxPlacements | undefined = $derived(
    memberLayout?.boxes,
  );
  let regionPlacements = $derived(
    memberLayout?.placements ?? baseRegionPlacements,
  );
  let setLabelPlacements = $derived(
    memberLayout?.setPlacements ?? baseSetLabelPlacements,
  );

  // Publish callout regions and any skipped rendering for the glyph controls.
  $effect(() => {
    if (style.glyphMode !== "members") {
      glyphStatus.clear();
      return;
    }
    if (memberBudgetExceeded) {
      const total = Object.values(memberLabels).reduce(
        (n, l) => n + l.length,
        0,
      );
      console.warn(
        `[glyph boxes] skipped: ${total} names exceed the interactive budget of ${MAX_MEMBER_LABELS}`,
      );
      glyphStatus.unplaced = {};
      glyphStatus.callouts = [];
      glyphStatus.skipped = `${total} names exceed the interactive budget of ${MAX_MEMBER_LABELS}.`;
      return;
    }
    glyphStatus.skipped = "";
    glyphStatus.unplaced = { ...(glyphBoxPlacements?.unplaced ?? {}) };
    glyphStatus.callouts = Object.keys(memberLayout?.callouts ?? {});
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
    counts: regionCounts,
    legend: { show: style.showLegend, position: style.legendPosition },
    padding: PADDING,
    placements: regionPlacements,
    labelSizes: measuredSizes,
    setLabelPlacements,
    setLabelSizes: measuredSetSizes,
    complement: result?.complement,
    glyphs: glyphPlacements,
    // The placer is font-blind, so the strings ride separately; they are
    // index-aligned with `boxes` because the packer returns a prefix of what we
    // measured. `renderGlyphBoxes` applies `fontSize * scale` itself, and the
    // family is inherited from the root `<svg>`.
    glyphBoxes: glyphBoxPlacements && {
      ...glyphBoxPlacements,
      labels: memberLabels,
      fontSize: style.memberLabelSize,
      // Regular weight regardless of the Bold toggle, which belongs to the set
      // names — and which would otherwise render member text heavier than the
      // 400-weight nodes we measured, overflowing every box.
      fontWeight: 400,
    },
    memberCallouts: memberLayout?.callouts,
  });

  let vb = $derived(
    result
      ? viewBox(result.layout, svgOptions)
      : { x: 0, y: 0, w: 100, h: 100 },
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
  {#if result && (isRegion || setLabelsOutside)}
    <g
      bind:this={measureContainer}
      visibility="hidden"
      aria-hidden="true"
      data-fit-measure
    >
      {#each regions as region}
        {#if !setLabelsOutside}
          {#each regionTitleLines(region.combination, nested) as title, i}
            <text
              data-fit-region={region.combination}
              y={i * style.labelSize}
              text-anchor="middle"
              dominant-baseline="central"
              font-size={style.labelSize}
              font-weight={fontWeight}
              font-style={fontItalic}
            >
              {title}
            </text>
          {/each}
        {/if}
        {#if style.showCounts}
          {@const count = countFor(region)}
          {#if count !== undefined}
            <text
              data-fit-region={region.combination}
              y={setLabelsOutside
                ? 0
                : regionTitleLines(region.combination, nested).length *
                  style.labelSize}
              text-anchor="middle"
              dominant-baseline="central"
              font-size={style.labelSize * 0.75}
            >
              {fmt(count)}
            </text>
          {/if}
        {/if}
      {/each}
      {#if !memberBudgetExceeded}
        {#each Object.entries(memberLabels) as [combination, members]}
          <text
            data-fit-missing={combination}
            font-size={style.memberLabelSize}
          >
            {combination || "Outside"} (no geometry)
          </text>
          {#each members as name}
            <text
              data-fit-member={combination}
              font-size={style.memberLabelSize}
            >
              {name}
            </text>
          {/each}
        {/each}
      {/if}
      {#if setLabelsOutside}
        {#each setOutlines as outline}
          <text
            data-fit-set={outline.label}
            font-size={style.labelSize}
            font-weight={fontWeight}
            font-style={fontItalic}
          >
            {outline.label}
          </text>
        {/each}
      {/if}
    </g>
  {/if}
  {#if result}
    <!-- eslint-disable-next-line svelte/no-at-html-tags -->
    {@html body}
  {/if}
</svg>
