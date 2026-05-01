<script lang="ts">
  import type {
    DiagramStyle,
    FitResult,
    Polygon,
  } from "../../types/diagram";
  import { defaultColorFor } from "../colors";

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
        for (const p of r.polygons) {
          for (const v of p.vertices) consume(v);
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
    }

    if (!isFinite(minX)) {
      return { minX: 0, minY: 0, maxX: 100, maxY: 100 };
    }
    return { minX, minY, maxX, maxY };
  });

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
    return result.polygons.map((p) => p.label);
  });

  let setColorMap: Map<string, string> = $derived.by(() => {
    const m = new Map<string, string>();
    setLabels.forEach((l, i) =>
      m.set(l, style.colors[l] ?? defaultColorFor(i)),
    );
    return m;
  });

  function regionFill(combination: string): string {
    const sets = combination
      .split("&")
      .map((s) => s.trim())
      .filter(Boolean);
    const fallback = defaultColorFor(0);
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

  let fontWeight = $derived(style.fontBold ? 700 : 400);
  let fontItalic = $derived(style.fontItalic ? "italic" : "normal");

  let viewBox = $derived.by(() => {
    let { minX, minY, maxX, maxY } = bounds;
    let w = maxX - minX;
    let h = maxY - minY;
    let lx = minX - PADDING;
    let ly = minY - PADDING;
    let lw = w + 2 * PADDING;
    let lh = h + 2 * PADDING;

    // Reserve space for the legend.
    if (result && style.showLegend && setLabels.length > 0) {
      const legendW = Math.max(20, style.labelSize * 2);
      const legendH =
        Math.max(8, style.labelSize * 1.4) * setLabels.length + 8;
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

  function calcLabelPos(p: Polygon): { x: number; y: number } {
    if (p.labelPosition) return p.labelPosition;
    let cx = 0,
      cy = 0;
    for (const v of p.vertices) {
      cx += v.x;
      cy += v.y;
    }
    return { x: cx / p.vertices.length, y: cy / p.vertices.length };
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
    if (!result || !style.showLegend || setLabels.length === 0) {
      return null;
    }
    const swatch = style.labelSize * 0.9;
    const gap = swatch * 0.4;
    const lineH = swatch + gap;
    const totalH = lineH * setLabels.length;
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
  {#if result}
    {#if result.shapeMode === "region"}
      {#each result.regions as region}
        {@const fill = regionFill(region.combination)}
        {#each region.polygons as poly}
          <path
            d={polygonPath(poly)}
            fill={fill}
            fill-opacity={style.alpha}
            stroke={showStroke ? "black" : "none"}
            stroke-width={strokeW}
          />
        {/each}
      {/each}
      {#each result.regions as region}
        {#each region.polygons as poly}
          {@const lp = calcLabelPos(poly)}
          <text
            x={lp.x}
            y={lp.y}
            text-anchor="middle"
            dominant-baseline="middle"
            font-size={style.labelSize}
            font-weight={fontWeight}
            font-style={fontItalic}
            fill="black"
          >
            {region.combination}
          </text>
          {#if style.showCounts}
            <text
              x={lp.x}
              y={lp.y + style.labelSize}
              text-anchor="middle"
              dominant-baseline="middle"
              font-size={style.labelSize * 0.75}
              fill="#374151"
            >
              {fmt(region.totalArea)}
            </text>
          {/if}
        {/each}
      {/each}
    {:else}
      {#each result.polygons as poly, i}
        {@const color = setColorMap.get(poly.label) || defaultColorFor(i)}
        <path
          d={polygonPath(poly)}
          fill={color}
          fill-opacity={style.alpha}
          stroke={showStroke ? color : "none"}
          stroke-width={strokeW}
        />
      {/each}
      {#each result.circles as circle, i}
        {@const color = setColorMap.get(circle.label) || defaultColorFor(i)}
        <circle
          cx={circle.x}
          cy={circle.y}
          r={circle.radius}
          fill={color}
          fill-opacity={style.alpha}
          stroke={showStroke ? color : "none"}
          stroke-width={strokeW}
        />
      {/each}
      {#each result.ellipses as ellipse, i}
        {@const color = setColorMap.get(ellipse.label) || defaultColorFor(i)}
        <ellipse
          cx={ellipse.x}
          cy={ellipse.y}
          rx={ellipse.semi_major}
          ry={ellipse.semi_minor}
          transform={`rotate(${(ellipse.rotation * 180) / Math.PI} ${ellipse.x} ${ellipse.y})`}
          fill={color}
          fill-opacity={style.alpha}
          stroke={showStroke ? color : "none"}
          stroke-width={strokeW}
        />
      {/each}
      {#each result.squares as square, i}
        {@const color = setColorMap.get(square.label) || defaultColorFor(i)}
        <rect
          x={square.x - square.side / 2}
          y={square.y - square.side / 2}
          width={square.side}
          height={square.side}
          fill={color}
          fill-opacity={style.alpha}
          stroke={showStroke ? color : "none"}
          stroke-width={strokeW}
        />
      {/each}
      {#each result.polygons as poly}
        {@const lp = calcLabelPos(poly)}
        <text
          x={lp.x}
          y={lp.y}
          text-anchor="middle"
          dominant-baseline="middle"
          font-size={style.labelSize}
          font-weight={fontWeight}
          font-style={fontItalic}
        >
          {poly.label}
        </text>
      {/each}
      {#each result.circles as circle}
        <text
          x={circle.x}
          y={circle.y}
          text-anchor="middle"
          dominant-baseline="middle"
          font-size={style.labelSize}
          font-weight={fontWeight}
          font-style={fontItalic}
        >
          {circle.label}
        </text>
      {/each}
      {#each result.ellipses as ellipse}
        <text
          x={ellipse.x}
          y={ellipse.y}
          text-anchor="middle"
          dominant-baseline="middle"
          font-size={style.labelSize}
          font-weight={fontWeight}
          font-style={fontItalic}
        >
          {ellipse.label}
        </text>
      {/each}
      {#each result.squares as square}
        <text
          x={square.x}
          y={square.y}
          text-anchor="middle"
          dominant-baseline="middle"
          font-size={style.labelSize}
          font-weight={fontWeight}
          font-style={fontItalic}
        >
          {square.label}
        </text>
      {/each}
      {#if style.showCounts && result.shapeMode === "outline"}
        {#each Object.entries(result.metrics?.fitted ?? {}) as [combo, area]}
          {#if combo.indexOf("&") < 0}
            {@const c = result.circles.find((c) => c.label === combo)}
            {@const e = result.ellipses.find((e) => e.label === combo)}
            {@const sq = result.squares.find((s) => s.label === combo)}
            {#if c}
              <text
                x={c.x}
                y={c.y + style.labelSize}
                text-anchor="middle"
                dominant-baseline="middle"
                font-size={style.labelSize * 0.75}
                fill="#374151"
              >
                {fmt(area)}
              </text>
            {:else if e}
              <text
                x={e.x}
                y={e.y + style.labelSize}
                text-anchor="middle"
                dominant-baseline="middle"
                font-size={style.labelSize * 0.75}
                fill="#374151"
              >
                {fmt(area)}
              </text>
            {:else if sq}
              <text
                x={sq.x}
                y={sq.y + style.labelSize}
                text-anchor="middle"
                dominant-baseline="middle"
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
        {#each setLabels as label, i}
          {@const yi = legendBox.y + i * legendBox.lineH}
          {@const color = setColorMap.get(label) || defaultColorFor(i)}
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
            dominant-baseline="middle"
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
