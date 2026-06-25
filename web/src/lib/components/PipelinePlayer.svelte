<script lang="ts">
  import type {
    TrajectoryFrame,
    TrajectoryShape,
  } from "@jolars/eunoia/trajectory";
  import { onMount, untrack } from "svelte";
  import { browser } from "$app/environment";
  import { defaultColorFor } from "$lib/colors";
  import { DEFAULT_FONT_FAMILY } from "$lib/fonts";

  interface Props {
    /** Set sizes keyed by combination, e.g. `{ A: 5, B: 3, "A&B": 1.5 }`. */
    sets: Record<string, number>;
    /** Default `"exclusive"`. */
    inputType?: "exclusive" | "inclusive";
    /** Initial shape primitive. Default `"circle"`. */
    shape?: "circle" | "ellipse";
    /** Initial RNG seed. The "Re-run" button advances it. Default `1`. */
    seed?: number;
    /** Caption rendered below the animation. */
    caption?: string;
    /** Milliseconds spent gliding through each optimizer step. Default `260`. */
    frameMs?: number;
  }

  let {
    sets,
    inputType = "exclusive",
    shape: initialShape = "circle",
    seed: initialSeed = 1,
    caption,
    frameMs = 260,
  }: Props = $props();

  // Lazy-loaded experimental trajectory entry point (kept out of the main bundle
  // until this component mounts — same pattern as DiagramExample's euler import).
  type TrajModule = typeof import("@jolars/eunoia/trajectory");
  let traj: TrajModule | null = $state(null);

  // `shape`/`seed` are seeded once from the props, then driven by the controls;
  // `untrack` makes that "initial value only" intent explicit (and silences the
  // state_referenced_locally hint).
  let shape = $state<"circle" | "ellipse">(untrack(() => initialShape));
  let seed = $state(untrack(() => initialSeed));
  let frames: TrajectoryFrame[] = $state([]);
  let error: string | null = $state(null);

  // Continuous playhead (a float frame index) so we can interpolate between the
  // recorded iterates and render smooth motion rather than discrete jumps.
  let playhead = $state(0);

  type Segment = "initial" | "final";
  let playing = $state(false);
  let activeSegment = $state<Segment | null>(null);
  let target = $state(0); // frame index the current run stops at

  // Brief pause on the first frame of a run so the starting configuration reads
  // before it animates.
  const START_HOLD_MS = 450;

  // Animation accumulators (not reactive — only read/written by the rAF loop and
  // the run handlers).
  let hold = 0;
  let lastT = 0;

  const STAGE_LABEL: Record<string, string> = {
    init: "Random start",
    mds: "Initial layout (MDS)",
    final: "Final optimization",
  };

  function recompute() {
    if (!traj) return;
    try {
      const result = traj.eulerTrajectory({ sets, inputType, shape, seed });
      frames = result.frames;
      playhead = 0;
      playing = false;
      activeSegment = null;
      hold = 0;
      error = frames.length === 0 ? "No frames recorded." : null;
    } catch (err) {
      error = err instanceof Error ? err.message : String(err);
      frames = [];
    }
  }

  onMount(() => {
    if (!browser) return;
    let raf = 0;

    const tick = (t: number) => {
      raf = requestAnimationFrame(tick);
      if (lastT === 0) lastT = t;
      const dt = t - lastT;
      lastT = t;
      if (!playing || frames.length < 2) return;
      if (hold > 0) {
        hold -= dt;
        return;
      }
      playhead = Math.min(playhead + dt / frameMs, target);
      if (playhead >= target) playing = false; // settle on the act's last frame
    };

    raf = requestAnimationFrame(tick);
    void (async () => {
      traj = await import("@jolars/eunoia/trajectory");
      // Compute the first trajectory once the module is in. Recompute is driven
      // imperatively from the controls (not a $effect) so it never forms a
      // read-write reactive loop.
      recompute();
    })();

    return () => cancelAnimationFrame(raf);
  });

  // Frame ranges for the two acts. The "initial layout" act is the random start
  // plus the MDS iterates; the "final layout" act is the optimization iterates
  // (whose first frame is the MDS result the optimizer refines from).
  const segments = $derived.by(() => {
    if (frames.length === 0) return null;
    let initEnd = 0;
    let finalStart = frames.length - 1;
    let seenFinal = false;
    for (let i = 0; i < frames.length; i++) {
      const st = frames[i].stage;
      if (st === "init" || st === "mds") initEnd = i;
      if (st === "final" && !seenFinal) {
        finalStart = i;
        seenFinal = true;
      }
    }
    if (!seenFinal) finalStart = initEnd;
    return { initStart: 0, initEnd, finalStart, finalEnd: frames.length - 1 };
  });

  const ready = $derived(frames.length > 0);

  // Set names in shape order (the recorder emits them consistently per frame),
  // used to assign a stable color per set. We use a vivid palette here rather
  // than the app default: the default palette's first color is white (eulerr's
  // convention, made visible elsewhere by dark region borders), but this player
  // draws colored *outlines*, so a white stroke would vanish on the light
  // `.paper` artboard. "tableau10" gives every set an always-visible color.
  const OUTLINE_PALETTE = "tableau10";
  const setNames = $derived(frames[0]?.shapes.map((s) => s.label) ?? []);
  function colorOf(label: string): string {
    const i = setNames.indexOf(label);
    return defaultColorFor(i < 0 ? 0 : i, OUTLINE_PALETTE);
  }

  const lerp = (a: number, b: number, f: number) => a + (b - a) * f;

  type DisplayShape =
    | { kind: "circle"; label: string; x: number; y: number; r: number }
    | {
        kind: "ellipse";
        label: string;
        x: number;
        y: number;
        a: number;
        b: number;
        phi: number;
      };

  // The shapes to draw at the current (fractional) playhead, linearly
  // interpolated between the two bracketing recorded frames. Within a single act
  // every frame is the same shape type, so the two brackets always match.
  const displayShapes: DisplayShape[] = $derived.by(() => {
    if (frames.length === 0) return [];
    const p = Math.max(0, Math.min(playhead, frames.length - 1));
    const i0 = Math.floor(p);
    const i1 = Math.min(i0 + 1, frames.length - 1);
    const f = p - i0;
    const a = frames[i0].shapes;
    const b = frames[i1].shapes;
    return a.map((sa: TrajectoryShape, idx: number): DisplayShape => {
      const sb = b[idx] ?? sa;
      if ("r" in sa && "r" in sb) {
        return {
          kind: "circle",
          label: sa.label,
          x: lerp(sa.x, sb.x, f),
          y: lerp(sa.y, sb.y, f),
          r: lerp(sa.r, sb.r, f),
        };
      }
      const ea = sa as Extract<TrajectoryShape, { a: number }>;
      const eb = ("a" in sb ? sb : sa) as Extract<
        TrajectoryShape,
        { a: number }
      >;
      return {
        kind: "ellipse",
        label: sa.label,
        x: lerp(ea.x, eb.x, f),
        y: lerp(ea.y, eb.y, f),
        a: lerp(ea.a, eb.a, f),
        b: lerp(ea.b, eb.b, f),
        phi: lerp(ea.phi, eb.phi, f),
      };
    });
  });

  // Shared viewBox over every frame so the view never rescales mid-animation.
  const viewBox = $derived.by(() => {
    let minX = Infinity,
      minY = Infinity,
      maxX = -Infinity,
      maxY = -Infinity;
    for (const fr of frames) {
      for (const s of fr.shapes) {
        const ext = "r" in s ? s.r : Math.max(s.a, s.b);
        if (s.x - ext < minX) minX = s.x - ext;
        if (s.y - ext < minY) minY = s.y - ext;
        if (s.x + ext > maxX) maxX = s.x + ext;
        if (s.y + ext > maxY) maxY = s.y + ext;
      }
    }
    if (!isFinite(minX)) return { x: 0, y: 0, w: 100, h: 100 };
    const w = maxX - minX || 1;
    const h = maxY - minY || 1;
    const pad = Math.max(w, h) * 0.08;
    return { x: minX - pad, y: minY - pad, w: w + 2 * pad, h: h + 2 * pad };
  });

  const viewBoxAttr = $derived(
    `${viewBox.x} ${viewBox.y} ${viewBox.w} ${viewBox.h}`,
  );
  const aspectRatio = $derived(viewBox.w / viewBox.h || 1);
  const unit = $derived(Math.max(viewBox.w, viewBox.h));
  const strokeW = $derived(unit * 0.006);
  const fontSize = $derived(unit * 0.05);

  const statusFrame = $derived(
    frames.length > 0
      ? frames[Math.min(Math.round(playhead), frames.length - 1)]
      : null,
  );
  const statusText = $derived.by(() => {
    if (!statusFrame) return "";
    const label = STAGE_LABEL[statusFrame.stage] ?? statusFrame.stage;
    return statusFrame.stage === "init"
      ? label
      : `${label} · step ${statusFrame.iteration}`;
  });

  // The final region-area loss at the current step. The recorder scores every
  // frame — random start, MDS, and final — by this same criterion, so the number
  // falls consistently all the way through the pipeline.
  const metricText = $derived.by(() => {
    if (!statusFrame) return "";
    const v = statusFrame.cost;
    return `loss ${v === 0 ? "0" : v.toExponential(2)}`;
  });

  // Progress within the currently active (or last-run) act.
  const progress = $derived.by(() => {
    if (!segments) return 0;
    const [start, end] =
      activeSegment === "final"
        ? [segments.finalStart, segments.finalEnd]
        : [segments.initStart, segments.initEnd];
    if (end <= start) return playhead >= end ? 1 : 0;
    return Math.min(1, Math.max(0, (playhead - start) / (end - start)));
  });

  function runSegment(seg: Segment) {
    if (!segments || playing) return;
    if (seg === "initial") {
      playhead = segments.initStart;
      target = segments.initEnd;
    } else {
      playhead = segments.finalStart;
      target = segments.finalEnd;
    }
    activeSegment = seg;
    hold = START_HOLD_MS;
    playing = true;
  }

  function rerun() {
    if (playing) return;
    seed = (seed + 1) % 1_000_000;
    recompute();
  }

  function setShape(s: "circle" | "ellipse") {
    if (playing || s === shape) return;
    shape = s;
    recompute();
  }
</script>

<!-- `paper`: keep the artboard light in dark mode so the dark labels/strokes
     stay readable (see app.css). -->
<figure
  class="paper not-prose my-6 rounded-lg border border-line bg-surface p-4"
>
  <div class="relative">
    {#if error}
      <div
        class="aspect-square w-full flex items-center justify-center text-sm text-red-600"
      >
        {error}
      </div>
    {:else if ready}
      <svg
        viewBox={viewBoxAttr}
        class="w-full"
        style="aspect-ratio: {aspectRatio}; max-height: 380px;"
        preserveAspectRatio="xMidYMid meet"
        font-family={DEFAULT_FONT_FAMILY}
        xmlns="http://www.w3.org/2000/svg"
      >
        {#each displayShapes as s (s.label)}
          {@const color = colorOf(s.label)}
          {#if s.kind === "circle"}
            <circle
              cx={s.x}
              cy={s.y}
              r={s.r}
              fill={color}
              fill-opacity="0.16"
              stroke={color}
              stroke-width={strokeW}
            />
          {:else}
            <ellipse
              cx={s.x}
              cy={s.y}
              rx={s.a}
              ry={s.b}
              transform="rotate({(s.phi * 180) / Math.PI} {s.x} {s.y})"
              fill={color}
              fill-opacity="0.16"
              stroke={color}
              stroke-width={strokeW}
            />
          {/if}
          <text
            x={s.x}
            y={s.y}
            text-anchor="middle"
            dominant-baseline="central"
            font-size={fontSize}
            fill="#111827">{s.label}</text
          >
        {/each}
      </svg>
    {:else}
      <div
        class="aspect-square w-full flex items-center justify-center text-sm text-muted"
      >
        Loading…
      </div>
    {/if}
  </div>

  <!-- Run controls: two acts, each glides to its end and pauses there. -->
  <div class="mt-3 flex flex-wrap items-center gap-2 text-sm">
    <button
      type="button"
      onclick={() => runSegment("initial")}
      disabled={playing}
      class="rounded px-3 py-1 font-medium text-white bg-accent hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed"
      >▶ Initial layout</button
    >
    <button
      type="button"
      onclick={() => runSegment("final")}
      disabled={playing}
      class="rounded px-3 py-1 font-medium text-white bg-accent hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed"
      >▶ Final layout</button
    >
    <button
      type="button"
      onclick={rerun}
      disabled={playing}
      class="rounded border border-line bg-inset px-3 py-1 font-medium text-ink hover:bg-accent-soft disabled:opacity-50 disabled:cursor-not-allowed"
      >Re-run</button
    >

    <div class="ml-1 inline-flex overflow-hidden rounded border border-line">
      <button
        type="button"
        onclick={() => setShape("circle")}
        disabled={playing}
        class="px-3 py-1 disabled:cursor-not-allowed {shape === 'circle'
          ? 'bg-accent text-white'
          : 'bg-inset text-ink hover:bg-accent-soft'}">Circles</button
      >
      <button
        type="button"
        onclick={() => setShape("ellipse")}
        disabled={playing}
        class="border-l border-line px-3 py-1 disabled:cursor-not-allowed {shape ===
        'ellipse'
          ? 'bg-accent text-white'
          : 'bg-inset text-ink hover:bg-accent-soft'}">Ellipses</button
      >
    </div>

    <span class="ml-auto flex items-center gap-2">
      <span class="font-medium text-ink">{statusText}</span>
      {#if metricText}
        <span
          class="rounded bg-inset px-2 py-0.5 font-mono text-xs tabular-nums text-muted"
          >{metricText}</span
        >
      {/if}
    </span>
  </div>

  <!-- Progress through the current act -->
  <div class="mt-2 h-1.5 w-full overflow-hidden rounded bg-inset">
    <div
      class="h-full bg-accent transition-[width] duration-75"
      style="width: {progress * 100}%"
    ></div>
  </div>

  {#if caption}
    <figcaption class="mt-3 text-sm text-muted text-center">
      {caption}
    </figcaption>
  {/if}
</figure>
