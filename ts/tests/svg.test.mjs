// Tests for the pure SVG serializer (`@jolars/eunoia/svg`).
//
// Run with `node --test` (zero deps). Imports the *compiled* output in
// `../../npm/svg.js`, which is pure and wasm-free, so it loads in bare node —
// run `task build-wasm` (or `node ts/prepare-package.mjs`) first.

import assert from "node:assert/strict";
import { test } from "node:test";

import {
  boundingBox,
  defaultColorFor,
  leaderPath,
  mixColors,
  nestedSets,
  polygonPath,
  regionPath,
  regionTitleLines,
  svgBody,
  toSvg,
  viewBox,
} from "../../npm/svg.js";

const metrics = {
  loss: 0,
  stress: 0,
  diagError: 0,
  iterations: 0,
  targetAreas: {},
  fittedAreas: { A: 5 },
  regionError: {},
  residuals: {},
};

function circleLayout(label = "A") {
  return {
    mode: "shapes",
    shape: "circle",
    circles: [{ label, x: 0, y: 0, radius: 10, labelAnchor: { x: 0, y: 0 } }],
    metrics,
  };
}

function regionLayout() {
  return {
    mode: "regions",
    shape: "circle",
    regions: [
      {
        combination: "A",
        totalArea: 5,
        labelAnchor: { x: 5, y: 5 },
        pieces: [
          {
            outer: {
              label: "",
              area: 100,
              vertices: [
                { x: 0, y: 0 },
                { x: 10, y: 0 },
                { x: 10, y: 10 },
                { x: 0, y: 10 },
              ],
            },
            holes: [],
            area: 100,
          },
        ],
      },
    ],
    setAnchors: { A: { x: 5, y: 5 } },
    metrics,
  };
}

test("mixColors averages in sRGB", () => {
  assert.equal(mixColors(["#000000", "#ffffff"]), "rgb(128,128,128)");
  assert.equal(mixColors(["#ff0000", "#0000ff"]), "rgb(128,0,128)");
  // shorthand hex + rgb() form
  assert.equal(mixColors(["#000", "rgb(255,255,255)"]), "rgb(128,128,128)");
  // unparseable → first color back
  assert.equal(mixColors(["not-a-color"]), "not-a-color");
});

test("defaultColorFor wraps and reads the default palette", () => {
  assert.equal(defaultColorFor(0), "#ffffff");
  assert.equal(defaultColorFor(12), "#ffffff"); // wraps (12-color palette)
});

test("polygonPath closes the ring", () => {
  const d = polygonPath({
    vertices: [
      { x: 0, y: 0 },
      { x: 10, y: 0 },
      { x: 10, y: 10 },
    ],
  });
  assert.equal(d, "M 0,0 L 10,0 L 10,10 Z");
  assert.equal(polygonPath({ vertices: [] }), "");
});

test("regionPath concatenates outer + holes", () => {
  const d = regionPath({
    outer: {
      vertices: [
        { x: 0, y: 0 },
        { x: 4, y: 0 },
        { x: 4, y: 4 },
      ],
    },
    holes: [
      {
        vertices: [
          { x: 1, y: 1 },
          { x: 2, y: 1 },
          { x: 2, y: 2 },
        ],
      },
    ],
  });
  assert.equal(d, "M 0,0 L 4,0 L 4,4 Z M 1,1 L 2,1 L 2,2 Z");
});

test("leaderPath threads waypoints", () => {
  assert.equal(leaderPath({ x: 0, y: 0 }, { x: 5, y: 5 }), "M 0,0 L 5,5");
  assert.equal(
    leaderPath({ x: 0, y: 0 }, { x: 5, y: 5 }, [{ x: 2, y: 0 }]),
    "M 0,0 L 2,0 L 5,5",
  );
});

test("boundingBox covers a circle's extent", () => {
  assert.deepEqual(boundingBox(circleLayout()), {
    minX: -10,
    minY: -10,
    maxX: 10,
    maxY: 10,
  });
});

test("viewBox applies padding", () => {
  assert.deepEqual(viewBox(circleLayout(), { padding: 5 }), {
    x: -15,
    y: -15,
    w: 30,
    h: 30,
  });
});

test("toSvg renders a circle with palette fill and label", () => {
  const svg = toSvg(circleLayout(), { padding: 5 });
  assert.match(svg, /^<svg /);
  assert.match(svg, /viewBox="-15 -15 30 30"/);
  assert.match(svg, /<circle /);
  assert.match(svg, /r="10"/);
  assert.match(svg, /fill="#ffffff"/);
  assert.match(svg, />A<\/text>/);
});

test("toSvg renders a region as a path with a label", () => {
  const svg = svgBody(regionLayout(), { showCounts: true });
  assert.match(svg, /<path d="M 0,0 L 10,0 L 10,10 L 0,10 Z"/);
  assert.match(svg, />A<\/text>/);
  assert.match(svg, />5\.00<\/text>/); // count for totalArea 5
});

test("legend is drawn only when requested", () => {
  const without = toSvg(circleLayout(), { padding: 5 });
  assert.ok(!without.includes("<g>"));
  const withLegend = toSvg(circleLayout(), {
    padding: 5,
    legend: { show: true },
  });
  assert.match(withLegend, /<g>/);
});

test("legend gains a dashed complement entry when a container is present", () => {
  const base = circleLayout();
  const withContainer = {
    ...base,
    container: { x: 0, y: 0, width: 40, height: 40 },
  };

  const legend = toSvg(withContainer, { padding: 5, legend: { show: true } });
  assert.match(legend, /stroke-dasharray="2 2"/);
  assert.match(legend, />Complement<\/text>/);

  // The label is overridable.
  const custom = toSvg(withContainer, {
    padding: 5,
    legend: { show: true },
    complementLabel: "Outside",
  });
  assert.match(custom, />Outside<\/text>/);

  // No container → no complement entry.
  const plain = toSvg(base, { padding: 5, legend: { show: true } });
  assert.ok(!plain.includes("stroke-dasharray"));
  assert.ok(!plain.includes(">Complement</text>"));
});

test("label text is XML-escaped", () => {
  const svg = toSvg(circleLayout("A&B"), { padding: 5 });
  assert.match(svg, />A&amp;B<\/text>/);
  assert.ok(!/>A&B</.test(svg));
});

// B is fully nested in A: it has no exclusive region, only "A" and "A&B".
function nestedRegionLayout(withCoreMap) {
  const layout = {
    mode: "regions",
    shape: "circle",
    regions: [
      {
        combination: "A",
        totalArea: 8,
        labelAnchor: { x: 2, y: 5 },
        pieces: [],
      },
      {
        combination: "A&B",
        totalArea: 3,
        labelAnchor: { x: 7, y: 5 },
        pieces: [],
      },
    ],
    setAnchors: { A: { x: 2, y: 5 }, B: { x: 7, y: 5 } },
    metrics,
  };
  // The core records B's label as anchored to region "A&B"; A keeps its own.
  if (withCoreMap) layout.setAnchorRegions = { A: "A", B: "A&B" };
  return layout;
}

test("nestedSets folds a nested set using the core setAnchorRegions map", () => {
  const nested = nestedSets(nestedRegionLayout(true));
  assert.deepEqual(nested, { "A&B": ["B"] });
  // A is titled by its own exclusive region; B is folded into A&B.
  assert.deepEqual(regionTitleLines("A", nested), ["A"]);
  assert.deepEqual(regionTitleLines("A&B", nested), ["B"]);
});

test("nestedSets falls back to area scan without setAnchorRegions", () => {
  // Same result as the core map, re-derived from region areas.
  assert.deepEqual(nestedSets(nestedRegionLayout(false)), { "A&B": ["B"] });
});

// --- interactivity hooks (tooltip / interactive / regionAttrs) ---------------

test("no interactivity options leave fills as self-closing tags", () => {
  const region = svgBody(regionLayout());
  assert.match(
    region,
    /<path d="[^"]*" fill="[^"]*" fill-opacity="1" stroke="none" \/>/,
  );
  assert.ok(!region.includes("<title>"));
  assert.ok(!region.includes("data-combination"));

  const shape = svgBody(circleLayout());
  assert.match(shape, /<circle [^>]*stroke="none" \/>/);
  assert.ok(!shape.includes("<title>"));
});

test("tooltip adds an XML-escaped <title> to a region fill", () => {
  const info = [];
  const svg = svgBody(regionLayout(), {
    tooltip: (r) => {
      info.push(r);
      return `${r.combination} <${r.area}>`;
    },
  });
  assert.match(
    svg,
    /<path [^>]*stroke="none"><title>A &lt;5&gt;<\/title><\/path>/,
  );
  assert.ok(!svg.includes("<title>A <5>"));
  // The hook receives the region descriptor.
  assert.deepEqual(info, [{ combination: "A", sets: ["A"], area: 5 }]);
});

test("empty or nullish tooltip return adds no <title>", () => {
  assert.ok(
    !svgBody(regionLayout(), { tooltip: () => "" }).includes("<title>"),
  );
  assert.ok(
    !svgBody(regionLayout(), { tooltip: () => null }).includes("<title>"),
  );
  assert.ok(
    !svgBody(regionLayout(), { tooltip: () => undefined }).includes("<title>"),
  );
});

test("interactive emits data-combination/data-area on fills but not strokes", () => {
  const svg = svgBody(regionLayout(), { interactive: true, strokeWidth: 1 });
  // Fill path carries the data attributes.
  assert.match(
    svg,
    /<path [^>]*fill-opacity="1" stroke="none" data-combination="A" data-area="5">/,
  );
  // The stroke pass (fill="none") must not carry them.
  const strokePath = svg
    .split("\n")
    .find((l) => l.includes('fill="none"') && l.includes("stroke-width"));
  assert.ok(strokePath, "expected a stroke path");
  assert.ok(!strokePath.includes("data-combination"));
});

test("regionAttrs adds custom data-*, skips nullish, and overrides defaults", () => {
  const svg = svgBody(regionLayout(), {
    interactive: true,
    regionAttrs: (r) => ({
      "data-members": r.combination === "A" ? 3 : undefined,
      "data-skip": undefined,
      "data-area": "override",
    }),
  });
  assert.match(svg, /data-members="3"/);
  assert.ok(!svg.includes("data-skip"));
  // Explicit key wins over the interactive default.
  assert.match(svg, /data-area="override"/);
  assert.ok(!svg.includes('data-area="5"'));
});

test("shape-mode fills carry tooltip/data-* keyed on the set and fitted area", () => {
  const svg = svgBody(circleLayout(), {
    interactive: true,
    tooltip: (r) => `${r.combination}: ${r.area}`,
  });
  // fittedAreas.A === 5 from the shared metrics fixture.
  assert.match(
    svg,
    /<circle [^>]*stroke="none" data-combination="A" data-area="5"><title>A: 5<\/title><\/circle>/,
  );
});

test("glyphs render as one circle per point, above fills and below labels", () => {
  const svg = svgBody(regionLayout(), {
    glyphs: {
      radius: 0.5,
      positions: {
        A: [
          { x: 2, y: 2 },
          { x: 5, y: 5 },
          { x: 8, y: 8 },
        ],
      },
    },
  });
  const circles = svg.match(/<circle class="eunoia-glyph"/g) ?? [];
  assert.equal(circles.length, 3);
  assert.match(svg, /<circle class="eunoia-glyph" cx="2" cy="2" r="0.5" \/>/);
  // Layering: glyph group after the region fill path, before the label text.
  const glyphAt = svg.indexOf("data-glyphs");
  assert.ok(svg.indexOf("<path") < glyphAt);
  assert.ok(glyphAt < svg.indexOf("<text"));
});

test("glyphs default to a tinted region fill with a finer, darker edge", () => {
  const svg = svgBody(regionLayout(), {
    strokeWidth: 0.8,
    colors: { A: "#808080" },
    glyphs: { radius: 0.5, positions: { A: [{ x: 2, y: 2 }] } },
  });
  // Mid gray lightened by the default tint of 0.45, edged with the same
  // color darkened, at half the shape stroke width.
  assert.match(
    svg,
    /<g data-glyphs="A" fill="rgb\(185,185,185\)" stroke="rgb\(77,77,77\)" stroke-width="0.4">/,
  );
});

test("glyph fill, tint, stroke, opacity, and class are configurable; empty regions skipped", () => {
  const svg = svgBody(regionLayout(), {
    glyphs: {
      radius: 1,
      positions: { A: [{ x: 3, y: 3 }], "A&B": [] },
      fill: "#ff0000",
      stroke: "none",
      opacity: 0.5,
      className: "dot",
    },
  });
  assert.match(svg, /<g data-glyphs="A" fill="#ff0000" fill-opacity="0.5">/);
  assert.match(svg, /<circle class="dot" cx="3" cy="3" r="1" \/>/);
  assert.ok(
    !svg.includes('data-glyphs="A&amp;B"'),
    "empty region emits no group",
  );

  // `tint` drives the derived fill when `fill` is omitted, and goes negative
  // for marks that sit darker than their region.
  const darker = svgBody(regionLayout(), {
    colors: { A: "#808080" },
    glyphs: {
      radius: 1,
      positions: { A: [{ x: 3, y: 3 }] },
      tint: -0.5,
      strokeWidth: 0,
    },
  });
  assert.match(darker, /<g data-glyphs="A" fill="rgb\(64,64,64\)">/);
});

test("glyph boxes render one text per box, above fills and below labels", () => {
  const svg = svgBody(regionLayout(), {
    glyphBoxes: {
      scale: 1,
      boxes: {
        A: [
          { x: 3, y: 3, width: 2, height: 1 },
          { x: 7, y: 7, width: 2, height: 1 },
        ],
      },
      labels: { A: ["Ada", "Grace"] },
    },
  });
  const texts = svg.match(/<text class="eunoia-glyph-label"/g) ?? [];
  assert.equal(texts.length, 2);
  assert.match(svg, /<text class="eunoia-glyph-label" x="3" y="3">Ada<\/text>/);
  // Layering: after the region fill path, before the region label text.
  const boxesAt = svg.indexOf("data-glyph-boxes");
  assert.ok(svg.indexOf("<path") < boxesAt);
  assert.ok(boxesAt < svg.lastIndexOf("<text"));
});

test("glyph box font size is the reference size times the packed scale", () => {
  const svg = svgBody(regionLayout(), {
    glyphBoxes: {
      scale: 0.5,
      fontSize: 4,
      boxes: { A: [{ x: 3, y: 3, width: 2, height: 1 }] },
      labels: { A: ["Ada"] },
    },
  });
  // One `font-size` on the group, not per `<text>`.
  assert.match(svg, /<g data-glyph-boxes="A" font-size="2"/);
  assert.ok(!/<text class="eunoia-glyph-label"[^>]*font-size/.test(svg));

  // Omitted, `fontSize` falls back to the diagram's `labelSize`.
  const fallback = svgBody(regionLayout(), {
    labelSize: 3,
    glyphBoxes: {
      scale: 2,
      boxes: { A: [{ x: 3, y: 3, width: 2, height: 1 }] },
      labels: { A: ["Ada"] },
    },
  });
  assert.match(fallback, /<g data-glyph-boxes="A" font-size="6"/);
});

test("glyph box backgrounds are off by default and opt-in", () => {
  const opts = {
    scale: 1,
    boxes: { A: [{ x: 3, y: 3, width: 2, height: 1 }] },
    labels: { A: ["Ada"] },
  };
  assert.ok(!svgBody(regionLayout(), { glyphBoxes: opts }).includes("<rect"));

  const chips = svgBody(regionLayout(), {
    colors: { A: "#808080" },
    glyphBoxes: { ...opts, background: true },
  });
  // Same tinted region color the glyph discs derive, `rx` a quarter of the
  // short side, and the box is emitted corner-anchored.
  assert.match(
    chips,
    /<rect class="eunoia-glyph-label-bg" x="2" y="2.5" width="2" height="1" rx="0.25" fill="rgb\(185,185,185\)" \/>/,
  );

  const custom = svgBody(regionLayout(), {
    glyphBoxes: {
      ...opts,
      background: { fill: "#ff0000", rx: 0, opacity: 0.5 },
      className: "member",
    },
  });
  assert.match(
    custom,
    /<rect class="member-bg" [^>]*rx="0" fill="#ff0000" fill-opacity="0.5" \/>/,
  );
  assert.match(custom, /<text class="member" x="3" y="3">Ada<\/text>/);
});

test("glyph box labels and combination keys are escaped", () => {
  const layout = regionLayout();
  layout.regions[0].combination = "A&B";
  const svg = svgBody(layout, {
    glyphBoxes: {
      scale: 1,
      boxes: { "A&B": [{ x: 3, y: 3, width: 2, height: 1 }] },
      labels: { "A&B": ['Ada <3 "Lovelace" & Co'] },
    },
  });
  assert.match(svg, /data-glyph-boxes="A&amp;B"/);
  assert.match(
    svg,
    /<text class="eunoia-glyph-label" x="3" y="3">Ada &lt;3 "Lovelace" &amp; Co<\/text>/,
  );
});

test("glyph boxes without labels emit background chips only", () => {
  const svg = svgBody(regionLayout(), {
    glyphBoxes: {
      scale: 1,
      boxes: { A: [{ x: 3, y: 3, width: 2, height: 1 }], "A&B": [] },
      background: true,
    },
  });
  assert.ok(!svg.includes('eunoia-glyph-label"'), "no text without labels");
  assert.match(svg, /<rect class="eunoia-glyph-label-bg"/);
  assert.ok(
    !svg.includes('data-glyph-boxes="A&amp;B"'),
    "empty region emits no group",
  );
});

test("glyphs and glyph boxes can both render, discs first", () => {
  const svg = svgBody(regionLayout(), {
    glyphs: { radius: 0.5, positions: { A: [{ x: 2, y: 2 }] } },
    glyphBoxes: {
      scale: 1,
      boxes: { A: [{ x: 3, y: 3, width: 2, height: 1 }] },
      labels: { A: ["Ada"] },
    },
  });
  assert.ok(svg.indexOf("data-glyphs") < svg.indexOf("data-glyph-boxes"));
});
