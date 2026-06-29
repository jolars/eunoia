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
