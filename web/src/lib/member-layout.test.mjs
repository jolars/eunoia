import assert from "node:assert/strict";
import { test } from "node:test";

import { boundingBox, toSvg } from "@jolars/eunoia/svg";
import { placeMemberLabels } from "./member-layout.ts";
import { memberListsForRegions } from "./members.ts";

function square(combination, x, side) {
  const vertices = [
    { x, y: 0 },
    { x: x + side, y: 0 },
    { x: x + side, y: side },
    { x, y: side },
  ];
  return {
    combination,
    totalArea: side ** 2,
    labelAnchor: { x: x + side / 2, y: side / 2 },
    pieces: [{ outer: { vertices }, holes: [], area: side ** 2 }],
  };
}

function fixture() {
  return {
    layout: {
      mode: "regions",
      shape: "circle",
      regions: [square("A", 0, 2), square("B", 10, 20)],
      shapeOutlines: [],
      setAnchors: {},
      setAnchorRegions: {},
      metrics: {},
    },
    labels: { A: ["Oversized", "Ada", "Grace"], B: ["Barbara"] },
    sizes: {
      A: [
        { w: 7, h: 0.3 },
        { w: 0.4, h: 0.3 },
        { w: 0.4, h: 0.3 },
      ],
      B: [{ w: 0.5, h: 0.3 }],
    },
    headerSizes: { A: { w: 0.3, h: 0.3 }, B: { w: 0.3, h: 0.3 } },
    headerLines: {
      A: [{ text: "A", w: 0.3, h: 0.3, fontSize: 0.3 }],
      B: [{ text: "B", w: 0.3, h: 0.3, fontSize: 0.3 }],
    },
    missingHeaders: {
      A: { text: "A (no geometry)", w: 3, h: 0.3, fontSize: 0.3 },
      B: { text: "B (no geometry)", w: 3, h: 0.3, fontSize: 0.3 },
    },
    fontSize: 0.3,
    padding: 0.05,
    boxOptions: {},
    strategy: { tether: "boundary" },
  };
}

for (const arrangement of ["uniform", "lowDiscrepancy", "random"]) {
  for (const placement of ["raycast", "forceDirected", "matched", "elbow"]) {
    test(`complete rosters survive oversized first items (${arrangement}, ${placement})`, () => {
      const options = fixture();
      options.boxOptions = { arrangement, seed: 42 };
      options.strategy.leader =
        placement === "elbow"
          ? { type: "elbow" }
          : { type: "straight", placement };
      const placed = placeMemberLabels(options);
      assert.equal(placed.boxes.boxes.A, undefined);
      assert.equal(placed.boxes.boxes.B.length, 1);
      assert.equal(placed.boxes.scale, 1);
      assert.equal(placed.boxes.unplaced, undefined);
      assert.deepEqual(
        placed.callouts.A.lines.map((l) => l.text),
        ["A", "Oversized", "Ada", "Grace"],
      );
      assert.notEqual(placed.callouts.A.placement.kind, "interior");
      assert.ok(placed.callouts.A.placement.tether);
      assert.deepEqual(placeMemberLabels(options), placed);
      const svg = toSvg(options.layout, {
        placements: placed.placements,
        memberCallouts: placed.callouts,
        glyphBoxes: { ...placed.boxes, labels: options.labels, fontSize: 0.3 },
      });
      for (const names of Object.values(options.labels)) {
        for (const name of names)
          assert.equal(svg.split(`>${name}</text>`).length - 1, 1);
      }
      assert.equal(svg.split(">A</text>").length - 1, 1);
    });
  }
}

test("missing, empty, and degenerate regions retain their complete rosters", () => {
  const options = fixture();
  options.layout.regions[0].pieces = [];
  const degenerate = square("C", 40, 0);
  options.layout.regions.push(degenerate);
  for (const key of ["A&B", "C"]) {
    options.labels[key] = [`Member of ${key}`];
    options.sizes[key] = [{ w: 1, h: 0.3 }];
    options.missingHeaders[key] = {
      text: `${key} (no geometry)`,
      w: 4,
      h: 0.3,
      fontSize: 0.3,
    };
  }
  const placed = placeMemberLabels(options);
  for (const key of ["A", "A&B", "C"]) {
    const callout = placed.callouts[key];
    assert.ok(callout);
    assert.equal(callout.placement.tether, undefined);
    assert.equal(callout.lines[0].text, `${key} (no geometry)`);
    assert.deepEqual(
      callout.lines.slice(1).map((l) => l.text),
      options.labels[key],
    );
    assert.ok(callout.placement.anchor.x - callout.size.w / 2 > 40);
  }
  const [a, ab, c] = ["A", "A&B", "C"].map((k) => placed.callouts[k]);
  for (const [above, below] of [
    [a, ab],
    [ab, c],
  ]) {
    assert.ok(
      above.placement.anchor.y + above.size.h / 2 <
        below.placement.anchor.y - below.size.h / 2,
    );
  }
  const bounds = boundingBox(options.layout, {
    memberCallouts: placed.callouts,
  });
  assert.ok(bounds.maxX >= ab.placement.anchor.x + ab.size.w / 2);
});

test("strict label obstacles cannot hide a partially placed roster", () => {
  const options = fixture();
  options.labels.A = ["Ada", "Grace"];
  options.sizes.A = [
    { w: 0.4, h: 0.3 },
    { w: 0.4, h: 0.3 },
  ];
  options.boxOptions.scale = 1;
  options.headerSizes.A = { w: 1.3, h: 1.3 };
  options.headerLines.A = [{ text: "Heading", w: 1.3, h: 1.3, fontSize: 1.3 }];
  const placed = placeMemberLabels(options);
  assert.equal(placed.boxes.boxes.A, undefined);
  assert.deepEqual(
    placed.callouts.A.lines.map((l) => l.text),
    ["Heading", "Ada", "Grace"],
  );
  assert.ok(placed.boxes.boxes.B.length > 0);
});

test("roster matching preserves missing combinations and merges reordered keys", () => {
  assert.deepEqual(
    memberListsForRegions(
      [
        { input: "B & A", members: "Ada, Grace" },
        { input: "A&B", members: "Barbara" },
        { input: "C", members: "Alan" },
        { input: "A|B", members: "Not a region" },
      ],
      [square("C", 0, 10)],
    ),
    {
      "A&B": ["Ada", "Grace", "Barbara"],
      C: ["Alan"],
    },
  );
});

test("callouts respect the container and keep exterior set names clear", () => {
  const options = fixture();
  options.layout.shapeOutlines = options.layout.regions.map((r) => ({
    label: r.combination,
    vertices: r.pieces[0].outer.vertices,
  }));
  options.layout.container = { x: 15, y: 10, width: 40, height: 30 };
  options.setSizes = { A: { w: 1, h: 0.3 }, B: { w: 1, h: 0.3 } };
  const placed = placeMemberLabels(options);
  const callout = placed.callouts.A;
  const { x, y } = callout.placement.anchor;
  assert.ok(
    x + callout.size.w / 2 <= -5 ||
      x - callout.size.w / 2 >= 35 ||
      y + callout.size.h / 2 <= -5 ||
      y - callout.size.h / 2 >= 25,
  );
  for (const [key, placement] of Object.entries(placed.setPlacements)) {
    const size = options.setSizes[key];
    assert.ok(
      Math.abs(placement.anchor.x - x) >= (size.w + callout.size.w) / 2 ||
        Math.abs(placement.anchor.y - y) >= (size.h + callout.size.h) / 2,
    );
  }
});
