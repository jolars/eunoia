import assert from "node:assert/strict";
import { test } from "node:test";

import { venn } from "../../npm/index.js";

test("four-set rotated-rectangle Venn returns rotated outlines", () => {
  const layout = venn({ n: 4, shape: "rotatedRectangle" });

  assert.equal(layout.mode, "polygons");
  assert.equal(layout.shape, "rotatedRectangle");
  assert.equal(layout.polygons.length, 4);
  assert.equal(layout.rotatedRectangles.length, 4);
  assert.ok(layout.rotatedRectangles.some((rect) => rect.rotation !== 0));
  assert.ok(layout.polygons.every((polygon) => polygon.area > 0));
});

test("four-set rotated-rectangle Venn contains all 15 exclusive regions", () => {
  const layout = venn({ n: 4, shape: "rotatedRectangle", output: "regions" });

  assert.equal(layout.mode, "regions");
  assert.equal(layout.shape, "rotatedRectangle");
  assert.equal(layout.shapeOutlines.length, 4);
  assert.equal(layout.regions.length, 15);
  assert.equal(
    new Set(layout.regions.map((region) => region.combination)).size,
    15,
  );
  assert.ok(layout.regions.every((region) => region.totalArea > 0));
});

test("rotated-rectangle Venn rejects five sets with its four-set limit", () => {
  assert.throws(() => venn({ n: 5, shape: "rotatedRectangle" }), {
    name: "RangeError",
    message: /1\.\.=4/,
  });
});
