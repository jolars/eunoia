import assert from "node:assert/strict";
import { test } from "node:test";

import * as wasm from "../../npm/eunoia_wasm.js";
import {
  placeGlyphBoxesForRegions,
  placeGlyphsForRegions,
} from "../../npm/index.js";

const outer = [
  [0, 0],
  [2, 0],
  [2, 2],
  [0, 2],
];
const polygons = { A: [{ outer, holes: [] }] };
const regions = [
  {
    combination: "A",
    pieces: [
      { outer: { vertices: outer.map(([x, y]) => ({ x, y })) }, holes: [] },
    ],
  },
];
const obstacles = [{ x: 1, y: 1, width: 2, height: 2 }];

for (const boxes of [false, true]) {
  const place = boxes ? placeGlyphBoxesForRegions : placeGlyphsForRegions;
  const direct = boxes
    ? wasm.place_region_glyph_boxes
    : wasm.place_region_glyphs;
  const output = boxes ? "boxes" : "positions";
  const input = boxes
    ? { sizes: { A: Array.from({ length: 3 }, () => ({ w: 0.5, h: 0.2 })) } }
    : { counts: { A: 3 } };
  const wireInput = boxes
    ? {
        A: [
          [0.5, 0.2],
          [0.5, 0.2],
          [0.5, 0.2],
        ],
      }
    : { A: 3 };

  for (const arrangement of ["uniform", "random"]) {
    for (const fixed of [false, true]) {
      test(`${output}: strict obstacles report overflow (${arrangement}, fixed=${fixed})`, () => {
        const options = { arrangement, obstacles };
        if (fixed) options[boxes ? "scale" : "radius"] = boxes ? 0.5 : 0.2;
        const plain = place({ regions, ...input, options });
        assert.equal(plain[output].A.length, 3);
        assert.deepEqual(
          place({
            regions,
            ...input,
            options: { ...options, obstaclePolicy: "bestEffort" },
          }),
          plain,
        );
        const strict = place({
          regions,
          ...input,
          options: { ...options, obstaclePolicy: "strict" },
        });
        assert.equal(strict[output].A.length, 0);
        assert.deepEqual(strict.unplaced, { A: 3 });
        assert.equal(
          strict[boxes ? "scale" : "radius"],
          plain[boxes ? "scale" : "radius"],
        );

        const raw = JSON.parse(
          direct(
            JSON.stringify(polygons),
            JSON.stringify(wireInput),
            JSON.stringify({
              ...options,
              arrangement: arrangement === "uniform" ? "Uniform" : "Random",
              obstaclePolicy: "Strict",
            }),
          ),
        );
        assert.equal(raw[output].A.length, 0);
        assert.deepEqual(raw.unplaced, { A: 3 });
      });
    }
  }

  test(`${output}: invalid obstacle policies are rejected by TypeScript and WASM`, () => {
    for (const obstaclePolicy of ["unknown", "Strict", "best_effort"]) {
      assert.throws(
        () => place({ regions, ...input, options: { obstaclePolicy } }),
        /unknown obstaclePolicy/,
      );
    }
    for (const obstaclePolicy of ["unknown", "strict", "bestEffort"]) {
      assert.throws(
        () =>
          direct(
            JSON.stringify(polygons),
            JSON.stringify(wireInput),
            JSON.stringify({ obstaclePolicy }),
          ),
        /invalid options.obstaclePolicy/,
      );
    }
  });
}
