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
        if (boxes) options.targetAspectRatio = 1;
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

test("member boxes balance rows through TypeScript and raw WASM", () => {
  for (const [side, defaultRows] of [
    [20, 2],
    [40, 1],
  ]) {
    const outer = [
      [0, 0],
      [side, 0],
      [side, side],
      [0, side],
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
    const sizes = { A: Array.from({ length: 12 }, () => ({ w: 2, h: 0.5 })) };
    const wireSizes = { A: sizes.A.map(({ w, h }) => [w, h]) };
    const baseline = placeGlyphBoxesForRegions({ regions, sizes });
    const rows = (result) => new Set(result.boxes.A.map((b) => b.y)).size;
    assert.equal(rows(baseline), defaultRows);
    for (const [targetAspectRatio, expectedRows] of [
      [1, 6],
      [4, 3],
    ]) {
      const result = placeGlyphBoxesForRegions({
        regions,
        sizes,
        options: { targetAspectRatio },
      });
      assert.equal(result.scale, baseline.scale);
      assert.equal(result.boxes.A.length, 12);
      assert.equal(rows(result), expectedRows);
      assert.equal(result.unplaced, undefined);
      const direct = JSON.parse(
        wasm.place_region_glyph_boxes(
          JSON.stringify(polygons),
          JSON.stringify(wireSizes),
          JSON.stringify({ targetAspectRatio }),
        ),
      );
      assert.equal(direct.scale, result.scale);
      assert.deepEqual(
        direct.boxes.A.map(([x, y, width, height]) => ({
          x,
          y,
          width,
          height,
        })),
        result.boxes.A,
      );
    }
    for (const targetAspectRatio of [0, -1, NaN, Infinity]) {
      assert.deepEqual(
        placeGlyphBoxesForRegions({
          regions,
          sizes,
          options: { targetAspectRatio },
        }),
        baseline,
      );
    }
    const random = { arrangement: "random", seed: 3 };
    assert.deepEqual(
      placeGlyphBoxesForRegions({
        regions,
        sizes,
        options: { ...random, targetAspectRatio: 1 },
      }),
      placeGlyphBoxesForRegions({ regions, sizes, options: random }),
    );
  }
});
