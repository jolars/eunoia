import assert from "node:assert/strict";
import { test } from "node:test";

import * as wasm from "../../npm/eunoia_wasm.js";
import { euler } from "../../npm/index.js";

test("invalid tolerances throw without trapping the WASM instance", () => {
  const sets = { A: 5, B: 3, "A&B": 1 };
  for (const tolerance of [-1, NaN, Infinity, -Infinity]) {
    assert.throws(
      () => euler({ sets, seed: 1, tolerance }),
      /tolerance must be finite and nonnegative/,
    );
  }
  assert.ok(Number.isFinite(euler({ sets, seed: 1 }).metrics.loss));
});

test("zero tolerance reaches the solver unchanged", () => {
  const sets = { A: 2.2, B: 2, C: 3, "A&B&C": 1 };
  const specs = Object.entries(sets).map(
    ([name, size]) => new wasm.DiagramSpec(name, size),
  );
  // The WASM function consumes the spec handles. Only its result needs freeing.
  const direct = wasm.generate_circles_as_polygons(
    specs,
    "exclusive",
    16,
    1n,
    wasm.WasmOptimizer.LevenbergMarquardt,
    wasm.WasmLossType.SumSquared,
    0,
    1,
    undefined,
  );
  try {
    const layout = euler({
      sets,
      seed: 1,
      optimizer: "levenbergMarquardt",
      loss: "sumSquared",
      tolerance: 0,
      restarts: 1,
      polygonVertices: 16,
    });
    assert.equal(layout.metrics.loss, direct.loss);
  } finally {
    direct.free();
  }
});
