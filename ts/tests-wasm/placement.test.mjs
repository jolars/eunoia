import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import * as wasm from "../../npm/eunoia_wasm.js";
import { placeLabelsForRegions } from "../../npm/index.js";

const fixture = JSON.parse(
  readFileSync(
    new URL(
      "../../crates/eunoia/tests/fixtures/leader_routing_gallery.json",
      import.meta.url,
    ),
    "utf8",
  ),
);
const point = ([x, y]) => ({ x, y });
const regions = Object.entries(fixture.regions).map(
  ([combination, pieces]) => ({
    combination,
    pieces: pieces.map(({ outer, holes }) => ({
      outer: { vertices: outer.map(point) },
      holes: holes.map((ring) => ({ vertices: ring.map(point) })),
    })),
  }),
);
const sizes = Object.fromEntries(
  Object.entries(fixture.sizes).map(([key, [w, h]]) => [key, { w, h }]),
);

function inRing([x, y], ring) {
  let inside = false;
  for (let i = 0; i < ring.length; i++) {
    const [ax, ay] = ring[i];
    const [bx, by] = ring[(i + 1) % ring.length];
    if (ay > y !== by > y && x < ax + ((bx - ax) * (y - ay)) / (by - ay)) {
      inside = !inside;
    }
  }
  return inside;
}

function traversal(start, end, own) {
  const d = [end[0] - start[0], end[1] - start[1]];
  const cross = (a, b) => a[0] * b[1] - a[1] * b[0];
  let total = 0;
  for (const [key, pieces] of Object.entries(fixture.regions)) {
    if (key === own) continue;
    for (const { outer, holes } of pieces) {
      const cuts = [0, 1];
      for (const ring of [outer, ...holes]) {
        for (let i = 0; i < ring.length; i++) {
          const a = ring[i];
          const b = ring[(i + 1) % ring.length];
          const edge = [b[0] - a[0], b[1] - a[1]];
          const denom = cross(d, edge);
          if (Math.abs(denom) < 1e-12) continue;
          const delta = [a[0] - start[0], a[1] - start[1]];
          const t = cross(delta, edge) / denom;
          const u = cross(delta, d) / denom;
          if (t > 0 && t < 1 && u >= -1e-10 && u <= 1 + 1e-10) cuts.push(t);
        }
      }
      cuts.sort((a, b) => a - b);
      for (let i = 1; i < cuts.length; i++) {
        const lo = cuts[i - 1];
        const hi = cuts[i];
        const t = (lo + hi) / 2;
        const midpoint = [start[0] + t * d[0], start[1] + t * d[1]];
        if (
          inRing(midpoint, outer) &&
          !holes.some((h) => inRing(midpoint, h))
        ) {
          total += (hi - lo) * Math.hypot(...d);
        }
      }
    }
  }
  return total;
}

for (const [placement, wirePolicy] of [
  ["raycast", "Raycast"],
  ["forceDirected", "ForceDirected"],
  ["matched", "Matched"],
]) {
  for (const tether of ["poi", "boundary"]) {
    test(`gallery leader clears foreign regions through TS and WASM (${placement}/${tether})`, () => {
      const typed = placeLabelsForRegions({
        regions,
        sizes,
        strategy: { leader: { type: "straight", placement }, tether },
      });
      const raw = JSON.parse(
        wasm.place_region_labels(
          JSON.stringify(fixture.regions),
          undefined,
          JSON.stringify(fixture.sizes),
          JSON.stringify({
            leader: { type: "straight", placement: wirePolicy },
            tether: tether === "poi" ? "Poi" : "Boundary",
          }),
        ),
      );
      const key = "Anti-CCP&Treat";
      const p = typed[key];
      assert.equal(p.kind, `exterior${wirePolicy}`);
      assert.deepEqual(p.anchor, point(raw[key].anchor));
      assert.deepEqual(p.tether, point(raw[key].tether));
      assert.deepEqual(p.leaderEnd, point(raw[key].leaderEnd));
      assert.deepEqual(p.leaderWaypoints, []);
      const length = traversal(raw[key].tether, raw[key].leaderEnd, key);
      assert.ok(length < 1e-5, `foreign traversal ${length}`);

      const baseline = fixture.baseline["raycast/poi"][key].placement;
      assert.ok(traversal(baseline.tether, baseline.leader_end, key) > 2);
      assert.equal(Object.keys(typed).length, Object.keys(sizes).length);
    });
  }
}
