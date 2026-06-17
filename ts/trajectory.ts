// Experimental, internal entry point: the per-iteration fitter trajectory used
// by the `fitter-pipeline` docs animation. Exposed as `@jolars/eunoia/trajectory`.
//
// This is NOT part of the supported public API and may change or be removed
// without a major-version bump. It re-runs a single deterministic fit attempt
// with `basin` observers attached and returns the optimizer's parameter
// trajectory (random init → MDS → final) for replay. Use `euler()`/`venn()`
// from `@jolars/eunoia` for normal fitting.

import * as wasm from "./eunoia_wasm.js";

/** Pipeline stage a trajectory frame belongs to. */
export type TrajectoryStage = "init" | "mds" | "final";

/** A circle at one recorded frame. */
export interface TrajectoryCircle {
  label: string;
  x: number;
  y: number;
  r: number;
}

/** An ellipse at one recorded frame (`phi` in radians). */
export interface TrajectoryEllipse {
  label: string;
  x: number;
  y: number;
  a: number;
  b: number;
  phi: number;
}

export type TrajectoryShape = TrajectoryCircle | TrajectoryEllipse;

/** One recorded optimizer iterate. */
export interface TrajectoryFrame {
  /** Which pipeline stage produced this frame. */
  stage: TrajectoryStage;
  /** Solver iteration counter (resets per stage). */
  iteration: number;
  /** Objective value; only comparable within a stage. */
  cost: number;
  /** Shapes at this frame, in set order. */
  shapes: TrajectoryShape[];
}

export interface TrajectoryOptions {
  /** Set sizes keyed by combination expression (e.g. `{ A: 5, "A&B": 1 }`). */
  sets: Record<string, number>;
  /** Whether `sets` values are exclusive subset sizes or full set unions. Default `"exclusive"`. */
  inputType?: "exclusive" | "inclusive";
  /** Shape primitive to fit. Only `"circle"` and `"ellipse"` are supported. Default `"circle"`. */
  shape?: "circle" | "ellipse";
  /** RNG seed; a fresh seed re-rolls the random start. Default `0`. */
  seed?: number | bigint;
}

export interface Trajectory {
  /** Frames in playback order: one `init`, then the `mds` iterates, then the `final` iterates. */
  frames: TrajectoryFrame[];
}

/**
 * Record the per-iteration fitter trajectory for a single deterministic attempt.
 *
 * The run is fully determined by `seed`. Pins Levenberg-Marquardt and the
 * sum-of-squares loss internally (so every iterate decodes to shapes) and skips
 * layout normalization — the frames are the optimizer's own raw coordinates,
 * which is what the animation should show.
 */
export function eulerTrajectory(options: TrajectoryOptions): Trajectory {
  const { sets, inputType = "exclusive", shape = "circle", seed = 0 } = options;

  const specs: wasm.DiagramSpec[] = [];
  for (const [input, size] of Object.entries(sets)) {
    if (!Number.isFinite(size) || size <= 0) continue;
    specs.push(new wasm.DiagramSpec(input, size));
  }

  const seedArg = typeof seed === "bigint" ? seed : BigInt(seed);
  const result = wasm.record_trajectory(specs, inputType, shape, seedArg);
  const frames = JSON.parse(result.frames_json) as TrajectoryFrame[];
  return { frames };
}
