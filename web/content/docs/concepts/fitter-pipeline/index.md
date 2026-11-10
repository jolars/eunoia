# Fitter Pipeline {#fitter-pipeline}

Fitting a diagram turns a set of region sizes into concrete shapes
(circles, ellipses, squares, axis-aligned rectangles, or rotated
rectangles) whose overlaps reproduce those sizes as closely as possible.
Eunoia runs this in four stages.

The widget below records a real fit of a three-set spec and lets you
play it back step by step. **Initial layout** runs the MDS stage: the
shapes start at random positions and spread out so overlapping sets sit
near each other. **Final layout** then runs the optimizer, refining
every shape to minimize the loss. Each button plays its stage and stops
on the result; **Re-run** draws a fresh random start, and you can switch
between circles and ellipses.

::: {.component name="eunoia" #docs-concepts-fitter-pipeline-1}

A real recording of the fitter, played back one optimizer step at a time. Run the initial layout, then the final layout. Re-run draws a new random start; a real fit keeps the lowest-loss of several such starts.

:::

## 1. Initial Layout (MDS) {#1-initial-layout-mds}

An initial arrangement is computed with **positional multidimensional
scaling**. Each shape's size is fixed up front from the requested set
areas, and only the center `(x, y)` of every shape is solved for, which
is why the shapes move into place at a fixed size during this stage in
the animation. Target pairwise distances come from inverting
each shape's own overlap formula, so the warm start already places
overlapping sets near each other and disjoint sets apart.

## 2. Optimizer Encoding {#2-optimizer-encoding}

The MDS positions are converted into the per-shape parameter encoding
the optimizer works in. An ellipse, for example, is optimized over
`[x, y, ln a, ln b, φ]` rather than its raw semi-axes. The log encoding
keeps widths and heights positive and balances the scales of the
parameters so a single solver setting works across positions, radii,
and angles.

## 3. Final Optimization {#3-final-optimization}

The shape parameters are refined by **minimizing a loss** that measures
the discrepancy between the fitted region areas and the requested ones.
This is the **final** stage in the animation, where the shapes resize
and shift to tighten the fit. The optimizer and the loss are both
selectable; circles and ellipses use analytical gradients where the
loss allows, and a global-escape stage can kick in when a plain local
solve stalls on a hard layout.

Rotated rectangles use exact polygon clipping for their overlap areas.
Because those areas have kinks when edges enter or leave an intersection,
the default optimizer pool uses derivative-free Nelder–Mead and CMA-ES.

A single fit actually runs this two-phase pipeline several times (ten by
default) from independent random starts and keeps the lowest-loss
result. Pressing **Re-run** above shows one such start, and you can
see how different starts converge differently.

## 4. Normalization {#4-normalization}

The fitted layout is post-processed for a stable, consistent
appearance: disconnected clusters are separated, the diagram is rotated
to a canonical orientation, centered, and the clusters are packed
together. The result is a layout you can polygonize, label, and render.
(The animation shows the optimizer's own raw coordinates, before this
cosmetic step.)

How faithfully the result reproduces the requested sizes is summarized
by a few scalar metrics; see [Goodness of Fit](../goodness-of-fit/index.md).

For the authoritative details (optimizers, losses, gradients, and the
shape-specific encodings) see the
[Architecture section in `CONTRIBUTING.md`](https://github.com/jolars/eunoia/blob/main/CONTRIBUTING.md).
