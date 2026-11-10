# Shapes {#shapes}

A `DiagramSpec` says *what* to draw (the size of each set and of each
overlap) but carries no geometry. Geometry only appears at **fit
time**, when you pick a shape primitive and the fitter solves for the
position and size of one such shape per set so that the overlaps
reproduce the requested areas as closely as possible. The same spec can
therefore be drawn with any of five primitives:

- **Circle:** the default, simplest, and fastest.
- **Ellipse:** flexible in size and orientation, with canonical Venn diagrams for up to five sets.
- **Square** and **Rectangle:** axis-aligned, for a layout-board look and exact, rotation-free regions.
- **Rotated rectangle:** a rectangle that can turn, with exact polygon overlaps and canonical Venn diagrams for up to four sets.

Every primitive is area-true: the fitter never distorts a shape's
reported area to improve a fit. What differs between them is how many
degrees of freedom each shape gives the optimizer, and that is what this
page is about.

## Circle {#circle}

A circle is `[x, y, r]`: a center and a radius, three degrees of
freedom per set. It has no orientation, so there is nothing to rotate
and nothing to skew; the fitter only chooses where each disc sits and
how big it is. That makes circles the quickest shape to fit and the
easiest to read, and for two or three sets they are usually all you
need. Circles are the default for `euler()`, and eunoia has canonical
Venn layouts for them up to three sets.

::: {.component name="eunoia" #docs-concepts-shapes-1}

A symmetric three-set spec fitted with circles.

:::

## Ellipse {#ellipse}

An ellipse is `[x, y, a, b, φ]`: a center, two semi-axes, and a
rotation angle, giving five degrees of freedom per set. The extra two
parameters let a set stretch and turn, which is what you need when the
pairwise overlaps are lopsided enough that no arrangement of round discs
can satisfy them. Eunoia ships canonical four- and five-set Venn
arrangements (the Wilkinson and Edwards ellipse layouts) for ellipses.
They are the only supported primitive with a canonical five-set Venn.

That freedom comes at a cost: more free parameters mean a rougher loss
surface, so ellipse fits benefit from more restarts and lean more often
on the optimizer's global-escape stage (see the
[fitter pipeline](../fitter-pipeline/index.md) for how that works).

::: {.component name="eunoia" #docs-concepts-shapes-2}

A four-set spec fitted with ellipses; no four-circle diagram realizes every region.

:::

## Square {#square}

A square is `[x, y, side]`: a center and a side length, three degrees
of freedom, the same count as a circle. Squares are kept **axis-aligned**:
rotation is not part of the parameter vector, so the diagram keeps a
tidy, grid-like orientation. Because the intersection of axis-aligned
squares is itself an axis-aligned rectangle, the overlap areas are
computed exactly in closed form, with no polygon clipping. Like circles,
squares support canonical Venn layouts up to three sets; there is no
axis-aligned-square Venn for four or more sets.

::: {.component name="eunoia" #docs-concepts-shapes-3}

The same three-set spec fitted with axis-aligned squares.

:::

## Rectangle {#rectangle}

A rectangle is `[x, y, w, h]`: a center, a width, and a height, giving
four degrees of freedom. It is the square with a free aspect ratio: wide
or tall, but, like the square, never rotated. It keeps the same exact
closed-form region areas and the same three-set limit on canonical Venn
layouts. Prefer rectangles over squares when your sets differ enough in
size that a fixed aspect ratio is too constraining.

::: {.component name="eunoia" #docs-concepts-shapes-4}

A two-set spec with unequal sizes fitted with rectangles.

:::

## Rotated Rectangle {#rotated-rectangle}

A rotated rectangle is `[x, y, w, h, φ]`: a center, a width, a height,
and a rotation angle in radians, giving five degrees of freedom per set.
The width and height describe the rectangle's local axes; `φ` turns
those axes counterclockwise from the positive x-axis. This lets each
box turn to adjust its overlaps while keeping straight edges.

Intersections of rotated rectangles are convex polygons. Eunoia clips
the corners and edges to compute their areas exactly, without sampling
the boundary. Those areas have kinks as edges enter or leave an
intersection, so the fitter uses derivative-free Nelder–Mead and
CMA-ES by default. More restarts can help with difficult layouts.

Rotation also makes a true four-set Venn arrangement possible. Eunoia
provides canonical rotated-rectangle Venn layouts for one through four
sets; the axis-aligned rectangle stops at three.

::: {.component name="eunoia" #docs-concepts-shapes-5}

The four-set spec fitted with rectangles that can turn as well as stretch.

:::

## Choosing a Shape {#choosing-a-shape}

A few rules of thumb:

- **Two or three sets:** start with circles. They are fast, legible, and usually faithful. Check the [goodness-of-fit metrics](../goodness-of-fit/index.md). If a three-set fit is poor, the overlaps are probably too lopsided for circles, and ellipses will do better.
- **Four-set Venn diagrams:** use ellipses or rotated rectangles. Both provide canonical arrangements with every intersection present.
- **Five-set Venn diagrams:** use ellipses, the only supported primitive with a canonical arrangement for five sets.
- **A grid or layout-board look, or exact rotation-free regions:** use squares or rectangles. Pick rectangles when the sets vary enough in size that a fixed aspect ratio would be too constraining.
- **Straight edges with freedom to turn:** use rotated rectangles. Check the fit metrics and increase restarts if the default fit is poor.

The canonical Venn support per primitive sets a hard ceiling on set
count for true Venn diagrams:

| Shape             | Parameters         | Degrees of freedom | Canonical Venn |
| ----------------- | ------------------ | ------------------ | -------------- |
| Circle            | `[x, y, r]`        | 3                  | up to 3 sets   |
| Ellipse           | `[x, y, a, b, φ]`  | 5                  | up to 5 sets   |
| Square            | `[x, y, side]`     | 3                  | up to 3 sets   |
| Rectangle         | `[x, y, w, h]`     | 4                  | up to 3 sets   |
| Rotated rectangle | `[x, y, w, h, φ]`  | 5                  | up to 4 sets   |

Area-proportional Euler diagrams are not bound by these ceilings (you
can fit more sets), but the more free parameters a shape has, the harder
the fit, so expect ellipse and rotated-rectangle fits to take longer and
to vary more between runs than circle fits.

## Selecting a Shape in Code {#selecting-a-shape-in-code}

In JavaScript, the shape is the `shape` option on `euler()` and
`venn()`. It accepts `"circle"`, `"ellipse"`, `"square"`,
`"rectangle"`, or `"rotatedRectangle"`; `euler()` defaults to `"circle"`
and `venn()` to `"ellipse"` (the only shape that reaches `n = 5`).

```js
import { euler } from "@jolars/eunoia";

const layout = euler({
  sets: { A: 10, B: 10, "A&B": 3 },
  shape: "rotatedRectangle",
});
layout.rotatedRectangles; // center, width, height, rotation, and label anchor
```

In Rust, the shape is a type parameter on the fitter, defaulting to
`Circle`:

```rust
use eunoia::geometry::shapes::RotatedRectangle;
use eunoia::Fitter;

let layout = Fitter::<RotatedRectangle>::new(&spec).fit()?;
```

The R, Python, Julia, and C APIs use `"rotated_rectangle"` for the
same shape. See the language quickstarts for examples.

A type becomes fittable by implementing the `DiagramShape` trait, which
also defines each shape's two parameter encodings (a human-friendly
geometric one and an internal optimizer one). For the trait and the
encodings in full, see the
[architecture notes in `CONTRIBUTING.md`](https://github.com/jolars/eunoia/blob/main/CONTRIBUTING.md)
and the [rustdoc](https://docs.rs/eunoia/). (`Polygon` is used for output
and region extraction, not for fitting, so it is not one of these
primitives.)

## See Also {#see-also}

- [Fitter Pipeline](../fitter-pipeline/index.md): how a spec becomes fitted geometry.
- [Goodness of Fit](../goodness-of-fit/index.md): how to tell a faithful diagram from a forced one.
