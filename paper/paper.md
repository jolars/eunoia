---
title: >-
  Eunoia: Area-Proportional Euler and Venn Diagrams in Rust, Julia, Python, R,
  and JavaScript
tags:
  - Rust
  - R
  - Python
  - Julia
  - JavaScript
  - WebAssembly
  - Euler diagrams
  - Venn diagrams
  - set visualization
  - data visualization
  - optimization
authors:
  - name: Johan Larsson
    orcid: 0000-0002-4029-5945
    affiliation: "1"
  - name: Nicole M. Lee
    orcid: 0000-0001-6953-2177
    affiliation: "2"
affiliations:
  - name: Department of Mathematical Sciences, University of Copenhagen, Denmark
    index: 1
    ror: 035b05819
  - name: Program in Neuroscience and Behavioral Disorders, Duke-NUS Medical School, Singapore
    index: 2
    ror: 02j1m6098
date: 11 July 2026
bibliography: paper.bib
---

# Summary

Eunoia is a [Rust library](https://github.com/jolars/eunoia)\ [@eunoia2026] for
area-proportional Euler and Venn diagrams. Given the sizes of a collection of
sets and their intersections, it fits a diagram of circles, ellipses, squares,
or rectangles whose overlapping areas match those quantities as closely as
possible. When an exact representation does not exist, Eunoia returns the fitted
layout plus residuals and goodness-of-fit statistics that indicate whether it
can be trusted. The core is a Rust crate, and the same engine powers an [R
package (eulerr)](https://cran.r-project.org/package=eulerr), a [Python
package](https://pypi.org/project/eunoia/), a [Julia
package](https://github.com/jolars/Eunoia.jl), a [JavaScript
package](https://www.npmjs.com/package/@jolars/eunoia) compiled to WebAssembly,
a C API for further bindings, and a [web app](https://eunoia.bz/app) (that uses
the JavaScript package). Because every binding calls the same implementation,
all of them produce the same layout given the same specification, options, and
seed.

# Statement of Need

Euler diagrams\ [@euler1802] depict relationships between sets, and
area-proportional Euler diagrams do so quantitatively: the area of each region
is drawn in proportion to the size of the corresponding set intersection. They
are common in the life sciences, for instance, where they are used to compare
gene and protein lists across conditions or studies. For most specifications
involving three or more sets, however, no exact area-proportional diagram
exists\ [@wilkinson2012], and the diagram must instead be constructed
numerically\ [@chow2007]. Producing one is therefore an optimization problem
where we need to choose positions and parameters of the shapes so as to minimize
the discrepancy between the fitted and requested areas of intersection.

Existing alternatives offer different compromises. venneuler introduced
multi-dimensional scaling\ (MDS) initialization followed by overlap-aware
refinement\ [@wilkinson2012]; venn.js improved that initialization and added
exact area calculations\ [@frederickson2015]. eulerAPE introduced ellipses for
three-set diagrams\ [@micallef2014], and eulerr generalized them to any number
of sets\ [@larsson2018].

Eunoia extends eulerr by

- replacing its fitting algorithm with a multi-stage local-and-global method,
- supporting squares and rectangles (with or without rotation), and
- providing analytical gradients.

Eunoia's target audience comprises researchers and others who need trustworthy
proportional set visualizations, irrespective of whether they use R, Python,
Julia, JavaScript, Rust, or C, and developers who want to embed a diagram fitter
and visualizer in their own tools, for instance in interactive visualizations on
the web.

# State of the Field

\autoref{tab:packages} compares Eunoia with representative area-proportional
diagram fitters: venneuler\ [@wilkinson2012], eulerAPE\ [@micallef2014],
venn.js\ [@frederickson2015], matplotlib-venn\ [@tretyakov2024],
nVenn\ [@perezsilva2018], Edeap\ [@wybrow2021], and eulerr\ [@larsson2018].

  | Package           | Algorithm                | Shapes                                 | Sets | Language                                  |
  | ----------------- | ------------------------ | -------------------------------------- | :--: | ----------------------------------------- |
  | `venneuler`       | Stress minimization      | Circles                                | Any  | `Java`, `R`                               |
  | `eulerAPE`        | Hill climbing            | Ellipses                               |  3   | `Java`                                    |
  | `venn.js`         | MDS + final layout       | Circles                                | Any  | `JavaScript`                              |
  | `matplotlib-venn` | Analytic placement       | Circles                                |  3   | `Python`                                  |
  | `nVenn`           | Physics-based simulation | Polygons                               | Any  | `C++`, `R`                                |
  | `Edeap`           | Hill climbing            | Ellipses                               | Any  | `JavaScript`                              |
  | **`Eunoia`**      | MDS + final layout       | Circles, ellipses, squares, rectangles | Any  | `Rust`, `R`, `Julia`, `Python`, `C`, `JS` |

  : Related software for area-proportional Euler and Venn diagrams: fitting
    algorithm, supported shapes, maximum number of sets, and implementation
    language. \label{tab:packages}

Eunoia is a separate library rather than another host-specific eulerr backend
because eulerr's C++ implementation was coupled to R and could not serve
WebAssembly and the other bindings directly. Centralizing fitting, diagnostics,
and geometry in an independent Rust crate keeps those interfaces consistent and
avoids duplication of effort.

# Example

A diagram is specified by naming the sets and their intersections and giving the
size of each region. Below, we give an example in Rust.

```rust
use eunoia::geometry::shapes::Ellipse;
use eunoia::{DiagramSpecBuilder, Fitter, InputType};

let spec = DiagramSpecBuilder::new()
    .set("SE", 13.0)
    .set("Treat", 28.0)
    .set("Anti-CCP", 101.0)
    .set("DAS28", 91.0)
    .intersection(&["SE", "Treat"], 1.0)
    .intersection(&["SE", "DAS28"], 14.0)
    .intersection(&["Treat", "Anti-CCP"], 6.0)
    .intersection(&["SE", "Anti-CCP", "DAS28"], 1.0)
    .input_type(InputType::Exclusive)
    .build()
    .unwrap();

let layout = Fitter::<Ellipse>::new(&spec).seed(1).fit().unwrap();
```

The four sets here---SE, Treat, Anti-CCP, and DAS28---are clinical variables
from rheumatoid-arthritis research, taken from @junta2009. The fitted layout is
shown in \autoref{fig:ellipse}. Eunoia reports how well it succeeded: here the
diagError is $9 \times 10^{-5}$, which means that the diagram reproduces the
data essentially exactly. The figure also shows the automatic label placement,
which positions set labels and region quantities at poles of inaccessibility of
the region polygons and moves labels that do not fit outside the diagram.

![A four-set Euler diagram fitted with ellipses, with region quantities shown.
Eunoia fits it to a diagError of $9 \times 10^{-5}$, an essentially exact
diagram.\label{fig:ellipse}](images/euler_4set.pdf)

The bindings mirror this interface; the equivalent call in Python is:

```python
import eunoia as eu

diagram = eu.euler(
    {
        "SE": 13,
        "Treat": 28,
        "Anti-CCP": 101,
        "DAS28": 91,
        "SE&Treat": 1,
        "SE&DAS28": 14,
        "Treat&Anti-CCP": 6,
        "SE&Anti-CCP&DAS28": 1,
    },
    shape="ellipse",
)
```

The JavaScript, Julia, and R packages follow the same pattern.

Eunoia also constructs Venn diagrams, in which every intersection is drawn
regardless of whether it is empty\ (\autoref{fig:venn}).

![A canonical five-set Venn diagram drawn with
ellipses.\label{fig:venn}](images/venn5.pdf)

# Software Design

The library follows a trait-based design, where shapes are provided at compile
time. Circles, ellipses, squares, and axis-aligned and rotated rectangles are
all implemented this way, and a given specification can be fitted with each of
them. \autoref{fig:shapes} illustrates the four axis-aligned families.

![A three-set specification fitted with circles, ellipses, squares, and
rectangles. Not every shape family can represent the input exactly: the
diagError is $1.2 \times 10^{-2}$ for circles, $1.7 \times 10^{-2}$ for squares,
and $4.5 \times 10^{-2}$ for rectangles, compared to $2.3 \times 10^{-14}$ for
ellipses.\label{fig:shapes}](images/shape_families.pdf)

Fitting proceeds in two phases, roughly following the approach of
venneuler\ [@wilkinson2012], venn.js\ [@frederickson2015] and the original
eulerr package\ [@larsson2018]. The first phase computes an initial layout by
multidimensional scaling: shapes of fixed size are placed so that their pairwise
distances approximate the distances required by the pairwise intersections,
using the relaxed loss suggested by @frederickson2015 for disjoint and contained
set pairs. The second phase refines all shape parameters to minimize a loss over
the differences between the fitted and requested region areas^[the default loss
is a normalized sum of squared errors]. Region areas for every shape are
computed analytically---ellipse intersections through a projective-conic
construction\ [@richtergebert2011] and the resulting overlaps from circular and
elliptical segments in closed form\ [@eberly2016]. For circles, ellipses,
squares, and axis-aligned rectangles, the smooth losses come with exact
analytical gradients.

For shapes with analytical gradients, refinement begins with Levenberg-Marquardt
least squares, with a bounded Levenberg-Marquardt retry if the loss remains
above a threshold. If both local fits remain above the threshold, a bounded
variant of CMA-ES\ [@hansen1996] searches for another basin, which bounded
Levenberg-Marquardt then polishes. Eunoia keeps the solution with the lowest
loss. The pipeline uses several seeded restarts, which can run in parallel when the optional `parallel` feature is enabled (but remain
serial in WebAssembly). For small set counts, one restart begins from a
canonical Venn layout. Non-smooth losses---such as the maximum absolute region
error---use derivative-free methods (Nelder--Mead or mesh-adaptive direct
search) or can be replaced by a smooth surrogate.

The fitted layout is returned with its residuals, loss, and the stress (a
venneuler-style normalized least-squares measure) and diagError (the maximum
absolute difference between requested and fitted region shares) statistics, so
that the quality of the diagram can be assessed numerically rather than by eye.
A plotting module extracts the region polygons through polygon clipping and
computes label positions, which is what the R, Python, Julia, and JavaScript
packages use for rendering. Because the core has no dependency on any host
language's runtime, it compiles to WebAssembly, and the [web
app](https://eunoia.bz/app) fits diagrams entirely in the browser without a
server.

# Features

Area-proportional Euler diagrams encode the size of each set intersection as the
area of a region. Eunoia extends this with three complementary additions. Unit
glyphs show how many elements a region contains, member labels show which
elements they are, and interactive output exposes regions that are too small to
annotate (\autoref{fig:features}).

![A three-set Euler diagram fitted with ellipses, shown with glyphs (left)
and member labels (right).\label{fig:features}](images/features.pdf)

## Glyphs

Glyphs are equally sized circular marks, one per data unit, packed inside the
region they belong to, following the eulerGlyphs approach of @micallef2012.
Glyphs can also be added to Venn diagrams.

All glyphs in a diagram are the same size, so that counts can be compared across
regions. By default, Eunoia searches for the largest common size that fits every
region's count. In Python, a fixed size can be set instead, and a warning is
given if any glyphs do not fit. In Python and the web app, glyphs are kept clear
of the diagram's labels where space allows.

The web app and the JavaScript package offer four arrangements, namely a
hexagonal grid (the default), a rectangular grid, an even scatter, and a random
scatter. The Python package offers the hexagonal grid and the random scatter.

## Member Labels

Member labels add the elements themselves to the diagram, so that each region
shows which elements it contains rather than only a set name or a count. In
Python, member names are available when the input is given as member lists, or
as array or DataFrame input with identifiers. They are drawn with
`plot(members=...)` in one of two modes.

List mode *(the default)*

:   Draws each region's names as a single block of text. A block that does not
    fit inside its region is placed outside it with a leader line. If a maximum
    number of names per region is set, only that many are shown, followed by a
    "+N more" line.

Packed mode

:   Places each name individually inside its region. When space is limited, all
    names are reduced by the same factor, down to a minimum size, and names that
    still do not fit are left out with a warning.

In the web app, names are typed in for each region and drawn in packed mode. If
a region's names do not all fit, they are shown as a single block instead, as in
list mode, so no names are left out. Names are not drawn when a diagram has more
than 500 in total.

## Interactivity

The Python package provides an interactive plotly backend through
`fit.plot_plotly()`. It adds a hover tooltip to each region showing the
intersection label, its sample size, and its members. This is primarily useful
for dense multi-set diagrams, in which small regions cannot hold any text. The
JavaScript SVG serializer similarly supports native tooltips and data attributes
on each region, so that host pages can add their own interaction. The web app
provides controls for glyph mode, spacing, and styling.

# Research Impact Statement

Eunoia descends from eulerr, which has been distributed on CRAN since 2016 and
has been cited in over 700 academic publications^[Google Scholar, accessed Aug
11, 2026], predominantly in the life sciences. Since version\ 8.0, eulerr's C++
backend has been replaced by Eunoia, which means that all current eulerr users
are now also using Eunoia. Other packages build on eulerr, and therefore on
Eunoia, as a dependency, including the Bioconductor genomics packages cola,
hicVennDiagram, and seqsetvis and the CRAN package RulesTools, which use it to
draw the area-proportional diagrams in their own analyses. Six more
packages---DOTSeq, highdir, IlluminaHumanMethylationEPICv2manifest, ISAnalytics,
overviewR, and pcutils---use eulerr optionally and therefore can also invoke
Eunoia.

The Python, Julia, and JavaScript packages and the web app extend the same
underlying library to communities that previously had no access to ellipse-based
area-proportional diagrams.

# AI Usage Disclosure

The predecessor eulerr, which Eunoia builds on, was written entirely by JL,
without the use of AI. For Eunoia, JL used generative AI tools including Claude
Code Opus (versions 4.5, 4.8, and 5.0), Claude Code Fable 5, GitHub Copilot GPT
5.1 and 5.4, and Codex GPT 5.6-sol and GPT 6.1-sol, and NL used Claude Code Opus
5.5. These tools were used to assist with writing code, parts of the
documentation, unit tests, and reviewing and editing the manuscript. NL
performed user testing and bug detection independently of AI. JL and NL
reviewed, modified, and validated all AI-assisted content and made the final
design and implementation decisions.

# Acknowledgements

Eunoia builds on a decade of feedback from users of eulerr, whose bug reports
and feature requests have shaped the fitting engine, and on Peter Gustafsson's
contributions to the original eulerr package.

# References
