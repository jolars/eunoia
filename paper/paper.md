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
affiliations:
  - name: Department of Mathematical Sciences, University of Copenhagen, Denmark
    index: 1
    ror: 035b05819
date: 11 July 2026
bibliography: paper.bib
---

# Summary

Eunoia is a [Rust library](https://github.com/jolars/eunoia)\ [@eunoia2026] for
area-proportional Euler and Venn diagrams. Given the sizes of a collection of
sets and their intersections, it fits a diagram of circles, ellipses, squares,
or rectangles whose overlapping areas match those quantities as closely as
possible. When an exact representation does not exist, Eunoia returns both the
fitted layout and residuals and goodness-of-fit statistics that indicate whether
it can be trusted. The core is a Rust crate, and the same engine powers an [R
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

Existing fitters offer different compromises. venneuler introduced MDS
initialization followed by overlap-aware refinement\ [@wilkinson2012]; venn.js
improved that initialization and added exact area
calculations\ [@frederickson2015]. eulerAPE introduced ellipses for three-set
diagrams\ [@micallef2014], and eulerr generalized them to any number of
sets\ [@larsson2018]. Circles remain familiar and legible\ [@blake2016], whereas
ellipses can represent more specifications accurately.

Eunoia extends eulerr by

- replacing its fitting algorithm with a multistage local-and-global pipeline,
- expanding its circles and ellipses to include squares and both axis-aligned
  and rotated rectangles, and
- providing analytical gradients for its four axis-aligned shape families.

Eunoia's target audience comprises researchers and others who need trustworthy
proportional set visualizations, irrespective of whether they use R, Python,
Julia, JavaScript, Rust, or C, and developers who want to embed a diagram fitter
in their own tools, for instance in interactive visualizations on the web.

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
    language. \*Rectangles may be axis-aligned or rotated.\label{tab:packages}

Eunoia is a separate core library rather than another host-specific eulerr
backend because eulerr's C++ implementation was coupled to R and could not serve
WebAssembly and the other bindings directly. Centralizing fitting, diagnostics,
and geometry in an independent Rust crate keeps those interfaces consistent; the
alternatives above are limited either in shape family, set count, deployment
target, or some combination of the three.

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

For shapes with analytical gradients, refinement begins with a bounded
trust-region-reflective least-squares solver. If the loss remains above a
threshold, a bounded variant of CMA-ES\ [@hansen1996] searches for another
basin, which the trust-region method then polishes; Eunoia keeps whichever
solution has the lower loss. The pipeline uses several seeded restarts, which
can run in parallel when the optional `parallel` feature is enabled (but remain
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

The predecessor eulerr, which Eunoia builds on, was written entirely by the
author, without the use of AI. For Eunoia, the author used generative AI tools
including Claude Code Opus (versions 4.5, 4.8, and 5.0), Claude Code Fable 5,
GitHub Copilot GPT 5.1 and 5.4, and Codex GPT 5.6-sol. They were used to assist
with writing code, parts of the documentation, unit tests, and reviewing and
editing the manuscript. The author reviewed, modified, and validated all
AI-assisted content and made the final design and implementation decisions.

# Acknowledgements

Eunoia builds on a decade of feedback from users of eulerr, whose bug reports
and feature requests have shaped the fitting engine, and on Peter Gustafsson's
contributions to the original eulerr package.

# References
