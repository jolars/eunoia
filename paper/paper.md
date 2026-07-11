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

Eunoia is a [Rust library](https://github.com/jolars/eunoia) for
area-proportional Euler and Venn diagrams. Given the sizes of a collection of
sets and their intersections, it fits a diagram of circles, ellipses, squares,
or rectangles whose overlapping areas match those quantities as closely as
possible. When a specification admits an exact diagram, the optimizer typically
finds it. When it does not---the common case for four or more sets---Eunoia
returns the closest approximation it can find, together with residuals and
goodness-of-fit statistics that indicate whether the diagram can be trusted. The
core is a Rust crate, and the same engine powers an [R package
(eulerr)](https://cran.r-project.org/package=eulerr), a [Python
package](https://pypi.org/project/eunoia/), a [Julia
package](https://github.com/jolars/Eunoia.jl), a [JavaScript
package](https://www.npmjs.com/package/@jolars/eunoia) compiled to WebAssembly,
a C API for further bindings, and a [web app](https://eunoia.bz/app)\ (that uses
the JavaScript package). Because every binding calls the same implementation,
all of them produce the same layout from the same specification.

# Statement of Need

Euler diagrams\ [@euler1802] depict relationships between sets, and
area-proportional Euler diagrams do so quantitatively: the area of each region
is drawn in proportion to the size of the corresponding set intersection. They
are for instance common in the life sciences, where they are used to compare
gene and protein lists across conditions or studies. For most specifications
involving three or more sets, however, no exact area-proportional diagram
exists\ [@wilkinson2012], and the diagram must instead be constructed
numerically\ [@chow2007]. Producing one is therefore an optimization problem
where we need to choose positions and parameters of the shapes so as to minimize
the discrepancy between the fitted and requested areas of intersection.

Fitting Euler diagrams has been an active research area, starting roughly with
the seminal work by @wilkinson2012, who wrote and published venneuler for R.
venneuler took a principled two-step approach two fitting Euler diagrams, where
an initial layout based on multi-dimensional scaling\ (MDS) is refined in a
later optimization step that accounts for all overlaps in the layout. This
approach has since evolved. @frederickson2015's venn.js added a refinement of
the MDS step, which enables better initial configurations.
eulerAPE\ [@micallef2014], meanwhile, introduced ellipses for up to three sets,
which are able to accurately fit a larger number of set combinations. This lets
the user trade accuracy against legibility: circles are the most familiar and
often the easiest to read\ [@blake2016], while ellipses can fit configurations
that circles cannot.

Unlike venneuler, which uses quad-tree approximations for the intersection
areas, both EulerAPE and venn.js also introduced exact area computations.
eulerr\ [@larsson2018] in turn extended the ellipse-based approach from EulerAPE
to any number of sets.

This brings us to Eunoia, which has improved upon eulerr by

- introducing a new, more robust algorithm for fitting the diagram,
- expanded the set of shapes from circles and ellipes to squares and rectangles
  as well, and
- improved performance by relying on analytical gradients.

Eunoia's target audience is researchers or anyone else who need trustworthy
proportional set visualizations, irrespective of whether they use R, Python,
Julia, JavaScript, Rust, or C, and developers who want to embed a diagram fitter
in their own tools, for instance in interactive visualizations on the web.

# State of the Field

Several programs fit area-proportional diagrams and we summarize a selection of
them in \autoref{tab:packages}. venneuler\ [@wilkinson2012] fits circles for any
number of sets and reports a stress statistic. eulerAPE\ [@micallef2014]
introduced ellipses but is limited to three-set Venn diagrams in which all
intersections are present. venn.js\ [@frederickson2015] fits circles in
JavaScript, and matplotlib-venn\ [@tretyakov2024] draws two- and three-set
circle diagrams in Python. nVenn\ [@perezsilva2018] produces quasi-proportional
diagrams with polygonal set boundaries for any number of sets, and
Edeap\ [@wybrow2021] fits ellipses for any number of sets in a web application.
eulerr [@larsson2018] fits both circles and ellipses for any number of sets and
reports stress and diagError diagnostics.

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
    language.\label{tab:packages}

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
from rheumatoid-arthritis research, taken from one of eulerr's test cases. The
fitted layout is shown in \autoref{fig:ellipse}. Eunoia reports how well it
succeeded: here the diagError is $9 \times 10^{-5}$, so the diagram reproduces
the data essentially exactly. The figure also shows the automatic label placement, which positions
set labels and region quantities at poles of
inaccessibility\ [@agafonkin2016] of the region polygons and moves labels that
do not fit outside the diagram.

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
time. Circles, ellipses, squares, and rectangles are all implemented this way,
and a given specification can be fitted with each of
them\ (\autoref{fig:shapes}).

![A three-set specification fitted with circles, ellipses, squares, and
rectangles. Not every shape family can represent the input exactly: the
diagError is $1.2 \times 10^{-2}$ for circles, $1.7 \times 10^{-2}$ for squares,
and $4.5 \times 10^{-2}$ for rectangles, compared to $2.3 \times 10^{-14}$ for
ellipses.\label{fig:shapes}](images/shape_families.pdf)

Fitting proceeds in two phases, roughly following the approach of
venneuler\ [@wilkinson2012], venn.js\ [@frederickson2015] and
eulerr\ [@larsson2018]. The first phase computes an initial layout by
multidimensional scaling: shapes of fixed size are placed so that their pairwise
distances approximate the distances required by the pairwise intersections,
using the relaxed loss suggested by @frederickson2015 for disjoint and contained
set pairs. The second phase refines all shape parameters to minimize a loss over
the differences between the fitted and requested region areas; the default loss
is a normalized sum of squared errors. Region areas for every shape are computed
analytically---ellipse intersections through a projective-conic
construction\ [@richtergebert2011] and the resulting overlaps from circular and
elliptical segments in closed form\ [@eberly2016]---and the smooth losses come
with exact analytical gradients.

The default optimizer for the refinement phase is Levenberg--Marquardt, which is
specially designed to handle the least-squares residuals of the default loss. If
the loss remains above a threshold, Eunoia runs a fallback strategy, using a
bounded variant of CMA-ES\ [@hansen1996], which searches for a better basin, and
a trust-region method polishes the result, keeping whichever solution has the
lower loss. This two-phase pipeline runs for several random restarts in
parallel, and for small set counts one restart is seeded with a canonical Venn
layout, which often leads directly to an exact solution when one exists. Loss
functions that are not smooth---such as the maximum absolute region error---are
minimized with derivative-free methods (Nelder--Mead or mesh-adaptive direct
search) or can optionally be replaced by a smooth surrogate.

The fitted layout is returned with its residuals, loss, and the
stress\ [@wilkinson2012] and diagError\ [@micallef2014] statistics, so that the
quality of the diagram can be assessed numerically rather than by eye. A
plotting module extracts the region polygons through polygon clipping and
computes label positions, which is what the R, Python, Julia, and JavaScript
packages use for rendering. Because the core has no dependency on any host
language's runtime, it compiles to WebAssembly, and the [web
app](https://eunoia.bz/app) fits diagrams entirely in the browser without a
server.

# Research Impact Statement

Eunoia descends from eulerr, which has been distributed on CRAN since 2016 and
has been used in at least 600 academic publications, predominantly in the life
sciences. Since version\ 8.0, eulerr's C++ backend has been replaced by Eunoia,
which means that all current eulerr users are now also using Eunoia. Other
packages build on eulerr, and therefore on Eunoia, as a dependency, including
the Bioconductor genomics packages cola, hicVennDiagram, and seqsetvis and the
CRAN package RulesTools, which use it to draw the area-proportional diagrams in
their own analyses. Five other packages, DOTSeq,
IlluminaHumanMethylationEPICv2manifest, ISAnalytics, overviewR, and
pcutils---the latter two on BioConductor, all take optional dependencies on
Eunoia through eulerr.

The Python, Julia, and JavaScript packages and the web app extend the same
underlying library to communities that previously had no access to ellipse-based
area-proportional diagrams with fit diagnostics in their own language.

# AI Usage Disclosure

eulerr was designed, built, and maintained over many years without any help of
AI. When creating Eunoia, generative AI tools (Claude, Anthropic; Copilot,
GitHub) have been used to assist in writing parts of the code, documentation,
and editing this manuscript.

# Acknowledgements

Eunoia builds on a decade of feedback from users of eulerr, whose bug reports
and feature requests have shaped the fitting engine, and on Peter Gustafsson's
contributions to the original eulerr package.

# References
