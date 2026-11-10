# Quickstart: R {#quickstart-r}

[`eulerr`](https://cran.r-project.org/package=eulerr) is the R binding: the
original package, now backed by the eunoia engine. You hand `euler()` a named
vector of set sizes, it fits an area-proportional diagram, and `plot()` draws it
with grid graphics.

::: {.component name="eunoia" #docs-quickstart-r-1}

The diagram fitted by the first example below.

:::

## Install {#install}

```r
install.packages("eulerr")
```

CRAN ships binary builds, so no Rust toolchain is needed. (Installing from
source does require `rustc` ≥ 1.88; see the package README.)

## Fit a Diagram {#fit-a-diagram}

```r
library(eulerr)

fit <- euler(c(A = 5, B = 3, "A&B" = 1.5))
fit # prints the residual table and goodness-of-fit metrics
plot(fit)
```

The input is a named numeric vector. Names are **combination expressions**: a
single set (`A`) or sets joined with `&` (`"A&B"`). Values are *disjoint*
(exclusive) by default: the part of `A` outside `B` is `5`, the overlap is
`1.5`. Pass `input = "union"` if your numbers are full set sizes instead.

`fit` carries the fitted geometry and fit statistics:

```r
fit$shapes # data frame of fitted geometry for the selected shape
fit$stress # goodness-of-fit (venneuler-style)
fit$diagError # largest single-region error
```

## Pick a Different Shape {#pick-a-different-shape}

```r
fit <- euler(c(A = 5, B = 3, "A&B" = 1.5), shape = "ellipse")
plot(fit)
```

`shape` is one of `"circle"` (default), `"ellipse"`, `"square"`,
`"rectangle"`, or `"rotated_rectangle"`. Ellipses are more flexible and often
fit three or more sets noticeably better. Rotated rectangles add an angle to
the width and height and use derivative-free fitting; see
[Shapes](../../concepts/shapes/index.md).

## Customize the Plot {#customize-the-plot}

`plot()` takes styling arguments for fills, edges, labels, and quantities:

```r
plot(
  fit,
  quantities = TRUE,
  fills = c("#e41a1c", "#377eb8"),
  labels = list(fontsize = 14)
)
```

## Venn Diagrams {#venn-diagrams}

For a fixed, *non*-proportional Venn template (every intersection shown
regardless of size), use `venn()`:

```r
plot(venn(c("A", "B", "C"))) # three-set Venn
plot(venn(3)) # same, by set count
plot(venn(4, shape = "rotated_rectangle"))
```

`venn()` defaults to ellipses, supporting up to five sets. Rotated rectangles
support up to four sets.

## Next Steps {#next-steps}

Read the [introductory
vignette](https://cran.r-project.org/package=eulerr/vignettes/introduction.html)
and the full reference on the [eulerr
website](https://jolars.github.io/eulerr/). For how the fit works across all
bindings, see the [Fitter Pipeline](../../concepts/fitter-pipeline/index.md) and
[Goodness of Fit](../../concepts/goodness-of-fit/index.md) chapters.
