test_that("Monte Carlo area approximation approximates the real solution", {
  s <- c(
    A = 1,
    B = 2,
    C = 3,
    "A&B" = 0.2,
    "A&C" = 0.1,
    "B&C" = 0.3,
    "A&B&C" = 0.01
  )
  fit <- euler(s, shape = "circle")

  set.seed(1)

  circles <- t(as.matrix(coef(fit)))

  exact <- eulerr:::intersect_ellipses(circles, FALSE, FALSE)
  approx <- eulerr:::intersect_ellipses(circles, FALSE, TRUE)

  expect_equal(exact, approx, tolerance = 1e-3)
})

circles <- data.frame(
  A = c(-1, 1, 2, 2, 0),
  B = c(0, 0, 1, -1, 0),
  C = c(2, 2, 2, 2, 3),
  row.names = c("h", "k", "a", "b", "phi")
)
eulerr_areas <- eulerr:::intersect_ellipses(as.matrix(circles), FALSE, FALSE)
