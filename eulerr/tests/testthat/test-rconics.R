library(RConics)

c1 <- matrix(c(0.04, 0, 0, 0, 0.1111111, 0, 0, 0, -1), nrow = 3, byrow = TRUE)
c2 <- matrix(c(0.1111111, 0, 0, 0, 0.25, 0, 0, 0, -1), nrow = 3, byrow = TRUE)
# intersection conic C with conic C2
intersectConicConic(c1, c2)
