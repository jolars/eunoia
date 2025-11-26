#!/usr/bin/env Rscript

# Script to generate test cases from eulerr for comparison with eunoia
# This creates multiple circle configurations and computes their areas using eulerr

library(eulerr)

# Helper function to format circle data for Rust tests
format_circle <- function(h, k, r, name) {
  sprintf('Circle::new(Point::new(%.10f, %.10f), %.10f)', h, k, r)
}

# Helper function to format areas for Rust tests
format_areas <- function(areas, set_names) {
  result <- character()
  n <- length(set_names)
  
  # Single sets
  for (i in seq_along(set_names)) {
    result <- c(result, sprintf('(Combination::new(&["%s"]), %.10f),', set_names[i], areas[i]))
  }
  
  # All combinations - generate them properly
  idx <- n + 1
  
  # Pairwise intersections
  for (i in 1:(n-1)) {
    for (j in (i+1):n) {
      if (idx <= length(areas)) {
        result <- c(result, sprintf('(Combination::new(&["%s", "%s"]), %.10f),', 
                                    set_names[i], set_names[j], areas[idx]))
        idx <- idx + 1
      }
    }
  }
  
  # Three-way intersections
  if (n >= 3) {
    for (i in 1:(n-2)) {
      for (j in (i+1):(n-1)) {
        for (k in (j+1):n) {
          if (idx <= length(areas)) {
            result <- c(result, sprintf('(Combination::new(&["%s", "%s", "%s"]), %.10f),', 
                                        set_names[i], set_names[j], set_names[k], areas[idx]))
            idx <- idx + 1
          }
        }
      }
    }
  }
  
  # Four-way intersections
  if (n >= 4) {
    for (i in 1:(n-3)) {
      for (j in (i+1):(n-2)) {
        for (k in (j+1):(n-1)) {
          for (l in (k+1):n) {
            if (idx <= length(areas)) {
              result <- c(result, sprintf('(Combination::new(&["%s", "%s", "%s", "%s"]), %.10f),', 
                                          set_names[i], set_names[j], set_names[k], set_names[l], areas[idx]))
              idx <- idx + 1
            }
          }
        }
      }
    }
  }
  
  # Five-way intersection
  if (n >= 5 && idx <= length(areas)) {
    result <- c(result, sprintf('(Combination::new(&["%s", "%s", "%s", "%s", "%s"]), %.10f),', 
                                set_names[1], set_names[2], set_names[3], set_names[4], set_names[5], areas[idx]))
  }
  
  result
}

# Test Case 1: Your example
cat("=== Test Case 1: Simple overlapping circles ===\n")
circles1 <- data.frame(
  A = c(-1, 1, 2, 2, 0),
  B = c(0, 0, 1, -1, 0),
  C = c(2, 2, 2, 2, 3),
  row.names = c("h", "k", "a", "b", "phi")
)
areas1 <- eulerr:::intersect_ellipses(as.matrix(circles1), FALSE, FALSE)
cat("Circles:\n")
print(circles1)
cat("\nAreas:", areas1, "\n\n")

# Test Case 2: Well-separated circles
cat("=== Test Case 2: Well-separated circles ===\n")
circles2 <- data.frame(
  A = c(0, 0, 1, 1, 0),
  B = c(5, 0, 1, 1, 0),
  C = c(2.5, 4, 1, 1, 0),
  row.names = c("h", "k", "a", "b", "phi")
)
areas2 <- eulerr:::intersect_ellipses(as.matrix(circles2), FALSE, FALSE)
cat("Circles:\n")
print(circles2)
cat("\nAreas:", areas2, "\n\n")

# Test Case 3: Highly overlapping circles
cat("=== Test Case 3: Highly overlapping circles ===\n")
circles3 <- data.frame(
  A = c(0, 0, 2, 2, 0),
  B = c(1, 0, 2, 2, 0),
  C = c(0.5, 0.866, 2, 2, 0),
  row.names = c("h", "k", "a", "b", "phi")
)
areas3 <- eulerr:::intersect_ellipses(as.matrix(circles3), FALSE, FALSE)
cat("Circles:\n")
print(circles3)
cat("\nAreas:", areas3, "\n\n")

# Test Case 4: One small circle inside intersection
cat("=== Test Case 4: Nested configuration ===\n")
circles4 <- data.frame(
  A = c(0, 0, 3, 3, 0),
  B = c(2, 0, 3, 3, 0),
  C = c(1, 0, 0.5, 0.5, 0),
  row.names = c("h", "k", "a", "b", "phi")
)
areas4 <- eulerr:::intersect_ellipses(as.matrix(circles4), FALSE, FALSE)
cat("Circles:\n")
print(circles4)
cat("\nAreas:", areas4, "\n\n")

# Test Case 5: From the original failing test (for verification)
cat("=== Test Case 5: Original failing test ===\n")
s <- c(
  A = 1,
  B = 2,
  C = 3,
  "A&B" = 0.2,
  "A&C" = 0.1,
  "B&C" = 0.3,
  "A&B&C" = 0.01
)
set.seed(1)
fit <- euler(s, shape = "circle")
circles5 <- t(as.matrix(coef(fit)))
areas5 <- eulerr:::intersect_ellipses(circles5, FALSE, FALSE)
cat("Circles:\n")
print(circles5)
cat("\nAreas:", areas5, "\n\n")

# Test Case 6: Four circles
cat("=== Test Case 6: Four circles - square arrangement ===\n")
circles6 <- data.frame(
  A = c(0, 0, 1.5, 1.5, 0),
  B = c(2, 0, 1.5, 1.5, 0),
  C = c(0, 2, 1.5, 1.5, 0),
  D = c(2, 2, 1.5, 1.5, 0),
  row.names = c("h", "k", "a", "b", "phi")
)
areas6 <- eulerr:::intersect_ellipses(as.matrix(circles6), FALSE, FALSE)
cat("Circles:\n")
print(circles6)
cat("\nAreas:", areas6, "\n\n")

# Test Case 7: Four circles - one in center
cat("=== Test Case 7: Four circles - one in center ===\n")
circles7 <- data.frame(
  A = c(-2, 0, 1.5, 1.5, 0),
  B = c(2, 0, 1.5, 1.5, 0),
  C = c(0, 2, 1.5, 1.5, 0),
  D = c(0, 0, 1, 1, 0),
  row.names = c("h", "k", "a", "b", "phi")
)
areas7 <- eulerr:::intersect_ellipses(as.matrix(circles7), FALSE, FALSE)
cat("Circles:\n")
print(circles7)
cat("\nAreas:", areas7, "\n\n")

# Test Case 8: Five circles - circular arrangement
cat("=== Test Case 8: Five circles - circular arrangement ===\n")
# Create 5 circles in a circular pattern
angles <- seq(0, 2*pi, length.out = 6)[1:5]
radius_from_center <- 2
circle_radius <- 1.5
circles8 <- data.frame(
  A = c(radius_from_center * cos(angles[1]), radius_from_center * sin(angles[1]), circle_radius, circle_radius, 0),
  B = c(radius_from_center * cos(angles[2]), radius_from_center * sin(angles[2]), circle_radius, circle_radius, 0),
  C = c(radius_from_center * cos(angles[3]), radius_from_center * sin(angles[3]), circle_radius, circle_radius, 0),
  D = c(radius_from_center * cos(angles[4]), radius_from_center * sin(angles[4]), circle_radius, circle_radius, 0),
  E = c(radius_from_center * cos(angles[5]), radius_from_center * sin(angles[5]), circle_radius, circle_radius, 0),
  row.names = c("h", "k", "a", "b", "phi")
)
areas8 <- eulerr:::intersect_ellipses(as.matrix(circles8), FALSE, FALSE)
cat("Circles:\n")
print(circles8)
cat("\nAreas:", areas8, "\n\n")

# Test Case 9: Five circles - cross arrangement with center
cat("=== Test Case 9: Five circles - cross with center ===\n")
circles9 <- data.frame(
  A = c(0, 0, 1, 1, 0),      # Center
  B = c(1.2, 0, 0.8, 0.8, 0),   # Right
  C = c(-1.2, 0, 0.8, 0.8, 0),  # Left
  D = c(0, 1.2, 0.8, 0.8, 0),   # Top
  E = c(0, -1.2, 0.8, 0.8, 0),  # Bottom
  row.names = c("h", "k", "a", "b", "phi")
)
areas9 <- eulerr:::intersect_ellipses(as.matrix(circles9), FALSE, FALSE)
cat("Circles:\n")
print(circles9)
cat("\nAreas:", areas9, "\n\n")

# Generate Rust test code
cat("\n=== RUST TEST CODE ===\n\n")

test_cases <- list(
  list(name = "simple_overlapping", circles = circles1, areas = areas1),
  list(name = "well_separated", circles = circles2, areas = areas2),
  list(name = "highly_overlapping", circles = circles3, areas = areas3),
  list(name = "nested_configuration", circles = circles4, areas = areas4),
  list(name = "original_failing", circles = circles5, areas = areas5),
  list(name = "four_circles_square", circles = circles6, areas = areas6),
  list(name = "four_circles_center", circles = circles7, areas = areas7),
  list(name = "five_circles_circular", circles = circles8, areas = areas8),
  list(name = "five_circles_cross", circles = circles9, areas = areas9)
)

for (tc in test_cases) {
  cat(sprintf("#[test]\nfn test_eulerr_comparison_%s() {\n", tc$name))
  
  set_names <- colnames(tc$circles)
  n_sets <- length(set_names)
  
  for (i in seq_along(set_names)) {
    cat(sprintf("    let c%d = %s; // %s\n", 
                i, 
                format_circle(tc$circles[1, i], tc$circles[2, i], tc$circles[3, i], set_names[i]),
                set_names[i]))
  }
  
  cat(sprintf("    let circles = vec![%s];\n\n", 
              paste0("c", 1:n_sets, collapse = ", ")))
  cat(sprintf("    let areas = compute_exclusive_areas_from_layout(\n"))
  cat(sprintf("        &circles,\n"))
  cat(sprintf("        &[%s],\n", paste(sprintf('"%s".to_string()', set_names), collapse = ", ")))
  cat(sprintf("    );\n\n"))
  cat(sprintf("    let expected_areas = vec![\n"))
  
  formatted_areas <- format_areas(tc$areas, set_names)
  for (area_line in formatted_areas) {
    cat(sprintf("        %s\n", area_line))
  }
  
  cat(sprintf("    ];\n\n"))
  cat(sprintf("    for (combo, expected) in expected_areas {\n"))
  cat(sprintf("        let computed = areas.get(&combo).copied().unwrap_or(0.0);\n"))
  cat(sprintf("        let error = if expected > 1e-10 {\n"))
  cat(sprintf("            (computed - expected).abs() / expected\n"))
  cat(sprintf("        } else {\n"))
  cat(sprintf("            (computed - expected).abs()\n"))
  cat(sprintf("        };\n"))
  cat(sprintf("        assert!(\n"))
  cat(sprintf("            error < 0.01,\n"))
  cat(sprintf('            "Area for {{:?}} should match: {} vs {}",\n'))
  cat(sprintf("            combo.sets(),\n"))
  cat(sprintf("            computed,\n"))
  cat(sprintf("            expected\n"))
  cat(sprintf("        );\n"))
  cat(sprintf("    }\n"))
  cat(sprintf("}\n\n"))
}
