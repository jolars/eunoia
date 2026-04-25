{
  pkgs,
  ...
}:
{
  packages = with pkgs; [
    go-task
    llvmPackages.bintools
    cargo-llvm-cov
    cargo-flamegraph
    cargo-audit
    cargo-deny
    nodejs
    wasm-pack
    bashInteractive
    perf
    go-task
    quartoMinimal
    shfmt
  ];

  languages = {
    rust = {
      enable = true;
      toolchainFile = ./rust-toolchain.toml;
    };

    javascript = {
      enable = true;
    };

    typescript = {
      enable = true;
    };

    r = {
      enable = true;
      package = (
        pkgs.rWrapper.override {
          packages = with pkgs.rPackages; [
            GenSA
            polyclip
            polylabelr
            Rcpp
            RcppArmadillo
            RConics
            devtools
            languageserver
            SLOPE
            tidyverse
            usethis
            testthat
          ];
        }
      );
    };
  };

  git-hooks = {
    hooks = {
      clippy = {
        enable = true;

        settings = {
          allFeatures = true;
        };
      };

      rustfmt = {
        enable = true;
      };
    };
  };
}
