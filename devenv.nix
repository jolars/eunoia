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
    cargo-msrv
    nodejs
    gnuplot
    samply
    pprof
    cargo-show-asm
    wasm-pack
    bashInteractive
    perf
    shfmt
    biome
  ];

  languages = {
    rust = {
      enable = true;
      channel = "stable";
      version = "1.88.0";
      targets = [ "wasm32-unknown-unknown" ];
    };

    julia = {
      enable = true;
    };

    javascript = {
      enable = true;

      corepack.enable = true;
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

      biome = {
        enable = true;
      };

      # Svelte is not handled by Biome; format it with prettier-plugin-svelte
      # using the web/ project's local prettier (and its .prettierrc.json).
      prettier-svelte = {
        enable = true;
        name = "prettier (svelte)";
        entry = "web/node_modules/.bin/prettier --write";
        files = "^web/.*\\.svelte$";
        pass_filenames = true;
      };
    };
  };
}
