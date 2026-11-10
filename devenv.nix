{
  config,
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
    google-lighthouse
    chromium
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
    uv
  ];

  env.LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
    pkgs.stdenv.cc.cc.lib
    pkgs.zlib
  ];

  # Run the lockfile's Workerd with Nix's loader, keeping its protocol version
  # matched to Wrangler without relying on a system-wide nix-ld installation.
  env.MINIFLARE_WORKERD_PATH = pkgs.writeShellScript "eunoia-workerd" ''
    set -euo pipefail
    eunoia_workerd="$(${pkgs.nodejs}/bin/node -e '
      const { createRequire } = require("node:module");
      const project = createRequire(process.argv[1]);
      const wrangler = createRequire(project.resolve("wrangler/package.json"));
      process.stdout.write(wrangler("workerd").default);
    ' "${config.env.DEVENV_ROOT}/web/package.json")"
    exec ${pkgs.stdenv.cc.bintools.dynamicLinker} "$eunoia_workerd" "$@"
  '';

  languages = {
    python.enable = true;

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
