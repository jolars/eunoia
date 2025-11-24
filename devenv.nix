{
  pkgs,
  ...
}:

{
  packages = with pkgs; [
    go-task
    llvmPackages.bintools
    cargo-llvm-cov
    prettierd
    nodejs
  ];

  languages = {
    rust = {
      enable = true;
    };

    javascript = {
      enable = true;
    };

    typescript = {
      enable = true;
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
