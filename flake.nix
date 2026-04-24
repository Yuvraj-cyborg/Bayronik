{
  description = "Bayronik development shell: Rust/WASM + Python/PyTorch tooling";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };

        darwinPackages = pkgs.lib.optionals pkgs.stdenv.isDarwin [
          pkgs.libiconv
          pkgs.darwin.apple_sdk.frameworks.CoreServices
          pkgs.darwin.apple_sdk.frameworks.Security
          pkgs.darwin.apple_sdk.frameworks.SystemConfiguration
        ];
      in
      {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            # Rust + WASM
            rustc
            cargo
            clippy
            rustfmt
            wasm-pack
            binaryen

            # Python workflow
            python313
            uv

            # Native build/link tooling
            pkg-config
            cmake
            openssl
            git
            wget
          ] ++ darwinPackages;

          RUST_BACKTRACE = "1";
          UV_LINK_MODE = "copy";

          shellHook = ''
            echo "Bayronik dev shell"
            echo "  Rust:   $(rustc --version)"
            echo "  Python: $(python --version)"
            echo ""
            echo "Common commands:"
            echo "  make help          # all targets"
            echo "  make build         # all Rust crates (release)"
            echo "  make test          # engine + registry + fast model"
            echo "  make server        # HTTP inference on :8000"
            echo "  make client        # native egui app"
            echo "  make wasm          # WASM client on :8080"
            echo ""
            echo "tch crates (server, infer) need libtorch from model/.venv:"
            echo "  export LIBTORCH_USE_PYTORCH=1 LIBTORCH_BYPASS_VERSION_CHECK=1"
          '';
        };
      });
}
