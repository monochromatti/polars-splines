{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs";
    devenv.url = "github:cachix/devenv";
  };

  outputs = inputs@{ flake-parts, nixpkgs, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      imports = [
        inputs.devenv.flakeModule
      ];
      systems = nixpkgs.lib.systems.flakeExposed;

      perSystem = { config, self', inputs', pkgs, system, ... }: {
        devenv.shells.default = {
          packages = with pkgs; [ maturin uv ];
          languages.rust.enable = true;
        };
      };
    };
}
