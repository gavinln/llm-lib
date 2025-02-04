# vim: sw=2 ts=2 sts=2 et
{
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

  outputs = { self, nixpkgs }:
    let
      supportedSystems = [ "x86_64-linux" ];
      forAllSystems = nixpkgs.lib.genAttrs supportedSystems;
      pkgs = forAllSystems (system: nixpkgs.legacyPackages.${system});
    in {
      # used by: nix develop
      devShells = forAllSystems (system:
        let
          pythonEnv = pkgs.${system}.python313.withPackages
            (ps: with ps; [ isort black vulture ]);
        in {
          default = pkgs.${system}.mkShellNoCC {
            packages = with pkgs.${system}; [
              bashInteractive # for nested interactive shells: poetry shell
              nixfmt-rfc-style
              poetry
              pre-commit
              pythonEnv
              ruff
            ];
            shellHook = ''
              # set SHELL to interactive bash
              export SHELL=`which bash`
            '';
          };
        });
      # not supported
      # nix build
      # nix run
      # nix flake check
    };
}
