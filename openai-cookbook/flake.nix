# vim: sw=2 ts=2 sts=2 et

# Setup the environment
# nix flake lock  # create the flake.lock file
# nix develop  # setup the development environment

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
              pythonEnv
              ruff
              # poetry  # does not work. use uv tool install poetry
              # pre-commit  # does not work. use uv tool install pre-commit
            ];
            shellHook = ''
              # set SHELL to interactive bash
              export SHELL=`which bash`
              export UV_PROJECT_ENVIRONMENT=~/.cache/venv/openai-cookbook
            '';
          };
        });
      # not supported
      # nix build
      # nix run
      # nix flake check
    };
}
