{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    # nix-ros-overlay.url = "github:lopsided98/nix-ros-overlay/master";
    flake-utils.url = "github:numtide/flake-utils";
    # nixpkgs.follows = "nix-ros-overlay/nixpkgs";
  };
  nixConfig = {
    # extra-substituters = ["https://ros.cachix.org"];
    # extra-trusted-public-keys = ["ros.cachix.org-1:dSyZxI8geDCJrwgvCOHDoAfOm5sV1wCPjBkKL+38Rvo="];
  };
  outputs = {
    self,
    nixpkgs,
    flake-utils,
    # nix-ros-overlay,
  }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = import nixpkgs {
          inherit system;
          # overlays = [nix-ros-overlay.overlays.default];
        };
        buildInputs = with pkgs; [
          # stdenv.cc.cc
          # libuv
          nodejs
          openssl
          zlib
          libGL
          glib
          cmake
          mkcert
          vulkan-headers
          vulkan-loader
          vulkan-tools
          pcl
          eigen
          boost
          ghc_filesystem
          opencv
          libjpeg_turbo
          #xorg.libX1Z
          xorg.libX11

          SDL2
          SDL2_image
          SDL2_mixer
          SDL2_ttf
          alsa-lib
          libpulseaudio
          xorg.libXext
          xorg.libXrandr
          xorg.libXrender
          xorg.libXfixes
          xorg.libXdamage
          xorg.libXcomposite
          xorg.libXcursor
          xorg.libXi
          xorg.libXinerama
          xorg.libXxf86vm
        ];
      in {
        devShells.default = pkgs.mkShell {
          packages = [
            pkgs.micromamba
            pkgs.cmake
            pkgs.pcl
            pkgs.basedpyright
            pkgs.cyclonedds
            pkgs.openssl

            # pkgs.colcon
            # (with pkgs.rosPackages.humble;
            #   buildEnv {
            #     paths = [
            #       ros-core
            #     ];
            #   })
          ];
          shellHook = ''
            export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath buildInputs}:$LD_LIBRARY_PATH
            export CMAKE_PREFIX_PATH=${pkgs.pcl}/share/pcl-1.14:${pkgs.eigen}/share/eigen3/cmake:${pkgs.boost}/share/boost/cmake:${pkgs.opencv}/lib/cmake/opencv4:$CMAKE_PREFIX_PATH
            export CYCLONEDX_URI="<CycloneDX><Domain><General><NetworkInterfaceAddress>192.168.123.123</NetworkInterfaceAddress></General></Domain></CycloneDX>"
            # Ensure display is available
            export DISPLAY=''${DISPLAY:-:0}
            set -e
            eval "$(micromamba shell hook --shell zsh)"
            if ! test -d ~/.cache/micromamba/envs/tv; then
              micromamba create --yes -q -n tv python==3.8
            fi
            micromamba activate tv
            set +e
          '';
        };
      }
    );
}
