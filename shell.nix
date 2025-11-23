{ pkgs ? import <nixpkgs> {} }:

(pkgs.buildFHSEnv {
  name = "statistik-fhs";

  targetPkgs = pkgs: with pkgs; [
    python3Full
    stdenv.cc.cc
    zlib
  ];

  runScript = "fish";
}).env
