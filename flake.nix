{
  description = "QCalEval — Vision-language model evaluation on quantum calibration plots";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        python = pkgs.python312;

        pythonEnv = python.withPackages (ps: with ps; [
          datasets
          httpx
          pillow
          huggingface-hub
        ]);

        mkBenchmark = name: scriptPath:
          pkgs.writeShellApplication {
            inherit name;
            runtimeInputs = [ pythonEnv ];
            text = ''
              exec python ${scriptPath} "$@"
            '';
          };

        zeroshot = mkBenchmark "qcaleval-zeroshot" ./benchmark_zeroshot.py;
        icl = mkBenchmark "qcaleval-icl" ./benchmark_icl.py;
        judge = mkBenchmark "qcaleval-judge" ./benchmark_judge.py;
      in
      {
        devShells.default = pkgs.mkShell {
          packages = [ pythonEnv ];
          shellHook = ''
            echo "QCalEval dev shell — Python ${python.version}"
            echo "Scripts: benchmark_zeroshot.py, benchmark_icl.py, benchmark_judge.py"
          '';
        };

        packages = {
          inherit zeroshot icl judge;
          default = zeroshot;
        };

        apps = {
          zeroshot = {
            type = "app";
            program = "${zeroshot}/bin/qcaleval-zeroshot";
          };
          icl = {
            type = "app";
            program = "${icl}/bin/qcaleval-icl";
          };
          judge = {
            type = "app";
            program = "${judge}/bin/qcaleval-judge";
          };
          default = {
            type = "app";
            program = "${zeroshot}/bin/qcaleval-zeroshot";
          };
        };
      });
}
