# deepire2.0-cade2025-paper-supplementary-materials
Supplementary materials for reproducing experiments reported in the the CADE 2025 paper "Efficient Neural Clause-Selection Reinforcement"

### Vampire

Vampire's modified executable came from the https://github.com/vprover/vampire/tree/mtpa-gnn branch (commit tagged as https://github.com/vprover/vampire/releases/tag/mtpa-gnn-cade2025).

It was compiled using the inofficial Makefile path (not via cmake). See https://github.com/vprover/vampire/blob/mtpa-gnn-cade2025/Makefile
By mistake, on linux, this Makefile hardwires the path to libtorch. You will need to install libtorch on your own (ideally version 2.5) and update this path there.

### Traning Scritps

Training scripts come from the repo https://github.com/quickbeam123/lawa, branch https://github.com/quickbeam123/lawa/tree/mtpa-gnn, tag https://github.com/quickbeam123/lawa/releases/tag/CADE2025

The main entry point is the script `elooper.py` to be run as in `elooper.py <num_loops> <num_cores>`. We used 20-30 loops and ran on servers with 128 available cores using 120 of these for the training.

This scripts looks into `hyperparams.py` for all other configuration.

### Additional Info

In the repository, under `newSplit30k` you can find a list of the used TPTP problems (`problemsSTDclean.txt`) and the used split into `train.txt` and `test.txt`.

The example `hyperparams.py` is the version used by the main experiment.

### Continuous Development

Since the submission both the vampire branch https://github.com/vprover/vampire/tree/mtpa-gnn and the training scripts branch https://github.com/quickbeam123/lawa/tree/mtpa-gnn advanced quite a bit, contains bugfixes and new (mostly undocumented) features. It's probably better to try using the most recent version (of both), unless reproducibility is the main concern.

Feel free to let me know (using github issues is an option) should you have any problems running this.

### Note

The subdirectory `tptpOverfit100` contains proofs of the 49 hard TPTP problems (rating 1.0, never re-rated), as well as the models used to find them. The first line in each proof file is how Vampire was invoked to find the proof. (This includes random_seed values and sometimes non-zero temperature settting.)