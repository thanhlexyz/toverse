# Code rules

Distilled from the code rules of `mpts4aoii`, `kfold`, `vqmix` and `nnv`. Rules that were specific to another project are left out.

## Scope

- `core/` is the code of the AAAI 2027 version of the tool, then called Verca and now Toverse, moved from `../resource/code/core` (branch `aaai2027`). Only `scenario/plot/` has changed since: the font, the sizes, the labels and the tool name, for the ICC figures. The results of the paper are final and no new experiment is run unless the user asks.
- `core/abcrown` and `core/neuralsat` are vendored verifiers. Treat them as third-party baselines: never change their behavior.

## Style

- No comments and no docstrings, in Python, the makefile, YAML, Lean or `lakefile.toml`, and no commented-out code. Variable, function, class and flag names carry the meaning. Explanations live in `README.md`, `lean/README.md` and this file. The vendored `core/abcrown` and `core/neuralsat` keep theirs.
- Use single quotes for strings.
- Order import lines from longest to shortest within each import group.
- Do not implement edge cases until they actually happen. Write the straightforward path first.
- New folder and file names are singular: `test/`, `doc/`, `requirement.txt`. Stored result files keep their frozen names.
- Wording and notation follow the paper; the Overview section of `../paper/paper.tex` is the reference. The glossary below maps the paper to the code, which keeps its older names.

## Glossary

| Paper | Paper wording | Code name | Flag |
|---|---|---|---|
| \tool | Toverse | `e2e_mip` | `--verifier=e2e_mip` |
| \neuralsat | NeuralSAT | `e2e` | `--verifier=e2e` |
| \crown | $\alpha\beta$-CROWN | `abcrown` | `--verifier=abcrown` |
| \pgm, $\mathcal{G}_a$ | noise generator | `pgm` | `--trainer=pgm`, `--verifier=pgm` |
| $\mathcal{E}, \mathcal{D}, \mathcal{F}$ | semantic encoder, semantic decoder, task model | `semcom` | `--trainer=semcom` |
| $z$ | semantic features | latent | `--n_latent` |
| PNR | peak noise-to-signal ratio, in dB | `adv_snr_threshold`, stored as `pnr` | `--adv_snr_threshold` |
| $P_z$ | upper bound on the power of the benign features | `z2_ub` | |
| $\rho$ | power budget | `max_power` | |
| $[r_L, r_U]$ | trigger range | `lb`, `ub`, `LBS`, `UBS` | `experiment/*.yaml` |
| $\varepsilon$ | input bound | `linf_eps`, `LINF_EPS` | `experiment/*.yaml` |

## Frozen data schema

These names are how the results are stored, so they never change, even where the paper uses another word:

- Summary files `data/csv/{verifier}_{experiment}.csv` with columns `instance`, `onnx`, `vnnlib`, `pnr`, `n_latent`, `lb`, `ub`, `dt`, `sat`, `unsat`, `timeout`.
- Verifier keys `pgm`, `abcrown`, `e2e`, `e2e_mip`; experiment keys `fashion_mnist`, `cifar10`, `ablation_fashion_mnist`, `ablation_cifar10`.

`data/csv/` is committed and holds the paper's results. The raw logs (`data/log/`, needed by `check_cex` and the search plots) stay in the original repository, `../resource/code/data/log` on this machine. The generated folders `data/onnx/`, `data/vnnlib/`, `data/data/`, checkpoints and figures are not committed.

## Plots

`make plot` in `core/` writes the paper's result figures straight into `../../paper/figure/result/` (`FIGURE_DIR` in `core/scenario/plot/util.py`), so run it only when the figures should change. `get_dfs` in the same file applies the rules the paper reports: baseline `sat` results on FashionMNIST are the spurious attacks, and a result at or above the 60-second timeout is a timeout.

## Lean

- `lean/` holds the machine-checked proof of Theorem 1 of the paper. It uses Lean 4.33.1 (`/opt/lean`) and Mathlib `v4.33.1` through the shared package cache `~/.cache/lake/packages` (`lakefile.toml`, `lake-manifest.json` copied from `mpts4aoii`); `.lake/` is git-ignored.
- `make proof` in `core/` runs `lake build` and `lake env lean Check.lean`. Every axiom line must list only `propext`, `Classical.choice` and `Quot.sound`, never `sorryAx`.
- Check a single file with `lake env lean Noise.lean` (it elaborates and writes nothing). Run `lake build` from one agent only, and at most one Lean process at a time on this machine.
- The theorem-to-file map and the modelling assumptions are in `lean/README.md`. When the paper's Theorem 1, its proof sketch or the Lean hypotheses change, change the others in the same edit; the paper's third part assumes a sound bounding step and a covering search (`Bounder.sound`, `Search.cover`), which are hypotheses, not conclusions.

## Environment and experiments

- This machine (corsair) is an AMD Ryzen 9 5900X. The AMD machines crash under parallel load: run one heavy process at a time, at most 2 in total, with no `make -j` and at most 23 threads (`OMP_NUM_THREADS`).
- The user runs experiments by hand. Do not run experiments, training or baselines unless asked.
- Never create a virtual environment and never install or uninstall packages. When code needs a dependency, give the user the exact install command and add it to `requirement.txt`.
- Check `nvidia-smi` and `pgrep -af python` before a heavy job, and do not kill processes you did not start.
- Two jobs must never append to the same result file.

## Syncthing and git

`~/Project` is a Syncthing folder shared with other machines, and a sync once deleted this project. `code/` and `paper/` are separate git repositories, so they can be restored from their remotes. Agents commit and push only when the user asks.
