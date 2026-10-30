# Soundness of the verifier, in Lean 4

Machine-checked proofs behind Theorem 1 (Section IV-E, "Soundness of the
Verification") of the paper. The paper states the theorem in mathematical
notation with a proof sketch and names this folder as the full proof.

## Build

Lean `v4.33.1` and Mathlib `v4.33.1` through the shared package cache
`~/.cache/lake/packages` (see `lakefile.toml`, as in `~/Project/lean4aoi`).

    lake build                  # both files
    lake env lean Check.lean    # prints the axioms of every theorem
    make proof                  # both, from code/core/

Every `#print axioms` line must list only `propext`, `Classical.choice`
and `Quot.sound`: no `sorryAx`.

## Files and what they prove

| File | Paper | Content |
|---|---|---|
| `Noise.lean` | Eq. (1), Eq. (6), Eq. (7), Theorem 1 first and second parts | The feasible triggers `Feasible` (in the trigger range and within the power budget) and the feasible noise `Noise` (the set 𝒩). `Bounds` are coordinatewise bounds of the feasible noise, which the extreme values of Eq. (6) are (`Bounds.ofExtrema`). `noise_subset_Icc` is 𝒩 ⊆ [n_L, n_U]. `inject_subset` is Eq. (7): an interval that contains the feasible noise of a subproblem still contains it after the intersection with [n_L, n_U]. `noise_empty` is the infeasible MIP of Section V-G: no feasible trigger, no feasible noise. |
| `Verifier.lean` | Algorithm 1, Theorem 1 third part | `FeasibleCex` is a feasible counterexample. `Bounder` is the bounding step of a BaB verifier with its soundness, `Search` a finished search whose subproblems cover the input region, each with a noise interval that is sound for it, and `Run` says when Algorithm 1 returns each result. `falsified_feasible`: a falsified result carries a feasible counterexample. `verified_no_cex`: a verified result excludes every feasible counterexample. `sound` states both. |

## What is modelled and what is proved

- The generator `G : T → Fin k → ℝ` is any function from triggers to noise
  vectors; its ReLU structure and the MIP encoding of Eq. (5) are not
  modelled. Eq. (6) enters as the hypothesis that `n_L` and `n_U` are
  coordinatewise bounds of the feasible noise (`Bounds`), which the least
  and greatest values returned by the MIP are. This hypothesis also holds
  when no trigger is feasible, where the paper calls the system safe.
- The power budget is the joint constraint `∑ i, G r i ^ 2 ≤ ρ` of Eq. (1),
  as checked by `noise_constraint_fn` in `core/scenario/check_cex.py`.
- An interval is sound for a subproblem when it contains the noise of every
  feasible trigger of that subproblem (`Search.sound`); bound propagation
  returns such intervals. This is weaker than containing all of 𝒩, which a
  subproblem's interval does not after a split, and it is all that Eq. (7)
  needs.
- The bounding step of the underlying verifier is sound (`Bounder.sound`)
  and its subproblems cover the input region (`Search.cover`). Both are
  hypotheses: they are properties of NeuralSAT, on which the tool runs, and
  the paper's theorem assumes them.
- The constrained attack of Section IV-B and the BaB candidate of Line 6 are
  the same case of `Run`: a falsified result needs the pair to lie in the
  input region, to flip the prediction, and to pass the exact power check.
- Nothing is proved about termination, about the tightness of the intervals,
  or about the 60-second time budget.
