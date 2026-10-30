import Mathlib

namespace Toverse

variable {T : Type*} {k : ℕ}

def Feasible (G : T → Fin k → ℝ) (R : Set T) (ρ : ℝ) : Set T :=
  {r | r ∈ R ∧ ∑ i, G r i ^ 2 ≤ ρ}

def Noise (G : T → Fin k → ℝ) (R : Set T) (ρ : ℝ) : Set (Fin k → ℝ) :=
  G '' Feasible G R ρ

structure Bounds (G : T → Fin k → ℝ) (R : Set T) (ρ : ℝ) (nL nU : Fin k → ℝ) : Prop where
  lo : ∀ i, nL i ∈ lowerBounds ((fun r => G r i) '' Feasible G R ρ)
  hi : ∀ i, nU i ∈ upperBounds ((fun r => G r i) '' Feasible G R ρ)

theorem Bounds.ofExtrema {G : T → Fin k → ℝ} {R : Set T} {ρ : ℝ} {nL nU : Fin k → ℝ}
    (hL : ∀ i, IsLeast ((fun r => G r i) '' Feasible G R ρ) (nL i))
    (hU : ∀ i, IsGreatest ((fun r => G r i) '' Feasible G R ρ) (nU i)) :
    Bounds G R ρ nL nU :=
  ⟨fun i => (hL i).2, fun i => (hU i).2⟩

theorem noise_subset_Icc {G : T → Fin k → ℝ} {R : Set T} {ρ : ℝ} {nL nU : Fin k → ℝ}
    (hb : Bounds G R ρ nL nU) : Noise G R ρ ⊆ Set.Icc nL nU := by
  rintro _ ⟨r, hr, rfl⟩
  exact ⟨fun i => hb.lo i ⟨r, hr, rfl⟩, fun i => hb.hi i ⟨r, hr, rfl⟩⟩

theorem inject_subset {G : T → Fin k → ℝ} {R : Set T} {ρ : ℝ} {nL nU : Fin k → ℝ}
    (hb : Bounds G R ρ nL nU) {M : Set (Fin k → ℝ)} (hM : M ⊆ Noise G R ρ)
    {l h : Fin k → ℝ} (hlh : M ⊆ Set.Icc l h) :
    M ⊆ Set.Icc (l ⊔ nL) (h ⊓ nU) := by
  rw [← Set.Icc_inter_Icc]
  exact Set.subset_inter hlh (hM.trans (noise_subset_Icc hb))

theorem noise_empty {G : T → Fin k → ℝ} {R : Set T} {ρ : ℝ} (h : Feasible G R ρ = ∅) :
    Noise G R ρ = ∅ := by
  simp [Noise, h]

end Toverse
