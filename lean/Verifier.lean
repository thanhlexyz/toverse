import Noise

namespace Toverse

variable {X T : Type*} {k : ℕ}

def FeasibleCex (G : T → Fin k → ℝ) (R : Set T) (ρ : ℝ) (D : Set X) (bad : X → T → Prop)
    (x : X) (r : T) : Prop :=
  x ∈ D ∧ r ∈ Feasible G R ρ ∧ bad x r

structure Bounder (G : T → Fin k → ℝ) (bad : X → T → Prop) where
  verifies : Set (X × T) → (Fin k → ℝ) → (Fin k → ℝ) → Prop
  sound : ∀ S l h, verifies S l h → ∀ p ∈ S, G p.2 ∈ Set.Icc l h → ¬ bad p.1 p.2

inductive Result (X T : Type*)
  | verified
  | falsified (x : X) (r : T)
  | timeout

structure Search (G : T → Fin k → ℝ) (R : Set T) (ρ : ℝ) (D : Set X) where
  ι : Type
  finite : Finite ι
  sub : ι → Set (X × T)
  cover : D ×ˢ R ⊆ ⋃ j, sub j
  l : ι → Fin k → ℝ
  h : ι → Fin k → ℝ
  sound : ∀ j, ∀ p ∈ sub j, p.2 ∈ Feasible G R ρ → G p.2 ∈ Set.Icc (l j) (h j)

inductive Run (G : T → Fin k → ℝ) (R : Set T) (ρ : ℝ) (D : Set X) (bad : X → T → Prop)
    (B : Bounder G bad) (nL nU : Fin k → ℝ) : Result X T → Prop
  | falsified {x : X} {r : T} (hx : x ∈ D) (hr : r ∈ R) (hbad : bad x r)
      (hpow : ∑ i, G r i ^ 2 ≤ ρ) : Run G R ρ D bad B nL nU (.falsified x r)
  | verified (s : Search G R ρ D)
      (hall : ∀ j, B.verifies (s.sub j) (s.l j ⊔ nL) (s.h j ⊓ nU)) :
      Run G R ρ D bad B nL nU .verified
  | timeout : Run G R ρ D bad B nL nU .timeout

variable {G : T → Fin k → ℝ} {R : Set T} {ρ : ℝ} {D : Set X} {bad : X → T → Prop}
  {B : Bounder G bad} {nL nU : Fin k → ℝ}

theorem falsified_feasible {x : X} {r : T}
    (hrun : Run G R ρ D bad B nL nU (.falsified x r)) : FeasibleCex G R ρ D bad x r := by
  cases hrun with
  | falsified hx hr hbad hpow => exact ⟨hx, ⟨hr, hpow⟩, hbad⟩

theorem verified_no_cex (hb : Bounds G R ρ nL nU)
    (hrun : Run G R ρ D bad B nL nU .verified) :
    ¬ ∃ x r, FeasibleCex G R ρ D bad x r := by
  rintro ⟨x, r, hx, hr, hbad⟩
  cases hrun with
  | verified s hall =>
    obtain ⟨j, hj⟩ := Set.mem_iUnion.1 (s.cover (Set.mk_mem_prod hx hr.1))
    have hn : G r ∈ Set.Icc (s.l j ⊔ nL) (s.h j ⊓ nU) := by
      rw [← Set.Icc_inter_Icc]
      exact ⟨s.sound j (x, r) hj hr, noise_subset_Icc hb ⟨r, hr, rfl⟩⟩
    exact B.sound _ _ _ (hall j) (x, r) hj hn hbad

theorem sound (hb : Bounds G R ρ nL nU) {res : Result X T} (hrun : Run G R ρ D bad B nL nU res) :
    (∀ x r, res = .falsified x r → FeasibleCex G R ρ D bad x r) ∧
      (res = .verified → ¬ ∃ x r, FeasibleCex G R ρ D bad x r) :=
  ⟨fun _ _ h => falsified_feasible (h ▸ hrun), fun h => verified_no_cex hb (h ▸ hrun)⟩

end Toverse
