import json
import re
import pandas as pd
from tqdm import tqdm
import sys, os
import argparse
import re
import google.generativeai as genai

# put your API key here
genai.configure(api_key="XX")

SAVE_FREQ = 4

# Add the directory containing checkLEAN.py to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../LEAN_interaction')))
from checkLEAN import *

def saveData(data_list, output_file):
    # Convert back to DataFrame
    df_out = pd.DataFrame(data_list)
    # Save to CSV
    df_out.to_csv(output_file, index=False)
    # Count combinations of has_fl_proof? and fl_proof_compiles?
    counts = df_out.groupby(["has_fl_proof?", "fl_proof_compiles?", "has_nl_proof?"]).size().reset_index(name="count")
    print(counts)

def extract_nl_proof(text, TAG):
    if text is not None:
        text = str(text)
    else:
        text = ""
    START_DELIMITER = "<{}>".format(TAG)
    END_DELIMITER = "</{}>".format(TAG)
    if (START_DELIMITER in text) and (END_DELIMITER in text):
        inner_str = (text.split(START_DELIMITER)[-1].split(END_DELIMITER)[0]).strip()
        return inner_str
    return ""


def prompt_LLM(model, fl_proof, nl_statement, soln_description = None):
    prompt = """
You are an expert in formal mathematics and Lean-4 theorem proving.

Your task is to take as input a Lean-4 proof (in version v4.15.0) and produce a clear, detailed, and unambiguous informal proof in natural language. The informal proof should be precise enough that a human or another language model could reconstruct the original Lean-4 proof from it.

### Instructions
1. Proof Plan: Before writing the informal proof, analyze the Lean-4 proof and provide a structured plan.
    - Identify the main proof strategy.
    - List the key tactics and proof-steps used.
    - Highlight intermediate lemmas, subcases, or structural reasoning.
    - Summarize how these pieces fit together to establish the final proof.

2. Informal Proof Generation: After the plan, write the informal proof in natural language.
    - Be precise and avoid ambiguity.
    - Ensure the structure of reasoning mirrors the Lean-4 proof.
    - Use mathematical prose (definitions, cases, chains of inequalities, induction, etc.) to explain each step. 
    - **ABSOLUTELY DO NOT** mention or reference ANYTHING about LEAN, the LEAN proof or the corresponding tactics in the output informal proof. Prove it in natural language like a mathematician.

3. Formatting:
    - Input Lean-4 proof will be wrapped within <lean4_proof> ... </lean4_proof> tags.
    - Input informal theorem statement (in natural language) will be given within <natural_language_statement> ... </natural_language_statement> tags.
    - Sometimes, an additional solution description may be provided as input inside <solution_desc> ... </solution_desc> tags. You may optionally use this to refine or supplement your informal proof, but it is not mandatory.
    - Wrap the output informal proof between <natural_language_proof> and </natural_language_proof>.

Before producing the informal proof from the Lean-4 proof, provide a detailed understanding outlining the main proof steps (tactics) and strategies in the Lean-4 proof.
The plan should highlight key ideas, intermediate lemmas, and proof structures that will guide the construction of the final informal proof. Then output the informal proof within these tags: <natural_language_proof> and </natural_language_proof>

See Example 1, Example 2, Example 3 below — for style, structure, and level of detail.

####### Example 1 starts #######
<lean4_proof>
```lean4
import Mathlib.Analysis.Calculus.FDeriv.Add
import Mathlib.Analysis.Calculus.FDeriv.Equiv
import Mathlib.Analysis.Calculus.FDeriv.Prod
import Mathlib.Analysis.Calculus.Monotone
import Mathlib.Data.Set.Function
import Mathlib.Algebra.Group.Basic
import Mathlib.Tactic.WLOG
import Mathlib.Analysis.BoundedVariation

open eVariationOn
open scoped NNReal ENNReal Topology UniformConvergence
open Set MeasureTheory Filter
variable {{α : Type*}} [LinearOrder α] {{E : Type*}} [PseudoEMetricSpace E]

theorem eVariationOn.sum_le_of_monotoneOn_Icc (f : α → E) {{s : Set α}} {{m n : ℕ}} {{u : ℕ → α}}
(hu : MonotoneOn u (Icc m n)) (us : ∀ i ∈ Icc m n, u i ∈ s) :
(∑ i ∈ Finset.Ico m n, edist (f (u (i + 1))) (f (u i))) ≤ eVariationOn f s := by
rcases le_total n m with hnm | hmn
· simp [Finset.Ico_eq_empty_of_le hnm]
let π := projIcc m n hmn
let v i := u (π i)
calc
∑ i ∈ Finset.Ico m n, edist (f (u (i + 1))) (f (u i))
= ∑ i ∈ Finset.Ico m n, edist (f (v (i + 1))) (f (v i)) :=
Finset.sum_congr rfl fun i hi ↦ by
rw [Finset.mem_Ico] at hi
simp only [v, π, projIcc_of_mem hmn ⟨hi.1, hi.2.le⟩,
projIcc_of_mem hmn ⟨hi.1.trans i.le_succ, hi.2⟩]
_ ≤ ∑ i ∈ Finset.range n, edist (f (v (i + 1))) (f (v i)) :=
Finset.sum_mono_set _ (Nat.Iio_eq_range ▸ Finset.Ico_subset_Iio_self)
_ ≤ eVariationOn f s :=
sum_le _ _ (fun i j h ↦ hu (π i).2 (π j).2 (monotone_projIcc hmn h)) fun i ↦ us _ (π i).2
```
</lean4_proof>

<natural_language_statement>
Sum of Extended Distances on Monotone Sequence in Closed Interval is Less Than or Equal to Extended Variation on Set : For any function \( f : \alpha \to E \) where \( \alpha \) is a linearly ordered type and \( E \) is a pseudo-extended metric space, and for any set \( s \subseteq \alpha \) and any natural numbers \( m \) and \( n \) with \( m \leq n \), if \( u : \mathbb{{N}} \to \alpha \) is a function that is monotone on the closed interval \([m, n]\) and for all \( i \in [m, n] \), \( u(i) \in s \), then the sum of the extended distances \( \text{{edist}}(f(u(i+1)), f(u(i))) \) over the range \( \{{m, m+1, \ldots, n-1\}} \) is less than or equal to the extended variation of \( f \) on \( s \). Formally, this is expressed as:
\[
\sum_{{i=m}}^{{n-1}} \text{{edist}}(f(u(i+1)), f(u(i))) \leq \text{{eVariationOn}}(f, s)
\]
</natural_language_statement>

<natural_language_proof>
We start by considering two cases based on the total order of \( n \) and \( m \): either \( n \leq m \) or \( m \leq n \).

**Case 1: \( n \leq m \)**
In this case, the interval \( \{{i \mid m \leq i < n\}} \) is empty. Therefore, the sum of extended distances over this interval is trivially zero, which is less than or equal to any extended variation.

**Case 2: \( m \leq n \)**
Let \( \pi \) denote the projection function from \( \mathbb{{N}} \) onto the closed interval \([m, n]\) with respect to the order \( m \leq n \). Define a new sequence \( v \) such that \( v(i) = u(\pi(i)) \) for each \( i \in \mathbb{{N}} \).

We proceed with a chain of inequalities:

1. The sum of extended distances over the interval \( \{{i \mid m \leq i < n\}} \) for the sequence \( u \) is equal to the sum of extended distances over the same interval for the sequence \( v \), because \( v(i) = u(\pi(i)) \) and \( \pi \) maps each \( i \) in the interval to itself.

2. The sum of extended distances over the interval \( \{{i \mid m \leq i < n\}} \) for the sequence \( v \) is less than or equal to the sum of extended distances over the interval \( \{{i \mid 0 \leq i < n\}} \) for the sequence \( v \), due to the monotonicity of the sum over finite subsets.

3. The sum of extended distances over the interval \( \{{i \mid 0 \leq i < n\}} \) for the sequence \( v \) is less than or equal to the extended variation of \( f \) on \( s \), by the definition of extended variation and the monotonicity of \( u \) on the interval \([m, n]\).

Thus, in both cases, the sum of extended distances is less than or equal to the extended variation of \( f \) on \( s \), completing the proof.
</natural_language_proof>
####### Example 1 ends #######

####### Example 2 starts #######
<lean4_proof>
```lean4
import Mathlib.CategoryTheory.Comma.Over
import Mathlib.Tactic.CategoryTheory.Elementwise
import Mathlib.CategoryTheory.Comma.Presheaf

open CategoryTheory
open OverPresheafAux
open Category Opposite
variable {{C : Type u}} [Category.{{v}} C] {{A : Cᵒᵖ ⥤ Type v}}
variable {{F : (CostructuredArrow yoneda A)ᵒᵖ ⥤ Type v}} {{X : C}}

theorem CategoryTheory.OverPresheafAux.app_unitForward {{F : Cᵒᵖ ⥤ Type v}} (η : F ⟶ A) (X : Cᵒᵖ)
(p : YonedaCollection (restrictedYonedaObj η) X.unop) :
η.app X (unitForward η X.unop p) = p.yonedaEquivFst := by
simpa [unitForward] using p.snd.app_val
```
</lean4_proof>

<natural_language_statement>
For any contravariant functor \(F : C^{{op}} \to \mathit{{Type}}_v\), a morphism \(\eta : F \to A\) in the category of presheaves over \(C\), an object \(X\) in \(C^{{op}}\), and an element \(p\) in the Yoneda collection of the object obtained by restricting \(\eta\) through the unopposite of \(X\), show that applying \(\eta\) at \(X\) to the unit forward morphism determined by \(\eta\), \(X\)'s unopposite, and \(p\), is equal to \(p\)'s Yoneda equivalence first component.
</natural_language_statement>

<natural_language_proof>
We start by noting that the unit forward map \( \text{{unitForward}} \, \eta \, X \, p \) is defined as the second component of \( p \), which is an element of \( F(X^\text{{op}}) \). Specifically, \( \text{{unitForward}} \, \eta \, X \, p = p.snd.val \).

To prove the desired equality:
\[
\eta.app(X) (\text{{unitForward}} \, \eta \, X \, p) = p.yonedaEquivFst
\]
we use the definition of the unit forward map and the property of the second component of the Yoneda collection element. This simplifies our goal to:
\[
\eta.app(X) (p.snd.val) = p.yonedaEquivFst
\]
We observe that this is exactly the same as the given hypothesis \( p.snd.app_val \), which states:
\[
\eta.app(X) (p.snd.val) = p.yonedaEquivFst
\]
Thus, the proposition is trivially true, and the proof is complete.
</natural_language_proof>
####### Example 2 ends #######

####### Example 3 starts #######
<lean4_proof>
```lean4
import Mathlib.CategoryTheory.Functor.KanExtension.Basic
import Mathlib.CategoryTheory.Functor.KanExtension.Pointwise

open CategoryTheory
open Functor
open LeftExtension
open IsPointwiseLeftKanExtension
open Category Limits
variable {{C D H : Type*}} [Category C] [Category D] [Category H] (L : C ⥤ D) (F : C ⥤ H)
variable {{F L}}
variable (E : LeftExtension L F)
variable (E : LeftExtension L F)
variable (L F) in
/-- The cocones for `CostructuredArrow.proj L Y ⋙ F`, as a functor from `LeftExtension L F`. -/
@[simps]
def coconeAtFunctor (Y : D) :
LeftExtension L F ⥤ Cocone (CostructuredArrow.proj L Y ⋙ F) where
obj E := E.coconeAt Y
map {{E E'}} φ := CoconeMorphism.mk (φ.right.app Y) (fun G => by
dsimp
rw [← StructuredArrow.w φ]
simp)
variable {{E}} in
lemma IsPointwiseLeftKanExtensionAt.hasPointwiseLeftKanExtensionAt
{{Y : D}} (h : E.IsPointwiseLeftKanExtensionAt Y) :
HasPointwiseLeftKanExtensionAt L F Y := ⟨_, h⟩
variable {{E E'}}
variable (h : E.IsPointwiseLeftKanExtension)
include h

theorem CategoryTheory.Functor.LeftExtension.IsPointwiseLeftKanExtension.hom_ext
{{G : LeftExtension L F}} {{f₁ f₂ : E ⟶ G}} : f₁ = f₂ := by
ext Y
apply (h Y).hom_ext
intro X
have eq₁ := congr_app (StructuredArrow.w f₁) X.left
have eq₂ := congr_app (StructuredArrow.w f₂) X.left
dsimp at eq₁ eq₂ ⊢
simp only [assoc, NatTrans.naturality]
rw [reassoc_of% eq₁, reassoc_of% eq₂]
```
</lean4_proof>

<natural_language_statement>
Uniqueness of Morphisms from Pointwise Left Kan Extension : For any categories \( \mathcal{{C}} \), \( \mathcal{{D}} \), and \( \mathcal{{H}} \), and functors \( L : \mathcal{{C}} \to \mathcal{{D}} \) and \( F : \mathcal{{C}} \to \mathcal{{H}} \), if \( E \) is a pointwise left Kan extension of \( F \) along \( L \), then for any other left extension \( G \) of \( F \) along \( L \) and any two natural transformations \( f_1, f_2 : E \to G \), it holds that \( f_1 = f_2 \).
</natural_language_statement>

<natural_language_proof>
We start by noting that the assumption that the cardinality of \( n \) is zero is equivalent to \( n \) being an empty type. Therefore, we convert the assumption \( \left| n \right| = 0 \) into the assumption that \( n \) is empty.

To prove that the determinant of \( M \) is \( 1 \), it suffices to show that \( M \) is the identity matrix. This is because, if \( M \) is the identity matrix, then by the definition of the determinant, \(\det M = 1\).

Using extensionality, to show that \( M \) is the identity matrix, we need to show that for every \( i \in n \), \( M i = 1 i \).

Since \( n \) is empty, the statement \( M i = 1 i \) holds for every \( i \in n \) vacuously. Therefore, \( M \) is the identity matrix, and hence \(\det M = 1\).

This completes the proof. \(\blacksquare\)
</natural_language_proof>
####### Example 3 ends #######

### Actual Input
Here is the **actual** formal proof (in LEAN-4):
<lean4_proof>
```lean4
{fl_proof}
```
</lean4_proof>

Here is the **actual** theorem statement in natural language:
<natural_language_statement>
{nl_statement}
</natural_language_statement>

{additional_info}

Now, first write the proof plan. Then output the informal proof within the following tags:
<natural_language_proof>
(Your detailed natural language proof goes here.)
</natural_language_proof>
    """.strip()

    if soln_description is not None:
        additional_info = f"<solution_desc>\n{soln_description}\n</solution_desc>".strip()
    else:
        additional_info = ""

    formatted_prompt = prompt.format(
        fl_proof=fl_proof,
        nl_statement=nl_statement,
        additional_info=additional_info
    )

    print("\n", "-"*25, "PROMPT", "-"*25)
    print (formatted_prompt)
    print("-"*25, "PROMPT", "-"*25, "\n")

    nl_proof = None
    for tryNum in range(10):
        try:
            response = model.generate_content(formatted_prompt)
            nl_proof = extract_nl_proof(response.text, "natural_language_proof")
        except BaseException as e:
            print(f"Attempt {tryNum+1} failed: {e}")
            continue
        if len(nl_proof) > 0:
            break
    return nl_proof

def generate_NLproofs(model, list_of_dicts, output_file):
    #run_dir, project_dir, lean_file_path = bootstrap_project()
    for rowIndx, row in enumerate(tqdm(list_of_dicts, desc="Generating NL proofs")):
        print("\n" + "="*50)
        print(f"Row {rowIndx}")
        if row["has_nl_proof?"].strip() == "no" and row["has_fl_proof?"].strip() == "yes":
            fl_proof = row["formal_proof"].strip()
            nl_statement = row["informal_statement"].strip()
            if row["has_soln_description?"].strip() == "yes":
                soln_description = row["soln_description"].strip()
            else:
                soln_description = None
            nl_proof = prompt_LLM(model, fl_proof, nl_statement, soln_description)
            if (nl_proof is not None) and (len(nl_proof) > 0):
                row["has_nl_proof?"] = "yes_generated"
                row["informal_proof"] = nl_proof
                print("\n", "-"*25, "NL-PROOF", "-"*25)
                print (nl_proof)
                print("-"*25, "NL-PROOF", "-"*25, "\n")
            else:
                print("\n", "-"*25, "NL-PROOF", "-"*25)
                print ("Skipping, no parsable output from LLM")
                print("-"*25, "NL-PROOF", "-"*25, "\n")
        print("="*50 + "\n", flush = True)
        if rowIndx % SAVE_FREQ == 0:
            saveData(list_of_dicts, output_file)
    saveData(list_of_dicts, output_file)

if __name__ == "__main__":

    #------------------parse command line arguments------------------
    model = genai.GenerativeModel("gemini-2.5-pro")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strt_indx", 
        type=int, 
        default=0,
        help="in multiples of 1000"
    )
    args = parser.parse_args()
    strt_indx = args.strt_indx
    strt_indx_padded = f"{strt_indx:06d}"

    # Get the "AI-MO/NuminaMath-LEAN" dataset
    output_file = f"./datasets_training/NuminaMath-LEAN-PF/step3Parts/dataset_step3Part-{strt_indx_padded}.csv"
    main_dataset = f"./datasets_training/NuminaMath-LEAN-PF/step2Parts/dataset_step2Part-{strt_indx_padded}.csv"
    if os.path.exists(output_file):
        # Load the previously written file
        df_out = pd.read_csv(output_file, dtype=str).fillna("")
        data_list = df_out.to_dict(orient="records")

        # # Load the base dataset slice
        # df_base = pd.read_csv(main_dataset, dtype=str).fillna("")
        # data_base = df_base.to_dict(orient="records")[strt_indx : strt_indx + 1000]

        # # Fill missing rows/uuids
        # uuids_out = {row.get("uuid") for row in data_out if "uuid" in row}
        # merged_list = []
        # for row in data_base:
        #     if row.get("uuid") not in uuids_out or not row.get("uuid"):
        #         # take from base if missing in output
        #         merged_list.append(row)
        #     else:
        #         # take from output (overrides base)
        #         found = next((r for r in data_out if r.get("uuid") == row.get("uuid")), row)
        #         merged_list.append(found)

        # data_list = merged_list
    else:
        df = pd.read_csv(main_dataset, dtype=str).fillna("")
        data_list = df.to_dict(orient="records")
    generate_NLproofs(model, data_list, output_file)
    print ("Done!")
