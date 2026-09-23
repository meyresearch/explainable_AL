# Supervisor review of the TYK2 work

Use [review_tyk2_results.ipynb](review_tyk2_results.ipynb) to inspect the full analysis. The repository is public; the research data and trained model artifacts are shared separately. Extract `supervisor_tyk2_review_bundle_20260923.zip` under `deliverables/lindsey_corrected_all_models_20260923/` at the repository root, then run the notebook. Its nine files are listed in [tyk2_artifact_manifest.csv](tyk2_artifact_manifest.csv). The notebook verifies each file's size and SHA-256 hash.

The separate `supervisor_tyk2_source_20260923.zip` contains the training, inference, comparison, and packaging scripts plus the methods log and research diary. Extract it into `supervisor_tyk2_source_20260923/` at the repository root. The notebook then checks its 18 files against [tyk2_source_manifest.csv](tyk2_source_manifest.csv). These scripts document the exact local run; several use workspace-specific paths and need the separately held source data and checkpoints to rerun.

The notebook checks the 555,332-row ranking, each model's prediction coverage, consecutive selection ranks, the selected 400, their overlap with the 1,109-ligand training set, held-out metrics, recorded training curves, and the round-1 FEP retrospective. It displays the 12-model pool distribution figure. Its checks run on saved outputs; the notebook does not retrain the research models.

## What was evaluated

The original labeled set has 753 molecules. Its frozen connectivity-aware split has 678 training and 75 test molecules. The cleaned prediction pool has 555,332 unique stereochemistry-aware SMILES after removing five labeled overlaps. The six historical full-library models have exact predictions for 547,490 of those molecules. Four adapted-S2 predictions, a five-member Chemprop MVE ensemble, and the constrained consensus cover all 555,332. Two later RF fits trained on 753 and 1,109 labels also cover the full cleaned pool. They are included for comparison; the constrained consensus supplies the delivered selection rank.

The constrained consensus was fit from connectivity-grouped out-of-fold Chemprop, adapted-S2 RF, and adapted-S2+ECFP RF predictions. Its fitted coefficients were 0.723, 0, and 0.518 with intercept 0.014. The selection score is the percentile rank of consensus predicted dG plus 0.25 times the mean percentile of Chemprop MVE variance, Chemprop ensemble variance, and model-family disagreement. Lower selection rank is better. The selected 400 are the first 400 by that rank; 33 are already in the 1,109-ligand round-1 training set.

The adapted-S2 ridge output is retained for audit but was excluded from candidate selection because of unstable pool extrapolation. The 1,109-label RF fit has seen round-1 ligands and must not be assessed on them as if they were an untouched test set. The retrospective FEP comparison is also based on selected molecules, not a prospective random sample. The original six-model 400 and the updated constrained-consensus 400 are different cohorts.

## Source map

| Stage | Source code in research workspace | Saved result in shared package |
| --- | --- | --- |
| Original six-model screen and split | `al_for_lindsey/final_dg_workflow.py`, `al_for_lindsey/train_*_final.py` | Original model columns, held-out predictions, metrics |
| Masked S2 adaptation and pool inference | `adapt_molertd_s2_masked.py`, `predict_pool_molertd_s2_masked_stack.py` | Four adapted S2 columns, training curves |
| Chemprop MVE ensemble | `run_chemprop_dg_ensemble.py`, `run_chemprop_grouped_oof.py` | Ensemble column, held-out predictions, metrics |
| Constrained ensemble and selection | `build_constrained_consensus.py` | Full ranking, selected 400, uncertainty columns |
| Round-1 FEP comparison | `compare_fep_round1.py`, `plot_all_models_vs_fep.py` | Retrospective model comparison |
| Later 753 versus 1,109 RF comparison | `compare_rf_753_vs_1109_pool.py` | Two RF columns |

The source data and model checkpoints are currently local workspace material. Before calling the pipeline independently reproducible from a fresh clone, they need a separate release and a validated environment. The two review bundles let a supervisor inspect the delivered calculations and source without implying that all training artifacts are public.
