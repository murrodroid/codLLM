# Publication thoughts

Updated: 2026-09-06.

Keep substantial decisions and unresolved ideas, not an experiment diary. Update existing entries as
the discussion develops. Results belong in [publication_progress.md](publication_progress.md);
the agreed study design belongs in [publication_plan.md](publication_plan.md).

## 1. Pretraining targets and duration

Status: full-code targets adopted; category-only pretraining rejected; the benefit of longer doses remains open.

Keep full ICD10h targets, including their category prefixes. Lucas does not want an A00-only stage.
Token-level learning can exploit the hierarchy, but it does not by itself prove hierarchical understanding.
Three-character evaluation can remain a diagnostic without changing training targets.

Retain 48 epochs alongside a matched 4-epoch comparison. Their recorded validation macro F1 differs by
only 0.000488; this does not establish equivalence or a general advantage for either duration.
Longer exposure may help descriptions or codes poorly covered by historical training, but repeated
epochs add no new descriptions by themselves. Test source transfer, verified language transfer, and
masterlist-exposure slices before claiming deeper generalization. A near-tie may justify a declared
product preference for the longer dose, not a claim that it demonstrably improves performance.

## 2. Estimate seed variability with less data

Status: implemented with whole-cohort sampling; usable fraction awaits 25%/40% calibration.

Reduce historical dataset size so several seeds can run to convergence; do not replace convergence
with a short epoch budget. Separate split/preparation variability from training-pipeline variability.

Check 25% and 40%, preserve meaningful rare/transfer support, and pin the cohort while varying split
seeds. Calibration and seed-777 anchors must use identical sampling semantics to be reusable.
Reduced-data intervals describe reduced-data behavior, not the full-data model's variance.
Reusing a pretrained checkpoint is a compute-saving option only if the result is labeled
fine-tuning-only sensitivity rather than end-to-end sensitivity.

## 3. Evaluate new descriptions, sources, and languages separately

Status: grouped splitting and language-aware metrics implemented; aggregate audit complete;
split-specific support review and external curation remain.

Use COD-only novelty, globally COD-grouped splitting, and early paired leave-one-source-out comparisons.
Retain row-split performance as a complementary setting. Age/sex differences must not create novelty.

The existing transfer metric indexes sources, not languages. Add verified label-language exposure
and report true cross-lingual eligibility separately, including whether the masterlist supplied
target-language supervision. Unknown/mixed language needs an explicit unresolved category.
If natural support is sparse, consider controlled code-language withholding as a later experiment;
do not manufacture a cross-lingual claim by renaming source transfer.

Lucas clarified that the Belgian dataset is Flemish. Treat it as Dutch (`nl`), like Amsterdam,
with the Flemish variety recorded separately. It is not a fifth language or automatically French/Dutch
mixed supervision. Reviewed exceptions would require explicit row annotations.

Use existing archive folds for development transparently. Freeze a genuinely external evaluation such
as UniCin before viewing results; availability, provenance, label mapping, and integration remain open.

## 4. Test the combined recipe and what makes it work

Status: placeholder replaced by an eight-cell grouped-COD design; mechanism follow-ups are conditional.

The revised comparison is floor 0/450, synthesis .30/.60, and full-code pretraining 4/48, with
no-pretraining and no-synthesis controls. The single-variable curve winners were measured under
different other settings and cannot simply be combined and called optimal.
Floor 450 is the strongest observed positive-floor alternative, not an assumed final choice.

Floor changes also change perturbed upsample copies; synthesis changes training exposure.
Separate those mechanisms only when the observed behavior justifies a controlled follow-up.
Natural multi-COD performance matters more than success solely on synthetically composed evaluation.
COD-only versus COD+age+sex is a useful portability comparison, not a reason to silently reinterpret
the old metadata-assisted curves.

The [completed audit](publication_data_audit.md) adds a source-selection question: Copenhagen has no
recorded natural multi-COD targets, whereas Belgium has 16.05% and Ipswich 42.28%. The Phase 2 transfer
pilots and Phase 3 mechanism tests need not use the same pair. Keeping Copenhagen tests single-COD
harm; replacing it with Ipswich tests benefit on a second multi-COD-rich source. Adding Ipswich costs
two more ablation runs. This choice remains with Lucas; the source TOMLs are unchanged.

Also consider a secondary equal-description analysis if a few recurrent strings dominate case-weighted
scores: Copenhagen has 501,855 rows but only 2,525 distinct CODs. Retain natural case-weighted results
for archival workload; do not delete legitimate repeated cases or silently change the primary estimand.

## 5. Use Bayesian search selectively

Status: optional; outside the core confirmation sequence.

The existing short-budget, single-allocation search can propose candidates, but might favor fast
starters over the best converged model. Validate reduced-fidelity ranking before relying on it and
confirm promotions at full fidelity. Freeze the evaluation contract first; never optimize on the
final external archive. Prioritize paired source comparisons over a large search.

## 6. Build a strong research tool and a coherent engineering paper

Status: adopted; deadline-first scope rejected.

The central question is how to train and evaluate a multilingual historical coding system that remains
useful on new descriptions and archives. This supports a broad account of what works and fails,
organized around generalization rather than a chronological list of experiments.

Useful open extensions include metadata robustness, uncertainty/abstention, a strong multilingual
architecture comparator, and carefully scoped expert error review. Add them when they answer a
remaining question; scale alone does not establish reliability. Choose a suitable venue when evidence,
documentation, and the model release are ready, rather than compressing the study into a 12-day target.
