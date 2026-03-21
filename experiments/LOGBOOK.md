# Experiment Logbook

## 2026-03-19 — SDU visit, upsampling redesign, and open questions

### SDU presentation and data leakage observation

Visited SDU and presented our current results to Christian Møller Dahl, who asked why the model performs so well. This prompted a closer look at the data and we identified **inherent data leakage** as a likely contributor: in the largest classes (up to 28,767 samples), there is substantial natural duplication — many records share identical cause-of-death strings. After a random train/test split, exact copies of test inputs will appear in the training data, giving the model a memorisation shortcut for high-frequency classes. This is not an error in our pipeline (the duplicates are real historical records), but it inflates headline metrics and must be disclosed and accounted for when reporting results.

### Upsampling strategy redesign

Replaced the previous quantile-pressure upsampling (`select_upsample_targets` with `target_quantile`, `inverse_power`, `budget_ratio`) with a simpler **floor-based** approach (`select_floor_upsample_targets`). The old method had three interacting parameters whose combined effect was unintuitive — even at budget_ratio=0.9 the output barely changed because a per-class quantile cap dominated. The new method has two parameters:

| Parameter | Meaning |
|---|---|
| `floor` | Every class with fewer samples is brought up to this count |
| `decay` | Smallest original classes receive slightly less than the full floor (e.g. 10% less), preserving the original size ordering after upsampling |

The decay follows a log curve so that the reduction is concentrated on the very rarest classes. This ensures that a class with 1 original sample stays smaller than a class with 5 original samples after upsampling, maintaining the natural ordering.

**Justification:**
1. **Simpler** — two parameters instead of three, no hidden interactions
2. **Predictable** — "floor=10" means every class gets at least ~10 samples, full stop
3. **Ordering-preserving** — the decay ensures no rank inversions among minority classes

Reference: the square-root / power-law approach from Kang et al. (ICLR 2020, "Decoupling Representation and Classifier for Long-Tailed Recognition") was considered but rejected as overkill for our use case — it redistributes budget across the entire distribution, which is hard to reason about when you just want a minimum viable representation for rare classes.

### Open issues to address

**1. Inherent data leakage in high-frequency classes**

The largest classes contain many near-identical or identical records. After a random split, the model can achieve high accuracy on these classes by memorisation rather than generalisation. This does not invalidate the results (the duplicates reflect real-world data), but we need to:
- Quantify the duplication rate per class
- Consider reporting metrics with and without duplicate-contaminated test samples
- Potentially introduce a deduplication-aware split (group identical inputs together so all copies land in the same split)

**2. Single-sample classes and the stratification problem**

Many classes have only 1 sample. After splitting, these land entirely in either train or test — proper stratification is impossible without upsampling before the split, which we refuse to do (augmenting test/val data is methodologically unsound). This creates two distinct evaluation scenarios:

- **Seen classes**: classes that appear in both train and test. Standard macro recall applies.
- **Unseen classes**: classes that appear only in test (or only in train). The model has zero training signal for these.

We need two new metrics:
1. **Seen-class macro recall** — macro recall computed only over classes present in the training data. This measures how well the model generalises within its training distribution.
2. **Unseen-class recall** — recall on test samples whose class was never seen during training. This measures zero-shot generalisation and provides a lower bound on performance for rare codes.

These metrics together give a more honest picture than a single macro recall that conflates the two regimes.
