# Tree Search Experiment Logbook

## 2026-03-04 — Full tree search (v1): initial run, abandoned

**Setup**: Qwen2.5-7B-Instruct via LM Studio (localhost:1234), beam search
with width=3, top_choices=3, 5% sample of single-CoD Belgian records (1,764 records).

**Result: Total failure.** Every LLM call failed JSON parsing. The prompt
asks for `{"choices": [{"code": "A", "reason": "..."}]}` but Qwen 7B
consistently returns flat arrays like `["I", "J", "G"]` instead. The
`_parse_response` function expects dicts with a `"code"` key, so every
response is rejected and the system falls back to picking the first 3
children in order — effectively random.

**Performance**: ~37s per record (9-10 LLM calls each), projected 18+ hours
for the 1,764 sample. Stopped after 7 records (262s total, 70 LLM calls,
75,870 tokens).

**Root cause**: Two compounding issues:
1. **Prompt/parse mismatch**: The structured JSON format
   (`{"choices": [...]}`) is too complex for a 7B model to follow reliably.
   It ignores the schema and returns simpler formats.
2. **Speed**: ~4s per LLM call on local hardware is too slow for a
   full-depth beam search (5 levels × 3 beams = ~15 calls per record).

**Decision**: Abandon full-depth tree search for now. Simplify to
**chapter-level classification only** (25-way, single LLM call per record).
This tests whether the LLM can do the fundamental task — map CoD strings to
ICD-10 chapters — before investing in multi-level search.

---

## 2026-03-04 — Chapter classification (v2): 46% accuracy

**Setup**: Same model, simplified to 25-way chapter classification.
Single LLM call per record, prompt returns one letter. 200 samples, 0.09s/record.

**Result: 46% accuracy**, 100% parse rate.

| Chapter | Acc | n | Note |
|---------|-----|---|------|
| G,N,C,E | 80–100% | 38 | Clear medical terms → works well |
| J,B,X | 67–78% | 34 | Decent |
| I,K,T | 50–64% | 54 | Mixed |
| P,R,A | 0–3% | 68 | **Total failure — 38% of sample** |

**Top confusions**: P→O (stillbirth↔pregnancy, 14x), A→Z/B (Dutch TB terms, 21x),
R→G (sénilité→nervous system, 8x), I→T (asystolie→poisoning, 6x).

**Takeaway**: The LLM handles recognisable medical terms but fails on
(1) historical Dutch/French vocabulary, (2) ICD-10 coding conventions
where the "correct" chapter is unintuitive (R=symptoms catch-all, P vs O).

---

## 2026-03-04 — Diagnosis & prompt engineering (v2→v3)

### Diagnosis

Tested translation (Helsinki-NLP offline, LLM-as-translator) and
chain-of-thought. Translation helps marginally (~1-2 cases) but the core
problem is **ICD-10 coding convention knowledge**, not language:

- The model correctly translates "doodgeboorte" → stillbirth → perinatal,
  but still picks O (pregnancy) instead of P. **Coding convention failure.**
- "Kinkhoest" → whooping cough is correct, but maps to J (respiratory)
  instead of A (infectious). **Same issue.**
- Helsinki-NLP Dutch model is unreliable ("Long tering" → "Long f***ing shit").

### Fix: enhanced chapter descriptions (v3)

Replaced the auto-generated chapter labels (random category name soup) with
proper ICD-10 chapter descriptions including scope, key inclusions, and
disambiguation hints (e.g. "O = conditions affecting the MOTHER",
"P = conditions affecting the BABY", "R = use when no specific chapter applies").

**Result: 71.5% accuracy** (up from 46%), same speed (0.12s/record).

| Chapter | v2 | v3 | n | Note |
|---------|-----|-----|---|------|
| C,E | 80% | 100% | 25 | Already good, now perfect |
| I | 64% | 84% | 44 | Big improvement |
| J | 78% | 89% | 27 | Solid |
| P | 0% | 74% | 19 | **Massive fix** (was all P→O) |
| R | 0% | 65% | 20 | **Big fix** (was all R→G) |
| W | 0% | 100% | 3 | Fixed |
| A | 3% | 28% | 29 | Improved but still worst chapter |
| B | 67% | 0% | 3 | Regressed (B→A, only 3 samples) |

**Remaining issues**: A (28%, mostly "Long tering"=TB not recognised as
infectious), B→A confusion (A/B boundary unclear), I→G (5x).

**On translation**: Tested Helsinki-NLP (opus-mt-nl-en, opus-mt-fr-en) and
LLM-as-translator. Neither solves the problem — the model doesn't know
"Long tering", "Kinkhoest", "Roos", "Mazelen" at all (returns "not a
standard term"). A dictionary approach would overfit to this dataset; with
5+ datasets and 10+ languages coming, **fine-tuning is the right long-term
fix for coarse-level classification**.

---

## 2026-03-04 — Leaf traversal given gold chapter (v4)

**Question**: If we assume the chapter is already correct (future fine-tuned
model), can the LLM handle the remaining traversal using English descriptions?

**Setup**: Gold chapter given → LLM picks block_group → block → category →
leaf. 200 samples, ~3 LLM calls per record, same Qwen 7B model.

**Result**:

| Level | Accuracy |
|-------|----------|
| Block group (2 chars) | 47.5% |
| Block (3 chars) | 22.5% |
| Category (5 chars) | 15.0% |
| Leaf (full code) | 12.5% |

**Miss depth analysis** (of 175 misses):
- 60% go wrong at the very first step (block group)
- 29% get block group right but pick wrong block
- 9% get block right but pick wrong category
- 3% get category right but pick wrong leaf

**Key observations**:
- The **block group labels are still garbage** (same auto-generated name
  soup). This is why 60% of errors happen at the first step after chapter.
  Enhanced descriptions (like we did for chapters) would likely help.
- When the model gets to the right block, the remaining steps mostly work —
  only 12% of misses happen below block level.
- Some near-misses are defensible: C16.907 (Carcinoma, stomach) vs gold
  C16.901 (Cancer, stomach), I46.100 (Sudden cardiac death) vs gold I46.900
  (Cardiac arrest, unspecified).
- Language barrier persists: "Long tering" → A49 (bacterial infection
  unspecified) instead of A16 (tuberculosis).

**Conclusion**: The tree traversal approach has potential but needs:
1. **Fine-tuned coarse classifier** for chapter level (language barrier)
2. **Better node descriptions** at all levels (not just chapter)
3. Possibly **enhanced descriptions at block_group level** like we did for
   chapters — this is where most errors cascade from

---

## 2026-03-04 — H-part final disambiguation given gold category (v5)

**Question**: Assuming a future system gets us to the correct ICD-10 category
(e.g. A01.0), can the LLM pick the right historical synonym (h-part) from
the English descriptions in the masterlist?

**Setup**: Gold ICD-10 category given → LLM picks from 1–18 h-part options
using English descriptions. 200 samples, Qwen 7B.

**Result**:

| Subset | Accuracy | n |
|--------|----------|---|
| Trivial (1 h-part, no choice) | 100% | 40 |
| Non-trivial (>1 h-part) | **88.3%** | 103 |
| Records with no h-parts (code not in masterlist) | skipped | 57 |
| **Overall** | **65.5%** | 200 |

The 65.5% overall is misleading — 57 records have gold codes not in the
masterlist (0 h-parts), dragging the average down. On records where the
task is well-defined, the LLM gets **88.3%** on non-trivial disambiguation.

**Nature of misses** (12/103 non-trivial):
- Most are **defensible near-misses** between very similar options:
  - "Measles, no mention of complication" vs "stated to be without complication"
  - "Heart failure, unspecified" vs "Weak heart"
  - "Diabetes mellitus, no mention of complications" vs "stated to be without"
  - "Acute myocardial infarction, unspecified" vs "Heart apoplexy"
- A few are **genuine translation failures**: "Levenszwakte" (= weakness of
  life/congenital debility) mapped to "Poor viability" instead of "Neonatorum"
- One **ambiguous case**: "Gangène sénile" → "Senile gangrene" (I70.201)
  instead of gold "Atherosclerosis of arteries" (I70.200) — arguably the
  LLM's pick is more appropriate

### Baselines and pipeline-level accuracy

The 88.3% on non-trivial needs context. The dataset is skewed: 68.6% of
gold codes are the default .X00 (first option).

**Non-trivial only (>1 h-part, n=103 in test, 994 in full sample)**:

| Method | Accuracy |
|--------|----------|
| Random guessing | 31.6% |
| Always pick first (.X00) | 68.6% |
| **LLM (Qwen 7B)** | **88.3%** |

LLM's real lift: +19.7pp over always-first, +56.7pp over random.

**Full pipeline view** (all 1,764 sample records, assuming gold category):

| Record type | n | % of sample | Handling |
|-------------|---|-------------|----------|
| 0 h-parts (not in masterlist) | 475 | 26.9% | Auto-correct (no h-step) |
| 1 h-part (trivial) | 295 | 16.7% | Auto-correct (only one choice) |
| >1 h-parts (LLM decides) | 994 | 56.3% | LLM picks |

| Method | Pipeline accuracy |
|--------|-------------------|
| Random | 61.5% |
| Always-first | 82.3% |
| **LLM** | **93.4%** |

So as a pipeline component (given gold category), the h-part step would
contribute **93.4% accuracy** — but 82.3% of that is achievable by just
always picking the default code.

### Suggestions to improve further

1. **Structured output / constrained decoding**: Force the LLM to output
   one of the valid codes exactly (LM Studio supports grammar-based
   sampling). Eliminates parse failures and "close but wrong code" errors.

2. **Inject the CoD string's detected language**: Many misses are language
   failures ("Levenszwakte", "Stuipen"). If the pipeline detects Dutch and
   tells the LLM "this is Dutch for...", it can match better. Even a simple
   language tag ("Language: Dutch") in the prompt may help.

3. **Re-rank with embedding similarity**: For the ambiguous near-miss cases
   ("no mention of complication" vs "stated to be without complication"),
   compute cosine similarity between the CoD string embedding and each
   h-part description embedding. Use as a tiebreaker or second opinion.
   The embedding model is already loaded in LM Studio.

**Emerging architecture**:
1. Fine-tuned coarse classifier → chapter (language barrier requires training)
2. Enhanced descriptions / fine-tuned → block group + block (label quality)
3. LLM with English h-part descriptions → leaf (**already works at 88%**)
