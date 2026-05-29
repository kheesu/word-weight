Below is a repository-oriented research outline focused on **experiments**, not prose. The project’s core idea is to compare two notions of word “weight”: **document-level importance** from TF-IDF and **tokenization cost** from BPE/subword fragmentation, which matches your original project goal of exploring word weight through BPE and TF-IDF on Wikipedia. 

# Project title

**Information-Weighted Tokenization Friction**

Working subtitle:

**Auditing whether subword tokenizers spend tokens on document-informative words**

---

# 1. Main research question

## Core question

> Do subword tokenizers allocate representational budget efficiently with respect to document-level word importance?

In simpler terms:

> Are the words that matter most to documents also expensive for the tokenizer to represent?

This project should not claim that TF-IDF is “semantic importance” in a deep sense. Instead, define it carefully as **document specificity** or **document-level information value**.

---

# 2. Main hypothesis

## H1: Informative words are more fragmented than common words

Words with high TF-IDF will, on average, require more BPE tokens than low-TF-IDF words.

This is expected because high-TF-IDF words tend to be rarer, and BPE tends to fragment rare words.

## H2: Some words are over-fragmented even after controlling for length and frequency

The more interesting claim is not that rare words are split.

The interesting claim is:

> Some words are more fragmented than expected for words of similar length and frequency.

These words are the project’s main object of study.

## H3: High-friction words are useful candidates for vocabulary adaptation

Words with high information friction should produce better token savings than selecting vocabulary additions by frequency alone, TF-IDF alone, or token count alone.

---

# 3. Dataset plan

## Dataset A: Wikipedia baseline corpus

Use Wikipedia as the main general-domain corpus.

Recommended unit:

```text
document = Wikipedia article
term = normalized word type
```

You can start with a manageable subset:

```text
10k articles
50k articles
100k articles
```

Use increasing corpus sizes as scalability experiments.

## Dataset B: Domain corpus

Add at least one specialized domain corpus.

Good options:

```text
biomedical abstracts
arXiv abstracts
legal documents
software documentation
StackExchange posts
```

The domain corpus is important because the most compelling use case is tokenizer auditing for specialized vocabulary.

## Dataset C: Optional multilingual or non-English slice

Only add this if time permits.

Useful for showing that tokenization friction can vary across language, but this may expand the project too much.

---

# 4. Preprocessing

Keep preprocessing simple and reproducible.

## Document cleaning

For each document:

```text
lowercase text
remove markup
normalize whitespace
strip punctuation for word-level TF-IDF
preserve original word forms for tokenizer evaluation if needed
```

You should probably store two versions:

```text
clean_text_for_tfidf
raw_or_lightly_cleaned_text_for_tokenization
```

Reason: TF-IDF benefits from normalized word types, but tokenizer behavior may depend on casing, punctuation, and spacing.

## Vocabulary filtering

Create a word-level vocabulary with filters:

```text
minimum document frequency: 3 or 5
minimum character length: 2
maximum character length: maybe 40 or 60
alphabetic-only version
full version including digits/hyphens
```

Run experiments on both:

```text
clean alphabetic vocabulary
full noisy vocabulary
```

The noisy version will reveal artifacts. The clean version will make the main analysis more defensible.

---

# 5. Metrics

## 5.1 TF-IDF importance

For each word (w), compute document-level TF-IDF.

You need one aggregate score per word.

Recommended options:

### Option A: maximum TF-IDF

[
I_{\max}(w) = \max_d \text{TF-IDF}(w,d)
]

Interpretation:

> How important can this word be to its most representative document?

This is good for identifying highly document-specific terms.

### Option B: mean nonzero TF-IDF

[
I_{\text{mean}}(w) = \frac{1}{|{d:w\in d}|}\sum_{d:w\in d}\text{TF-IDF}(w,d)
]

Interpretation:

> How important is this word when it appears?

### Option C: top-k mean TF-IDF

[
I_{\text{top-k}}(w) = \text{mean of top } k \text{ TF-IDF values for } w
]

This is more stable than max TF-IDF.

Recommended main metric:

```text
top-k mean TF-IDF, with k = 5 or 10
```

Use max TF-IDF as a secondary analysis.

---

## 5.2 BPE/tokenizer cost

For each word (w), compute:

[
C(w)=\text{number of subword tokens used to encode }w
]

Also compute normalized variants:

```text
tokens_per_character = C(w) / len(w)
characters_per_token = len(w) / C(w)
is_split = 1 if C(w) > 1 else 0
```

Use **token count** as the primary metric.

Use **tokens per character** as a robustness metric.

---

## 5.3 Information-weighted fragmentation

Simple metric:

[
\text{IWF}(w)=I(w)\cdot(C(w)-1)
]

Interpretation:

> How much document-level information is carried by a word that the tokenizer fragments?

This is easy to explain and good for the blog/repository.

---

## 5.4 Residual information friction

This should be the serious paper-style metric.

First fit a model predicting expected token count:

[
\hat{C}(w)=f(\text{length}(w), \log \text{frequency}(w), \text{character features})
]

Then compute:

[
\text{RIF}(w)=I(w)\cdot\max(0, C(w)-\hat{C}(w))
]

Interpretation:

> A word has high residual information friction if it is important and more fragmented than expected.

Suggested features for the expected-cost model:

```text
character length
log corpus frequency
document frequency
number of digits
number of uppercase letters
contains hyphen
contains non-ascii character
suffix/prefix features, optional
```

Suggested model:

```text
linear regression
Poisson regression
negative binomial regression
random forest regression
```

For interpretability, start with linear or Poisson regression.

---

# 6. Tokenizers to compare

Do not only use your own BPE implementation.

Use your own BPE for explanation and reproducibility, but compare against established tokenizers.

## Minimum set

```text
custom BPE trained on your corpus
GPT-style BPE tokenizer
BERT WordPiece tokenizer
SentencePiece unigram tokenizer
```

## Why this matters

If you only evaluate your own tokenizer, reviewers/readers may think the result is an implementation artifact.

The core comparison should ask:

> Do different tokenizers produce different information-friction profiles?

---

# 7. Experiment 1: Basic distribution analysis

## Goal

Understand the relationship between TF-IDF importance and tokenization cost.

## Procedure

For each corpus and tokenizer:

1. Compute word-level TF-IDF importance.
2. Compute token count for each word.
3. Plot distributions:

   ```text
   TF-IDF score distribution
   token count distribution
   token count by TF-IDF percentile
   ```
4. Compute correlations:

   ```text
   Pearson correlation: TF-IDF vs token count
   Spearman correlation: TF-IDF vs token count
   correlation after controlling for word length
   correlation after controlling for frequency
   ```

## Expected result

Raw TF-IDF and token count probably correlate weakly to moderately.

After controlling for length and frequency, the relationship will likely weaken.

That is fine. The point is not that TF-IDF perfectly predicts tokenization cost. The point is that the residual outliers are interesting.

## Repository outputs

```text
results/distributions/
results/correlations.csv
figures/tfidf_vs_token_count_scatter.png
figures/token_count_by_tfidf_decile.png
```

---

# 8. Experiment 2: Quadrant analysis

## Goal

Classify words by information value and tokenization cost.

## Axes

Use:

```text
x-axis: TF-IDF importance
y-axis: BPE token count or residual token cost
```

Create four quadrants:

| Quadrant                     | Meaning                     |
| ---------------------------- | --------------------------- |
| High TF-IDF, low token cost  | efficient informative words |
| High TF-IDF, high token cost | information friction        |
| Low TF-IDF, low token cost   | common vocabulary           |
| Low TF-IDF, high token cost  | dead-weight fragmentation   |

## Procedure

Define thresholds:

```text
high TF-IDF = top 10%
high token cost = top 10%
```

or use percentile-based bins:

```text
top 5%
top 10%
top 25%
```

For each quadrant, output top examples.

## What to inspect

For each quadrant, categorize words as:

```text
technical terms
named entities
morphologically complex words
foreign words
hyphenated compounds
numbers/codes
markup or artifacts
misspellings
```

## Repository outputs

```text
results/quadrants/high_info_high_cost.csv
results/quadrants/high_info_low_cost.csv
results/quadrants/low_info_high_cost.csv
results/quadrants/low_info_low_cost.csv
figures/quadrant_scatter.png
```

This should be one of the most important visualizations in the project.

---

# 9. Experiment 3: Residual friction analysis

## Goal

Separate genuinely interesting fragmentation from obvious effects of word length and frequency.

## Procedure

For every word, fit:

[
C(w) \sim \text{length}(w) + \log \text{frequency}(w) + \text{character features}
]

Then calculate:

```text
residual_cost = actual_token_count - expected_token_count
positive_residual_cost = max(0, residual_cost)
RIF = TF-IDF_importance * positive_residual_cost
```

Rank words by RIF.

## Baselines for comparison

Compare top-RIF words against:

```text
top TF-IDF words
top token-count words
top length-normalized token-count words
top rare words
top frequent words
```

## Evaluation

Manual inspection:

```text
Are top-RIF words meaningful?
Are they domain-specific?
Are they cleaner than top token-count words?
Are they less obvious than top rare words?
```

Automatic summaries:

```text
average frequency
average length
average token count
percentage alphabetic
percentage named entities, if using NER
percentage domain terms, if using domain lexicon
```

## Repository outputs

```text
results/residual_model_coefficients.csv
results/top_rif_words.csv
results/baseline_rankings/
figures/residual_cost_histogram.png
figures/rif_top_words_barplot.png
```

---

# 10. Experiment 4: Vocabulary adaptation simulation

This is the most important experiment if you want the project to look like a research contribution.

## Goal

Test whether high-friction words are good candidates for vocabulary expansion.

## Basic idea

Pretend you add the top (K) selected words as single tokens.

Then measure how many tokens the corpus would save.

## Selection strategies

Compare these strategies:

| Strategy   | Description                                |
| ---------- | ------------------------------------------ |
| Random     | randomly chosen words                      |
| Frequency  | most frequent words                        |
| TF-IDF     | highest document-importance words          |
| Token cost | most fragmented words                      |
| IWF        | high TF-IDF × token fragmentation          |
| RIF        | high TF-IDF × residual token fragmentation |

## Procedure

For each strategy and each (K):

```text
K = 100, 500, 1000, 5000, 10000
```

Simulate vocabulary additions.

For each word selected:

```text
old_cost = C(w)
new_cost = 1
savings_per_occurrence = old_cost - 1
total_savings = savings_per_occurrence * corpus_frequency(w)
```

Total corpus token savings:

[
\text{Savings} =
\frac{
\sum_w f(w)(C(w)-C_{\text{new}}(w))
}{
\sum_w f(w)C(w)
}
]

## Important distinction

You should report two types of savings:

### Corpus-wide token savings

This favors frequent words.

[
\text{corpus savings}
]

### Information-weighted token savings

This favors informative words.

[
\text{information savings}
==========================

\sum_w I(w) \cdot f(w) \cdot (C(w)-C_{\text{new}}(w))
]

The frequency baseline may win corpus-wide savings. That is okay.

Your metric should do better on **information-weighted savings**.

## Repository outputs

```text
results/vocab_simulation/token_savings_by_strategy.csv
results/vocab_simulation/information_savings_by_strategy.csv
figures/token_savings_curve.png
figures/information_savings_curve.png
```

This experiment gives the project a practical conclusion:

> Information friction is useful for selecting candidate vocabulary additions when the goal is not merely compression, but compression of document-informative terms.

---

# 11. Experiment 5: Cross-tokenizer comparison

## Goal

Measure whether different tokenizers have different friction profiles.

## Procedure

For each tokenizer:

```text
compute token count C(w)
compute IWF
compute RIF
run quadrant analysis
run vocabulary simulation
```

Then compare:

```text
average token count
average token count for high-TF-IDF words
percentage of high-TF-IDF words split
top-RIF overlap across tokenizers
vocabulary-simulation savings
```

## Useful metrics

```text
mean fertility
median fertility
split rate
high-info split rate
residual friction mean
top-k RIF overlap
```

## Key table

| Tokenizer | Mean cost | Split rate | High-info split rate | Mean RIF | Top-1k vocab savings |
| --------- | --------: | ---------: | -------------------: | -------: | -------------------: |

## Repository outputs

```text
results/tokenizer_comparison.csv
figures/tokenizer_high_info_split_rate.png
figures/tokenizer_rif_comparison.png
```

---

# 12. Experiment 6: Cross-domain comparison

## Goal

Show that tokenization friction increases in specialized domains.

## Procedure

Run the full pipeline on:

```text
Wikipedia
biomedical corpus
legal corpus
software corpus
```

For each corpus, compute:

```text
top-RIF words
high-info split rate
mean RIF
vocabulary-simulation savings
```

## Expected result

Specialized corpora should produce more high-information, high-fragmentation terms.

Examples:

```text
biomedical: gene names, diseases, chemical compounds
legal: statute names, case names, formal compounds
software: package names, identifiers, API terms
```

## Repository outputs

```text
results/domain_comparison.csv
figures/domain_high_info_split_rate.png
figures/domain_vocab_savings.png
```

This experiment is what makes the project useful for tokenizer adaptation.

---

# 13. Experiment 7: Robustness checks

## Robustness check A: Different TF-IDF aggregations

Repeat main results using:

```text
max TF-IDF
mean nonzero TF-IDF
top-k mean TF-IDF
IDF only
```

The top words should not completely change.

## Robustness check B: Different vocabulary filters

Repeat with:

```text
alphabetic-only words
all words
minimum frequency thresholds
lowercased vs cased words
```

## Robustness check C: Different BPE vocabulary sizes

For your custom BPE tokenizer, train with different vocabulary sizes:

```text
1k
5k
10k
30k
50k
```

Analyze:

```text
Does larger vocabulary reduce friction?
Which words remain high-friction even at larger vocab sizes?
```

## Robustness check D: Different corpora sizes

Use:

```text
10k articles
50k articles
100k articles
```

Check whether top-RIF words stabilize.

## Repository outputs

```text
results/robustness/
figures/robustness_topk_overlap.png
figures/bpe_vocab_size_vs_friction.png
```

---

# 14. Optional downstream experiment

Only do this if the rest is complete.

## Option A: Retrieval experiment

Use TF-IDF retrieval.

Compare document retrieval using:

```text
original word-level TF-IDF
subword-level TF-IDF
word-level TF-IDF with high-friction terms added/handled specially
```

This may be too far from the core tokenizer audit.

## Option B: Classification experiment

Use a small text classification dataset.

Test whether documents with high average information friction are harder for a classifier.

Possible measurement:

```text
document_friction = average or sum RIF of words in document
```

Then analyze:

```text
Does document friction correlate with classification error?
```

## Option C: Language model surprisal experiment

Use a pretrained language model to compute word or token surprisal.

Ask:

> Do high-RIF words also have high model surprisal?

This would connect your metric to model behavior, but it is more complex.

---

# 15. Recommended repository structure

```text
information-tokenization-friction/
│
├── README.md
├── pyproject.toml
├── requirements.txt
│
├── configs/
│   ├── wikipedia.yaml
│   ├── biomedical.yaml
│   ├── tokenizers.yaml
│   └── experiments.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
│
├── src/
│   ├── preprocessing/
│   │   ├── clean_text.py
│   │   └── build_corpus.py
│   │
│   ├── tokenization/
│   │   ├── bpe_from_scratch.py
│   │   ├── train_bpe.py
│   │   ├── load_tokenizers.py
│   │   └── compute_token_costs.py
│   │
│   ├── tfidf/
│   │   ├── compute_tfidf.py
│   │   └── aggregate_word_scores.py
│   │
│   ├── metrics/
│   │   ├── information_weighted_fragmentation.py
│   │   ├── residual_friction.py
│   │   └── expected_cost_model.py
│   │
│   ├── experiments/
│   │   ├── exp01_distribution_analysis.py
│   │   ├── exp02_quadrant_analysis.py
│   │   ├── exp03_residual_friction.py
│   │   ├── exp04_vocab_simulation.py
│   │   ├── exp05_tokenizer_comparison.py
│   │   ├── exp06_domain_comparison.py
│   │   └── exp07_robustness.py
│   │
│   └── visualization/
│       ├── plot_distributions.py
│       ├── plot_quadrants.py
│       └── plot_savings_curves.py
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_bpe_demo.ipynb
│   ├── 03_metric_exploration.ipynb
│   └── 04_final_figures.ipynb
│
├── results/
│   ├── correlations.csv
│   ├── tokenizer_comparison.csv
│   ├── domain_comparison.csv
│   ├── top_rif_words.csv
│   └── vocab_simulation/
│
├── figures/
│   ├── tfidf_vs_token_count.png
│   ├── quadrant_scatter.png
│   ├── token_savings_curve.png
│   └── domain_comparison.png
│
└── paper/
    ├── outline.md
    ├── abstract.md
    └── related_work.md
```

---

# 16. Main tables to generate

## Table 1: Dataset statistics

| Corpus | Documents | Tokens | Word types | Mean doc length |
| ------ | --------: | -----: | ---------: | --------------: |

## Table 2: Tokenizer statistics

| Tokenizer | Vocab size | Mean word cost | Split rate | Median cost |
| --------- | ---------: | -------------: | ---------: | ----------: |

## Table 3: High-friction examples

| Word | Corpus | TF-IDF | Token count | Expected cost | RIF | Category |
| ---- | ------ | -----: | ----------: | ------------: | --: | -------- |

## Table 4: Vocabulary simulation

| Strategy |  K | Token savings | Information-weighted savings |
| -------- | -: | ------------: | ---------------------------: |

## Table 5: Cross-domain comparison

| Corpus | High-info split rate | Mean RIF | Top-1k savings |
| ------ | -------------------: | -------: | -------------: |

---

# 17. Main figures to generate

## Figure 1: TF-IDF vs token count scatter plot

Shows the main conceptual relationship.

## Figure 2: Quadrant plot

Labels:

```text
efficient signal words
information friction
common vocabulary
dead-weight fragmentation
```

## Figure 3: Token count by TF-IDF decile

Shows whether high-information words are more fragmented.

## Figure 4: Residual friction ranking

Bar plot of top high-RIF words.

## Figure 5: Vocabulary simulation curve

X-axis:

```text
number of added vocabulary items
```

Y-axis:

```text
token savings or information-weighted savings
```

Lines:

```text
frequency
TF-IDF
token cost
IWF
RIF
random
```

## Figure 6: Domain comparison

Shows whether specialized corpora have higher friction.

---

# 18. Main claims you should be able to support

By the end, you want to be able to say:

## Claim 1

> High-TF-IDF words are more likely to be split by subword tokenizers than low-TF-IDF words.

## Claim 2

> However, raw token count is heavily confounded by word length and frequency.

## Claim 3

> Residual information friction identifies words that are both document-informative and unexpectedly expensive to tokenize.

## Claim 4

> These words are often technical terms, named entities, domain-specific compounds, or corpus artifacts.

## Claim 5

> For vocabulary adaptation, information-friction-based selection is more targeted than frequency-only or fragmentation-only selection.

The last claim is the most important one if you want the project to feel like a research contribution.

---

# 19. Minimal viable version

Build this first.

```text
1. Wikipedia subset
2. custom BPE + one pretrained tokenizer
3. TF-IDF word scores
4. token count per word
5. IWF metric
6. quadrant analysis
7. top high-friction word list
8. vocabulary simulation with 4 baselines
```

This is enough for a strong technical blog and a plausible workshop-style project.

---

# 20. Strong version

Add these once the minimal version works:

```text
multiple pretrained tokenizers
residual expected-cost model
domain corpus
cross-domain comparison
robustness checks
information-weighted vocabulary simulation
```

This is the version I would aim for if the repository is meant to support an ACL workshop submission.

---

# 21. Suggested experiment order

Implement in this order:

```text
01_preprocess_corpus
02_compute_tfidf
03_train_or_load_tokenizers
04_compute_word_token_costs
05_merge_word_metrics
06_distribution_analysis
07_quadrant_analysis
08_information_weighted_fragmentation
09_residual_friction_model
10_vocab_adaptation_simulation
11_tokenizer_comparison
12_domain_comparison
13_robustness_checks
```

The first real milestone should be a single CSV like:

```text
word, frequency, document_frequency, tfidf_score, token_count, token_count_normalized, expected_token_count, residual_cost, IWF, RIF
```

Once you have that table, almost every experiment becomes straightforward.

