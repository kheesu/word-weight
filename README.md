# word-weight

**Information-Weighted Tokenization Friction**

*Auditing whether subword tokenizers spend tokens on document-informative words.*

---

## Research question

> Do subword tokenizers allocate representational budget efficiently with respect to document-level word importance?

Words that are most distinctive to a document (high TF-IDF) tend to be rarer, and BPE tokenizers tend to fragment rare words. This project measures whether that fragmentation is worse than expected after controlling for word length and frequency — and whether friction-aware selection outperforms naive strategies for vocabulary adaptation.

## Key metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **IWF** | `I(w) × (C(w) − 1)` | Raw information-weighted fragmentation |
| **RIF** | `I(w) × max(0, C(w) − Ĉ(w))` | Residual friction after controlling for length and frequency |

- `I(w)` — top-k mean TF-IDF score for word *w*
- `C(w)` — BPE token count for *w*
- `Ĉ(w)` — expected token count from a Poisson regression on length, log-frequency, and character features

## Repository structure

```
word-weight/
├── pipeline.py              # end-to-end runner
├── pyproject.toml
├── configs/
│   ├── wikipedia.yaml       # corpus paths and sizes
│   ├── tokenizers.yaml      # BPE vocab sizes and paths
│   └── experiments.yaml     # metric thresholds, model choice
├── src/
│   ├── preprocessing/       # corpus download and cleaning
│   ├── tokenization/        # BPE training, GPT-2/BERT/SP loaders, cost computation
│   ├── tfidf/               # TF-IDF computation and word-level aggregation
│   ├── metrics/             # IWF, RIF, residual cost model
│   ├── experiments/         # exp01–exp05 analysis scripts
│   └── visualization/       # plot distributions, quadrants, savings curves
├── rusty-bpe/               # fast Rust BPE tokenizer (Python bindings via maturin)
├── data/processed/          # parquet outputs, trained BPE models
├── results/                 # CSVs: correlations, quadrants, RIF rankings, simulations
└── figures/                 # saved plots
```

## Setup

Requires Python ≥ 3.10 and Rust (for `rusty-bpe`).

```sh
# Install Python dependencies (uv recommended)
uv sync

# Build the Rust BPE extension
cd rusty-bpe && maturin develop --release --features python && cd ..
```

## Running the pipeline

```sh
# Full pipeline on the small corpus (default)
python pipeline.py

# Larger corpus
python pipeline.py --size medium

# Show all steps
python pipeline.py --list

# Run specific steps only
python pipeline.py --steps 1 2 3
```

**Pipeline steps:**

| # | Step |
|---|------|
| 1 | Preprocess corpus |
| 2 | Train BPE |
| 3 | Compute TF-IDF scores |
| 4 | Compute token costs (custom BPE, GPT-2, BERT, SentencePiece) |
| 5 | Merge word metrics + IWF |
| 6 | Exp01: distribution analysis |
| 7 | Exp02: quadrant analysis |
| 8 | Exp03: residual friction model |
| 9 | Exp04: vocabulary simulation |
| 10 | Exp05: tokenizer comparison |
| 11–13 | Plots |

## Experiments

### Exp01 — Distribution analysis
Correlation between TF-IDF importance and token count, with and without controlling for length and frequency.
Outputs: `results/correlations.csv`, `figures/tfidf_vs_token_count_*.png`, `figures/token_count_by_tfidf_decile_*.png`

### Exp02 — Quadrant analysis
Classifies words into four quadrants by information value and tokenization cost.

| Quadrant | Meaning |
|----------|---------|
| High TF-IDF, low cost | efficient signal words |
| High TF-IDF, high cost | **information friction** |
| Low TF-IDF, low cost | common vocabulary |
| Low TF-IDF, high cost | dead-weight fragmentation |

Outputs: `results/quadrants/`, `figures/quadrant_scatter_*.png`

### Exp03 — Residual friction model
Fits a Poisson regression to predict expected token count from word features, then ranks words by RIF.
Outputs: `results/top_rif_words.csv`, `results/residual_model_coefficients.csv`

### Exp04 — Vocabulary simulation
Simulates adding the top-K words as single tokens and measures corpus-wide and information-weighted token savings across six selection strategies: random, frequency, TF-IDF, token cost, IWF, RIF.
Outputs: `results/vocab_simulation/`, `figures/token_savings_curve_*.png`, `figures/information_savings_curve_*.png`

### Exp05 — Tokenizer comparison
Compares friction profiles across custom BPE, GPT-2, BERT WordPiece, and SentencePiece.
Outputs: `results/tokenizer_comparison.csv`

## Subproject: rusty-bpe

A fast, byte-level BPE tokenizer written in Rust with Python bindings. Supports parallel training and batch encoding via rayon. See [`rusty-bpe/README.md`](rusty-bpe/README.md) for full documentation.

## Main findings (small corpus, custom BPE)

- High-TF-IDF words are more likely to be split than low-TF-IDF words, but much of this is explained by word length and frequency.
- After controlling for length and frequency, a subset of high-RIF words remains: predominantly technical terms, named entities, and domain-specific compounds.
- For vocabulary adaptation, IWF- and RIF-based selection outperforms frequency-only selection on information-weighted token savings, while frequency wins on raw corpus-wide savings.
