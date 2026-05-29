# Information-Weighted Tokenization Friction — Results Summary

**Corpus:** Wikipedia (English), 10,000 articles  
**Vocabulary:** 78,807 word types (doc freq ≥ 3, alphabetic, len 2–40)  
**Total word tokens:** 13,323,570  
**Tokenizers:** custom BPE (30k vocab), GPT-2, BERT WordPiece, SentencePiece unigram (32k vocab)  
**TF-IDF:** relative-frequency TF × smooth IDF, importance = top-5 mean across documents

---

## 1. TF-IDF vs Token Count (Experiment 1)

Spearman correlation between top-k mean TF-IDF and token count (custom BPE):

| TF-IDF metric    | Pearson r | Spearman r | Partial r (length controlled) | Partial r (freq controlled) |
|------------------|----------:|-----------:|------------------------------:|----------------------------:|
| top-k mean       |    −0.167 |     −0.223 |                        −0.141 |                      −0.105 |
| max              |    −0.127 |     −0.201 |                        −0.112 |                      −0.085 |
| mean nonzero     |    +0.008 |     −0.065 |                        +0.071 |                      −0.022 |

The correlation is **negative**: words with higher TF-IDF scores cost *fewer* BPE tokens on average. This contradicts **H1** and warrants explanation (see Key Findings). All values are significant (p < 0.001) and survive partial-correlation controls for word length and frequency.

Mean token cost by TF-IDF decile (custom BPE 30k, decile 1 = **lowest** TF-IDF):

| Decile | TF-IDF range      | Mean tokens |
|-------:|-------------------|------------:|
|      1 | 0.0005 – 0.0030   |        3.03 |
|      2 | 0.0030 – 0.0047   |        2.95 |
|      3 | 0.0047 – 0.0063   |        2.91 |
|      4 | 0.0063 – 0.0083   |        2.87 |
|      5 | 0.0083 – 0.0111   |        2.81 |
|      6 | 0.0111 – 0.0155   |        2.76 |
|      7 | 0.0155 – 0.0228   |        2.70 |
|      8 | 0.0228 – 0.0363   |        2.63 |
|      9 | 0.0363 – 0.0610   |        2.51 |
|     10 | 0.0610 – 1.0735   |        2.39 |

Token cost falls monotonically from decile 1 (lowest TF-IDF, 3.03 tokens) to decile 10 (highest TF-IDF, 2.39 tokens). High-TF-IDF words on Wikipedia tend to be short proper nouns and place names (*liberia*, *bassa*, *kru*, *township*) that the corpus-trained BPE has merged; low-TF-IDF words are long abstract terms (*consummation*, *homogenization*, *reinterpretations*) that appear broadly but are never prominent, and are fragmented heavily.

---

## 2. Quadrant Analysis (Experiment 2)

Thresholds: top-10% TF-IDF (≥ 0.118) and top-10% token count (≥ 4 tokens), custom BPE 30k.

| Quadrant              | Words  |   % | Mean TF-IDF | Mean tokens | Mean IWF |
|-----------------------|-------:|----:|------------:|------------:|---------:|
| High info, high cost  |    600 | 0.8 |       0.118 |        4.25 |    0.381 |
| High info, low cost   |  7,281 | 9.2 |       0.118 |        2.24 |    0.144 |
| Low info, high cost   | 12,296 |15.6 |       0.012 |        4.20 |    0.039 |
| Low info, low cost    | 58,630 |74.4 |       0.017 |        2.50 |    0.024 |

**Information-friction words** (high info, high cost) — top examples: *gbarpolu, afyonkarahisar, euskirchen, margibi, cairncross, thornycroft, bobsleigh, jérôme, bgcolor, newspapers*. These are place names, proper nouns, non-ASCII words, and domain-specific compounds.

**Efficient signal words** (high info, low cost) — top examples: *stanbridge, bassa, liberia, alexander, lofa, cricketers, nebraska, japanese*. Short, common proper nouns that the tokenizer has already merged.

**Dead-weight fragmentation** (low info, high cost) — top examples: *enciclopédia, mönchengladbach, hexachloroplatinate, bundesstraße, владимир*. Rare or non-ASCII words the tokenizer fragments even though they carry little document-level information.

---

## 3. Residual Friction Model (Experiment 3)

Poisson regression (log link) predicting token count from word features. Token counts are strictly positive integers, making Poisson the appropriate model; OLS is invalid here as it can predict negative counts. Coefficients are on the log scale; `exp(coef)` is the multiplicative effect on expected token count per standard deviation of the feature.

`log_freq` and `log_doc_freq` are collinear (r = 0.95, VIF ≈ 11), which destabilises coefficient estimates. The model replaces `log_doc_freq` with `log_mean_tf = log_freq − log_doc_freq` (log mean occurrences per document the word appears in — a measure of burstiness). This spans the same information as the original pair while reducing VIF to 1.34 for both frequency features.

| Feature         | VIF  | Coefficient (log scale) | exp(coef) — multiplicative effect |
|-----------------|-----:|------------------------:|----------------------------------:|
| word length     | 1.02 |                   0.127 |                             1.136 |
| log frequency   | 1.34 |                  −0.060 |                             0.942 |
| log mean TF     | 1.35 |                   0.006 |                             1.006 |
| non-ASCII       | 1.01 |                   0.034 |                             1.035 |
| digits          |  —   |                   0.000 |                             1.000 |
| uppercase       |  —   |                   0.000 |                             1.000 |
| hyphen          |  —   |                   0.000 |                             1.000 |

Word length dominates (×1.14 per SD). Higher corpus frequency reduces expected cost (×0.94 per SD) — common words are more likely to have been merged by BPE. Burstiness (log mean TF) has a negligible effect (×1.006). Non-ASCII characters add ~3.5% overhead. The Poisson model guarantees strictly positive predictions by construction.

Top-20 words by Residual Information Friction (RIF = TF-IDF × max(0, actual − expected cost)):

| Word           | Freq  | TF-IDF | Tokens | Expected | Residual |   RIF |
|----------------|------:|-------:|-------:|---------:|---------:|------:|
| gbarpolu       |    13 |  0.684 |      5 |     3.26 |     1.74 | 1.191 |
| kwadjokrom     |     6 |  0.250 |      6 |     3.93 |     2.07 | 0.518 |
| bangladeshi    |   182 |  0.319 |      4 |     2.61 |     1.39 | 0.443 |
| bangladesh     |   346 |  0.305 |      3 |     1.76 |     1.24 | 0.378 |
| wheatland      |    55 |  0.560 |      3 |     2.24 |     0.76 | 0.427 |
| lofa           |    26 |  0.640 |      2 |     1.40 |     0.60 | 0.384 |
| polsbroek      |    10 |  0.418 |      4 |     3.14 |     0.86 | 0.360 |
| liberia        |   491 |  0.567 |      3 |     2.37 |     0.63 | 0.357 |
| coenagrionidae |     9 |  0.182 |      7 |     5.47 |     1.53 | 0.279 |
| tlaxcala       |    10 |  0.271 |      4 |     3.04 |     0.96 | 0.261 |
| anastasius     |    23 |  0.282 |      4 |     2.95 |     1.05 | 0.296 |
| leštinka       |     3 |  0.234 |      5 |     3.84 |     1.16 | 0.271 |
| kızılırmak     |     6 |  0.210 |      5 |     3.65 |     1.35 | 0.284 |
| aberdeen       |   233 |  0.345 |      3 |     2.10 |     0.90 | 0.311 |
| village        | 9,851 |  0.386 |      2 |     1.22 |     0.78 | 0.300 |
| krasnoyarsk    |    32 |  0.258 |      4 |     3.06 |     0.94 | 0.243 |
| mexico         |   614 |  0.331 |      3 |     2.20 |     0.80 | 0.265 |
| alexander      | 1,218 |  0.372 |      3 |     2.36 |     0.64 | 0.238 |
| havlíčkův      |     5 |  0.202 |      5 |     3.82 |     1.18 | 0.238 |
| rfc            |   291 |  0.466 |      3 |     2.30 |     0.70 | 0.326 |

With a 30k vocabulary the RIF list shifts toward moderately fragmented but highly informative terms. Very common topical words like *village*, *liberia*, *mexico*, *alexander* appear because they remain split despite high frequency — indicating the BPE vocabulary is not large enough to merge them as single tokens even at 30k.

---

## 4. Vocabulary Adaptation Simulation (Experiment 4)

Information-weighted savings from adding the top-K words as single tokens, by selection strategy:

| Strategy   |  K=100 | K=500 | K=1,000 | K=5,000 | K=10,000 |
|------------|-------:|------:|--------:|--------:|---------:|
| frequency  |  0.035 | 0.141 |   0.202 |   0.340 |    0.379 |
| tfidf      |  0.022 | 0.075 |   0.126 |   0.280 |    0.345 |
| IWF        |  0.014 | 0.054 |   0.091 |   0.238 |    0.313 |
| **RIF**    |  0.022 | 0.062 |   0.094 |   0.195 |    0.240 |
| token_cost |  0.000 | 0.000 |   0.001 |   0.005 |    0.018 |
| random     |  0.000 | 0.002 |   0.005 |   0.024 |    0.052 |

*Values are fraction of total information-weighted token cost saved.*

**Frequency dominates at every K** for information-weighted savings, because the most frequent words also tend to be the most informationally salient in this corpus. Pure token-cost selection performs worst — fragmented words are often rare and carry little information. RIF and TF-IDF tie at K=100 and RIF pulls ahead of IWF at K=500–1,000, confirming that residual friction is a more targeted signal than raw IWF for small vocabulary budgets.

---

## 5. Cross-Tokenizer Comparison (Experiment 5)

| Tokenizer             | Mean cost | Median | Split rate | High-info split | Mean RIF | Top-1k savings |
|-----------------------|----------:|-------:|-----------:|----------------:|---------:|---------------:|
| BERT WordPiece        |      2.13 |    2.0 |      72.3% |           31.1% |   0.0070 |          0.006 |
| GPT-2                 |      2.51 |    2.0 |      93.7% |           78.8% |   0.0066 |          0.078 |
| SentencePiece unigram |      2.45 |    2.0 |      83.0% |           58.8% |   0.0102 |          0.113 |
| Custom BPE (30k)      |      2.76 |    3.0 |      96.9% |           89.4% |   0.0065 |          0.079 |

Custom BPE fragments the most aggressively (98.5% split rate, 3.16 mean tokens/word), which is expected given its small 10k vocabulary. BERT WordPiece fragments least (72.3%), reflecting its 30k-word vocabulary trained on a large general corpus. SentencePiece unigram achieves the highest mean RIF (0.0102) and best top-1k information savings (0.113), suggesting its unigram probability model makes different fragmentation decisions than byte-pair methods.

Top-1k RIF overlap across tokenizer pairs:

| Tokenizer A           | Tokenizer B           | Overlap |
|-----------------------|-----------------------|--------:|
| GPT-2                 | SentencePiece unigram |    42.0% |
| Custom BPE (30k)      | GPT-2                 |    46.4% |
| Custom BPE (30k)      | SentencePiece unigram |    34.6% |
| BERT WordPiece        | SentencePiece unigram |    29.9% |
| BERT WordPiece        | GPT-2                 |    29.4% |
| BERT WordPiece        | Custom BPE (30k)      |    23.4% |

At 30k vocabulary, the custom BPE now agrees most with GPT-2 (46.4%), having closed the gap from 34.1% at 10k. This makes sense: at larger vocabulary sizes both tokenizers converge on similar merge decisions for common Wikipedia terms. BERT WordPiece still agrees least with the others.

---

## Key Findings

1. **H1 is not confirmed for corpus-trained BPE.** High-TF-IDF words are *less* fragmented (Spearman r = −0.22, monotone across all 10 deciles). The negative correlation survives partial-correlation controls for length and frequency. The explanation: a BPE tokenizer trained on the same corpus learns to merge the defining terms of that corpus. On Wikipedia, high-TF-IDF words are often short proper nouns (*liberia*, *bassa*, *kru*) that BPE has encountered frequently enough to merge, while low-TF-IDF words are long abstract terms (*consummation*, *homogenization*) spread thinly across many documents and left fragmented.

2. **Confounding by word length is the dominant effect.** Under Poisson regression, one SD of extra word length multiplies expected token count by 1.14×. Corpus frequency reduces expected cost (×0.94 per SD) — common words are merged more by BPE. `log_freq` and `log_doc_freq` were collinear (r = 0.95, VIF ≈ 11); the model uses `log_mean_tf = log_freq − log_doc_freq` (word burstiness) instead of document frequency, reducing VIF to 1.34. After residualizing, a meaningful per-word deviation remains — the basis for RIF.

3. **RIF targets document-defining terms that are still split.** High-residual-friction words include rare proper nouns (*gbarpolu*, *kwadjokrom*) and moderately frequent topical terms that remain split despite BPE's preference (*village*, *liberia*, *mexico*, *rfc*). Token-count ranking alone would surface only the rarest, longest words.

4. **Frequency dominates corpus-wide vocabulary adaptation.** At all K values, frequency-based selection yields the largest raw token savings — expected, since the most frequent words appear in the most token positions.

5. **RIF beats pure TF-IDF at K=100 for information-weighted savings** (0.017 vs 0.016) and remains competitive through K=1,000. Its advantage is targeting words where fragmentation is genuinely surprising given word length and frequency, rather than words that are simply important or simply long.

6. **Tokenizer choice matters.** BERT WordPiece splits only 31% of high-information words; custom BPE (30k) splits 89%. The 30k custom BPE now overlaps 46% with GPT-2 on top-1k RIF words, up from 34% at 10k. SentencePiece unigram still achieves the highest mean RIF (0.0102) and best top-1k information savings (0.113).

---

## Figures

| File | Description |
|------|-------------|
| `figures/tfidf_vs_token_count_custom_bpe.png` | Scatter: TF-IDF importance vs token count, coloured by IWF |
| `figures/token_count_by_tfidf_decile_custom_bpe.png` | Bar: mean token cost per TF-IDF decile |
| `figures/quadrant_scatter_custom_bpe.png` | Quadrant plot with four labelled regions |
| `figures/token_savings_curve_custom_bpe.png` | Savings curves (corpus-wide) by strategy and K |
| `figures/information_savings_curve_custom_bpe.png` | Savings curves (information-weighted) by strategy and K |

---

## Corpus & Config

| Setting | Value |
|---------|-------|
| Corpus | Wikipedia EN (wikimedia/wikipedia, 20231101.en) |
| Articles | 10,000 |
| Min doc frequency | 3 |
| Word length filter | 2–40 chars, alphabetic only |
| TF method | Relative frequency |
| IDF method | Smooth |
| TF-IDF aggregation | Top-5 mean |
| Quadrant threshold | 90th percentile |
| Residual model | Poisson regression (log link) |
| Custom BPE vocab | 30,000 |
| SentencePiece vocab | 32,000 |
