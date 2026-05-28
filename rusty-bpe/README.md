# rusty-bpe

A fast, byte-level Byte Pair Encoding (BPE) tokenizer in Rust, built for statistical analysis of text corpora.

## Features

- **Efficient training** — lazy max-heap with a pair→word index; only affected words are updated after each merge
- **Parallel** — word counting and batch encoding use rayon
- **Byte-level** — operates on raw UTF-8 bytes; handles any language or script
- **Compact model format** — JSON file stores only the merge list; token strings are reconstructed at load time
- **Library + CLI** — use as a Rust crate or from the command line

## Building

```sh
cargo build --release
```

The binary lands at `target/release/rusty-bpe`.

## CLI

### Train

Learn BPE merges from a text file.

```sh
rusty-bpe train <input> <vocab_size> <output_model> [min_freq]
```

| Argument | Description |
|---|---|
| `input` | Path to the training corpus (plain text, UTF-8) |
| `vocab_size` | Target vocabulary size (base 256 bytes + learned merges) |
| `output_model` | Path to write the model JSON |
| `min_freq` | Minimum word frequency to include in training (default: 2) |

```sh
rusty-bpe train corpus.txt 8000 model.json
```

### Encode

Tokenize a text file and print space-separated token IDs.

```sh
rusty-bpe encode <model> <input>
```

```sh
rusty-bpe encode model.json text.txt > ids.txt
```

### Decode

Reconstruct text from a file of space-separated token IDs.

```sh
rusty-bpe decode <model> <input>
```

```sh
rusty-bpe decode model.json ids.txt
```

### Stats

Print corpus statistics for a text file using a trained model.

```sh
rusty-bpe stats <model> <input>
```

```
Total tokens   : 1482301
Unique tokens  : 4217
Vocab size     : 8000
Total bytes    : 8943210
Bytes/token    : 6.033
Token entropy  : 9.2841 bits
Top-10 coverage: 18.34%

Top 20 tokens:
    48201  " the"
    21045  ","
    ...
```

## Model format

The model is a JSON file with a single `merges` field — a list of `[id_a, id_b]` pairs in the order they were learned.

```json
{
  "merges": [
    [116, 104],
    [256, 101],
    ...
  ]
}
```

Token 0–255 are single bytes. Token `256 + i` is the result of `merges[i]`.

## Library usage

```rust
use rusty_bpe::{PreTokenizer, Trainer, Tokenizer, CorpusStats};

// Train
let pretok = PreTokenizer::new();
let trainer = Trainer::new(8000, 2);
let vocab = trainer.train(&text, &pretok);
vocab.save("model.json").unwrap();

// Parallel train (split corpus into chunks first)
let vocab = trainer.train_parallel(&chunks, &pretok);

// Encode / decode
let tok = Tokenizer::new(vocab, pretok);
let ids = tok.encode("hello world");
let text = tok.decode(&ids);

// Parallel batch encode
let results = tok.encode_batch(&["doc one", "doc two", "doc three"]);

// Corpus statistics
let ids = tok.encode(&corpus);
let stats = CorpusStats::from_tokens(&ids, corpus.len() as u64);
stats.print_summary(&tok);
```

## Module overview

| Module | Description |
|---|---|
| `pretok` | GPT-2 style regex pre-tokenizer; splits text into word-level units before BPE |
| `trainer` | BPE training: lazy max-heap + pair→word index for efficient merge updates |
| `tokenizer` | Encode (greedy lowest-rank merge) and decode; parallel batch encoding |
| `stats` | Token frequency, Shannon entropy, compression ratio, top-N tokens, coverage@k |
