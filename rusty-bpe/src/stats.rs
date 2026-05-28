use hashbrown::HashMap;
use rayon::prelude::*;

use crate::tokenizer::Tokenizer;

pub struct CorpusStats {
    pub total_tokens: u64,
    pub unique_tokens: usize,
    pub total_bytes: u64,
    pub token_freqs: Vec<(u32, u64)>, // sorted by frequency descending
    pub entropy: f64,
    pub compression_ratio: f64, // bytes per token (higher = less compression)
}

impl CorpusStats {
    pub fn from_tokens(ids: &[u32], total_bytes: u64) -> Self {
        let mut freqs: HashMap<u32, u64> = HashMap::new();
        for &id in ids {
            *freqs.entry(id).or_insert(0) += 1;
        }

        let total = ids.len() as u64;
        let unique = freqs.len();

        let entropy = freqs.values().fold(0.0f64, |acc, &c| {
            let p = c as f64 / total as f64;
            acc - p * p.log2()
        });

        let mut sorted: Vec<(u32, u64)> = freqs.into_iter().collect();
        sorted.sort_unstable_by(|a, b| b.1.cmp(&a.1));

        Self {
            total_tokens: total,
            unique_tokens: unique,
            total_bytes,
            token_freqs: sorted,
            entropy,
            compression_ratio: total_bytes as f64 / total as f64,
        }
    }

    /// Compute stats directly from text using parallel tokenization.
    pub fn from_text_parallel(texts: &[&str], tok: &Tokenizer) -> Self {
        let total_bytes: u64 = texts.iter().map(|t| t.len() as u64).sum();

        let all_ids: Vec<u32> = texts
            .par_iter()
            .flat_map(|text| tok.encode(text))
            .collect();

        Self::from_tokens(&all_ids, total_bytes)
    }

    pub fn top_tokens<'a>(&'a self, n: usize, tok: &'a Tokenizer) -> Vec<(String, u64)> {
        self.token_freqs
            .iter()
            .take(n)
            .map(|&(id, freq)| (tok.token_str(id), freq))
            .collect()
    }

    /// Fraction of token mass covered by the top-k tokens.
    pub fn coverage(&self, k: usize) -> f64 {
        let top: u64 = self.token_freqs.iter().take(k).map(|(_, f)| f).sum();
        top as f64 / self.total_tokens as f64
    }

    pub fn print_summary(&self, tok: &Tokenizer) {
        println!("Total tokens   : {}", self.total_tokens);
        println!("Unique tokens  : {}", self.unique_tokens);
        println!("Vocab size     : {}", tok.vocab_size());
        println!("Total bytes    : {}", self.total_bytes);
        println!("Bytes/token    : {:.3}", self.compression_ratio);
        println!("Token entropy  : {:.4} bits", self.entropy);
        println!("Top-10 coverage: {:.2}%", self.coverage(10) * 100.0);
        println!("\nTop 20 tokens:");
        for (s, freq) in self.top_tokens(20, tok) {
            println!("  {:>8}  {:?}", freq, s);
        }
    }
}
