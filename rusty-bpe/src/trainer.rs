use std::cmp::Ordering;
use std::collections::BinaryHeap;

use hashbrown::{HashMap, HashSet};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::pretok::PreTokenizer;

#[derive(Serialize, Deserialize, Clone)]
pub struct Vocab {
    pub merges: Vec<(u32, u32)>,
}

impl Vocab {
    pub fn token_bytes(&self) -> Vec<Vec<u8>> {
        let mut bytes: Vec<Vec<u8>> = (0u8..=255).map(|b| vec![b]).collect();
        for &(a, b) in &self.merges {
            let merged = [bytes[a as usize].as_slice(), bytes[b as usize].as_slice()].concat();
            bytes.push(merged);
        }
        bytes
    }

    pub fn vocab_size(&self) -> usize {
        256 + self.merges.len()
    }

    pub fn save(&self, path: &str) -> std::io::Result<()> {
        std::fs::write(path, serde_json::to_string_pretty(self).unwrap())
    }

    pub fn load(path: &str) -> std::io::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&json).unwrap())
    }
}

#[derive(Eq, PartialEq)]
struct HeapEntry {
    freq: u64,
    pair: (u32, u32),
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.freq
            .cmp(&other.freq)
            .then_with(|| other.pair.cmp(&self.pair))
    }
}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub struct Trainer {
    pub vocab_size: usize,
    pub min_frequency: u64,
}

impl Default for Trainer {
    fn default() -> Self {
        Self {
            vocab_size: 1000,
            min_frequency: 2,
        }
    }
}

impl Trainer {
    pub fn new(vocab_size: usize, min_frequency: u64) -> Self {
        Self {
            vocab_size,
            min_frequency,
        }
    }

    pub fn train(&self, text: &str, pretok: &PreTokenizer) -> Vocab {
        let mut word_counts: HashMap<Vec<u8>, u64> = HashMap::new();
        for word in pretok.split(text) {
            *word_counts.entry(word.as_bytes().to_vec()).or_insert(0) += 1;
        }
        self.train_from_counts(word_counts)
    }

    /// Convenience wrapper: splits `text` into per-thread chunks then calls `train_parallel`.
    pub fn train_text_parallel(&self, text: &str, pretok: &PreTokenizer) -> Vocab {
        let num_threads = rayon::current_num_threads().max(1);
        let chunk_size = (text.len() / num_threads).max(1);
        let chunks: Vec<&str> = text
            .as_bytes()
            .chunks(chunk_size)
            .map(|c| {
                std::str::from_utf8(c).unwrap_or_else(|e| {
                    std::str::from_utf8(&c[..e.valid_up_to()]).unwrap_or("")
                })
            })
            .collect();
        self.train_parallel(&chunks, pretok)
    }

    /// Parallel training — splits text into chunks and merges word counts via rayon.
    pub fn train_parallel(&self, chunks: &[&str], pretok: &PreTokenizer) -> Vocab {
        let word_counts = chunks
            .par_iter()
            .fold(
                || HashMap::<Vec<u8>, u64>::new(),
                |mut acc, chunk| {
                    for word in pretok.split(chunk) {
                        *acc.entry(word.as_bytes().to_vec()).or_insert(0) += 1;
                    }
                    acc
                },
            )
            .reduce(
                || HashMap::new(),
                |mut a, b| {
                    for (k, v) in b {
                        *a.entry(k).or_insert(0) += v;
                    }
                    a
                },
            );
        self.train_from_counts(word_counts)
    }

    fn train_from_counts(&self, word_counts: HashMap<Vec<u8>, u64>) -> Vocab {
        if self.vocab_size <= 256 {
            return Vocab { merges: Vec::new() };
        }
        let num_merges = self.vocab_size - 256;

        // Each word is a sequence of token IDs (initially byte values) plus its corpus frequency.
        let mut words: Vec<(Vec<u32>, u64)> = word_counts
            .into_iter()
            .filter(|(_, v)| *v >= self.min_frequency)
            .map(|(k, v)| (k.iter().map(|&b| b as u32).collect(), v))
            .collect();

        // Global pair frequencies (signed so we can subtract safely).
        let mut pair_freqs: HashMap<(u32, u32), i64> = HashMap::new();
        // Index: pair -> set of word indices that contain it.
        let mut pair_to_words: HashMap<(u32, u32), HashSet<usize>> = HashMap::new();

        for (wi, (tokens, freq)) in words.iter().enumerate() {
            for w in tokens.windows(2) {
                let p = (w[0], w[1]);
                *pair_freqs.entry(p).or_insert(0) += *freq as i64;
                pair_to_words.entry(p).or_default().insert(wi);
            }
        }

        // Lazy max-heap: stale entries are skipped when the stored freq diverges from pair_freqs.
        let mut heap: BinaryHeap<HeapEntry> = pair_freqs
            .iter()
            .filter(|&(_, &f)| f > 0)
            .map(|(&pair, &freq)| HeapEntry {
                freq: freq as u64,
                pair,
            })
            .collect();

        let mut merges: Vec<(u32, u32)> = Vec::with_capacity(num_merges);

        while merges.len() < num_merges {
            // Pop until we find a valid (non-stale) entry.
            let best = loop {
                let Some(entry) = heap.pop() else {
                    break None;
                };
                let cur = pair_freqs.get(&entry.pair).copied().unwrap_or(0);
                if cur <= 0 {
                    continue;
                }
                if entry.freq as i64 == cur {
                    break Some(entry.pair);
                }
                // Stale — re-insert with current frequency.
                heap.push(HeapEntry {
                    freq: cur as u64,
                    pair: entry.pair,
                });
            };
            let Some(pair) = best else { break };

            let new_id = (256 + merges.len()) as u32;
            merges.push(pair);

            // Drain the affected word set — pair_to_words[pair] is now stale and will be removed.
            let affected: Vec<usize> = pair_to_words
                .remove(&pair)
                .map(|s| s.into_iter().collect())
                .unwrap_or_default();

            for wi in affected {
                let (tokens, freq) = &mut words[wi];
                let freq = *freq;
                let old_tokens = std::mem::take(tokens);

                // Count every adjacent pair in the old sequence.
                let mut old_counts: HashMap<(u32, u32), i64> = HashMap::new();
                for w in old_tokens.windows(2) {
                    *old_counts.entry((w[0], w[1])).or_insert(0) += 1;
                }

                // Apply the merge (greedy left-to-right, non-overlapping).
                let mut new_tokens: Vec<u32> = Vec::with_capacity(old_tokens.len());
                let mut i = 0;
                while i < old_tokens.len() {
                    if i + 1 < old_tokens.len()
                        && old_tokens[i] == pair.0
                        && old_tokens[i + 1] == pair.1
                    {
                        new_tokens.push(new_id);
                        i += 2;
                    } else {
                        new_tokens.push(old_tokens[i]);
                        i += 1;
                    }
                }

                // Count every adjacent pair in the new sequence.
                let mut new_counts: HashMap<(u32, u32), i64> = HashMap::new();
                for w in new_tokens.windows(2) {
                    *new_counts.entry((w[0], w[1])).or_insert(0) += 1;
                }

                // Subtract old pair contributions.
                for (&p, &cnt) in &old_counts {
                    *pair_freqs.entry(p).or_insert(0) -= freq as i64 * cnt;
                    // Don't touch pair_to_words[pair] — it was already removed above.
                    if p != pair {
                        if let Some(set) = pair_to_words.get_mut(&p) {
                            set.remove(&wi);
                        }
                    }
                }

                // Add new pair contributions.
                for (&p, &cnt) in &new_counts {
                    let e = pair_freqs.entry(p).or_insert(0);
                    *e += freq as i64 * cnt;
                    if *e > 0 {
                        heap.push(HeapEntry {
                            freq: *e as u64,
                            pair: p,
                        });
                    }
                    pair_to_words.entry(p).or_default().insert(wi);
                }

                *tokens = new_tokens;
            }

            // Ensure the merged pair won't appear again.
            pair_freqs.insert(pair, 0);
        }

        Vocab { merges }
    }
}
