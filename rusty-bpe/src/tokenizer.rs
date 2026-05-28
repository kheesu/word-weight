use hashbrown::HashMap;
use rayon::prelude::*;

use crate::pretok::PreTokenizer;
use crate::trainer::Vocab;

pub struct Tokenizer {
    token_bytes: Vec<Vec<u8>>,
    merge_rank: HashMap<(u32, u32), u32>, // pair -> merge index (lower = higher priority)
    pretok: PreTokenizer,
}

impl Tokenizer {
    pub fn new(vocab: Vocab, pretok: PreTokenizer) -> Self {
        let merge_rank: HashMap<(u32, u32), u32> = vocab
            .merges
            .iter()
            .enumerate()
            .map(|(i, &pair)| (pair, i as u32))
            .collect();
        let token_bytes = vocab.token_bytes();
        Self {
            token_bytes,
            merge_rank,
            pretok,
        }
    }

    pub fn from_file(path: &str) -> std::io::Result<Self> {
        let vocab = Vocab::load(path)?;
        Ok(Self::new(vocab, PreTokenizer::new()))
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        self.pretok
            .split(text)
            .flat_map(|word| self.encode_word(word.as_bytes()))
            .collect()
    }

    /// Parallel batch encode — each string is processed independently.
    pub fn encode_batch(&self, texts: &[&str]) -> Vec<Vec<u32>> {
        texts
            .par_iter()
            .map(|text| self.encode(text))
            .collect()
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let bytes: Vec<u8> = ids
            .iter()
            .flat_map(|&id| self.token_bytes.get(id as usize).map(|b| b.as_slice()).unwrap_or(&[]))
            .copied()
            .collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }

    pub fn vocab_size(&self) -> usize {
        self.token_bytes.len()
    }

    pub fn token_str(&self, id: u32) -> String {
        String::from_utf8_lossy(&self.token_bytes[id as usize]).into_owned()
    }

    fn encode_word(&self, bytes: &[u8]) -> Vec<u32> {
        let mut tokens: Vec<u32> = bytes.iter().map(|&b| b as u32).collect();

        loop {
            if tokens.len() < 2 {
                break;
            }
            // Find the pair with the lowest merge rank (= earliest merge rule).
            let best = tokens
                .windows(2)
                .enumerate()
                .filter_map(|(i, w)| {
                    let pair = (w[0], w[1]);
                    self.merge_rank.get(&pair).map(|&rank| (i, pair, rank))
                })
                .min_by_key(|&(_, _, rank)| rank);

            let Some((_, pair, rank)) = best else { break };

            let new_id = 256 + rank;
            let mut new_tokens: Vec<u32> = Vec::with_capacity(tokens.len());
            let mut i = 0;
            while i < tokens.len() {
                if i + 1 < tokens.len() && tokens[i] == pair.0 && tokens[i + 1] == pair.1 {
                    new_tokens.push(new_id);
                    i += 2;
                } else {
                    new_tokens.push(tokens[i]);
                    i += 1;
                }
            }
            tokens = new_tokens;
        }

        tokens
    }
}
