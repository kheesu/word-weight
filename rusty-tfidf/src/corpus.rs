use std::collections::{HashMap, HashSet};

/// Tracks document frequencies across a corpus for IDF computation.
#[derive(Clone, Debug, Default)]
pub struct Corpus {
    doc_count: u64,
    /// term → number of documents that contain it
    doc_frequencies: HashMap<String, u64>,
}

impl Corpus {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a tokenized document, updating document frequencies.
    pub fn add_document(&mut self, tokens: &[String]) {
        self.doc_count += 1;
        let unique: HashSet<&str> = tokens.iter().map(String::as_str).collect();
        for term in unique {
            *self.doc_frequencies.entry(term.to_string()).or_insert(0) += 1;
        }
    }

    pub fn doc_count(&self) -> u64 {
        self.doc_count
    }

    /// Number of documents in the corpus that contain `term`.
    pub fn doc_frequency(&self, term: &str) -> u64 {
        self.doc_frequencies.get(term).copied().unwrap_or(0)
    }

    pub fn vocabulary_size(&self) -> usize {
        self.doc_frequencies.len()
    }

    pub fn terms(&self) -> impl Iterator<Item = &str> {
        self.doc_frequencies.keys().map(String::as_str)
    }
}
