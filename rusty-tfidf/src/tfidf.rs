use std::collections::HashMap;

use crate::{Corpus, IdfMethod, IdfWeighting, TfMethod, TfWeighting};

/// Per-document term statistics needed by all TF weighting methods.
pub struct DocumentStats<'a> {
    pub counts: HashMap<&'a str, u64>,
    pub total_terms: u64,
    pub max_term_count: u64,
}

impl<'a> DocumentStats<'a> {
    pub fn from_tokens(tokens: &'a [String]) -> Self {
        let mut counts: HashMap<&str, u64> = HashMap::new();
        for token in tokens {
            *counts.entry(token.as_str()).or_insert(0) += 1;
        }
        let total_terms = tokens.len() as u64;
        let max_term_count = counts.values().copied().max().unwrap_or(0);
        Self { counts, total_terms, max_term_count }
    }
}

/// TF-IDF engine with pluggable weighting strategies.
///
/// `T` implements [`TfWeighting`] and `I` implements [`IdfWeighting`].
/// For runtime-selected strategies use the [`TfIdfEngine`] type alias.
pub struct TfIdf<T: TfWeighting, I: IdfWeighting> {
    tf_method: T,
    idf_method: I,
    corpus: Corpus,
}

impl<T: TfWeighting, I: IdfWeighting> TfIdf<T, I> {
    pub fn new(tf_method: T, idf_method: I) -> Self {
        Self { tf_method, idf_method, corpus: Corpus::new() }
    }

    /// Add a pre-tokenized document to the IDF corpus.
    pub fn add_document(&mut self, tokens: &[String]) {
        self.corpus.add_document(tokens);
    }

    pub fn corpus(&self) -> &Corpus {
        &self.corpus
    }

    /// TF-IDF score for a single term against a document.
    pub fn score(&self, term: &str, document_tokens: &[String]) -> f64 {
        let stats = DocumentStats::from_tokens(document_tokens);
        let term_count = stats.counts.get(term).copied().unwrap_or(0);
        let tf = self.tf_method.compute(term_count, stats.total_terms, stats.max_term_count);
        let idf = self
            .idf_method
            .compute(self.corpus.doc_count(), self.corpus.doc_frequency(term));
        tf * idf
    }

    /// TF-IDF scores for every unique term in a document.
    pub fn scores_for_document(&self, tokens: &[String]) -> HashMap<String, f64> {
        let stats = DocumentStats::from_tokens(tokens);
        stats
            .counts
            .iter()
            .map(|(term, &count)| {
                let tf = self.tf_method.compute(count, stats.total_terms, stats.max_term_count);
                let idf = self
                    .idf_method
                    .compute(self.corpus.doc_count(), self.corpus.doc_frequency(term));
                (term.to_string(), tf * idf)
            })
            .collect()
    }

    /// Top `n` terms by descending TF-IDF score.
    pub fn top_terms(&self, tokens: &[String], n: usize) -> Vec<(String, f64)> {
        let mut scores: Vec<(String, f64)> = self.scores_for_document(tokens).into_iter().collect();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(n);
        scores
    }
}

/// Convenience alias using [`TfMethod`] and [`IdfMethod`] enum dispatch.
pub type TfIdfEngine = TfIdf<TfMethod, IdfMethod>;
