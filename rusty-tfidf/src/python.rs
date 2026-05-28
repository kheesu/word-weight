use pyo3::prelude::*;

use crate::{tokenize as rust_tokenize, IdfMethod, TfIdfEngine, TfMethod};

// ── TF method enum ─────────────────────────────────────────────────────────────

#[pyclass(name = "TfMethod", eq)]
#[derive(Clone, Debug, PartialEq)]
pub enum PyTfMethod {
    RawCount,
    RelativeFrequency,
    LogNormalized,
    Binary,
    Augmented,
}

#[pymethods]
impl PyTfMethod {
    fn __repr__(&self) -> &'static str {
        match self {
            PyTfMethod::RawCount => "TfMethod.RawCount",
            PyTfMethod::RelativeFrequency => "TfMethod.RelativeFrequency",
            PyTfMethod::LogNormalized => "TfMethod.LogNormalized",
            PyTfMethod::Binary => "TfMethod.Binary",
            PyTfMethod::Augmented => "TfMethod.Augmented",
        }
    }
}

// ── IDF method enum ────────────────────────────────────────────────────────────

#[pyclass(name = "IdfMethod", eq)]
#[derive(Clone, Debug, PartialEq)]
pub enum PyIdfMethod {
    Unary,
    Standard,
    Smooth,
    Probabilistic,
    MaxNormalized,
}

#[pymethods]
impl PyIdfMethod {
    fn __repr__(&self) -> &'static str {
        match self {
            PyIdfMethod::Unary => "IdfMethod.Unary",
            PyIdfMethod::Standard => "IdfMethod.Standard",
            PyIdfMethod::Smooth => "IdfMethod.Smooth",
            PyIdfMethod::Probabilistic => "IdfMethod.Probabilistic",
            PyIdfMethod::MaxNormalized => "IdfMethod.MaxNormalized",
        }
    }
}

// ── TfIdf class ────────────────────────────────────────────────────────────────

#[pyclass(name = "TfIdf")]
pub struct PyTfIdf {
    inner: TfIdfEngine,
}

#[pymethods]
impl PyTfIdf {
    /// Create a TF-IDF engine.
    ///
    /// Args:
    ///   tf_method    TfMethod variant (default: RelativeFrequency)
    ///   idf_method   IdfMethod variant (default: Smooth)
    ///   augmented_k  K parameter for TfMethod.Augmented (default: 0.5)
    #[new]
    #[pyo3(signature = (tf_method = None, idf_method = None, augmented_k = 0.5))]
    fn new(
        tf_method: Option<PyTfMethod>,
        idf_method: Option<PyIdfMethod>,
        augmented_k: f64,
    ) -> Self {
        let tf = match tf_method.unwrap_or(PyTfMethod::RelativeFrequency) {
            PyTfMethod::RawCount => TfMethod::RawCount,
            PyTfMethod::RelativeFrequency => TfMethod::RelativeFrequency,
            PyTfMethod::LogNormalized => TfMethod::LogNormalized,
            PyTfMethod::Binary => TfMethod::Binary,
            PyTfMethod::Augmented => TfMethod::Augmented(augmented_k),
        };
        let idf = match idf_method.unwrap_or(PyIdfMethod::Smooth) {
            PyIdfMethod::Unary => IdfMethod::Unary,
            PyIdfMethod::Standard => IdfMethod::Standard,
            PyIdfMethod::Smooth => IdfMethod::Smooth,
            PyIdfMethod::Probabilistic => IdfMethod::Probabilistic,
            PyIdfMethod::MaxNormalized => IdfMethod::MaxNormalized,
        };
        PyTfIdf { inner: TfIdfEngine::new(tf, idf) }
    }

    /// Add a pre-tokenized document to the IDF corpus.
    fn add_document(&mut self, tokens: Vec<String>) {
        self.inner.add_document(&tokens);
    }

    /// TF-IDF score for a single term against a document.
    fn score(&self, term: &str, document_tokens: Vec<String>) -> f64 {
        self.inner.score(term, &document_tokens)
    }

    /// TF-IDF scores for every unique term in the document.
    /// Returns a dict mapping term → score.
    fn scores_for_document(&self, tokens: Vec<String>) -> std::collections::HashMap<String, f64> {
        self.inner.scores_for_document(&tokens)
    }

    /// Top `n` terms by descending TF-IDF score.
    /// Returns a list of (term, score) tuples.
    #[pyo3(signature = (tokens, n = 10))]
    fn top_terms(&self, tokens: Vec<String>, n: usize) -> Vec<(String, f64)> {
        self.inner.top_terms(&tokens, n)
    }

    #[getter]
    fn doc_count(&self) -> u64 {
        self.inner.corpus().doc_count()
    }

    #[getter]
    fn vocabulary_size(&self) -> usize {
        self.inner.corpus().vocabulary_size()
    }
}

// ── Free functions ─────────────────────────────────────────────────────────────

/// Tokenize a string into lowercase alphanumeric tokens.
#[pyfunction]
fn tokenize(text: &str) -> Vec<String> {
    rust_tokenize(text)
}

// ── Module entry point ─────────────────────────────────────────────────────────

#[pymodule]
pub fn rusty_tfidf(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTfIdf>()?;
    m.add_class::<PyTfMethod>()?;
    m.add_class::<PyIdfMethod>()?;
    m.add_function(wrap_pyfunction!(tokenize, m)?)?;
    Ok(())
}
