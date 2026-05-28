use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

use crate::pretok::PreTokenizer;
use crate::stats::CorpusStats;
use crate::tokenizer::Tokenizer;
use crate::trainer::{Trainer, Vocab};

// ── Vocab ──────────────────────────────────────────────────────────────────

#[pyclass(name = "Vocab")]
pub struct PyVocab {
    pub inner: Vocab,
}

#[pymethods]
impl PyVocab {
    /// Persist the model to a JSON file.
    fn save(&self, path: &str) -> PyResult<()> {
        self.inner
            .save(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
    }

    /// Load a model from a JSON file produced by `save` or `rusty-bpe train`.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        Vocab::load(path)
            .map(|inner| Self { inner })
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
    }

    #[getter]
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    /// Ordered merge rules as a list of (id_a, id_b) tuples.
    #[getter]
    fn merges(&self) -> Vec<(u32, u32)> {
        self.inner.merges.clone()
    }
}

// ── Tokenizer ─────────────────────────────────────────────────────────────

#[pyclass(name = "Tokenizer")]
pub struct PyTokenizer {
    inner: Tokenizer,
}

#[pymethods]
impl PyTokenizer {
    /// Load a tokenizer directly from a saved model file.
    #[new]
    fn new(model_path: &str) -> PyResult<Self> {
        Tokenizer::from_file(model_path)
            .map(|inner| Self { inner })
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
    }

    /// Build a tokenizer from a `Vocab` returned by `train()`.
    #[staticmethod]
    fn from_vocab(vocab: &PyVocab) -> Self {
        Self {
            inner: Tokenizer::new(vocab.inner.clone(), PreTokenizer::new()),
        }
    }

    /// Encode a single string → list of integer token IDs.
    fn encode(&self, text: &str) -> Vec<u32> {
        self.inner.encode(text)
    }

    /// Decode a list of token IDs back to a string.
    fn decode(&self, ids: Vec<u32>) -> String {
        self.inner.decode(&ids)
    }

    /// Encode a list of strings in parallel → list of list[int].
    fn encode_batch(&self, texts: Vec<String>) -> Vec<Vec<u32>> {
        texts.par_iter().map(|t| self.inner.encode(t)).collect()
    }

    #[getter]
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    /// Return the string representation of a single token ID.
    fn token_str(&self, id: u32) -> String {
        self.inner.token_str(id)
    }

    /// Tokenize `text` and return a stats dict suitable for pandas analysis.
    ///
    /// Keys:
    ///   total_tokens      int
    ///   unique_tokens     int
    ///   total_bytes       int
    ///   entropy           float   (bits, Shannon)
    ///   compression_ratio float   (bytes per token)
    ///   token_freqs       list[(token_id, count)]  sorted by count desc
    fn stats<'py>(&self, py: Python<'py>, text: &str) -> PyResult<Bound<'py, PyDict>> {
        let ids = self.inner.encode(text);
        let s = CorpusStats::from_tokens(&ids, text.len() as u64);

        let d = PyDict::new(py);
        d.set_item("total_tokens", s.total_tokens)?;
        d.set_item("unique_tokens", s.unique_tokens)?;
        d.set_item("total_bytes", s.total_bytes)?;
        d.set_item("entropy", s.entropy)?;
        d.set_item("compression_ratio", s.compression_ratio)?;
        d.set_item("token_freqs", s.token_freqs)?;
        Ok(d)
    }

    /// Like `stats` but tokenizes multiple text chunks in parallel.
    fn stats_parallel<'py>(
        &self,
        py: Python<'py>,
        texts: Vec<String>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let total_bytes: u64 = texts.iter().map(|t| t.len() as u64).sum();
        let ids: Vec<u32> = texts.par_iter().flat_map(|t| self.inner.encode(t)).collect();
        let s = CorpusStats::from_tokens(&ids, total_bytes);

        let d = PyDict::new(py);
        d.set_item("total_tokens", s.total_tokens)?;
        d.set_item("unique_tokens", s.unique_tokens)?;
        d.set_item("total_bytes", s.total_bytes)?;
        d.set_item("entropy", s.entropy)?;
        d.set_item("compression_ratio", s.compression_ratio)?;
        d.set_item("token_freqs", s.token_freqs)?;
        Ok(d)
    }
}

// ── Module-level functions ─────────────────────────────────────────────────

/// Train BPE on a text string. Returns a `Vocab` that can be saved or
/// passed directly to `Tokenizer.from_vocab()`.
///
/// Args:
///   text       Training corpus as a single UTF-8 string.
///   vocab_size Target vocabulary size (base 256 bytes + learned merges).
///   min_freq   Minimum word frequency to include (default 2).
#[pyfunction]
#[pyo3(signature = (text, vocab_size, min_freq=2))]
fn train(text: &str, vocab_size: usize, min_freq: u64) -> PyVocab {
    let vocab = Trainer::new(vocab_size, min_freq)
        .train_text_parallel(text, &PreTokenizer::new());
    PyVocab { inner: vocab }
}

// ── Module entry point ─────────────────────────────────────────────────────

#[pymodule]
pub fn rusty_bpe(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyVocab>()?;
    m.add_class::<PyTokenizer>()?;
    m.add_function(wrap_pyfunction!(train, m)?)?;
    Ok(())
}
