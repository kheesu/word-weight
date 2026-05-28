/// Computes a term's weight within a single document.
///
/// Parameters passed to every implementation:
///   term_count    – how many times the term appears in the document
///   total_terms   – total number of tokens in the document
///   max_term_count – count of the most frequent term in the document
///                   (used by AugmentedFrequency; others may ignore it)
pub trait TfWeighting: Send + Sync {
    fn compute(&self, term_count: u64, total_terms: u64, max_term_count: u64) -> f64;
}

// ── Built-in implementations ──────────────────────────────────────────────────

pub struct RawCount;
pub struct RelativeFrequency;
pub struct LogNormalized;
pub struct BinaryTf;
/// tf = K + (1 − K) × count / max_count.  Typical K = 0.5.
pub struct AugmentedFrequency(pub f64);

impl TfWeighting for RawCount {
    fn compute(&self, term_count: u64, _total: u64, _max: u64) -> f64 {
        term_count as f64
    }
}

impl TfWeighting for RelativeFrequency {
    fn compute(&self, term_count: u64, total_terms: u64, _max: u64) -> f64 {
        if total_terms == 0 {
            return 0.0;
        }
        term_count as f64 / total_terms as f64
    }
}

impl TfWeighting for LogNormalized {
    fn compute(&self, term_count: u64, _total: u64, _max: u64) -> f64 {
        if term_count == 0 {
            return 0.0;
        }
        1.0 + (term_count as f64).ln()
    }
}

impl TfWeighting for BinaryTf {
    fn compute(&self, term_count: u64, _total: u64, _max: u64) -> f64 {
        if term_count > 0 { 1.0 } else { 0.0 }
    }
}

impl TfWeighting for AugmentedFrequency {
    fn compute(&self, term_count: u64, _total: u64, max_term_count: u64) -> f64 {
        if max_term_count == 0 {
            return 0.0;
        }
        let k = self.0;
        k + (1.0 - k) * (term_count as f64 / max_term_count as f64)
    }
}

// ── Enum for runtime / CLI / Python dispatch ──────────────────────────────────

#[derive(Clone, Debug, PartialEq)]
pub enum TfMethod {
    RawCount,
    RelativeFrequency,
    LogNormalized,
    Binary,
    /// K parameter (typically 0.5) for the augmented frequency formula.
    Augmented(f64),
}

impl Default for TfMethod {
    fn default() -> Self {
        TfMethod::RelativeFrequency
    }
}

impl TfWeighting for TfMethod {
    fn compute(&self, term_count: u64, total_terms: u64, max_term_count: u64) -> f64 {
        match self {
            TfMethod::RawCount => RawCount.compute(term_count, total_terms, max_term_count),
            TfMethod::RelativeFrequency => {
                RelativeFrequency.compute(term_count, total_terms, max_term_count)
            }
            TfMethod::LogNormalized => {
                LogNormalized.compute(term_count, total_terms, max_term_count)
            }
            TfMethod::Binary => BinaryTf.compute(term_count, total_terms, max_term_count),
            TfMethod::Augmented(k) => {
                AugmentedFrequency(*k).compute(term_count, total_terms, max_term_count)
            }
        }
    }
}

impl std::str::FromStr for TfMethod {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "raw" | "raw_count" => Ok(TfMethod::RawCount),
            "relative" | "relative_frequency" => Ok(TfMethod::RelativeFrequency),
            "log" | "log_normalized" => Ok(TfMethod::LogNormalized),
            "binary" => Ok(TfMethod::Binary),
            "augmented" => Ok(TfMethod::Augmented(0.5)),
            _ => Err(format!(
                "unknown tf method '{s}'; expected one of: raw, relative, log, binary, augmented"
            )),
        }
    }
}

impl std::fmt::Display for TfMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TfMethod::RawCount => write!(f, "raw"),
            TfMethod::RelativeFrequency => write!(f, "relative"),
            TfMethod::LogNormalized => write!(f, "log"),
            TfMethod::Binary => write!(f, "binary"),
            TfMethod::Augmented(k) => write!(f, "augmented(k={k})"),
        }
    }
}
