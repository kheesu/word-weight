/// Computes the inverse document frequency weight for a term across a corpus.
///
/// Parameters:
///   num_documents  – total number of documents in the corpus
///   docs_with_term – number of documents containing the term
pub trait IdfWeighting: Send + Sync {
    fn compute(&self, num_documents: u64, docs_with_term: u64) -> f64;
}

// ── Built-in implementations ──────────────────────────────────────────────────

/// idf = 1 (no corpus weighting)
pub struct Unary;
/// idf = ln(N / df)
pub struct StandardIdf;
/// idf = ln(N / (1 + df)) + 1
pub struct SmoothIdf;
/// idf = ln((N − df) / df), clamped to 0
pub struct ProbabilisticIdf;
/// idf = ln((N + 1) / df)
pub struct MaxNormalizedIdf;

impl IdfWeighting for Unary {
    fn compute(&self, _n: u64, _df: u64) -> f64 {
        1.0
    }
}

impl IdfWeighting for StandardIdf {
    fn compute(&self, n: u64, df: u64) -> f64 {
        if n == 0 || df == 0 {
            return 0.0;
        }
        (n as f64 / df as f64).ln()
    }
}

impl IdfWeighting for SmoothIdf {
    fn compute(&self, n: u64, df: u64) -> f64 {
        if n == 0 {
            return 0.0;
        }
        (n as f64 / (1.0 + df as f64)).ln() + 1.0
    }
}

impl IdfWeighting for ProbabilisticIdf {
    fn compute(&self, n: u64, df: u64) -> f64 {
        if df == 0 || df >= n {
            return 0.0;
        }
        ((n - df) as f64 / df as f64).ln().max(0.0)
    }
}

impl IdfWeighting for MaxNormalizedIdf {
    fn compute(&self, n: u64, df: u64) -> f64 {
        if df == 0 {
            return 0.0;
        }
        ((n + 1) as f64 / df as f64).ln()
    }
}

// ── Enum for runtime / CLI / Python dispatch ──────────────────────────────────

#[derive(Clone, Debug, PartialEq)]
pub enum IdfMethod {
    Unary,
    Standard,
    Smooth,
    Probabilistic,
    MaxNormalized,
}

impl Default for IdfMethod {
    fn default() -> Self {
        IdfMethod::Smooth
    }
}

impl IdfWeighting for IdfMethod {
    fn compute(&self, num_documents: u64, docs_with_term: u64) -> f64 {
        match self {
            IdfMethod::Unary => Unary.compute(num_documents, docs_with_term),
            IdfMethod::Standard => StandardIdf.compute(num_documents, docs_with_term),
            IdfMethod::Smooth => SmoothIdf.compute(num_documents, docs_with_term),
            IdfMethod::Probabilistic => ProbabilisticIdf.compute(num_documents, docs_with_term),
            IdfMethod::MaxNormalized => MaxNormalizedIdf.compute(num_documents, docs_with_term),
        }
    }
}

impl std::str::FromStr for IdfMethod {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "unary" => Ok(IdfMethod::Unary),
            "standard" => Ok(IdfMethod::Standard),
            "smooth" => Ok(IdfMethod::Smooth),
            "probabilistic" => Ok(IdfMethod::Probabilistic),
            "max" | "max_normalized" => Ok(IdfMethod::MaxNormalized),
            _ => Err(format!(
                "unknown idf method '{s}'; expected one of: unary, standard, smooth, probabilistic, max"
            )),
        }
    }
}

impl std::fmt::Display for IdfMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IdfMethod::Unary => write!(f, "unary"),
            IdfMethod::Standard => write!(f, "standard"),
            IdfMethod::Smooth => write!(f, "smooth"),
            IdfMethod::Probabilistic => write!(f, "probabilistic"),
            IdfMethod::MaxNormalized => write!(f, "max_normalized"),
        }
    }
}
