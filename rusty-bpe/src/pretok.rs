use regex::Regex;
use std::sync::OnceLock;

// GPT-2 style pattern (simplified — regex crate lacks lookahead support)
static PATTERN: OnceLock<Regex> = OnceLock::new();

fn pattern() -> &'static Regex {
    PATTERN.get_or_init(|| {
        Regex::new(
            r"(?:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}+| ?[^\s\p{L}\p{N}]+[\r\n]*|\s+"
        )
        .unwrap()
    })
}

pub struct PreTokenizer;

impl PreTokenizer {
    pub fn new() -> Self {
        Self
    }

    pub fn split<'a>(&self, text: &'a str) -> impl Iterator<Item = &'a str> {
        pattern().find_iter(text).map(|m| m.as_str())
    }

    pub fn split_owned(&self, text: &str) -> Vec<Vec<u8>> {
        pattern()
            .find_iter(text)
            .map(|m| m.as_str().as_bytes().to_vec())
            .collect()
    }
}
