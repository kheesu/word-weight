use std::fs;
use std::io::{self, Read};
use std::path::PathBuf;
use std::process;

use clap::Parser;
use rusty_tfidf::{tokenize, IdfMethod, TfIdfEngine, TfMethod};

#[derive(Parser)]
#[command(
    name = "rusty-tfidf",
    about = "Compute TF-IDF scores across a set of documents.\n\
             Each FILE is treated as one document. Reads stdin as a single document when no files are given."
)]
struct Args {
    /// Input files (one document per file)
    files: Vec<PathBuf>,

    /// TF weighting: raw, relative, log, binary, augmented
    #[arg(long, default_value = "relative")]
    tf: String,

    /// IDF weighting: unary, standard, smooth, probabilistic, max
    #[arg(long, default_value = "smooth")]
    idf: String,

    /// Show top N terms per document
    #[arg(long, short = 'n', default_value = "10")]
    top: usize,

    /// Output format: text or json
    #[arg(long, default_value = "text")]
    output: String,
}

fn main() {
    let args = Args::parse();

    let tf: TfMethod = args.tf.parse().unwrap_or_else(|e: String| {
        eprintln!("error: {e}");
        process::exit(1);
    });
    let idf: IdfMethod = args.idf.parse().unwrap_or_else(|e: String| {
        eprintln!("error: {e}");
        process::exit(1);
    });

    // ── Load documents ────────────────────────────────────────────────────────

    let documents: Vec<(String, Vec<String>)> = if args.files.is_empty() {
        let mut text = String::new();
        io::stdin().read_to_string(&mut text).unwrap_or_else(|e| {
            eprintln!("error reading stdin: {e}");
            process::exit(1);
        });
        vec![("<stdin>".to_string(), tokenize(&text))]
    } else {
        args.files
            .iter()
            .map(|path| {
                let text = fs::read_to_string(path).unwrap_or_else(|e| {
                    eprintln!("error reading {}: {e}", path.display());
                    process::exit(1);
                });
                let label = path.display().to_string();
                (label, tokenize(&text))
            })
            .collect()
    };

    // ── Build corpus ──────────────────────────────────────────────────────────

    let mut engine = TfIdfEngine::new(tf, idf);
    for (_, tokens) in &documents {
        engine.add_document(tokens);
    }

    // ── Output ────────────────────────────────────────────────────────────────

    match args.output.as_str() {
        "json" => print_json(&engine, &documents, args.top),
        _ => print_text(&engine, &documents, args.top),
    }
}

fn print_text(engine: &TfIdfEngine, documents: &[(String, Vec<String>)], top: usize) {
    for (label, tokens) in documents {
        println!("=== {label} ===");
        for (term, score) in engine.top_terms(tokens, top) {
            println!("  {term:<30} {score:.6}");
        }
        println!();
    }
}

fn print_json(engine: &TfIdfEngine, documents: &[(String, Vec<String>)], top: usize) {
    let results: Vec<serde_json::Value> = documents
        .iter()
        .map(|(label, tokens)| {
            let terms: Vec<serde_json::Value> = engine
                .top_terms(tokens, top)
                .into_iter()
                .map(|(term, score)| serde_json::json!({"term": term, "score": score}))
                .collect();
            serde_json::json!({"document": label, "terms": terms})
        })
        .collect();

    println!("{}", serde_json::to_string_pretty(&results).unwrap());
}
