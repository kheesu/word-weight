use rusty_bpe::{CorpusStats, PreTokenizer, Tokenizer, Trainer};

fn usage() -> ! {
    eprintln!(
        "Usage:
  rusty-bpe train   <input> <vocab_size> <output_model> [min_freq]
  rusty-bpe encode  <model> <input>
  rusty-bpe decode  <model> <input>
  rusty-bpe stats   <model> <input>"
    );
    std::process::exit(1);
}

fn read_file(path: &str) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|e| {
        eprintln!("Error reading {path}: {e}");
        std::process::exit(1);
    })
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        usage();
    }

    match args[1].as_str() {
        "train" => {
            if args.len() < 5 {
                usage();
            }
            let input = &args[2];
            let vocab_size: usize = args[3].parse().expect("vocab_size must be an integer");
            let output = &args[4];
            let min_freq: u64 = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(2);

            let text = read_file(input);
            let pretok = PreTokenizer::new();
            let trainer = Trainer::new(vocab_size, min_freq);

            eprintln!(
                "Training BPE: vocab_size={vocab_size} min_freq={min_freq} input_bytes={}",
                text.len()
            );

            // Split text into chunks for parallel training.
            let chunk_size = (text.len() / rayon::current_num_threads()).max(1);
            let chunks: Vec<&str> = text
                .as_bytes()
                .chunks(chunk_size)
                .map(|c| {
                    // Align to valid UTF-8 boundary.
                    let s = std::str::from_utf8(c).unwrap_or_else(|e| {
                        std::str::from_utf8(&c[..e.valid_up_to()]).unwrap_or("")
                    });
                    s
                })
                .collect();

            let vocab = trainer.train_parallel(&chunks, &pretok);
            eprintln!("Learned {} merges → {} tokens", vocab.merges.len(), vocab.vocab_size());
            vocab.save(output).expect("Failed to save model");
            eprintln!("Model saved to {output}");
        }

        "encode" => {
            if args.len() < 4 {
                usage();
            }
            let tok = Tokenizer::from_file(&args[2]).expect("Failed to load model");
            let text = read_file(&args[3]);
            let ids = tok.encode(&text);
            let out: Vec<String> = ids.iter().map(|id| id.to_string()).collect();
            println!("{}", out.join(" "));
        }

        "decode" => {
            if args.len() < 4 {
                usage();
            }
            let tok = Tokenizer::from_file(&args[2]).expect("Failed to load model");
            let input = read_file(&args[3]);
            let ids: Vec<u32> = input
                .split_whitespace()
                .filter_map(|s| s.parse().ok())
                .collect();
            print!("{}", tok.decode(&ids));
        }

        "stats" => {
            if args.len() < 4 {
                usage();
            }
            let tok = Tokenizer::from_file(&args[2]).expect("Failed to load model");
            let text = read_file(&args[3]);
            let total_bytes = text.len() as u64;
            let ids = tok.encode(&text);
            let stats = CorpusStats::from_tokens(&ids, total_bytes);
            stats.print_summary(&tok);
        }

        _ => usage(),
    }
}
