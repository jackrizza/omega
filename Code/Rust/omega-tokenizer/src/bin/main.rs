use clap::Parser;
use omega_tokenizer::Tokens;
use std::path::Path;
use std::process::ExitCode;

/// Omega Tokenizer
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Tokenizer JSON file, relative to the project datasets directory or absolute
    #[arg(short, long)]
    file_name: String,

    /// Text to encode and decode
    #[arg(long)]
    test_string: Option<String>,

    /// Add special tokens according to the loaded tokenizer's post-processor
    #[arg(long)]
    add_special_tokens: bool,

    /// Token IDs to decode (comma-separated)
    #[arg(long, value_delimiter = ',', num_args = 1.., conflicts_with = "test_string")]
    decode_ids: Option<Vec<u32>>,

    /// Omit special tokens from decoded text
    #[arg(long)]
    skip_special_tokens: bool,
}

fn run(args: Args) -> Result<(), String> {
    let file_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../datasets")
        .join(&args.file_name);

    let tokens = Tokens::new(&file_path)?;
    log::info!(
        "Loaded tokenizer with {} vocabulary entries",
        tokens.vocab_size()
    );

    if let Some(text) = args.test_string {
        let encoding = tokens.encode(&text, args.add_special_tokens)?;
        println!("Tokens: {:?}", encoding.get_tokens());
        println!("Input IDs: {:?}", encoding.get_ids());
        println!("Attention mask: {:?}", encoding.get_attention_mask());
        println!(
            "Special tokens mask: {:?}",
            encoding.get_special_tokens_mask()
        );
        println!("Offsets: {:?}", encoding.get_offsets());
        println!(
            "Decoded: {:?}",
            tokens.decode(encoding.get_ids(), args.skip_special_tokens)?
        );
    }

    if let Some(ids) = args.decode_ids {
        println!(
            "Decoded: {:?}",
            tokens.decode(&ids, args.skip_special_tokens)?
        );
    }

    Ok(())
}

fn main() -> ExitCode {
    main_with_args(std::env::args_os())
}

fn main_with_args(arguments: impl IntoIterator<Item = std::ffi::OsString>) -> ExitCode {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    match run(Args::parse_from(arguments)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            log::error!("{}", e);
            ExitCode::FAILURE
        }
    }
}
