use std::{
    path::{Path, PathBuf},
    process::ExitCode,
};

use burn::tensor::backend::Backend;
use clap::{Parser, Subcommand};
use omega_nn::{
    Cpu, GenerationOptions, GptConfig, encode_text, generate_with_options, token_tensor,
    train_on_tokens, validate_tokenizer,
};
use omega_tokenizer::Tokens;

/// CPU smoke tests for a small decoder-only GPT. Weights are initialized anew each run.
#[derive(Debug, Parser)]
#[command(version, about)]
struct Args {
    /// Tokenizer JSON, relative to the repository datasets directory or absolute
    #[arg(short = 'f', long, default_value = "test.json")]
    tokenizer: PathBuf,
    #[arg(long, default_value_t = 64)]
    context_length: usize,
    #[arg(long, default_value_t = 32)]
    d_model: usize,
    #[arg(long, default_value_t = 4)]
    heads: usize,
    #[arg(long, default_value_t = 2)]
    layers: usize,
    #[arg(long, default_value_t = 128)]
    d_ff: usize,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Encode text and inspect a randomly initialized model's logits
    Forward {
        #[arg(long)]
        text: String,
    },
    /// Overfit one short text, then generate with the in-memory trained model
    Train {
        /// Literal training text; must encode to 2..=context_length+1 tokens
        #[arg(long)]
        text: String,
        #[arg(long, default_value_t = 100)]
        steps: usize,
        #[arg(long, default_value_t = 0.003)]
        learning_rate: f64,
        #[arg(long)]
        prompt: String,
        #[arg(long, default_value_t = 8)]
        max_new_tokens: usize,
        /// Stop after generating this tokenizer ID; no EOS is assumed by default
        #[arg(long)]
        eos_token_id: Option<u32>,
        /// Sample tokens instead of choosing the highest logit
        #[arg(long)]
        sample: bool,
        /// Positive finite sampling temperature (default 1 with --sample)
        #[arg(long)]
        temperature: Option<f64>,
        /// Retain at most this many tokens before nucleus filtering
        #[arg(long)]
        top_k: Option<usize>,
        /// Nucleus probability in (0, 1], measured after top-k normalization
        #[arg(long)]
        top_p: Option<f64>,
        /// Request-local sampling seed (default 42); separate from model --seed
        #[arg(long)]
        sampling_seed: Option<u64>,
    },
}

fn run(args: Args) -> Result<(), String> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../datasets")
        .join(args.tokenizer);
    let tokenizer = Tokens::new(path)?;
    let config = GptConfig {
        vocab_size: validate_tokenizer(&tokenizer)?,
        context_length: args.context_length,
        d_model: args.d_model,
        num_heads: args.heads,
        num_layers: args.layers,
        d_ff: args.d_ff,
    };
    config.validate()?;
    println!("CPU model: {config:?}");
    match args.command {
        Command::Forward { text } => {
            let ids = encode_text(&tokenizer, &text)?;
            let device = Default::default();
            let input =
                token_tensor::<Cpu>(&ids, config.vocab_size, config.context_length, &device)?;
            Cpu::seed(args.seed);
            let model = config.init::<Cpu>(&device)?;
            let logits = model.forward(input);
            println!("Input IDs: {ids:?}");
            println!(
                "Logits shape [batch, sequence, vocabulary]: {:?}",
                logits.dims()
            );
            let next_ids = logits
                .argmax(2)
                .into_data()
                .to_vec::<i64>()
                .map_err(|e| format!("Failed to read predictions: {e:?}"))?;
            println!("Random-weight next-token predictions: {next_ids:?}");
        }
        Command::Train {
            text,
            steps,
            learning_rate,
            prompt,
            max_new_tokens,
            eos_token_id,
            sample,
            temperature,
            top_k,
            top_p,
            sampling_seed,
        } => {
            let options = GenerationOptions::from_sampling_flags(
                sample,
                temperature,
                top_k,
                top_p,
                sampling_seed,
            )?;
            options.validate(config.vocab_size)?;
            let ids = encode_text(&tokenizer, &text)?;
            let prompt_ids = encode_text(&tokenizer, &prompt)?;
            if prompt_ids
                .len()
                .checked_add(max_new_tokens)
                .is_none_or(|length| length > config.context_length)
            {
                return Err("Prompt plus max_new_tokens must fit context_length".into());
            }
            if eos_token_id.is_some_and(|id| id as usize >= config.vocab_size) {
                return Err("EOS token ID is outside the tokenizer vocabulary".into());
            }
            println!(
                "Training on {} tokens for {steps} steps (weights are not saved)",
                ids.len()
            );
            let (model, losses) = train_on_tokens(&config, &ids, steps, learning_rate, args.seed)?;
            println!(
                "Cross-entropy: {:.6} -> {:.6} (pre-update losses)",
                losses[0],
                losses[losses.len() - 1]
            );
            let output = generate_with_options(
                &model,
                &config,
                &prompt_ids,
                max_new_tokens,
                eos_token_id,
                &options,
            )?;
            println!("Generated IDs (including prompt): {output:?}");
            println!("Decoded: {}", tokenizer.decode(&output, false)?);
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    main_with_args(std::env::args_os())
}

fn main_with_args(args: impl IntoIterator<Item = std::ffi::OsString>) -> ExitCode {
    match run(Args::parse_from(args)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("Error: {error}");
            ExitCode::FAILURE
        }
    }
}
