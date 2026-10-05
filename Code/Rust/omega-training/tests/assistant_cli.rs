//! Real unique-binary CLI acceptance on tiny owned fixtures, not model-quality evidence.
use std::{
    fs,
    path::Path,
    process::{Command, Output, Stdio},
    sync::atomic::{AtomicU64, Ordering},
    time::{Duration, Instant},
};

use burn::{
    module::{Module, ModuleMapper, ParamId},
    tensor::{Tensor, TensorData},
};
use omega_nn::{Cpu, GptConfig};
use omega_tokenizer::{
    ByteBpeConfig, Tokens,
    chat::{CHAT_PROTOCOL_VERSION, CHAT_TARGET_OBJECTIVE, ChatProtocol},
    train_chat_byte_bpe,
};
use omega_training::checkpoint::sha256_bytes;
use serde_json::Value;

fn cli(root: &Path, arguments: &[&str], input: &str) -> Output {
    cli_with_deadline(root, arguments, input, Duration::from_secs(30))
}

fn cli_with_deadline(root: &Path, arguments: &[&str], input: &str, timeout: Duration) -> Output {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let index = NEXT.fetch_add(1, Ordering::Relaxed);
    let stdout = root.join(format!("child-{index}.stdout"));
    let stderr = root.join(format!("child-{index}.stderr"));
    let stdin = root.join(format!("child-{index}.stdin"));
    fs::write(&stdin, input).unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_omega-training"))
        .current_dir(root)
        .args(arguments)
        // Child-only settings avoid global RNG/pool mutation and CLI reexec.
        .env("RAYON_NUM_THREADS", "1")
        .env("RAYON_RS_NUM_CPUS", "1")
        .env("MATMUL_NUM_THREADS", "1")
        .env_remove("OMEGA_CPU_REEXEC_V1")
        .stdin(Stdio::from(fs::File::open(stdin).unwrap()))
        .stdout(Stdio::from(fs::File::create(&stdout).unwrap()))
        .stderr(Stdio::from(fs::File::create(&stderr).unwrap()))
        .spawn()
        .unwrap();
    let deadline = Instant::now() + timeout;
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        if Instant::now() >= deadline {
            child.kill().unwrap();
            child.wait().unwrap();
            panic!(
                "CLI deadline exceeded: {arguments:?}\nstdout:{}\nstderr:{}",
                fs::read_to_string(stdout).unwrap(),
                fs::read_to_string(stderr).unwrap()
            );
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    Output {
        status,
        stdout: fs::read(stdout).unwrap(),
        stderr: fs::read(stderr).unwrap(),
    }
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires explicitly qualified Vulkan GPU"]
fn vulkan_chat_uses_portable_protocol_and_preserves_turn_boundaries() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    let tokenizer = train_chat_byte_bpe(
        &["hello"],
        &ByteBpeConfig {
            vocab_size: 260,
            min_frequency: 1,
        },
    )
    .unwrap();
    let protocol = ChatProtocol::from_tokenizer(&tokenizer).unwrap();
    // CPU-written inference fixtures deliberately force an unambiguous output;
    // this tests actual Vulkan loading/forward/chat behavior, not model quality.
    constant_checkpoint(
        root,
        &tokenizer,
        "gpu-ending",
        protocol.token_ids().end_turn,
    );
    constant_checkpoint(root, &tokenizer, "gpu-role", protocol.token_ids().user);
    let run = |arguments: &[&str], input: &str| {
        let mut selected = vec!["--backend", "vulkan", "--device", "0"];
        selected.extend_from_slice(arguments);
        // First-use shader compilation gets a larger, still bounded deadline.
        cli_with_deadline(root, &selected, input, Duration::from_secs(90))
    };
    let reply = succeeds(run(
        &[
            "chat",
            "--weights-root",
            "weights",
            "--checkpoint",
            "gpu-ending-1",
            "--system",
            "Be brief",
            "--prompt",
            "café 世界 😀",
            "--max-new-tokens",
            "4",
        ],
        "",
    ));
    assert_eq!(reply.trim(), "");
    let interactive = succeeds(run(
        &[
            "chat",
            "--weights-root",
            "weights",
            "--checkpoint",
            "gpu-ending-1",
            "--system",
            "Be brief",
            "--max-new-tokens",
            "4",
        ],
        "hi\nagain\n/reset\nhello\n/exit\n",
    ));
    assert_eq!(interactive.matches("Assistant:").count(), 3);
    assert!(!interactive.contains("omega_v1"));
    fails(
        run(
            &[
                "chat",
                "--weights-root",
                "weights",
                "--checkpoint",
                "gpu-ending-1",
                "--prompt",
                "hi",
                "--max-new-tokens",
                "64",
            ],
            "",
        ),
        "exceeds the trained context",
    );
    fails(
        run(
            &[
                "chat",
                "--weights-root",
                "weights",
                "--checkpoint",
                "gpu-role-1",
                "--prompt",
                "hi",
                "--max-new-tokens",
                "4",
            ],
            "",
        ),
        "unexpected role token",
    );
}

fn succeeds(output: Output) -> String {
    assert!(output.status.success(), "{output:?}");
    String::from_utf8(output.stdout).unwrap()
}

fn fails(output: Output, message: &str) {
    assert!(!output.status.success(), "{output:?}");
    assert!(
        String::from_utf8_lossy(&output.stderr).contains(message),
        "{output:?}"
    );
}

fn json(path: impl AsRef<Path>) -> Value {
    serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
}

fn prepare(root: &Path) {
    fs::create_dir_all(root.join("data/base")).unwrap();
    fs::create_dir_all(root.join("data/chat")).unwrap();
    fs::write(root.join("data/base/train.txt"), "hi there\n").unwrap();
    let records = [
        serde_json::json!({"schema_version":1,"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"ok"}]}),
        serde_json::json!({"schema_version":1,"messages":[{"role":"user","content":"why"},{"role":"assistant","content":"yes"}]}),
    ];
    fs::write(
        root.join("data/chat/train.jsonl"),
        records
            .iter()
            .map(Value::to_string)
            .collect::<Vec<_>>()
            .join("\n"),
    )
    .unwrap();
}

#[test]
fn fresh_base_new_assistant_stage_evaluate_and_resume_preserve_lineage() {
    assistant_stage_workflow(false);
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires explicitly qualified Vulkan GPU"]
fn vulkan_fresh_base_new_assistant_stage_evaluate_and_resume_preserve_lineage() {
    assistant_stage_workflow(true);
}

fn assistant_stage_workflow(gpu: bool) {
    let cli = |root: &Path, arguments: &[&str], input: &str| {
        if gpu
            && arguments.first().is_some_and(|command| {
                matches!(
                    *command,
                    "train" | "train-stage" | "resume" | "evaluate" | "chat" | "generate"
                )
            })
        {
            let mut selected = vec!["--backend", "vulkan", "--device", "0"];
            selected.extend_from_slice(arguments);
            cli_with_deadline(root, &selected, input, Duration::from_secs(90))
        } else {
            self::cli(root, arguments, input)
        }
    };
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    prepare(root);
    succeeds(cli(
        root,
        &[
            "train-tokenizer",
            "--datasets-root",
            "data",
            "--dataset",
            "base",
            "--output",
            "data/chat-tokenizer.json",
            "--vocab-size",
            "260",
            "--chat-protocol",
        ],
        "",
    ));
    let tokenizer = Tokens::new(root.join("data/chat-tokenizer.json")).unwrap();
    assert_eq!(tokenizer.vocab_size(), 260);
    let protocol = ChatProtocol::from_tokenizer(&tokenizer).unwrap();
    succeeds(cli(
        root,
        &[
            "train",
            "--datasets-root",
            "data",
            "--weights-root",
            "weights",
            "--dataset",
            "base",
            "--tokenizer",
            "chat-tokenizer.json",
            "--name",
            "base",
            "--epochs",
            "1",
            "--max-updates",
            "1",
            "--d-model",
            "4",
            "--heads",
            "1",
            "--layers",
            "1",
            "--d-ff",
            "8",
            "--context-length",
            "64",
            "--quiet",
        ],
        "",
    ));
    let base = root.join("weights/base-1");
    let frozen: Vec<_> = [
        "manifest.json",
        "resume.json",
        "model.mpk",
        "optimizer.mpk",
        "tokenizer.json",
        "COMPLETE",
    ]
    .into_iter()
    .map(|name| (name, fs::read(base.join(name)).unwrap()))
    .collect();
    let manifest = json(base.join("manifest.json"));
    assert_eq!(manifest["schema_version"], 2);
    assert_eq!(manifest["chat"]["protocol"], CHAT_PROTOCOL_VERSION);
    assert_eq!(
        manifest["chat"]["token_ids"],
        serde_json::json!(protocol.token_ids().as_array())
    );
    succeeds(cli(
        root,
        &[
            "train-stage",
            "--datasets-root",
            "data",
            "--weights-root",
            "weights",
            "--checkpoint",
            "base-1",
            "--name",
            "assistant",
            "--dataset",
            "chat",
            "--epochs",
            "2",
            "--max-updates",
            "1",
            "--quiet",
        ],
        "",
    ));
    let stage = json(root.join("weights/assistant-1/resume.json"));
    assert_eq!(stage["schema_version"], if gpu { 6 } else { 5 });
    assert_eq!(stage["objective"], CHAT_TARGET_OBJECTIVE);
    assert_eq!(stage["parent"]["name"], "base-1");
    assert_eq!(
        stage["parent"]["manifest_sha256"],
        sha256_bytes(&fs::read(base.join("manifest.json")).unwrap())
    );
    assert_eq!(
        stage["parent"]["model_sha256"],
        sha256_bytes(&fs::read(base.join("model.mpk")).unwrap())
    );
    assert_eq!(
        stage["parent"]["resume_sha256"],
        sha256_bytes(&fs::read(base.join("resume.json")).unwrap())
    );
    assert_eq!(stage["progress"]["completed_updates"], 1);
    assert_eq!(stage["progress"]["completed_targets"], 3);
    assert_eq!(stage["targets_per_epoch"], 7);
    let evaluation = succeeds(cli(
        root,
        &[
            "evaluate",
            "--datasets-root",
            "data",
            "--weights-root",
            "weights",
            "--checkpoint",
            "assistant-1",
            "--dataset",
            "chat",
            "--dataset-format",
            "chat",
        ],
        "",
    ));
    assert!(
        evaluation.contains("7 targets, mean cross entropy"),
        "{evaluation}"
    );
    succeeds(cli(
        root,
        &[
            "resume",
            "--datasets-root",
            "data",
            "--weights-root",
            "weights",
            "--checkpoint",
            "assistant-1",
            "--name",
            "continued",
            "--epochs",
            "2",
            "--max-updates",
            "1",
            "--quiet",
        ],
        "",
    ));
    let continued = json(root.join("weights/continued-1/resume.json"));
    assert_eq!(continued["parent"], stage["parent"]);
    assert_eq!(continued["objective"], stage["objective"]);
    assert_eq!(continued["progress"]["completed_updates"], 2);
    assert_eq!(continued["progress"]["completed_targets"], 7);
    assert_eq!(continued["progress"]["completed_epochs"], 1);
    for (name, bytes) in frozen {
        assert_eq!(
            fs::read(base.join(name)).unwrap(),
            bytes,
            "parent {name} changed"
        );
    }
    fails(
        cli(
            root,
            &[
                "chat",
                "--weights-root",
                "weights",
                "--checkpoint",
                "assistant-1",
                "--prompt",
                "hello",
                "--max-new-tokens",
                "64",
            ],
            "",
        ),
        "exceeds the trained context",
    );
    fails(
        cli(
            root,
            &[
                "evaluate",
                "--datasets-root",
                "data",
                "--weights-root",
                "weights",
                "--checkpoint",
                "assistant-1",
                "--dataset",
                "chat",
                "--dataset-format",
                "chat",
                "--validation-count",
                "1",
            ],
            "",
        ),
        "source-group partitions",
    );
    fails(
        cli(
            root,
            &[
                "resume",
                "--datasets-root",
                "data",
                "--weights-root",
                "weights",
                "--checkpoint",
                "assistant-1",
                "--name",
                "bad-cache",
                "--epochs",
                "2",
                "--cache",
                "unused",
            ],
            "",
        ),
        "Conversation caches are unsupported",
    );
    assert!(!root.join("weights/bad-cache-1").exists());
}

struct ConstantOutput {
    vocabulary: usize,
    selected: usize,
}
impl ModuleMapper<Cpu> for ConstantOutput {
    fn map_float<const D: usize>(
        &mut self,
        _id: ParamId,
        tensor: Tensor<Cpu, D>,
    ) -> Tensor<Cpu, D> {
        let dims = tensor.dims();
        let mut values = vec![if D == 1 { 1.0 } else { 0.0 }; dims.iter().product()];
        if dims.as_slice() == [4, self.vocabulary] {
            for row in 0..4 {
                values[row * self.vocabulary + self.selected] = 1.0;
            }
        }
        Tensor::from_data(TensorData::new(values, dims), &tensor.device())
    }
}

fn constant_checkpoint(root: &Path, tokenizer: &Tokens, name: &str, selected: u32) {
    let config = GptConfig {
        vocab_size: tokenizer.vocab_size(),
        context_length: 64,
        d_model: 4,
        num_heads: 1,
        num_layers: 1,
        d_ff: 8,
    };
    let model = config
        .init::<Cpu>(&Default::default())
        .unwrap()
        .map(&mut ConstantOutput {
            vocabulary: config.vocab_size,
            selected: selected as usize,
        });
    omega_training::save_checkpoint(&root.join("weights"), name, model, &config, tokenizer)
        .unwrap();
}

#[test]
fn typed_chat_operation_retains_accepted_turns() {
    use omega_tokenizer::chat::{ChatMessage, ChatRole};
    use omega_training::operations::{
        self as op, Args, BackendChoice, CheckpointInput, Command, OperationControl,
    };
    use std::sync::{Arc, Mutex, atomic::AtomicBool};
    let root = tempfile::tempdir().unwrap();
    let tokenizer = train_chat_byte_bpe(
        &["Hi again x"],
        &ByteBpeConfig {
            vocab_size: 260,
            min_frequency: 1,
        },
    )
    .unwrap();
    let token = tokenizer.encode("x", false).unwrap().get_ids()[0];
    constant_checkpoint(root.path(), &tokenizer, "reply", token);
    let result = Arc::new(Mutex::new(serde_json::Value::Null));
    let observe = result.clone();
    let control = OperationControl {
        stop: Arc::new(AtomicBool::new(false)),
        observer: Box::new(move |event| {
            if event.kind == "chat" {
                *observe.lock().unwrap() = event.data;
            }
            Ok(())
        }),
    };
    let run = |history, prompt: &str| {
        op::execute(
            Args {
                cpu: Default::default(),
                backend: BackendChoice::Cpu,
                device: None,
                command: Command::Chat {
                    history,
                    weights_root: Some(root.path().join("weights")),
                    input: CheckpointInput {
                        checkpoint: Some("reply-1".into()),
                        latest_run: None,
                    },
                    prompt: Some(prompt.into()),
                    system: None,
                    max_new_tokens: 2,
                    generation: Default::default(),
                },
            },
            Some(&control),
        )
        .unwrap()
    };
    run(vec![], "Hi");
    let first = result.lock().unwrap().clone();
    assert_eq!(first["history"].as_array().unwrap().len(), 2);
    let history = vec![
        ChatMessage {
            role: ChatRole::User,
            content: "Hi".into(),
        },
        ChatMessage {
            role: ChatRole::Assistant,
            content: first["text"].as_str().unwrap().into(),
        },
    ];
    run(history, "again");
    assert_eq!(
        result.lock().unwrap()["history"].as_array().unwrap().len(),
        4
    );
}

#[test]
fn chat_stops_empty_replies_resets_history_and_rejects_role_leakage() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    let tokenizer = train_chat_byte_bpe(
        &["hello"],
        &ByteBpeConfig {
            vocab_size: 260,
            min_frequency: 1,
        },
    )
    .unwrap();
    let protocol = ChatProtocol::from_tokenizer(&tokenizer).unwrap();
    constant_checkpoint(root, &tokenizer, "ending", protocol.token_ids().end_turn);
    constant_checkpoint(root, &tokenizer, "role", protocol.token_ids().user);
    let greedy = succeeds(cli(
        root,
        &[
            "chat",
            "--weights-root",
            "weights",
            "--checkpoint",
            "ending-1",
            "--prompt",
            "café 世界 😀",
            "--max-new-tokens",
            "4",
        ],
        "",
    ));
    assert_eq!(greedy.trim(), "");
    let sampled = succeeds(cli(
        root,
        &[
            "chat",
            "--weights-root",
            "weights",
            "--checkpoint",
            "ending-1",
            "--prompt",
            "café 世界 😀",
            "--max-new-tokens",
            "4",
            "--sample",
            "--top-k",
            "1",
            "--sampling-seed",
            "91",
        ],
        "",
    ));
    assert_eq!(sampled, greedy);
    let interactive = succeeds(cli(
        root,
        &[
            "chat",
            "--weights-root",
            "weights",
            "--checkpoint",
            "ending-1",
            "--max-new-tokens",
            "4",
        ],
        "hi\nagain\n/reset\nhello\n/exit\n",
    ));
    assert_eq!(interactive.matches("Assistant:").count(), 3);
    assert!(!interactive.contains("omega_v1"));
    let overflow = "x".repeat(70);
    let recovered = succeeds(cli(
        root,
        &[
            "chat",
            "--weights-root",
            "weights",
            "--checkpoint",
            "ending-1",
            "--max-new-tokens",
            "4",
        ],
        &format!("{overflow}\nhi\n/exit\n"),
    ));
    assert_eq!(recovered.matches("Assistant:").count(), 1);
    fails(
        cli(
            root,
            &[
                "chat",
                "--weights-root",
                "weights",
                "--checkpoint",
                "role-1",
                "--prompt",
                "hi",
                "--max-new-tokens",
                "4",
            ],
            "",
        ),
        "unexpected role token",
    );
    let plain =
        Tokens::new(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../datasets/test.json"))
            .unwrap();
    constant_checkpoint(root, &plain, "plain", 1);
    fails(
        cli(
            root,
            &[
                "chat",
                "--weights-root",
                "weights",
                "--checkpoint",
                "plain-1",
                "--prompt",
                "hi",
            ],
            "",
        ),
        "no explicit chat protocol",
    );
}

#[test]
fn assistant_help_and_explicit_input_errors_are_actionable() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    prepare(root);
    for (command, required) in [
        ("train-tokenizer", "--chat-protocol"),
        ("train-stage", "--checkpoint"),
        ("chat", "--max-new-tokens"),
        ("train", "chat"),
        ("generate", "--eos-token-id"),
    ] {
        let help = succeeds(cli(root, &[command, "--help"], ""));
        assert!(help.contains(required), "{help}");
    }
    fails(
        cli(
            root,
            &[
                "train-stage",
                "--datasets-root",
                "data",
                "--weights-root",
                "weights",
                "--name",
                "missing",
                "--dataset",
                "chat",
            ],
            "",
        ),
        "required arguments",
    );
    fails(
        cli(
            root,
            &[
                "train-stage",
                "--datasets-root",
                "data",
                "--weights-root",
                "weights",
                "--checkpoint",
                "absent-1",
                "--name",
                "missing",
                "--dataset",
                "chat",
            ],
            "",
        ),
        "checkpoint",
    );
    fails(
        cli(
            root,
            &[
                "prepare-cache",
                "--datasets-root",
                "data",
                "--dataset",
                "chat",
                "--dataset-format",
                "chat",
                "--output",
                "cache",
            ],
            "",
        ),
        "Conversation caches are unsupported",
    );
    assert!(!root.join("cache").exists());
}
