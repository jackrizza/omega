use omega_training::operations::main_with_args;

fn main() -> std::process::ExitCode {
    main_with_args(std::env::args_os())
}
