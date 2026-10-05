# GitHub builds and initial push

The workspace is `Code/Rust`. The initial Git file set includes source, docs,
container/script tooling, `Cargo.lock`, and the tiny offline fixtures documented
in [datasets/README.md](../datasets/README.md). Generated checkpoints, corpora,
downloaded tokenizers, local benchmark captures, target directories and `.env`
files are excluded. The host-specific `Code/Rust/.cargo/config.toml` also stays
local; a fresh checkout uses Cargo's normal target directory.

Text is normalized to LF through `.gitattributes`, including Linux shell scripts.
Do not force-add ignored datasets, credentials or build artifacts.

## First push

The repository is initialized on `main`. Preparation does not create a commit or
contact a GitHub repository. Review the staged files, then create the initial
commit and push to your chosen empty GitHub repository:

```sh
git diff --cached --stat
git commit -m "Initial Omega workspace"
# Add origin only if it is not already configured; replace OWNER and REPOSITORY.
git remote add origin https://github.com/OWNER/REPOSITORY.git
git push -u origin main
```

## Builds on main

[The release workflow](../.github/workflows/release.yml) runs for every push to
`main`, for `v*` tags, and through **Run workflow** in GitHub Actions. It tests and
builds the Linux x86-64 `omega` executable with CPU, Vulkan and experimental CUDA
using Rust 1.93.0 and the pinned CUDA 12.5.1 Ubuntu 22.04 build container.

The workflow runs formatting, offline workspace tests, strict Clippy, separate
backend feature checks, PTY persistence tests, and a clean Ubuntu 22.04 CPU smoke
test without Rust or GPU libraries. Hosted runners do not qualify GPU hardware.

After a successful run, open **Actions → Omega Linux release → the run → Artifacts**
and download `omega-linux-x86_64`. It contains:

- `omega`: standalone executable (run `chmod +x omega` after downloading).
- `omega-main-COMMIT-linux-x86_64.tar.gz`: executable and runtime guide; the
  commit identifier is the first 12 characters of the source commit SHA.
- `SHA256SUMS` and `RUNTIME.md`.
- `clean_runtime.sh`, used by the clean-host acceptance job.

Artifacts are retained for 14 days. Extract the archive and read
[the runtime requirements](omega.md) before selecting a GPU backend. Main builds
are downloadable workflow artifacts; they do not publish a stable GitHub Release.
See [GitHub's push-trigger documentation](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#push).

## Tagged releases

Push a deliberate version tag such as `v0.1.0` to run the same build and tests.
Only after both build and clean-runtime jobs succeed does the workflow publish
the existing tag as a GitHub Release, with executable, versioned archive,
checksums and runtime guide. Publishing uses the repository's `GITHUB_TOKEN`;
no personal token or training credentials are required. No release is published
by merely preparing this checkout.
