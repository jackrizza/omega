//! Hugging Face Hub metadata and immutable file downloads. No dataset scripts run.
use std::{
    io::{Read, Write},
    time::Duration,
};

use anyhow::{Context, Result, bail, ensure};
use reqwest::{Url, blocking::Client};
use serde::{Deserialize, Serialize};

use crate::{
    config::{Config, Source, safe_relative},
    read_bounded,
};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RemoteFile {
    pub path: String,
    pub size: Option<u64>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ResolvedSource {
    pub source_id: String,
    pub repo: String,
    pub revision: String,
    pub files: Vec<RemoteFile>,
}

/// Injectable transport for offline tests or a caller's own Hub cache.
pub trait Hub {
    fn resolve(&self, source: &Source) -> Result<ResolvedSource>;
    /// Write at most `limit` bytes, returning an error if the file is larger.
    fn download(
        &self,
        source: &ResolvedSource,
        file: &RemoteFile,
        output: &mut dyn Write,
        limit: u64,
    ) -> Result<u64>;
}

pub struct HuggingFace {
    client: Client,
    endpoint: Url,
    token: Option<String>,
}

impl HuggingFace {
    /// A token is optional for public repositories; callers may read HF_TOKEN.
    /// Tokens are never serialized in plans, configurations or error URLs.
    pub fn new(token: Option<String>, timeout_seconds: u64) -> Result<Self> {
        Ok(Self {
            client: Client::builder()
                .https_only(true)
                .connect_timeout(Duration::from_secs(30))
                .timeout(Duration::from_secs(timeout_seconds))
                .user_agent(concat!("omega-datasets/", env!("CARGO_PKG_VERSION")))
                .build()?,
            endpoint: Url::parse("https://huggingface.co/")?,
            token,
        })
    }

    fn url(&self, segments: &[&str]) -> Result<Url> {
        let mut url = self.endpoint.clone();
        url.path_segments_mut()
            .map_err(|_| anyhow::anyhow!("Invalid Hub endpoint"))?
            .pop_if_empty()
            .extend(segments);
        Ok(url)
    }

    fn get(&self, url: Url) -> Result<reqwest::blocking::Response> {
        let mut request = self.client.get(url);
        if let Some(token) = &self.token {
            request = request.bearer_auth(token);
        }
        let response = request
            .send()
            .map_err(|e| e.without_url())
            .context("Hugging Face request failed")?;
        let status = response.status();
        match status.as_u16() {
            401 | 403 => bail!(
                "Hugging Face HTTP {status}: set HF_TOKEN and obtain repository/gated-dataset access"
            ),
            404 => bail!("Hugging Face HTTP 404: check repo, revision and file names"),
            429 => bail!("Hugging Face rate limit: retry later (no output is considered complete)"),
            _ => ensure!(status.is_success(), "Hugging Face HTTP {status}"),
        }
        Ok(response)
    }
}

#[derive(Deserialize)]
struct Info {
    sha: String,
    siblings: Vec<Sibling>,
}

#[derive(Deserialize)]
struct Sibling {
    rfilename: String,
    size: Option<u64>,
}

impl Hub for HuggingFace {
    fn resolve(&self, source: &Source) -> Result<ResolvedSource> {
        let mut parts = vec!["api", "datasets"];
        parts.extend(source.repo.split('/'));
        parts.extend(["revision", &source.revision]);
        let mut url = self.url(&parts)?;
        url.query_pairs_mut().append_pair("blobs", "true");
        let info: Info = serde_json::from_slice(&read_bounded(self.get(url)?, 16 * 1024 * 1024)?)
            .with_context(|| format!("Invalid Hub metadata for {}", source.id))?;
        ensure!(
            info.sha.len() == 40 && info.sha.bytes().all(|c| c.is_ascii_hexdigit()),
            "Hub did not return a full commit SHA for {}",
            source.id
        );
        let mut selected = std::collections::BTreeMap::new();
        for pattern in &source.files {
            let matcher = globset::GlobBuilder::new(pattern)
                .literal_separator(true)
                .build()?
                .compile_matcher();
            let matches: Vec<_> = info
                .siblings
                .iter()
                .filter(|f| matcher.is_match(&f.rfilename))
                .collect();
            ensure!(
                !matches.is_empty(),
                "{}: file pattern {pattern:?} matched no files at {}",
                source.id,
                info.sha
            );
            for file in matches {
                safe_relative(&file.rfilename)?;
                selected.insert(file.rfilename.clone(), file.size);
            }
        }
        Ok(ResolvedSource {
            source_id: source.id.clone(),
            repo: source.repo.clone(),
            revision: info.sha,
            files: selected
                .into_iter()
                .map(|(path, size)| RemoteFile { path, size })
                .collect(),
        })
    }

    fn download(
        &self,
        source: &ResolvedSource,
        file: &RemoteFile,
        output: &mut dyn Write,
        limit: u64,
    ) -> Result<u64> {
        let mut parts = vec!["datasets"];
        parts.extend(source.repo.split('/'));
        parts.extend(["resolve", &source.revision]);
        parts.extend(file.path.split('/'));
        let response = self.get(self.url(&parts)?)?;
        ensure!(
            response.content_length().is_none_or(|n| n <= limit),
            "{} exceeds remaining download/file byte limit {limit}",
            file.path
        );
        let mut stream = response;
        let mut buffer = [0u8; 64 * 1024];
        let mut count = 0u64;
        loop {
            let n = stream
                .read(&mut buffer)
                .context("Read Hugging Face download")?;
            if n == 0 {
                break;
            }
            count += n as u64;
            ensure!(
                count <= limit,
                "{} exceeds remaining download/file byte limit {limit}",
                file.path
            );
            output.write_all(&buffer[..n])?;
        }
        ensure!(
            file.size.is_none_or(|expected| expected == count),
            "Downloaded size differs from Hub metadata: {}",
            file.path
        );
        Ok(count)
    }
}

/// Resolve all patterns to sorted files and full commit SHAs before any output.
pub fn plan(config: &Config, hub: &impl Hub) -> Result<Vec<ResolvedSource>> {
    config.validate()?;
    let mut resolved = Vec::new();
    let mut count = 0usize;
    let mut bytes = 0u64;
    // Resolve each repo/ref only once so sources cannot observe different heads.
    let mut revisions: std::collections::BTreeMap<(String, String), String> =
        std::collections::BTreeMap::new();
    for source in &config.sources {
        let mut source = source.clone();
        let key = (source.repo.clone(), source.revision.clone());
        if let Some(revision) = revisions.get(&key) {
            source.revision = revision.clone();
        }
        let result = hub
            .resolve(&source)
            .with_context(|| format!("Resolve source {}", source.id))?;
        ensure!(
            result.source_id == source.id && result.repo == source.repo && !result.files.is_empty(),
            "invalid resolved source {}",
            source.id
        );
        ensure!(
            result.revision.len() == 40 && result.revision.bytes().all(|c| c.is_ascii_hexdigit()),
            "revision must be a full commit SHA"
        );
        if let Some(previous) = revisions.get(&key) {
            ensure!(
                previous == &result.revision,
                "Hub returned inconsistent revisions"
            );
        }
        revisions.insert(key, result.revision.clone());
        for file in &result.files {
            safe_relative(&file.path)?;
            count = count.checked_add(1).context("file count overflow")?;
            if let Some(size) = file.size {
                ensure!(
                    size <= config.limits.max_file_bytes,
                    "{} exceeds max_file_bytes",
                    file.path
                );
                bytes = bytes.checked_add(size).context("download size overflow")?;
            }
        }
        ensure!(
            count <= config.limits.max_files && bytes <= config.limits.max_download_bytes,
            "selected files exceed max_files or max_download_bytes; narrow files or raise explicit limits"
        );
        resolved.push(result);
    }
    Ok(resolved)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        io::{BufRead, BufReader},
        net::TcpListener,
        sync::{Arc, Mutex},
    };

    type Requests = Arc<Mutex<Vec<String>>>;

    fn server(
        replies: Vec<(u16, Vec<u8>)>,
    ) -> (HuggingFace, Requests, std::thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let captured = requests.clone();
        let thread = std::thread::spawn(move || {
            for (status, bytes) in replies {
                let (mut stream, _) = listener.accept().unwrap();
                stream
                    .set_read_timeout(Some(Duration::from_secs(10)))
                    .unwrap();
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut request = String::new();
                loop {
                    let mut line = String::new();
                    reader.read_line(&mut line).unwrap();
                    request.push_str(&line);
                    if line == "\r\n" || line.is_empty() {
                        break;
                    }
                }
                captured.lock().unwrap().push(request);
                write!(
                    stream,
                    "HTTP/1.1 {status} OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    bytes.len()
                )
                .unwrap();
                stream.write_all(&bytes).unwrap();
            }
        });
        let hub = HuggingFace {
            client: Client::builder()
                .no_proxy()
                .timeout(Duration::from_secs(10))
                .build()
                .unwrap(),
            endpoint: Url::parse(&format!("http://{address}/")).unwrap(),
            token: Some("test-token-never-persisted".into()),
        };
        (hub, requests, thread)
    }

    fn source() -> Source {
        let config: Config = toml::from_str(crate::EXAMPLE_CONFIG).unwrap();
        let mut source = config.sources[0].clone();
        source.repo = "owner/data".into();
        source.revision = "refs/convert/parquet".into();
        source.files = vec!["data/*.jsonl".into()];
        source
    }

    #[test]
    fn http_resolves_globs_and_downloads_pinned_revision_with_auth() {
        let sha = "a".repeat(40);
        let metadata = serde_json::json!({"sha":sha,"siblings":[{"rfilename":"README.md"},{"rfilename":"data/b.jsonl","size":4},{"rfilename":"data/a.jsonl","size":4}]}).to_string().into_bytes();
        let (hub, requests, thread) = server(vec![(200, metadata), (200, b"test".to_vec())]);
        let resolved = hub.resolve(&source()).unwrap();
        assert_eq!(resolved.files[0].path, "data/a.jsonl");
        let mut bytes = Vec::new();
        assert_eq!(
            hub.download(&resolved, &resolved.files[0], &mut bytes, 10)
                .unwrap(),
            4
        );
        assert_eq!(bytes, b"test");
        thread.join().unwrap();
        let requests = requests.lock().unwrap();
        assert!(requests[0].contains("/revision/refs%2Fconvert%2Fparquet?blobs=true"));
        assert!(
            requests[0]
                .to_lowercase()
                .contains("authorization: bearer test-token-never-persisted")
        );
        assert!(requests[1].contains(&format!("/datasets/owner/data/resolve/{sha}/data/a.jsonl")));
        assert!(
            !serde_json::to_string(&resolved)
                .unwrap()
                .contains("test-token")
        );
    }

    #[test]
    fn http_errors_are_actionable_and_do_not_echo_secret_bodies() {
        for status in [401, 403, 404, 429, 500] {
            let (hub, _, thread) = server(vec![(status, b"secret error body".to_vec())]);
            let error = format!("{:#}", hub.resolve(&source()).unwrap_err());
            assert!(!error.contains("secret"));
            assert!(!error.contains("test-token"));
            thread.join().unwrap();
        }
    }

    #[test]
    fn rejects_no_matches_bad_sha_size_mismatch_and_download_limits() {
        for metadata in [
            serde_json::json!({"sha":"a".repeat(40),"siblings":[{"rfilename":"README.md"}]}),
            serde_json::json!({"sha":"main","siblings":[{"rfilename":"data/a.jsonl"}]}),
        ] {
            let (hub, _, thread) = server(vec![(200, metadata.to_string().into_bytes())]);
            assert!(hub.resolve(&source()).is_err());
            thread.join().unwrap();
        }
        for (size, limit) in [(Some(99), 10), (None, 2)] {
            let (hub, _, thread) = server(vec![(200, b"test".to_vec())]);
            let resolved = ResolvedSource {
                source_id: "x".into(),
                repo: "owner/data".into(),
                revision: "a".repeat(40),
                files: vec![],
            };
            let file = RemoteFile {
                path: "data/a.jsonl".into(),
                size,
            };
            assert!(
                hub.download(&resolved, &file, &mut Vec::new(), limit)
                    .is_err()
            );
            thread.join().unwrap();
        }
    }
}
