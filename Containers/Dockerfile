# syntax=docker/dockerfile:1
# Use the repository root as the context, not Containers/.
FROM rust:1.93.0-bookworm AS builder
RUN apt-get update && apt-get install -y --no-install-recommends python3 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /omega/Code/Rust
COPY Code/Rust/ ./
COPY Containers/build-rust.sh /usr/local/bin/omega-build
# Windows checkouts may use CRLF; normalize only the copied build helper.
RUN sed -i 's/\r$//' /usr/local/bin/omega-build
ARG BUILD_JOBS=2
RUN --mount=type=cache,target=/usr/local/cargo/registry,sharing=locked \
    --mount=type=cache,target=/usr/local/cargo/git,sharing=locked \
    --mount=type=cache,target=/build,sharing=locked \
    BUILD_JOBS=${BUILD_JOBS} sh /usr/local/bin/omega-build
# Every binary is retained as /opt/omega/bin/<package>/<binary>.
CMD ["bash"]

FROM debian:bookworm-slim AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libgcc-s1 libstdc++6 libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /omega/datasets /omega/weights \
       /omega/Code/Rust/omega-datasets /omega/Code/Rust/omega-training
WORKDIR /omega
STOPSIGNAL SIGTERM

FROM runtime AS datasets
COPY --from=builder /opt/omega/bin/omega-datasets/omega-datasets /usr/local/bin/omega-datasets
ENTRYPOINT ["/usr/local/bin/omega-datasets"]
CMD ["--help"]

FROM runtime AS training
COPY --from=builder /opt/omega/bin/omega-training/omega-training /usr/local/bin/omega-training
ENTRYPOINT ["/usr/local/bin/omega-training"]
CMD ["--help"]
