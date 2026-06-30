# docker build -t makershop/llamanexus:beta .
#
# --- Build only LlamaNexus using llamanexus:base as builder ---
FROM llamanexus:base AS builder

WORKDIR /app

# Copy dependencies and source code (build only if changed)
COPY go.mod /app/
COPY main.go /app/main.go
COPY hf_progress_download.py /app/hf_progress_download.py

# Install pflag fro command line argument parsing
RUN go get github.com/spf13/pflag

# Install ini.v1 for configuration file parsing
RUN go get gopkg.in/ini.v1

# Build llamanexus software
RUN go build -o llamanexus main.go

# --- Minimal container ---
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    ca-certificates curl python3 python3-pip libgomp1 \
    && pip3 install huggingface_hub \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy built binaries from the builder stage and the new Go binary
COPY --from=builder /app/llama.cpp/build/bin/llama-server /usr/local/bin/llama-server
COPY --from=builder /app/llama.cpp/build/bin/llama-cli /usr/local/bin/llama-cli
COPY --from=builder /app/llama.cpp/build/bin/ggml-rpc-server /usr/local/bin/rpc-server
COPY --from=builder /app/llamanexus /app/llamanexus
COPY --from=builder /app/hf_progress_download.py /app/hf_progress_download.py

ENV HOME=/root

ENTRYPOINT ["/app/llamanexus"]
CMD ["serve"]
