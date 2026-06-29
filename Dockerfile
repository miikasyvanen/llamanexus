# docker-compose build
# --- Vaihe 1: Käännetään VAIN Go-sovellus hyödyntäen esikäännettyä pohjaa ---
FROM llamanexus-base:latest AS builder

WORKDIR /app

# Kopioidaan Go-riippuvuudet ja lähdekoodi (käännetään vain jos ne muuttuvat)
COPY go.mod /app/
COPY main.go /app/main.go
COPY hf_progress_download.py /app/hf_progress_download.py

# Install pflag fro command line argument parsing
RUN go get github.com/spf13/pflag

# Install ini.v1 for configuration file parsing
RUN go get gopkg.in/ini.v1

# Build llama-nexus software
RUN go build -o llama-nexus main.go

# --- Vaihe 2: Minimaalinen ajokontti (Pysyy samana) ---
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    ca-certificates curl python3 python3-pip libgomp1 \
    && pip3 install huggingface_hub \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Kopioidaan valmiit binäärit esikäännetystä pohjasta sekä uusi Go-binääri
COPY --from=builder /app/llama.cpp/build/bin/llama-server /usr/local/bin/llama-server
COPY --from=builder /app/llama.cpp/build/bin/llama-cli /usr/local/bin/llama-cli
COPY --from=builder /app/llama.cpp/build/bin/rpc-server /usr/local/bin/rpc-server
COPY --from=builder /app/llama-nexus /app/llama-nexus
COPY --from=builder /app/hf_progress_download.py /app/hf_progress_download.py

ENV HOME=/root
#RUN mkdir -p /root/models

ENTRYPOINT ["/app/llama-nexus"]
CMD ["serve"]
