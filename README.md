# LlamaNexus

LlamaNexus is a Go proxy that sits between [Open WebUI](https://github.com/open-webui/open-webui) and [llama.cpp](https://github.com/ggml-org/llama.cpp)'s `llama-server`, translating Ollama-style and OpenAI-style API calls into requests `llama-server` understands — while handling model downloads from Hugging Face, context-size overrides, and model unloading along the way.

It runs as a single Go binary (`llamanexus`) with four subcommands: `serve`, `run`, `pull`, and `worker`.

## Why this exists

The primary goal of LlamaNexus is to enable **distributed inference** across multiple machines using llama.cpp's [RPC backend](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc#overview). Worker machines run the same Docker image started with the `worker` command, exposing their GPU or CPU resources to the primary server via `ggml-rpc-server`. The primary server's RPC backend distributes model layer computations across all connected workers, allowing inference to scale beyond a single machine's memory and compute.

Beyond distributed inference, LlamaNexus acts as a proxy between Open WebUI and `llama-server`. Open WebUI speaks the Ollama API natively, while `llama-server` speaks an OpenAI-compatible API and runs in **router mode**, managing multiple GGUF models loaded from a local cache directory. LlamaNexus bridges the two: it presents an Ollama-compatible (and OpenAI-compatible) HTTP API to Open WebUI and forwards requests to `llama-server`'s router underneath, translating request/response shapes as needed.

It also takes over model downloading from Hugging Face Hub directly (rather than relying on `llama-server`'s own `--hf-repo` auto-pull), so that download progress can be reported back to Open WebUI's UI in real time.

## Architecture overview

```
Open WebUI  --(Ollama / OpenAI API)-->  LlamaNexus  --(OpenAI-compatible API)-->  llama-server (router mode)
                                             |
                                             +--> hf_progress_download.py (Hugging Face downloads)
                                             +--> router.preset.ini (per-model context size overrides)
```

- **LlamaNexus** listens on the Ollama-standard port (`11434` by default) and exposes both `/ollama/api/*` and `/openai/v1/*` routes.
- **llama-server** runs in router mode (`--models-dir` + `--models-preset`), auto-discovering GGUF files from the Hugging Face cache and managing per-model child processes, loading and unloading them on demand.
- **`hf_progress_download.py`** is a small Python sidecar, invoked as a subprocess, that downloads models via `huggingface_hub` directly (bypassing the `hf` CLI, which doesn't expose usable progress when piped) and reports real byte-level progress as NDJSON lines on stdout.
- **`router.preset.ini`** is an INI file LlamaNexus reads and writes to override `ctx-size` (and potentially other llama-server launch args) on a per-model basis, since llama.cpp allocates the KV cache at model-load time and can't resize it live.

## Commands

### `serve`

Starts the proxy and the underlying `llama-server` router. This is the main mode, used in production via Docker Compose alongside Open WebUI.

```bash
llamanexus serve --llamaport 8080 --port 11434 -- [llama-server args...]
```

- `--port` — port LlamaNexus itself listens on (default `11434`, matching Ollama's default).
- `--llamaport` — port `llama-server`'s router listens on internally (default `8080`).
- Any arguments after the `--` are passed straight through to `llama-server` (e.g. `--n-gpu-layers`, `-ngl`, etc.) — but **not** `--ctx-size`, since that conflicts with the per-model preset mechanism (see [Context size overrides](#context-size-overrides) below).

### `run`

One-shot CLI inference against a single model, without starting the full proxy/router. Downloads the model automatically if it isn't already cached.

```bash
llamanexus run -m <repo>:<tag> -- -p "Your prompt here"
```

- `-m` / `--model` — model identifier as `repo:tag`, e.g. `Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M`. The tag can be a quantization name (`Q4_K_M`) or an exact filename.
- Everything after `--` is passed directly to `llama-cli` (e.g. `-p` for the prompt). The `--` separator is required — without it, LlamaNexus's own flag parser will try to interpret `llama-cli` flags as its own and fail.
- LlamaNexus automatically adds `-st` (`--single-turn`) so `llama-cli` exits after one response instead of dropping into an interactive prompt.

### `pull`

Downloads a model from Hugging Face without running inference or starting the server. Useful for pre-warming the cache from a script or cron job.

```bash
llamanexus pull <repo>:<tag>
```

Prints a live percentage as the download progresses, then exits.

### `worker`

Starts a `ggml-rpc-server` instance for distributed/multi-machine inference via llama.cpp's RPC backend. Used in a separate Docker Compose file (`docker-compose-worker.yml`) on machines contributing compute to a primary `serve` instance.

```bash
llamanexus worker --port 50052
```

## Distributed inference and worker discovery

LlamaNexus supports two modes for connecting worker nodes to the primary server.

### Manual mode

Pass `--rpc` to `serve` with a comma-separated list of worker addresses. LlamaNexus starts `llama-server` immediately with those workers and does not listen for heartbeats.

```bash
llamanexus serve --rpc 192.168.0.120:50052,192.168.0.110:50052
```

### Auto-discovery mode

Pass `--discovery` to both `serve` and `worker`. Workers broadcast a UDP heartbeat once per second; the serve node listens, discovers workers automatically, and passes them to `llama-server` as `--rpc` arguments.

```bash
# On each worker machine
llamanexus --discovery worker

# On the primary server
llamanexus --discovery serve
```

On startup, `serve` waits 8 seconds (configurable with `--discovery-wait`) to collect heartbeats, then starts `llama-server` with all discovered workers. After that the watcher keeps running in the background: if a new worker appears or an existing one stops sending heartbeats for 5 seconds, `llama-server` is automatically restarted with the updated worker list.

#### Discovery flags

| Flag | Default | Purpose |
|---|---|---|
| `--discovery` | `false` | Enable auto-discovery (use on both `serve` and `worker`) |
| `--discovery-port` | `50051` | UDP port used for heartbeat packets |
| `--discovery-wait` | `8s` | How long `serve` waits for heartbeats before starting `llama-server` |
| `--advertise-addr` | _(auto-detected)_ | IP or `host:port` to advertise in heartbeats; overrides auto-detection |

#### Docker networking requirement

Discovery uses UDP broadcast (`255.255.255.255`), which requires containers to be on the host network so packets reach the physical LAN interface rather than staying inside a Docker bridge. Add `network_mode: host` to both the serve and worker services in your Compose file:

```yaml
services:
  llamanexus:
    network_mode: host
  worker:
    network_mode: host
```

#### Advertise address override

When running inside Docker, the auto-detected IP may resolve to a container bridge address (e.g. `172.19.0.x`) instead of the host\'s LAN IP. If that happens, override it explicitly:

```bash
llamanexus --discovery --advertise-addr 192.168.0.120 worker
```

Or via environment variable in your Compose file:

```yaml
environment:
  - LLAMANEXUS_ADVERTISE_ADDR=192.168.0.120
```

The port is appended automatically if you supply only an IP. Priority order is: `--advertise-addr` flag → `LLAMANEXUS_ADVERTISE_ADDR` env var → auto-detection.

## Build from source

### Docker

Repository contains three Dockerfiles. Dockerfile.base and Dockerfile are used together. Dockerfile.base builds Llama.cpp with CUDA support base image and Dockerfile builds LlamaNexus and combines these two images to one that can be used with docker / docker-compose. Dockerfile.full builds all in one run. Two step build is only to prevent llama.cpp/CUDA to rebuilt in any case to save time.

### Build with base

```bash
docker build --file Dockerfile.base -t llamanexus:base .
docker build -t llamanexus:beta .
```

### Build full

```bash
docker build --file Dockerfile.full -t llamanexus:full .
```

When building and using local images, two lines needs to be changed in docker-compose.yml
```bash
    image: llamanexus:beta
    pull_policy: never
```

## CPU architecture compatibility

By default, llama.cpp's CMake build detects the CPU features of the **build machine** and compiles for those — meaning a binary built on a modern CPU may crash with `signal: illegal instruction (SIGILL)` when deployed to an older one. This is especially relevant when running the `worker` (`ggml-rpc-server`) on a different machine than the primary server.

LlamaNexus explicitly disables CPU-specific instruction sets in its Dockerfiles to produce a portable binary that works across machines:

```dockerfile
cmake .. \
  -DGGML_NATIVE=OFF \   # Do not optimize for the build machine's CPU
  -DGGML_AVX=ON \       # Safe baseline, present on most modern CPUs
  -DGGML_AVX2=OFF \     # Disable — not available on all CPUs
  -DGGML_FMA=OFF \      # Disable — often paired with AVX2
  -DGGML_F16C=OFF \     # Disable — not universally available
  -DGGML_BMI2=OFF \     # Disable — missing on older Intel/AMD CPUs
  -DGGML_AVX512=OFF     # Disable — high-end CPUs only
```

> **Note:** GPU offload (CUDA) is unaffected by these CPU flags — the performance impact of disabling AVX2/FMA is minimal when inference is running on the GPU.

### Diagnosing SIGILL on a worker machine

If `ggml-rpc-server` crashes immediately after `ggml_cuda_init` with `signal: illegal instruction`, compare CPU flags between the build machine and the worker machine:

```bash
cat /proc/cpuinfo | grep -m1 flags | tr ' ' '\n' | grep -E 'avx|fma|bmi'
```

Any flag present on the build machine but missing on the worker is a potential SIGILL cause. The most common culprits are `avx2`, `fma`, and `bmi2`.

## Setup

LlamaNexus is designed to run in Docker alongside `llama-server` (CUDA-enabled) and Open WebUI. See `compose.yaml` / `docker-compose-server.yml` for the reference setup.

### Environment variables

| Variable | Purpose |
|---|---|
| `HF_TOKEN` | Hugging Face access token. Required for gated/private repos; recommended generally to avoid unauthenticated rate limits. If unset, LlamaNexus logs a warning at startup. |

### Volumes

Mount a persistent volume at `~/.cache/huggingface` inside the container. This holds:
- `hub/` — the standard Hugging Face cache layout (`models--org--repo/blobs|refs|snapshots/...`), where downloaded GGUF files live.
- `router.preset.ini` — per-model `ctx-size` overrides, written by LlamaNexus and read by `llama-server`'s router at its own startup.

## How model identifiers work

Models are addressed as `repo:tag`, e.g. `unsloth/Qwen3.5-4B-GGUF:Q4_K_M`. LlamaNexus scans the Hugging Face cache (`ScanHFCacheModels`) and infers a short quant-tag identifier from each cached file's name (`Q4_K_M`, `Q8_0`, `Q5_K_M`, `Q4_0`, or the full filename if none of those match). This is the identifier shown in Open WebUI's model dropdown and the one `llama-server`'s router uses as well, so the two stay in sync.

When pulling a model by a short tag that doesn't exactly match a real filename (e.g. `Q4_K_M` when the real file is `model-name-Q4_K_M.gguf`), LlamaNexus resolves the real filename against the repo's file listing before downloading.

## Context size overrides

Open WebUI lets users set a per-model `num_ctx` value. Since llama.cpp allocates the KV cache at model load time, changing context size requires:

1. Writing the new `ctx-size` into that model's section in `router.preset.ini`.
2. Restarting the `llama-server` router process (it only parses `--models-preset` once, at its own startup — a per-model unload/reload alone won't pick up an edited preset file).
3. Waiting briefly for the router to come back up before the triggering chat request proceeds.

This means the *first* message after changing context size pays a short restart cost (typically a few seconds); subsequent messages are unaffected until the value is changed again. A `[*]` global section in the preset file holds the baseline default for any model without a custom override.

**Important:** don't pass `--ctx-size` as a `serve`-level argument — it takes precedence over the preset file's per-model values and will silently defeat this whole mechanism.

## Model unloading

Open WebUI's "eject" action and Ollama's `keep_alive: 0` convention are both supported: LlamaNexus detects either signal (on `/ollama/api/chat` and `/ollama/api/generate`) and forwards it to `llama-server`'s router `/models/unload` endpoint, rather than treating it as a real generation request.

## Known quirks

- **`llama-server` router mode is experimental** (per llama.cpp's own startup warning) and not recommended for untrusted environments.
- **`huggingface_hub`'s Xet transfer backend** produces uneven, bursty progress updates; LlamaNexus disables it (`HF_HUB_DISABLE_XET=1`) in favor of smooth, evenly-spaced progress reporting, at some cost to raw download speed.
- **Cancelled or failed downloads** can leave orphaned `.incomplete` blob files in the Hugging Face cache due to a known upstream bug ([huggingface/huggingface_hub#4196](https://github.com/huggingface/huggingface_hub/issues/4196)) where resume doesn't work across separate download attempts. LlamaNexus cleans these up automatically on cancellation or failure.
- **Open WebUI's green "loaded" dot and eject button** may not render correctly in all Open WebUI versions due to upstream frontend bugs unrelated to LlamaNexus; LlamaNexus's `/ollama/api/ps` and unload handling are independently verified to work correctly via direct API calls regardless. **Open WebUI version 0.9.4 is tested to work, latest version 0.9.6 will not render correctly green "loaded" dot and eject button**

## License

You may copy, distribute and modify the software as long as you track changes/dates in source files. Any modifications to or software including (via compiler) GPL-licensed code must also be made available under the GPL along with build & install instructions.

Commercial uses is possible, but all code linked with GPL 3.0 source code must be disclosed under a GPL 3.0 compatible license.

## Support

If you find LlamaNexus useful, please consider supporting me as this software is developed on my free time and will always be open source and free of charge. Donations will be used to aquire a lots of coffee :) that plays important role during development! Oh, and also some hardware if needed.

https://www.paypal.com/paypalme/miikasyvanen