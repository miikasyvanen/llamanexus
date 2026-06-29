# LlamaNexus

LlamaNexus is a Go proxy that sits between [Open WebUI](https://github.com/open-webui/open-webui) and [llama.cpp](https://github.com/ggml-org/llama.cpp)'s `llama-server`, translating Ollama-style and OpenAI-style API calls into requests `llama-server` understands — while handling model downloads from Hugging Face, context-size overrides, and model unloading along the way.

It runs as a single Go binary (`llamanexus`) with four subcommands: `serve`, `run`, `pull`, and `worker`.

## Why this exists

Open WebUI speaks the Ollama API natively. `llama-server` speaks an OpenAI-compatible API and runs in **router mode**, managing multiple GGUF models loaded from a local cache directory. LlamaNexus bridges the two: it presents an Ollama-compatible (and OpenAI-compatible) HTTP API to Open WebUI, and forwards requests to `llama-server`'s router underneath, translating request/response shapes as needed.

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
llamanexus serve --llamaport 8080 --port 11434 [llama-server args...]
```

- `--port` — port LlamaNexus itself listens on (default `11434`, matching Ollama's default).
- `--llamaport` — port `llama-server`'s router listens on internally (default `8080`).
- Any arguments after the subcommand are passed straight through to `llama-server` (e.g. `--n-gpu-layers`, `-ngl`, etc.) — but **not** `--ctx-size`, since that conflicts with the per-model preset mechanism (see [Context size overrides](#context-size-overrides) below).

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

Starts a `rpc-server` instance for distributed/multi-machine inference via llama.cpp's RPC backend. Used in a separate Docker Compose file (`docker-compose-worker.yml`) on machines contributing compute to a primary `serve` instance.

```bash
llamanexus worker --port 50052
```

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

