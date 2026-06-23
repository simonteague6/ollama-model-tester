# omt — Ollama Model Tester

Benchmark your Ollama models. Measure time-to-first-token, tokens-per-second, and total latency across local and cloud endpoints.

## Install

**Go 1.25+ required.** If you don't have Go: [go.dev/dl](https://go.dev/dl/)

```bash
go install github.com/simonteague6/ollama-model-tester/cmd/omt@latest
```

This puts `omt` in `~/go/bin/`. Make sure `~/go/bin` is in your `PATH` (`export PATH="$HOME/go/bin:$PATH"` in your `.zshrc` or `.bashrc`).

To install from a local clone instead:

```bash
git clone https://github.com/simonteague6/ollama-model-tester.git
cd ollama-model-tester
go install ./cmd/omt
```

## Quick start

```bash
# See what models are available locally
omt list --local

# Benchmark one model (2 runs, no warmup, for a quick test)
omt run llama3.2 --local --runs 2 --warmup 0

# Benchmark with JSON output
omt run llama3.2 --local --json

# Launch the interactive TUI
omt
```

## Commands

### `omt` / `omt tui`

Launch the interactive terminal UI. Discover models, select which to benchmark, watch live progress, and browse sortable results.

### `omt list`

Discover available models.

```
omt list              # both local and cloud (default)
omt list --local      # local only
omt list --cloud      # cloud only (requires API key)
```

### `omt run [models...]`

Run benchmarks from the command line.

```
omt run model1 model2 --local
```

| Flag | Default | Description |
|------|---------|-------------|
| `--local` | — | Use local endpoint |
| `--cloud` | — | Use cloud endpoint (requires `OLLAMA_API_KEY`) |
| `--both` | — | Use both endpoints |
| `--runs` | `5` | Measured runs per model |
| `--warmup` | `1` | Warmup runs to discard |
| `--prompt` | `"Tell me a short story about a robot learning to paint."` | Prompt text |
| `--max-tokens` | `256` | Max tokens to generate |
| `--timeout` | `1m` | Per-generation timeout |
| `--parallel` | `1` | Max parallel cloud models |
| `--sort` | `ttft` | Sort key: `ttft`, `tok/s`, `total`, `model` |
| `--json` | — | Output JSON instead of table |
| `--config` | — | Path to TOML config file |

## Configuration

Precedence (highest to lowest): CLI flags → environment variables → config file → defaults.

### Environment variables

| Variable | Purpose |
|----------|---------|
| `OLLAMA_API_KEY` | Cloud API key |
| `OLLAMA_LOCAL_URL` | Local endpoint URL (default `http://localhost:11434`) |
| `OLLAMA_CLOUD_URL` | Cloud endpoint URL (default `https://ollama.com/api`) |
| `OLLAMA_RUNS` | Default number of measured runs |
| `OLLAMA_TIMEOUT` | Default per-generation timeout |
| `OMT_CONFIG` | Alternate config file path |

### Config file

`~/.omt/config.toml` (optional — defaults apply if missing):

```toml
local_url = "http://localhost:11434"
cloud_url = "https://ollama.com/api"
runs = 5
warmup = 1
prompt = "Tell me a short story about a robot learning to paint."
max_tokens = 256
timeout = "60s"
parallel = 1
sort_key = "ttft"
```

The API key is **never** stored in the config file — use `OLLAMA_API_KEY` env var only.

## Metrics

| Metric | What it measures |
|--------|-----------------|
| **TTFT** | Time to first token — client-side wall clock from request to first response chunk |
| **tok/s** | Tokens per second — `eval_count / eval_duration × 10⁹` |
| **Total** | End-to-end wall clock from request to final chunk |
| **Tokens** | Total tokens generated |
| **OK** | Successful runs / total runs |

Results are aggregated: mean, median, min, max for each metric across all successful runs.

## TUI screens

```
Select ──→ Running ──→ Results ──→ Detail
  ↑                      │  ↑         │
  └──────────────────────┘  └─────────┘
```

- **Select** — toggle local/cloud/both, filter models (`/`), toggle selection (`Space`), start (`Enter`)
- **Running** — spinner, progress bar, live per-model metrics. `Esc` cancels with partial results.
- **Results** — sortable table. `s` cycles sort key, `Enter` opens detail, `r` re-runs, `q` quits.
- **Detail** — aggregate stats + per-run breakdown for one model. `Esc` returns to Results.

## Cloud endpoint

To benchmark cloud models, set your Ollama API key:

```bash
export OLLAMA_API_KEY="sk-..."
omt list --cloud
omt run gemma3:cloud --cloud
```

Missing API key? `omt` exits with a clear error before making any network calls.

## Building from source

```bash
git clone https://github.com/simonteague6/ollama-model-tester.git
cd ollama-model-tester
go build ./cmd/omt      # binary at ./omt
go test ./...            # run all tests
```
