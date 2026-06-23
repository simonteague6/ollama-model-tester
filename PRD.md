# omt — Ollama Model Tester

## Problem Statement

I have an Ollama Pro subscription that gives me access to cloud-hosted models, but I have no way to measure which models are actually fast. I run models through Ollama's local API and cloud API, but I can't compare time-to-first-token, tokens-per-second, or total latency across models. Without this data, I'm guessing which model to use for a given task, and I can't tell if my Pro subscription is delivering the performance I'm paying for.

## Solution

A single Go binary called `omt` (Ollama Model Tester) with two faces:

- **Interactive TUI** (default) — discover models from local and cloud endpoints, select which to benchmark, watch progress live, and browse a sortable results table with per-model detail views.
- **Direct CLI mode** — `omt run model1 model2 --cloud` for quick, scriptable benchmarks with table or JSON output.

The app runs a fixed short prompt against each selected model (5 measured runs + 1 discarded warmup), streams the response via Ollama's NDJSON API, measures time-to-first-token client-side, and derives tokens-per-second from the server's final-chunk metrics. Results are aggregated (mean, median, min, max) and displayed in a sortable table. Generated text is displayed during the run but not persisted.

## User Stories

1. As an Ollama Pro subscriber, I want to discover all models available on my local and cloud endpoints, so that I can choose which ones to benchmark.

2. As a user, I want to select a subset of discovered models to benchmark, so that I don't waste time or tokens on models I don't care about.

3. As a user, I want to see a live progress view while benchmarks are running, so that I know how far along the test is and can cancel if needed.

4. As a user, I want to see time-to-first-token (TTFT) for each model, so that I can compare how quickly different models start responding.

5. As a user, I want to see tokens-per-second throughput for each model, so that I can compare generation speed.

6. As a user, I want to see total response time for each model, so that I can compare end-to-end latency.

7. As a user, I want results displayed in a sortable table, so that I can quickly find the fastest model by any metric.

8. As a user, I want to drill into a per-model detail view showing all individual runs and their metrics, so that I can assess consistency and variance.

9. As a user, I want to run benchmarks from the command line without the TUI, so that I can script or automate testing.

10. As a user, I want JSON output in CLI mode, so that I can pipe results into other tools or save them for later analysis.

11. As a user, I want to configure the benchmark prompt, number of runs, max tokens, and timeout, so that I can tailor tests to my use case.

12. As a user, I want to provide my Ollama Cloud API key via environment variable or config file, so that I don't have to type it every time.

13. As a user, I want the tool to handle errors gracefully (network failures, auth failures, rate limits, missing models), so that a single failure doesn't abort the entire benchmark.

14. As a user, I want to cancel a running benchmark and see partial results for models already tested, so that I don't lose data if I change my mind.

15. As a user, I want to see which endpoint (local vs cloud) each model was tested on, so that I can compare local hardware performance against cloud inference.

16. As a user, I want to run benchmarks against both local and cloud endpoints in a single session, so that I can directly compare the same model family across environments.

17. As a user, I want a warmup run to be discarded before measured runs, so that cold-start model loading doesn't skew my metrics.

18. As a user, I want to see per-run error details for failed runs, so that I can diagnose why a model underperformed.

19. As a user, I want the TUI to work in terminals as small as 60x12, so that I can use it in constrained environments.

20. As a user, I want to list available models from the CLI without running a benchmark, so that I can quickly check what's available.

## Implementation Decisions

### Language & Framework
- **Go 1.25+** — the entire application is a single Go binary.
- **bubbletea v2** (`charm.land/bubbletea/v2`) for the TUI, with components from `charm.land/bubbles/v2` (list, table, spinner, progress, viewport, key, help).
- **lipgloss v2** (`charm.land/lipgloss/v2`) for styling.
- **cobra** (`github.com/spf13/cobra`) for CLI command structure.
- **BurntSushi/toml** (`github.com/BurntSushi/toml`) for config file parsing.

### API Client
- Two `Client` implementations: one for local (`http://localhost:11434`, no auth), one for cloud (`https://ollama.com/api`, Bearer token auth).
- Both implement the same `Client` interface: `ListModels(ctx) ([]Model, error)` and `Generate(ctx, req) (Stream, error)`.
- Streaming via NDJSON over `POST /api/generate` with `stream: true`. Uses `bufio.Reader.ReadString('\n')` (not `bufio.Scanner`) to avoid 64 KiB line-length limits.
- Server metrics (`total_duration`, `load_duration`, `prompt_eval_duration`, `eval_duration`, `eval_count`, `prompt_eval_count`) are decoded from the final `done: true` chunk and converted from nanoseconds to `time.Duration` at the boundary.
- Typed errors: `ErrAuth`, `ErrNotFound`, `ErrRateLimit`, `ErrTimeout`, `ErrServer`.

### Benchmark Runner
- **Sequential by default** — one model at a time, runs within a model sequential. This is mandatory for local endpoints (single-GPU contention corrupts metrics).
- **Optional `--parallel N`** for cloud-only models (independent cloud infra).
- **1 warmup run** (discarded, configurable to 0) + **5 measured runs** per model.
- **Cooldown** of 150ms between runs within a model to let the GPU settle.
- **Rate-limit handling**: on 429, sleep `Retry-After` (or exponential backoff) and retry up to 2 times.
- **Partial results**: if some runs fail, the summary is computed over successful runs only. If all runs fail, the row is marked FAIL.
- **Cancellation**: context cancellation (Esc in TUI) returns partial results immediately.

### Metrics
- **Time to First Token (TTFT)**: measured client-side as `time.Since(requestStart)` at the first non-empty response chunk. This is the quantity users actually perceive.
- **Tokens per second**: computed per-run as `eval_count / eval_duration * 1e9`. Summary uses mean and median of per-run rates (not from aggregate counts/durations).
- **Total time**: client-side wall-clock from request send to final `done: true` chunk.
- All server durations converted from nanoseconds to `time.Duration` at decode time.

### Configuration
- Precedence (high to low): CLI flags > environment variables > TOML config file > built-in defaults.
- Env vars: `OLLAMA_API_KEY`, `OLLAMA_LOCAL_URL`, `OLLAMA_CLOUD_URL`, `OLLAMA_RUNS`, `OLLAMA_TIMEOUT`, `OMT_CONFIG`.
- Config file: `~/.omt/config.toml`.
- CLI flags use `cmd.Flags().Changed()` guards so unset flags never clobber file/env values.
- API key is never logged.

### CLI Commands
- `omt` (bare) — launches the TUI.
- `omt tui` — explicit TUI launch.
- `omt run [models...]` — direct CLI benchmark. Flags: `--local`, `--cloud`, `--both`, `--runs`, `--warmup`, `--prompt`, `--max-tokens`, `--timeout`, `--parallel`, `--sort`, `--json`, `--config`.
- `omt list` — list available models from selected endpoints.

### TUI Screens
1. **Select** — endpoint toggle (local/cloud/both) + model multi-select list with filter (`/`), toggle (`Space`), start (`Enter`).
2. **Running** — spinner + progress bar + live per-model metrics as runs complete. Esc cancels.
3. **Results** — sortable table (Model, Endpoint, TTFT, tok/s, Total, Tokens, OK). `s` cycles sort key, `Enter` opens detail, `r` returns to select, `q` quits.
4. **Detail** — scrollable viewport with aggregate stats + per-run breakdown for one model.

### State Machine
```
[*] → Select (load /api/tags)
Select → Running (models chosen)
Running → Results (all done or Esc with partial)
Results → Detail (Enter on row)
Detail → Results (Esc/Backspace)
Results → Select (r for re-run)
Results → [*] (q/Ctrl+C)
```

## Testing Decisions

### What makes a good test
- Test external behavior, not implementation details. The benchmark runner should be testable by providing a fake client and asserting the returned results are correct — not by inspecting internal goroutine state.
- Pure functions (metrics aggregation, config precedence) get table-driven tests with edge cases.
- I/O boundaries (API client) get `httptest.Server`-backed tests with canned responses.
- TUI tests drive the model with canned `Msg` values and assert screen transitions and content substrings — not ANSI pixel output.

### Modules to test
- **`model`** — compile-check only (pure types).
- **`config`** — precedence table (flag > env > file > default), missing file, malformed TOML.
- **`ollama`** — NDJSON streaming decode, status→error mapping, `/api/tags` parsing, auth header presence/absence, timeout behavior.
- **`metrics`** — `TokensPerSec` zero-division guard, `Aggregate` mean/median/min/max correctness, all-fail edge case.
- **`benchmark`** — warmup discard, run ordering, progress callback firing, partial failure aggregation, rate-limit retry, cancellation, parallel mode.
- **`cli`** — arg parsing, flag→Config binding, pre-flight API-key check, JSON golden output.
- **`tui`** — screen transitions (Select→Running→Results→Detail), table sorting, detail content, cancel flow.

### Testing seams (highest to lowest)
1. **`benchmark.Runner` with a fake `ollama.Client`** — the primary seam. The fake scripts per-call streams, delays, and errors. This tests the core business logic (warmup, aggregation, partial failure, rate-limit retry, cancellation) without any network or TUI.
2. **`ollama.Client` with `httptest.Server`** — tests the NDJSON wire protocol, status code mapping, and timeout behavior against a real HTTP server.
3. **`tui.AppModel` with canned `Msg` values** — tests the state machine and rendering by feeding `ListLoadedMsg`, `ProgressMsg`, `RunDoneMsg` through `Update` and asserting screen transitions and content.

### Prior art
This is a greenfield project — no prior tests exist. Tests use only the standard library (`testing`, `net/http/httptest`). No testify or mock frameworks.

## Out of Scope

- **Persistent storage** — no database, no history of past benchmark runs. Each run is ephemeral.
- **HTML artifacts** — generated text is displayed during the run but not saved to disk.
- **Multi-user or team features** — single-user tool.
- **Model comparison across sessions** — no way to compare today's results with yesterday's.
- **Custom prompt libraries** — one prompt per run, configured via flag/config.
- **GPU metrics** — no GPU utilization, temperature, or memory monitoring.
- **OpenAI-compatible endpoint** — uses the native Ollama API (`/api/generate`), not the `/v1/chat/completions` path.
- **Windows support** — not explicitly targeted (though Go + bubbletea may work).

## Further Notes

- The `:cloud` tag suffix on local models is passed through as-is when running against the local endpoint. When models are discovered via the cloud API, bare names are used.
- The same model on both endpoints is treated as two distinct rows (different `Endpoint` values) — never deduplicated.
- Parallel mode (`--parallel N`) is only safe for cloud models. If local endpoints are selected with `Parallel > 1`, the runner warns and clamps local models to sequential.
- The default prompt is: "Tell me a short story about a robot learning to paint." — chosen to be short, generate a predictable output length, and be interesting enough to watch stream in.
