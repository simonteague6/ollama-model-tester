// Package benchmark executes model benchmarks against an Ollama client.
package benchmark

import (
	"context"
	"errors"
	"log"
	"sync"
	"time"

	"github.com/simonteague6/ollama-model-tester/internal/config"
	"github.com/simonteague6/ollama-model-tester/internal/metrics"
	"github.com/simonteague6/ollama-model-tester/internal/model"
	"github.com/simonteague6/ollama-model-tester/internal/ollama"
)

// cooldown is the pause between measured runs for a single model so the
// serving infra (GPU, CPU cache) can settle.
const cooldown = 150 * time.Millisecond

// rateLimitRetries is the number of times a rate-limited generation request is
// retried after the initial attempt. Backoffs are exponential starting at 1s.
const rateLimitRetries = 2

// defaultPrompt is used when Config.Prompt is empty.
const defaultPrompt = config.DefaultPrompt

// Result summarizes a single model's benchmark.
type Result struct {
	Model     model.Model
	Aggregate model.AggregateResult
	Runs      []model.RunResult
}

// ProgressFn is called after each measured run completes.
type ProgressFn func(modelName string, runIndex int, result model.RunResult)

// Runner executes benchmarks sequentially or in parallel against a Client.
type Runner struct {
	Client   model.Client
	Config   model.Config
	Progress ProgressFn
	Logger   *log.Logger
}

// Run benchmarks every model in models and returns per-model results.
//
// Benchmarks are sequential by default. If Config.Parallel > 1 and every model
// is cloud-hosted, models are run concurrently with at most Config.Parallel
// goroutines. Local models force sequential execution and log a warning.
//
// Context cancellation stops new work immediately and returns partial results
// for models already tested. A per-model Result with some successful measured
// runs is aggregated over those successes; a model where every run failed has
// Aggregate.SuccessCount == 0 and Aggregate.FailCount == len(Runs).
func (r *Runner) Run(ctx context.Context, models []model.Model) ([]Result, error) {
	if r.Client == nil {
		return nil, errors.New("runner: nil client")
	}

	if len(models) == 0 {
		return nil, nil
	}

	logger := r.Logger
	if logger == nil {
		logger = log.Default()
	}

	if r.Config.Parallel > 1 {
		if allCloud(models) {
			return r.runParallel(ctx, models)
		}
		logger.Printf("Warning: parallel benchmarking requested but local models selected; clamping to sequential")
	}

	return r.runSequential(ctx, models)
}

func (r *Runner) runSequential(ctx context.Context, models []model.Model) ([]Result, error) {
	results := make([]Result, 0, len(models))
	for _, m := range models {
		select {
		case <-ctx.Done():
			return results, ctx.Err()
		default:
		}
		res, err := r.runModel(ctx, m)
		results = append(results, res)
		if err != nil {
			return results, err
		}
	}
	return results, nil
}

func (r *Runner) runParallel(ctx context.Context, models []model.Model) ([]Result, error) {
	results := make([]Result, len(models))

	sem := make(chan struct{}, r.Config.Parallel)
	var wg sync.WaitGroup
	var mu sync.Mutex

modelLoop:
	for i, m := range models {
		select {
		case <-ctx.Done():
			break modelLoop
		default:
		}

		wg.Add(1)
		sem <- struct{}{}
		go func(idx int, m model.Model) {
			defer wg.Done()
			defer func() { <-sem }()

			res, _ := r.runModel(ctx, m)
			mu.Lock()
			results[idx] = res
			mu.Unlock()
		}(i, m)
	}

	wg.Wait()

	if err := ctx.Err(); err != nil {
		return results, err
	}
	return results, nil
}

func (r *Runner) runModel(ctx context.Context, m model.Model) (Result, error) {
	runs := make([]model.RunResult, 0, r.Config.Runs)
	prompt := r.prompt()

	for i := 0; i < r.Config.Warmup; i++ {
		select {
		case <-ctx.Done():
			return Result{Model: m, Runs: runs, Aggregate: metrics.Aggregate(runs)}, ctx.Err()
		default:
		}
		_ = r.runOnce(ctx, m, prompt)
	}

	for i := 0; i < r.Config.Runs; i++ {
		select {
		case <-ctx.Done():
			return Result{Model: m, Runs: runs, Aggregate: metrics.Aggregate(runs)}, ctx.Err()
		default:
		}

		res := r.runOnce(ctx, m, prompt)
		runs = append(runs, res)

		if r.Progress != nil {
			r.Progress(m.Name, i, res)
		}

		if i < r.Config.Runs-1 {
			select {
			case <-time.After(cooldown):
			case <-ctx.Done():
				return Result{Model: m, Runs: runs, Aggregate: metrics.Aggregate(runs)}, ctx.Err()
			}
		}
	}

	return Result{
		Model:     m,
		Runs:      runs,
		Aggregate: metrics.Aggregate(runs),
	}, nil
}

func (r *Runner) runOnce(ctx context.Context, m model.Model, prompt string) model.RunResult {
	req := model.GenerateRequest{
		Model:     m.Name,
		Prompt:    prompt,
		MaxTokens: r.Config.MaxTokens,
		Stream:    true,
	}

	backoffs := []time.Duration{time.Second, 2 * time.Second}

	for attempt := 0; attempt <= rateLimitRetries; attempt++ {
		select {
		case <-ctx.Done():
			return model.RunResult{Error: ctx.Err().Error()}
		default:
		}

		var runCtx context.Context
		var cancel context.CancelFunc
		if r.Config.Timeout > 0 {
			runCtx, cancel = context.WithTimeout(ctx, r.Config.Timeout)
		} else {
			runCtx, cancel = ctx, func() {}
		}

		start := time.Now()
		stream, err := r.Client.Generate(runCtx, req)
		if err != nil {
			cancel()
			if errors.Is(err, ollama.ErrRateLimit) && attempt < rateLimitRetries {
				select {
				case <-time.After(backoffs[attempt]):
				case <-ctx.Done():
					return model.RunResult{Error: ctx.Err().Error()}
				}
				continue
			}
			return model.RunResult{Error: err.Error()}
		}

		res := consumeStream(runCtx, stream, start)
		cancel()
		return res
	}

	return model.RunResult{Error: ollama.ErrRateLimit.Error()}
}

func consumeStream(ctx context.Context, stream model.GenerateStream, start time.Time) model.RunResult {
	defer stream.Close()

	var ttft time.Duration
	var seenFirst bool
	var final model.GenerateResponse

	for {
		resp, err := stream.Next()
		if err != nil {
			return model.RunResult{Error: err.Error()}
		}

		if !seenFirst && resp.Token != "" {
			ttft = time.Since(start)
			seenFirst = true
		}

		final = resp
		if resp.Done {
			break
		}

		if err := ctx.Err(); err != nil {
			return model.RunResult{Error: err.Error()}
		}
	}

	return model.RunResult{
		TTFT:         ttft,
		TokensPerSec: metrics.TokensPerSec(final.EvalCount, final.EvalDuration),
		TotalTime:    time.Since(start),
		TokenCount:   final.EvalCount,
	}
}

func (r *Runner) prompt() string {
	if r.Config.Prompt != "" {
		return r.Config.Prompt
	}
	return defaultPrompt
}

func allCloud(models []model.Model) bool {
	for _, m := range models {
		if m.Endpoint != "cloud" {
			return false
		}
	}
	return len(models) > 0
}
