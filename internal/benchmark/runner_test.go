package benchmark

import (
	"bytes"
	"context"
	"errors"
	"log"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/simonteague6/ollama-model-tester/internal/model"
	"github.com/simonteague6/ollama-model-tester/internal/ollama"
)

// fakeStream is a scripted GenerateStream for tests.
type fakeStream struct {
	chunks []model.GenerateResponse
	index  int
	delay  time.Duration
	closed bool
}

func newFakeStream(delay time.Duration, chunks ...model.GenerateResponse) *fakeStream {
	return &fakeStream{chunks: chunks, delay: delay}
}

func (s *fakeStream) Next() (model.GenerateResponse, error) {
	if s.delay > 0 {
		time.Sleep(s.delay)
		s.delay = 0
	}
	if s.index >= len(s.chunks) {
		return model.GenerateResponse{}, errors.New("fake stream exhausted")
	}
	r := s.chunks[s.index]
	s.index++
	return r, nil
}

func (s *fakeStream) Close() error {
	s.closed = true
	return nil
}

// fakeClient records every Generate call and delegates to fn.
type fakeClient struct {
	mu        sync.Mutex
	calls     []model.GenerateRequest
	callTimes []time.Time
	fn        func(ctx context.Context, req model.GenerateRequest) (model.GenerateStream, error)
}

func (f *fakeClient) ListModels(ctx context.Context) ([]model.Model, error) {
	return nil, nil
}

func (f *fakeClient) Generate(ctx context.Context, req model.GenerateRequest) (model.GenerateStream, error) {
	f.mu.Lock()
	f.calls = append(f.calls, req)
	f.callTimes = append(f.callTimes, time.Now())
	fn := f.fn
	f.mu.Unlock()
	return fn(ctx, req)
}

func (f *fakeClient) generateCount() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return len(f.calls)
}

func successChunks(evalCount int, evalDuration time.Duration) []model.GenerateResponse {
	return []model.GenerateResponse{
		{Token: "Hello"},
		{Token: " world", Done: true, EvalCount: evalCount, EvalDuration: evalDuration},
	}
}

func constantSuccess(delay time.Duration) func(ctx context.Context, req model.GenerateRequest) (model.GenerateStream, error) {
	return func(ctx context.Context, req model.GenerateRequest) (model.GenerateStream, error) {
		return newFakeStream(delay, successChunks(10, time.Second)...), nil
	}
}

func TestRunnerNilClient(t *testing.T) {
	r := &Runner{}
	_, err := r.Run(context.Background(), []model.Model{{Name: "m"}})
	if err == nil {
		t.Fatal("expected error for nil client")
	}
}

func TestDefaultPrompt(t *testing.T) {
	var prompt string
	fc := &fakeClient{
		fn: func(ctx context.Context, req model.GenerateRequest) (model.GenerateStream, error) {
			prompt = req.Prompt
			return newFakeStream(0, successChunks(1, time.Second)...), nil
		},
	}
	r := &Runner{
		Client: fc,
		Config: model.Config{Runs: 1, Warmup: 0},
	}
	_, err := r.Run(context.Background(), []model.Model{{Name: "m1", Endpoint: "local"}})
	if err != nil {
		t.Fatal(err)
	}
	if prompt != defaultPrompt {
		t.Fatalf("expected default prompt %q, got %q", defaultPrompt, prompt)
	}
}

func TestCustomPrompt(t *testing.T) {
	want := "custom prompt"
	var got string
	fc := &fakeClient{
		fn: func(ctx context.Context, req model.GenerateRequest) (model.GenerateStream, error) {
			got = req.Prompt
			return newFakeStream(0, successChunks(1, time.Second)...), nil
		},
	}
	r := &Runner{
		Client: fc,
		Config: model.Config{Runs: 1, Warmup: 0, Prompt: want},
	}
	_, err := r.Run(context.Background(), []model.Model{{Name: "m1", Endpoint: "local"}})
	if err != nil {
		t.Fatal(err)
	}
	if got != want {
		t.Fatalf("expected prompt %q, got %q", want, got)
	}
}

func TestWarmupDiscarded(t *testing.T) {
	calls := 0
	fc := &fakeClient{
		fn: func(ctx context.Context, req model.GenerateRequest) (model.GenerateStream, error) {
			calls++
			if calls == 1 {
				return newFakeStream(0, model.GenerateResponse{Token: "warmup", Done: true, EvalCount: 100, EvalDuration: time.Second}), nil
			}
			return newFakeStream(0, model.GenerateResponse{Token: "measured", Done: true, EvalCount: 10, EvalDuration: time.Second}), nil
		},
	}
	r := &Runner{
		Client: fc,
		Config: model.Config{Runs: 3, Warmup: 1},
	}
	res, err := r.Run(context.Background(), []model.Model{{Name: "m1", Endpoint: "local"}})
	if err != nil {
		t.Fatal(err)
	}
	if len(res) != 1 {
		t.Fatalf("expected 1 result, got %d", len(res))
	}
	if calls != 4 {
		t.Fatalf("expected 4 Generate calls (1 warmup + 3 measured), got %d", calls)
	}
	if len(res[0].Runs) != 3 {
		t.Fatalf("expected 3 measured runs, got %d", len(res[0].Runs))
	}
	for i, run := range res[0].Runs {
		if run.TokenCount != 10 {
			t.Fatalf("run %d token count = %d, want 10 (warmup discarded)", i, run.TokenCount)
		}
	}
}

func TestMeasuredRunCount(t *testing.T) {
	fc := &fakeClient{fn: constantSuccess(0)}
	r := &Runner{
		Client: fc,
		Config: model.Config{Runs: 5, Warmup: 1},
	}
	res, err := r.Run(context.Background(), []model.Model{{Name: "m1", Endpoint: "local"}})
	if err != nil {
		t.Fatal(err)
	}
	if fc.generateCount() != 6 {
		t.Fatalf("expected 6 Generate calls (1 warmup + 5 measured), got %d", fc.generateCount())
	}
	if len(res[0].Runs) != 5 {
		t.Fatalf("expected 5 measured runs, got %d", len(res[0].Runs))
	}
}

func TestCooldownBetweenRuns(t *testing.T) {
	fc := &fakeClient{fn: constantSuccess(0)}
	r := &Runner{
		Client: fc,
		Config: model.Config{Runs: 3, Warmup: 0},
	}
	start := time.Now()
	_, err := r.Run(context.Background(), []model.Model{{Name: "m1", Endpoint: "local"}})
	elapsed := time.Since(start)
	if err != nil {
		t.Fatal(err)
	}
	want := 2 * cooldown
	if elapsed < want-10*time.Millisecond {
		t.Fatalf("expected elapsed >= %v for cooldown, got %v", want, elapsed)
	}
}

func TestRateLimitRetry(t *testing.T) {
	calls := 0
	fc := &fakeClient{
		fn: func(ctx context.Context, req model.GenerateRequest) (model.GenerateStream, error) {
			calls++
			if calls <= 2 {
				return nil, ollama.ErrRateLimit
			}
			return newFakeStream(0, successChunks(7, time.Second)...), nil
		},
	}
	r := &Runner{
		Client: fc,
		Config: model.Config{Runs: 1, Warmup: 0},
	}
	start := time.Now()
	res, err := r.Run(context.Background(), []model.Model{{Name: "m1", Endpoint: "cloud"}})
	elapsed := time.Since(start)
	if err != nil {
		t.Fatal(err)
	}
	if calls != 3 {
		t.Fatalf("expected 3 Generate calls (initial + 2 retries), got %d", calls)
	}
	if len(res) != 1 || len(res[0].Runs) != 1 {
		t.Fatal("expected one model with one run")
	}
	if res[0].Runs[0].Error != "" {
		t.Fatalf("expected run success, got error %q", res[0].Runs[0].Error)
	}
	if res[0].Runs[0].TokenCount != 7 {
		t.Fatalf("expected token count 7, got %d", res[0].Runs[0].TokenCount)
	}
	// 1s + 2s backoff.
	if elapsed < 2900*time.Millisecond {
		t.Fatalf("expected at least 2.9s elapsed for retry backoffs, got %v", elapsed)
	}
}

func TestPartialFailureAggregation(t *testing.T) {
	calls := 0
	fc := &fakeClient{
		fn: func(ctx context.Context, req model.GenerateRequest) (model.GenerateStream, error) {
			calls++
			if calls%2 == 1 {
				return newFakeStream(0, model.GenerateResponse{Token: "ok", Done: true, EvalCount: 10, EvalDuration: time.Second}), nil
			}
			return nil, errors.New("boom")
		},
	}
	r := &Runner{
		Client: fc,
		Config: model.Config{Runs: 3, Warmup: 0},
	}
	res, err := r.Run(context.Background(), []model.Model{{Name: "m1", Endpoint: "local"}})
	if err != nil {
		t.Fatal(err)
	}
	agg := res[0].Aggregate
	if agg.SuccessCount != 2 {
		t.Fatalf("expected 2 successes, got %d", agg.SuccessCount)
	}
	if agg.FailCount != 1 {
		t.Fatalf("expected 1 failure, got %d", agg.FailCount)
	}
	if len(res[0].Runs) != 3 {
		t.Fatalf("expected 3 runs, got %d", len(res[0].Runs))
	}
	if res[0].Runs[0].Error == "" && res[0].Runs[1].Error != "" && res[0].Runs[2].Error == "" {
		// ok
	} else {
		t.Fatalf("expected alternating success/failure/success, got errors: %q, %q, %q",
			res[0].Runs[0].Error, res[0].Runs[1].Error, res[0].Runs[2].Error)
	}
}

func TestAllFailFAILMarker(t *testing.T) {
	fc := &fakeClient{
		fn: func(ctx context.Context, req model.GenerateRequest) (model.GenerateStream, error) {
			return nil, errors.New("model unavailable")
		},
	}
	r := &Runner{
		Client: fc,
		Config: model.Config{Runs: 3, Warmup: 0},
	}
	res, err := r.Run(context.Background(), []model.Model{{Name: "m1", Endpoint: "local"}})
	if err != nil {
		t.Fatal(err)
	}
	agg := res[0].Aggregate
	if agg.SuccessCount != 0 {
		t.Fatalf("expected SuccessCount 0, got %d", agg.SuccessCount)
	}
	if agg.FailCount != 3 {
		t.Fatalf("expected FailCount 3, got %d", agg.FailCount)
	}
	for i, run := range res[0].Runs {
		if run.Error == "" {
			t.Fatalf("run %d expected error, got none", i)
		}
	}
}

func TestContextCancellationMidBenchmark(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	fc := &fakeClient{fn: constantSuccess(50 * time.Millisecond)}
	r := &Runner{
		Client: fc,
		Config: model.Config{Runs: 3, Warmup: 1},
	}

	go func() {
		time.Sleep(180 * time.Millisecond)
		cancel()
	}()

	start := time.Now()
	res, err := r.Run(ctx, []model.Model{{Name: "m1", Endpoint: "local"}})
	elapsed := time.Since(start)
	if err != context.Canceled {
		t.Fatalf("expected context.Canceled, got %v", err)
	}
	if elapsed > 400*time.Millisecond {
		t.Fatalf("expected quick return after cancel, got %v", elapsed)
	}
	if len(res) != 1 {
		t.Fatalf("expected partial result for one model, got %d", len(res))
	}
	if len(res[0].Runs) < 1 {
		t.Fatalf("expected at least one measured run before cancel, got %d", len(res[0].Runs))
	}
}

func TestProgressCallbackFires(t *testing.T) {
	fc := &fakeClient{fn: constantSuccess(0)}
	var mu sync.Mutex
	var calls []progressCall
	r := &Runner{
		Client: fc,
		Config: model.Config{Runs: 3, Warmup: 1},
		Progress: func(name string, idx int, result model.RunResult) {
			mu.Lock()
			defer mu.Unlock()
			calls = append(calls, progressCall{name, idx, result})
		},
	}
	_, err := r.Run(context.Background(), []model.Model{
		{Name: "m1", Endpoint: "local"},
		{Name: "m2", Endpoint: "local"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(calls) != 6 {
		t.Fatalf("expected 6 progress callbacks (2 models * 3 runs), got %d", len(calls))
	}
	for i, c := range calls[:3] {
		if c.model != "m1" || c.idx != i {
			t.Fatalf("call %d: expected m1/%d, got %s/%d", i, i, c.model, c.idx)
		}
	}
	for i, c := range calls[3:] {
		if c.model != "m2" || c.idx != i {
			t.Fatalf("call %d: expected m2/%d, got %s/%d", i+3, i, c.model, c.idx)
		}
	}
}

type progressCall struct {
	model  string
	idx    int
	result model.RunResult
}

func TestParallelCloudModels(t *testing.T) {
	fc := &fakeClient{fn: constantSuccess(20 * time.Millisecond)}
	r := &Runner{
		Client: fc,
		Config: model.Config{Runs: 2, Warmup: 1, Parallel: 2},
	}
	res, err := r.Run(context.Background(), []model.Model{
		{Name: "a", Endpoint: "cloud"},
		{Name: "b", Endpoint: "cloud"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(res) != 2 {
		t.Fatalf("expected 2 results, got %d", len(res))
	}
	for _, r := range res {
		if len(r.Runs) != 2 {
			t.Fatalf("expected 2 runs for %s, got %d", r.Model.Name, len(r.Runs))
		}
	}

	fc.mu.Lock()
	callRecords := append([]model.GenerateRequest(nil), fc.calls...)
	times := append([]time.Time(nil), fc.callTimes...)
	fc.mu.Unlock()
	if len(times) != 6 {
		t.Fatalf("expected 6 Generate calls, got %d", len(times))
	}

	var lastA, firstB time.Time
	foundA, foundB := false, false
	for i, req := range callRecords {
		if req.Model == "a" {
			if !foundA || times[i].After(lastA) {
				lastA = times[i]
			}
			foundA = true
		}
		if req.Model == "b" {
			if !foundB || times[i].Before(firstB) {
				firstB = times[i]
			}
			foundB = true
		}
	}
	if !foundA || !foundB {
		t.Fatalf("expected calls for both models, found a=%v b=%v", foundA, foundB)
	}
	if !firstB.Before(lastA) {
		t.Fatalf("expected model b to start before model a finished; b first=%v, a last=%v", firstB, lastA)
	}
}

func TestParallelClampWarningForLocalModels(t *testing.T) {
	var buf bytes.Buffer
	logger := log.New(&buf, "", 0)
	fc := &fakeClient{fn: constantSuccess(20 * time.Millisecond)}
	r := &Runner{
		Client: fc,
		Config: model.Config{Runs: 1, Warmup: 0, Parallel: 2},
		Logger: logger,
	}
	res, err := r.Run(context.Background(), []model.Model{
		{Name: "a", Endpoint: "local"},
		{Name: "b", Endpoint: "local"},
	})
	if err != nil {
		t.Fatal(err)
	}
	out := buf.String()
	if !strings.Contains(out, "Warning") || !strings.Contains(out, "local") {
		t.Fatalf("expected warning about local models, got %q", out)
	}
	fc.mu.Lock()
	times := append([]time.Time(nil), fc.callTimes...)
	fc.mu.Unlock()
	if len(times) != 2 {
		t.Fatalf("expected 2 Generate calls, got %d", len(times))
	}
	// Local models are sequential: b's call is after a's call.
	if !times[1].After(times[0]) {
		t.Fatalf("expected local models sequential; got %v then %v", times[0], times[1])
	}
	if len(res) != 2 {
		t.Fatalf("expected 2 results, got %d", len(res))
	}
}
