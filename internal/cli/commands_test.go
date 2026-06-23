package cli

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/simonteague6/ollama-model-tester/internal/benchmark"
	"github.com/simonteague6/ollama-model-tester/internal/model"
	"github.com/simonteague6/ollama-model-tester/internal/store"
	"github.com/spf13/cobra"
)

type fakeRunner struct {
	results []benchmark.Result
	err     error
}

func (f *fakeRunner) Run(ctx context.Context, models []model.Model) ([]benchmark.Result, error) {
	return f.results, f.err
}

type fakeClient struct {
	models []model.Model
}

func (f *fakeClient) ListModels(ctx context.Context) ([]model.Model, error) {
	return f.models, nil
}

func (f *fakeClient) Generate(ctx context.Context, req model.GenerateRequest) (model.GenerateStream, error) {
	return nil, errors.New("not implemented")
}

type fakeStore struct {
	sessions []store.Session
	results  [][]store.StoredResult
	err      error
}

func (f *fakeStore) SaveSession(session store.Session, results []store.StoredResult) error {
	f.sessions = append(f.sessions, session)
	f.results = append(f.results, results)
	return f.err
}

func (f *fakeStore) ListSessions(limit, offset int) ([]store.Session, error) {
	if offset >= len(f.sessions) {
		return nil, f.err
	}
	if limit <= 0 || offset+limit > len(f.sessions) {
		return f.sessions[offset:], f.err
	}
	return f.sessions[offset : offset+limit], f.err
}

func (f *fakeStore) GetSession(id string) (store.Session, []store.StoredResult, error) {
	for i, s := range f.sessions {
		if s.ID == id {
			return s, f.results[i], f.err
		}
	}
	return store.Session{}, nil, store.ErrSessionNotFound
}

func setFakeRunner(results []benchmark.Result) func() {
	orig := newRunner
	newRunner = func(c model.Client, cfg model.Config) runner {
		return &fakeRunner{results: results}
	}
	return func() { newRunner = orig }
}

func setFakeClients(local, cloud []model.Model) func() {
	origLocal := newLocalClient
	origCloud := newCloudClient
	newLocalClient = func(baseURL string) model.Client {
		return &fakeClient{models: local}
	}
	newCloudClient = func(baseURL, apiKey string) model.Client {
		return &fakeClient{models: cloud}
	}
	return func() {
		newLocalClient = origLocal
		newCloudClient = origCloud
	}
}

func TestBareRootPrintsStub(t *testing.T) {
	cfg := model.Config{}
	root := BuildRootCmd(&cfg, nil)
	var out bytes.Buffer
	root.SetOut(&out)
	root.SetArgs([]string{})

	// Override TUI launch to avoid blocking on real bubbletea program.
	orig := runTUIProgram
	runTUIProgram = func(cmd *cobra.Command, cfg *model.Config, st store.Store) {
		fmt.Fprintln(cmd.OutOrStdout(), "TUI launched")
	}
	defer func() { runTUIProgram = orig }()

	if err := root.Execute(); err != nil {
		t.Fatalf("execute: %v", err)
	}
	if !strings.Contains(out.String(), "TUI launched") {
		t.Errorf("expected TUI launch message, got %q", out.String())
	}
}

func TestTuiCommandPrintsStub(t *testing.T) {
	cfg := model.Config{}
	root := BuildRootCmd(&cfg, nil)
	var out bytes.Buffer
	root.SetOut(&out)
	root.SetArgs([]string{"tui"})

	orig := runTUIProgram
	runTUIProgram = func(cmd *cobra.Command, cfg *model.Config, st store.Store) {
		fmt.Fprintln(cmd.OutOrStdout(), "TUI launched")
	}
	defer func() { runTUIProgram = orig }()

	if err := root.Execute(); err != nil {
		t.Fatalf("execute: %v", err)
	}
	if !strings.Contains(out.String(), "TUI launched") {
		t.Errorf("expected TUI launch message, got %q", out.String())
	}
}

func TestFlagBindingOnlyChangesSetFlags(t *testing.T) {
	cfg := model.Config{
		Runs:      5,
		Warmup:    1,
		Prompt:    "default prompt",
		MaxTokens: 256,
		Timeout:   60 * time.Second,
		Parallel:  1,
		SortKey:   "ttft",
		LocalURL:  "http://localhost:11434",
		CloudURL:  "https://ollama.com/api",
		APIKey:    "secret",
	}
	defer setFakeRunner(nil)()

	root := BuildRootCmd(&cfg, nil)
	var out bytes.Buffer
	root.SetOut(&out)
	root.SetErr(&out)
	root.SetArgs([]string{
		"run", "model1",
		"--local",
		"--runs", "3",
		"--prompt", "custom prompt",
	})

	if err := root.Execute(); err != nil {
		t.Fatalf("execute: %v", err)
	}

	if cfg.Runs != 3 {
		t.Errorf("Runs changed unexpectedly: got %d, want 3", cfg.Runs)
	}
	if cfg.Prompt != "custom prompt" {
		t.Errorf("Prompt not updated: got %q, want %q", cfg.Prompt, "custom prompt")
	}
	if cfg.Warmup != 1 {
		t.Errorf("Warmup clobbered: got %d, want 1", cfg.Warmup)
	}
	if cfg.MaxTokens != 256 {
		t.Errorf("MaxTokens clobbered: got %d, want 256", cfg.MaxTokens)
	}
	if cfg.Timeout != 60*time.Second {
		t.Errorf("Timeout clobbered: got %v, want 60s", cfg.Timeout)
	}
	if cfg.Parallel != 1 {
		t.Errorf("Parallel clobbered: got %d, want 1", cfg.Parallel)
	}
	if cfg.SortKey != "ttft" {
		t.Errorf("SortKey clobbered: got %q, want ttft", cfg.SortKey)
	}
}

func TestCloudAPIKeyPreFlight(t *testing.T) {
	cfg := model.Config{CloudURL: "https://ollama.com/api"} // APIKey intentionally empty
	root := BuildRootCmd(&cfg, nil)
	var out bytes.Buffer
	root.SetOut(&out)
	root.SetErr(&out)
	root.SetArgs([]string{"run", "llama", "--cloud"})

	err := root.Execute()
	if err == nil {
		t.Fatal("expected error for missing cloud API key")
	}
	if !strings.Contains(err.Error(), "OLLAMA_API_KEY") {
		t.Errorf("error should mention OLLAMA_API_KEY: %v", err)
	}
}

func TestListOutputFormat(t *testing.T) {
	cfg := model.Config{
		APIKey:   "secret",
		LocalURL: "http://local",
		CloudURL: "https://cloud",
	}
	defer setFakeClients(
		[]model.Model{{Name: "local-model", Endpoint: "local"}},
		[]model.Model{{Name: "cloud-model", Endpoint: "cloud"}},
	)()

	root := BuildRootCmd(&cfg, nil)
	var out bytes.Buffer
	root.SetOut(&out)
	root.SetArgs([]string{"list", "--both"})

	if err := root.Execute(); err != nil {
		t.Fatalf("execute: %v", err)
	}

	output := out.String()
	if !strings.Contains(output, "MODEL") || !strings.Contains(output, "ENDPOINT") {
		t.Errorf("expected table headers, got:\n%s", output)
	}
	if !strings.Contains(output, "local-model") {
		t.Errorf("expected local row, got:\n%s", output)
	}
	if !strings.Contains(output, "cloud-model") {
		t.Errorf("expected cloud row, got:\n%s", output)
	}
}

func TestRunWithFakeClient(t *testing.T) {
	cfg := model.Config{
		APIKey:   "secret",
		CloudURL: "https://cloud",
		SortKey:  "ttft",
	}
	results := []benchmark.Result{
		{
			Model: model.Model{Name: "llama", Endpoint: "cloud"},
			Aggregate: model.AggregateResult{
				MeanTTFT:     100 * time.Millisecond,
				MedianTTFT:   100 * time.Millisecond,
				MeanTPS:      50.0,
				MedianTPS:    50.0,
				MeanTotal:    1 * time.Second,
				MedianTotal:  1 * time.Second,
				SuccessCount: 5,
				FailCount:    0,
			},
		},
	}
	defer setFakeRunner(results)()

	root := BuildRootCmd(&cfg, nil)
	var out bytes.Buffer
	root.SetOut(&out)
	root.SetArgs([]string{"run", "llama", "--cloud"})

	if err := root.Execute(); err != nil {
		t.Fatalf("execute: %v", err)
	}

	output := out.String()
	if !strings.Contains(output, "llama") {
		t.Errorf("expected model name, got:\n%s", output)
	}
	if !strings.Contains(output, "50.00") {
		t.Errorf("expected tok/s value, got:\n%s", output)
	}
	if !strings.Contains(output, "5/5") {
		t.Errorf("expected OK count, got:\n%s", output)
	}
}

func TestRunPrintsPartialResultsOnCancellation(t *testing.T) {
	cfg := model.Config{
		APIKey:   "secret",
		CloudURL: "https://cloud",
	}
	results := []benchmark.Result{
		{
			Model: model.Model{Name: "llama", Endpoint: "cloud"},
			Aggregate: model.AggregateResult{
				MeanTTFT:     100 * time.Millisecond,
				MedianTTFT:   100 * time.Millisecond,
				MeanTPS:      50.0,
				MedianTPS:    50.0,
				MeanTotal:    1 * time.Second,
				MedianTotal:  1 * time.Second,
				SuccessCount: 5,
				FailCount:    0,
			},
		},
	}
	defer setFakeRunnerWithErr(results, context.Canceled)()

	root := BuildRootCmd(&cfg, nil)
	var out bytes.Buffer
	root.SetOut(&out)
	root.SetArgs([]string{"run", "llama", "--cloud"})

	if err := root.Execute(); err != nil {
		t.Fatalf("execute: %v", err)
	}

	if !strings.Contains(out.String(), "llama") {
		t.Errorf("expected partial results to be printed, got:\n%s", out.String())
	}
}

func setFakeRunnerWithErr(results []benchmark.Result, err error) func() {
	orig := newRunner
	newRunner = func(c model.Client, cfg model.Config) runner {
		return &fakeRunner{results: results, err: err}
	}
	return func() { newRunner = orig }
}

func TestRunJSONGoldenOutput(t *testing.T) {
	cfg := model.Config{
		APIKey:   "secret",
		CloudURL: "https://cloud",
	}
	results := []benchmark.Result{
		{
			Model: model.Model{Name: "llama", Endpoint: "cloud"},
			Aggregate: model.AggregateResult{
				MeanTTFT:     100 * time.Millisecond,
				MedianTTFT:   100 * time.Millisecond,
				MinTTFT:      90 * time.Millisecond,
				MaxTTFT:      110 * time.Millisecond,
				MeanTPS:      50.0,
				MedianTPS:    50.0,
				MinTPS:       48.0,
				MaxTPS:       52.0,
				MeanTotal:    1 * time.Second,
				MedianTotal:  1 * time.Second,
				MinTotal:     900 * time.Millisecond,
				MaxTotal:     1100 * time.Millisecond,
				SuccessCount: 5,
				FailCount:    0,
			},
			Runs: []model.RunResult{
				{TTFT: 100 * time.Millisecond, TokensPerSec: 50.0, TotalTime: 1 * time.Second, TokenCount: 10},
			},
		},
	}
	defer setFakeRunner(results)()

	root := BuildRootCmd(&cfg, nil)
	var out bytes.Buffer
	root.SetOut(&out)
	root.SetArgs([]string{"run", "llama", "--cloud", "--json"})

	if err := root.Execute(); err != nil {
		t.Fatalf("execute: %v", err)
	}

	var rows []jsonRow
	if err := json.Unmarshal(out.Bytes(), &rows); err != nil {
		t.Fatalf("invalid JSON: %v\n%s", err, out.String())
	}
	if len(rows) != 1 {
		t.Fatalf("expected 1 JSON row, got %d", len(rows))
	}
	row := rows[0]
	if row.Model != "llama" {
		t.Errorf("Model: got %q, want llama", row.Model)
	}
	if row.Endpoint != "cloud" {
		t.Errorf("Endpoint: got %q, want cloud", row.Endpoint)
	}
	if row.TTFTMs != 100 {
		t.Errorf("TTFTMs: got %d, want 100", row.TTFTMs)
	}
	if row.TokensPerSec != 50.0 {
		t.Errorf("TokensPerSec: got %f, want 50.0", row.TokensPerSec)
	}
	if row.OK != "5/5" {
		t.Errorf("OK: got %q, want 5/5", row.OK)
	}
	if row.SuccessCount != 5 || row.FailCount != 0 {
		t.Errorf("counts: got %d/%d, want 5/0", row.SuccessCount, row.FailCount)
	}
}

func TestRunSavesSessionWithFakeStore(t *testing.T) {
	cfg := model.Config{
		APIKey:    "secret",
		CloudURL:  "https://cloud",
		Prompt:    "test prompt",
		Runs:      3,
		Warmup:    1,
		MaxTokens: 128,
		Timeout:   30 * time.Second,
		Parallel:  2,
	}
	results := []benchmark.Result{
		{
			Model: model.Model{Name: "llama", Endpoint: "cloud"},
			Aggregate: model.AggregateResult{
				MeanTTFT:     100 * time.Millisecond,
				MedianTTFT:   100 * time.Millisecond,
				MeanTPS:      50.0,
				MedianTPS:    50.0,
				MeanTotal:    1 * time.Second,
				MedianTotal:  1 * time.Second,
				SuccessCount: 3,
				FailCount:    0,
			},
			Runs: []model.RunResult{
				{TTFT: 100 * time.Millisecond, TokensPerSec: 50.0, TotalTime: 1 * time.Second, TokenCount: 10},
			},
		},
	}
	defer setFakeRunner(results)()

	fs := &fakeStore{}
	root := BuildRootCmd(&cfg, fs)
	var out bytes.Buffer
	root.SetOut(&out)
	root.SetErr(&out)
	root.SetArgs([]string{"run", "llama", "--cloud"})

	if err := root.Execute(); err != nil {
		t.Fatalf("execute: %v", err)
	}

	if len(fs.sessions) != 1 {
		t.Fatalf("expected 1 saved session, got %d", len(fs.sessions))
	}
	if len(fs.results) != 1 || len(fs.results[0]) != 1 {
		t.Fatalf("expected 1 stored result, got %d", len(fs.results))
	}
	if fs.results[0][0].ModelName != "llama" {
		t.Errorf("stored result model: got %q, want llama", fs.results[0][0].ModelName)
	}
	if fs.sessions[0].Prompt != "test prompt" {
		t.Errorf("session prompt: got %q, want %q", fs.sessions[0].Prompt, "test prompt")
	}
	if len(fs.sessions[0].ModelsTested) != 1 || fs.sessions[0].ModelsTested[0] != "llama" {
		t.Errorf("session models tested: got %v, want [llama]", fs.sessions[0].ModelsTested)
	}
}

func TestRunSaveFailureLogsWarning(t *testing.T) {
	cfg := model.Config{
		APIKey:   "secret",
		CloudURL: "https://cloud",
	}
	results := []benchmark.Result{
		{
			Model: model.Model{Name: "llama", Endpoint: "cloud"},
			Aggregate: model.AggregateResult{
				MeanTTFT:     100 * time.Millisecond,
				MedianTTFT:   100 * time.Millisecond,
				MeanTPS:      50.0,
				MedianTPS:    50.0,
				MeanTotal:    1 * time.Second,
				MedianTotal:  1 * time.Second,
				SuccessCount: 3,
				FailCount:    0,
			},
		},
	}
	defer setFakeRunner(results)()

	fs := &fakeStore{err: errors.New("disk full")}
	root := BuildRootCmd(&cfg, fs)
	var out bytes.Buffer
	var errOut bytes.Buffer
	root.SetOut(&out)
	root.SetErr(&errOut)
	root.SetArgs([]string{"run", "llama", "--cloud"})

	if err := root.Execute(); err != nil {
		t.Fatalf("execute: %v", err)
	}

	if len(fs.sessions) != 1 {
		t.Fatalf("expected save attempt, got %d sessions", len(fs.sessions))
	}
	if !strings.Contains(errOut.String(), "disk full") {
		t.Errorf("expected warning in stderr, got:\n%s", errOut.String())
	}
}

func TestSortResultsDescending(t *testing.T) {
	results := []benchmark.Result{
		{
			Model:     model.Model{Name: "a"},
			Aggregate: model.AggregateResult{MedianTPS: 10.0, MedianTTFT: 200 * time.Millisecond, MedianTotal: 2 * time.Second},
		},
		{
			Model:     model.Model{Name: "b"},
			Aggregate: model.AggregateResult{MedianTPS: 20.0, MedianTTFT: 100 * time.Millisecond, MedianTotal: 1 * time.Second},
		},
	}

	tests := []struct {
		key       string
		wantFirst string
	}{
		{"tok/s", "b"},
		{"ttft", "a"},
		{"total", "a"},
		{"model", "b"},
	}

	for _, tc := range tests {
		cp := append([]benchmark.Result{}, results...)
		sortResults(cp, tc.key)
		if cp[0].Model.Name != tc.wantFirst {
			t.Errorf("sort by %q: expected %s first, got %s", tc.key, tc.wantFirst, cp[0].Model.Name)
		}
	}
}


func makeHistoryStore(modelSets ...[]string) *fakeStore {
	base := time.Date(2024, 6, 1, 12, 0, 0, 0, time.UTC)
	fs := &fakeStore{}
	for i, models := range modelSets {
		id := fmt.Sprintf("session-%d", i+1)
		session := store.Session{
			ID:            id,
			Timestamp:     base.Add(time.Duration(i) * time.Hour),
			ModelsTested:  models,
			Prompt:        "test prompt",
			ConfigSummary: "runs=3",
		}
		results := make([]store.StoredResult, 0, len(models))
		for _, m := range models {
			results = append(results, store.StoredResult{
				ModelName: m,
				Endpoint:  "local",
				Aggregate: model.AggregateResult{
					MinTTFT:      100 * time.Millisecond,
					MaxTPS:       50.0,
					SuccessCount: 1,
				},
				Runs: []model.RunResult{
					{TTFT: 100 * time.Millisecond, TokensPerSec: 50.0, TotalTime: 1 * time.Second, TokenCount: 50},
				},
			})
		}
		fs.sessions = append(fs.sessions, session)
		fs.results = append(fs.results, results)
	}
	return fs
}

func TestHistoryPrintsSessionTable(t *testing.T) {
	fs := makeHistoryStore([]string{"llama3", "mistral"})
	cfg := model.Config{}
	root := BuildRootCmd(&cfg, fs)
	var out bytes.Buffer
	root.SetOut(&out)
	root.SetArgs([]string{"history"})

	if err := root.Execute(); err != nil {
		t.Fatalf("execute: %v", err)
	}

	output := out.String()
	if !strings.Contains(output, "2024-06-01") {
		t.Errorf("expected date in output, got:\n%s", output)
	}
	if !strings.Contains(output, "llama3") {
		t.Errorf("expected model name in output, got:\n%s", output)
	}
	if !strings.Contains(output, "BEST TTFT") {
		t.Errorf("expected BEST TTFT header, got:\n%s", output)
	}
}

func TestHistoryEmptyStore(t *testing.T) {
	fs := &fakeStore{}
	cfg := model.Config{}
	root := BuildRootCmd(&cfg, fs)
	var out bytes.Buffer
	root.SetOut(&out)
	root.SetArgs([]string{"history"})

	if err := root.Execute(); err != nil {
		t.Fatalf("execute: %v", err)
	}

	if !strings.Contains(out.String(), "No benchmark history yet") {
		t.Errorf("expected empty message, got:\n%s", out.String())
	}
}

func TestHistoryJSONOutput(t *testing.T) {
	fs := makeHistoryStore([]string{"llama3"})
	cfg := model.Config{}
	root := BuildRootCmd(&cfg, fs)
	var out bytes.Buffer
	root.SetOut(&out)
	root.SetArgs([]string{"history", "--json"})

	if err := root.Execute(); err != nil {
		t.Fatalf("execute: %v", err)
	}

	var sessions []store.Session
	if err := json.Unmarshal(out.Bytes(), &sessions); err != nil {
		t.Fatalf("invalid JSON: %v\n%s", err, out.String())
	}
	if len(sessions) != 1 {
		t.Errorf("expected 1 session, got %d", len(sessions))
	}
}

func TestHistoryFilterByModel(t *testing.T) {
	fs := makeHistoryStore([]string{"llama3", "mistral"}, []string{"mistral"}, []string{"llama3"})
	cfg := model.Config{}
	root := BuildRootCmd(&cfg, fs)
	var out bytes.Buffer
	root.SetOut(&out)
	root.SetArgs([]string{"history", "--model", "llama3", "--json"})

	if err := root.Execute(); err != nil {
		t.Fatalf("execute: %v", err)
	}

	var sessions []store.Session
	if err := json.Unmarshal(out.Bytes(), &sessions); err != nil {
		t.Fatalf("invalid JSON: %v\n%s", err, out.String())
	}
	if len(sessions) != 2 {
		t.Errorf("expected 2 sessions with llama3, got %d", len(sessions))
	}
	for _, s := range sessions {
		found := false
		for _, m := range s.ModelsTested {
			if m == "llama3" {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("session %s does not contain llama3", s.ID)
		}
	}
}

func TestHistoryLimit(t *testing.T) {
	modelSets := make([][]string, 10)
	for i := range modelSets {
		modelSets[i] = []string{"llama3"}
	}
	fs := makeHistoryStore(modelSets...)
	cfg := model.Config{}
	root := BuildRootCmd(&cfg, fs)
	var out bytes.Buffer
	root.SetOut(&out)
	root.SetArgs([]string{"history", "--limit", "5", "--json"})

	if err := root.Execute(); err != nil {
		t.Fatalf("execute: %v", err)
	}

	var sessions []store.Session
	if err := json.Unmarshal(out.Bytes(), &sessions); err != nil {
		t.Fatalf("invalid JSON: %v\n%s", err, out.String())
	}
	if len(sessions) != 5 {
		t.Errorf("expected 5 sessions, got %d", len(sessions))
	}
}

func TestHistoryDetail(t *testing.T) {
	fs := makeHistoryStore([]string{"llama3", "mistral"})
	cfg := model.Config{}
	root := BuildRootCmd(&cfg, fs)
	var out bytes.Buffer
	root.SetOut(&out)
	root.SetArgs([]string{"history", "--detail", "session-1"})

	if err := root.Execute(); err != nil {
		t.Fatalf("execute: %v", err)
	}

	output := out.String()
	if !strings.Contains(output, "Session: session-1") {
		t.Errorf("expected session header, got:\n%s", output)
	}
	if !strings.Contains(output, "llama3") {
		t.Errorf("expected model name, got:\n%s", output)
	}
	if !strings.Contains(output, "RUN") {
		t.Errorf("expected per-run breakdown header, got:\n%s", output)
	}
}

func TestHistoryNilStore(t *testing.T) {
	cfg := model.Config{}
	root := BuildRootCmd(&cfg, nil)
	var out bytes.Buffer
	root.SetOut(&out)
	root.SetArgs([]string{"history"})

	if err := root.Execute(); err != nil {
		t.Fatalf("execute: %v", err)
	}

	if !strings.Contains(out.String(), "History not available (database not initialized)") {
		t.Errorf("expected nil store message, got:\n%s", out.String())
	}
}