package tui

import (
	"regexp"
	"strings"
	"testing"
	"time"

	tea "charm.land/bubbletea/v2"

	"github.com/simonteague6/ollama-model-tester/internal/benchmark"
	"github.com/simonteague6/ollama-model-tester/internal/metrics"
	"github.com/simonteague6/ollama-model-tester/internal/model"
)

var ansiRE = regexp.MustCompile(`\x1b\[[0-9;]*[A-Za-z]`)

func stripANSI(s string) string {
	return ansiRE.ReplaceAllString(s, "")
}

func viewString(m *AppModel) string {
	return stripANSI(m.View().Content)
}

func viewContains(m *AppModel, substr string) bool {
	return strings.Contains(viewString(m), substr)
}

func update(m *AppModel, msg tea.Msg) (*AppModel, tea.Cmd) {
	newM, cmd := m.Update(msg)
	return newM.(*AppModel), cmd
}

func keyMsg(k string) tea.KeyPressMsg {
	switch k {
	case "enter":
		return tea.KeyPressMsg{Code: tea.KeyEnter}
	case "esc":
		return tea.KeyPressMsg{Code: tea.KeyEscape}
	case "space":
		return tea.KeyPressMsg{Code: tea.KeySpace}
	case "backspace":
		return tea.KeyPressMsg{Code: tea.KeyBackspace}
	case "tab":
		return tea.KeyPressMsg{Code: tea.KeyTab}
	case "up":
		return tea.KeyPressMsg{Code: tea.KeyUp}
	case "down":
		return tea.KeyPressMsg{Code: tea.KeyDown}
	case "ctrl+c":
		return tea.KeyPressMsg{Code: 'c', Mod: tea.ModCtrl}
	default:
		if len(k) == 1 {
			return tea.KeyPressMsg{Text: k, Code: rune(k[0])}
		}
	}
	return tea.KeyPressMsg{}
}

func pressKey(m *AppModel, k string) *AppModel {
	m, _ = update(m, keyMsg(k))
	return m
}

func pressKeys(m *AppModel, keys ...string) *AppModel {
	for _, k := range keys {
		m = pressKey(m, k)
	}
	return m
}

func testConfig() model.Config {
	return model.Config{Runs: 3}
}

func makeRun(ttft, total time.Duration, tps float64, tokens int) model.RunResult {
	return model.RunResult{
		TTFT:         ttft,
		TotalTime:    total,
		TokensPerSec: tps,
		TokenCount:   tokens,
	}
}

func TestSelectScreenLoadsAndDisplaysModels(t *testing.T) {
	m := New(testConfig(), nil, nil)
	m, _ = update(m, ListLoadedMsg{Models: []model.Model{
		{Name: "llama3", Endpoint: "local"},
		{Name: "mistral", Endpoint: "cloud"},
	}})

	if !viewContains(m, "llama3") {
		t.Errorf("expected view to contain llama3, got:\n%s", viewString(m))
	}
	if !viewContains(m, "mistral") {
		t.Errorf("expected view to contain mistral, got:\n%s", viewString(m))
	}
	if !viewContains(m, "Select models") {
		t.Errorf("expected view to contain Select header, got:\n%s", viewString(m))
	}
	if !viewContains(m, "2 models") {
		t.Errorf("expected model count summary, got:\n%s", viewString(m))
	}
}

func TestFilterNarrowsModelList(t *testing.T) {
	m := New(testConfig(), nil, nil)
	m, _ = update(m, ListLoadedMsg{Models: []model.Model{
		{Name: "alpha"},
		{Name: "alphabet"},
		{Name: "beta"},
	}})

	// Simulate the user typing a filter directly on the list component.
	m.list.SetFilterText("alpha")

	if viewContains(m, "[ ] beta") {
		t.Errorf("expected beta to be filtered out, got:\n%s", viewString(m))
	}
	if !viewContains(m, "alpha") {
		t.Errorf("expected alpha to remain visible, got:\n%s", viewString(m))
	}

	// Esc clears the filter.
	m = pressKey(m, "esc")
	if !viewContains(m, "[ ] beta") {
		t.Errorf("expected beta to return after clearing filter, got:\n%s", viewString(m))
	}
}

func TestSpaceTogglesSelection(t *testing.T) {
	m := New(testConfig(), nil, nil)
	m, _ = update(m, ListLoadedMsg{Models: []model.Model{
		{Name: "llama3", Endpoint: "local"},
		{Name: "mistral", Endpoint: "cloud"},
	}})

	m = pressKey(m, "space")
	if !viewContains(m, "[x] llama3") {
		t.Errorf("expected llama3 to be selected, got:\n%s", viewString(m))
	}

	m = pressKey(m, "space")
	if viewContains(m, "[x] llama3") {
		t.Errorf("expected llama3 selection to be cleared, got:\n%s", viewString(m))
	}
}

func TestFullFlowSelectRunningResultsDetailResults(t *testing.T) {
	m := New(testConfig(), nil, nil)
	m, _ = update(m, ListLoadedMsg{Models: []model.Model{
		{Name: "llama3", Endpoint: "local"},
		{Name: "mistral", Endpoint: "cloud"},
	}})

	// Select llama3 and start.
	m = pressKeys(m, "space", "enter")
	if m.state != screenRunning {
		t.Fatalf("expected state Running, got %d", m.state)
	}
	if !viewContains(m, "Running") {
		t.Errorf("expected running view, got:\n%s", viewString(m))
	}

	// Send progress for llama3.
	for i := 0; i < 3; i++ {
		m, _ = update(m, ProgressMsg{
			Name:     "llama3",
			Endpoint: "local",
			RunIndex: i,
			Result:   makeRun(10*time.Millisecond, 50*time.Millisecond, 20, 10),
		})
	}

	runs := []model.RunResult{
		makeRun(10*time.Millisecond, 50*time.Millisecond, 20, 10),
		makeRun(11*time.Millisecond, 51*time.Millisecond, 19, 10),
		makeRun(9*time.Millisecond, 49*time.Millisecond, 21, 10),
	}
	m, _ = update(m, RunDoneMsg{Results: []benchmark.Result{
		{Model: model.Model{Name: "llama3", Endpoint: "local"}, Runs: runs, Aggregate: metrics.Aggregate(runs)},
	}})

	if m.state != screenResults {
		t.Fatalf("expected state Results, got %d", m.state)
	}
	if !viewContains(m, "llama3") || !viewContains(m, "Results") {
		t.Errorf("expected results view, got:\n%s", viewString(m))
	}

	m = pressKey(m, "enter")
	if m.state != screenDetail {
		t.Fatalf("expected state Detail, got %d", m.state)
	}
	v := viewString(m)
	for _, want := range []string{"Aggregate stats", "Per-run breakdown", "run 1", "run 2", "run 3"} {
		if !strings.Contains(v, want) {
			t.Errorf("expected detail view to contain %q, got:\n%s", want, v)
		}
	}

	m = pressKey(m, "esc")
	if m.state != screenResults {
		t.Fatalf("expected state Results after detail back, got %d", m.state)
	}
}

func TestSortKeyCycling(t *testing.T) {
	m := New(testConfig(), nil, nil)
	m, _ = update(m, ListLoadedMsg{Models: []model.Model{
		{Name: "fast", Endpoint: "local"},
		{Name: "slow", Endpoint: "cloud"},
	}})

	// Select both and start.
	m = pressKey(m, "space")
	m = pressKey(m, "down")
	m = pressKey(m, "space")
	m = pressKey(m, "enter")

	fastRuns := []model.RunResult{makeRun(5*time.Millisecond, 30*time.Millisecond, 40, 10)}
	slowRuns := []model.RunResult{makeRun(15*time.Millisecond, 80*time.Millisecond, 10, 10)}
	m, _ = update(m, RunDoneMsg{Results: []benchmark.Result{
		{Model: model.Model{Name: "fast", Endpoint: "local"}, Runs: fastRuns, Aggregate: metrics.Aggregate(fastRuns)},
		{Model: model.Model{Name: "slow", Endpoint: "cloud"}, Runs: slowRuns, Aggregate: metrics.Aggregate(slowRuns)},
	}})

	if m.sortKey != sortTTFT {
		t.Fatalf("expected initial sort TTFT, got %s", m.sortKey)
	}

	cycles := []sortKey{sortTPS, sortTotal, sortModel, sortTTFT}
	for _, want := range cycles {
		m = pressKey(m, "s")
		if m.sortKey != want {
			t.Errorf("expected sort %s, got %s", want, m.sortKey)
		}
	}
}

func TestCancelDuringRunningReturnsPartialResults(t *testing.T) {
	m := New(testConfig(), nil, nil)
	m, _ = update(m, ListLoadedMsg{Models: []model.Model{
		{Name: "llama3", Endpoint: "local"},
	}})
	m = pressKeys(m, "space", "enter")

	// One partial run completes.
	m, _ = update(m, ProgressMsg{
		Name:     "llama3",
		Endpoint: "local",
		RunIndex: 0,
		Result:   makeRun(10*time.Millisecond, 50*time.Millisecond, 20, 10),
	})

	// Cancel and deliver the runner's partial results.
	m = pressKey(m, "esc")
	partialRuns := []model.RunResult{makeRun(10*time.Millisecond, 50*time.Millisecond, 20, 10)}
	m, _ = update(m, RunDoneMsg{Results: []benchmark.Result{
		{Model: model.Model{Name: "llama3", Endpoint: "local"}, Runs: partialRuns, Aggregate: metrics.Aggregate(partialRuns)},
	}})

	if m.state != screenResults {
		t.Fatalf("expected state Results after cancel, got %d", m.state)
	}
	if !viewContains(m, "llama3") {
		t.Errorf("expected partial results to show llama3, got:\n%s", viewString(m))
	}
}

func TestResultsRReturnsToSelectAndQQuits(t *testing.T) {
	m := New(testConfig(), nil, nil)
	m, _ = update(m, ListLoadedMsg{Models: []model.Model{
		{Name: "llama3", Endpoint: "local"},
	}})
	m = pressKeys(m, "space", "enter")

	runs := []model.RunResult{makeRun(10*time.Millisecond, 50*time.Millisecond, 20, 10)}
	m, _ = update(m, RunDoneMsg{Results: []benchmark.Result{
		{Model: model.Model{Name: "llama3", Endpoint: "local"}, Runs: runs, Aggregate: metrics.Aggregate(runs)},
	}})

	m = pressKey(m, "r")
	if m.state != screenSelect {
		t.Fatalf("expected state Select after r, got %d", m.state)
	}
	if !viewContains(m, "Select models") {
		t.Errorf("expected select view after r, got:\n%s", viewString(m))
	}

	_, cmd := update(m, keyMsg("q"))
	if cmd == nil {
		t.Fatalf("expected quit command, got nil")
	}
	msg := cmd()
	if _, ok := msg.(tea.QuitMsg); !ok {
		t.Errorf("expected QuitMsg, got %T", msg)
	}
}

func TestDetailViewIncludesAggregateAndPerRunData(t *testing.T) {
	m := New(testConfig(), nil, nil)
	m, _ = update(m, ListLoadedMsg{Models: []model.Model{
		{Name: "llama3", Endpoint: "local"},
	}})
	m = pressKeys(m, "space", "enter")

	runs := []model.RunResult{
		makeRun(10*time.Millisecond, 50*time.Millisecond, 20, 10),
		makeRun(12*time.Millisecond, 55*time.Millisecond, 18, 9),
		makeRun(9*time.Millisecond, 48*time.Millisecond, 22, 11),
	}
	m, _ = update(m, RunDoneMsg{Results: []benchmark.Result{
		{Model: model.Model{Name: "llama3", Endpoint: "local"}, Runs: runs, Aggregate: metrics.Aggregate(runs)},
	}})

	m = pressKey(m, "enter")
	v := viewString(m)
	required := []string{
		"Aggregate stats",
		"Per-run breakdown",
		"mean:",
		"median:",
		"min:",
		"max:",
		"run 1",
		"run 2",
		"run 3",
		"llama3",
	}
	for _, r := range required {
		if !strings.Contains(v, r) {
			t.Errorf("expected detail view to contain %q, got:\n%s", r, v)
		}
	}
}

func TestRunningViewShowsPerRunMetricsForInProgressModels(t *testing.T) {
	m := New(model.Config{Runs: 5}, nil, nil)
	m, _ = update(m, ListLoadedMsg{Models: []model.Model{
		{Name: "llama3", Endpoint: "local"},
	}})
	m = pressKeys(m, "space", "enter")

	for i := 0; i < 2; i++ {
		m, _ = update(m, ProgressMsg{
			Name:     "llama3",
			Endpoint: "local",
			RunIndex: i,
			Result:   makeRun(10*time.Millisecond, 50*time.Millisecond, 20, 10),
		})
	}

	v := viewString(m)
	for _, want := range []string{"llama3/local", "Run 1", "Run 2", "TTFT:", "tok/s:", "Total:"} {
		if !strings.Contains(v, want) {
			t.Errorf("expected running view to contain %q, got:\n%s", want, v)
		}
	}
}
