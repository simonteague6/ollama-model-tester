// Package tui implements the interactive bubbletea v2 interface for omt.
package tui

import (
	"context"
	"crypto/rand"
	"fmt"
	"io"
	"log"
	"sort"
	"strings"
	"sync"
	"time"

	"charm.land/bubbles/v2/key"
	list "charm.land/bubbles/v2/list"
	progress "charm.land/bubbles/v2/progress"
	spinner "charm.land/bubbles/v2/spinner"
	table "charm.land/bubbles/v2/table"
	viewport "charm.land/bubbles/v2/viewport"
	tea "charm.land/bubbletea/v2"
	lipgloss "charm.land/lipgloss/v2"

	"github.com/simonteague6/ollama-model-tester/internal/benchmark"
	"github.com/simonteague6/ollama-model-tester/internal/metrics"
	"github.com/simonteague6/ollama-model-tester/internal/model"
	"github.com/simonteague6/ollama-model-tester/internal/store"
)

// screen enumerates the five TUI states.
type screen int

const (
	screenSelect screen = iota
	screenRunning
	screenResults
	screenDetail
	screenHistory
)

// endpointMode selects which endpoints are discovered and benchmarked.
type endpointMode int

const (
	endpointLocal endpointMode = iota
	endpointCloud
	endpointBoth
)

func (e endpointMode) String() string {
	switch e {
	case endpointLocal:
		return "local"
	case endpointCloud:
		return "cloud"
	default:
		return "both"
	}
}

// sortKey selects the active results table sort column.
type sortKey int

const (
	sortTTFT sortKey = iota
	sortTPS
	sortTotal
	sortModel
)

func (s sortKey) String() string {
	switch s {
	case sortTTFT:
		return "TTFT"
	case sortTPS:
		return "tok/s"
	case sortTotal:
		return "Total"
	default:
		return "Model"
	}
}

// ListLoadedMsg is delivered when models have been fetched from endpoints.
type ListLoadedMsg struct {
	Models []model.Model
	Err    error
}

// ProgressMsg is delivered as each measured benchmark run completes.
type ProgressMsg struct {
	Name     string
	Endpoint string
	RunIndex int
	Result   model.RunResult
}

// RunDoneMsg is delivered when the full benchmark run finishes or is cancelled.
type RunDoneMsg struct {
	Results []benchmark.Result
	Err     error
}

// SessionsLoadedMsg is delivered when past sessions have been fetched.
type SessionsLoadedMsg struct {
	Sessions []store.Session
	Results  [][]store.StoredResult
	Err      error
}

// SessionModelsLoadedMsg is delivered when a session's results are loaded.
type SessionModelsLoadedMsg struct {
	Models []store.StoredResult
	Err    error
}

// listItem wraps a model.Model for display in the list component.
type listItem struct {
	model    model.Model
	selected bool
}

func (i listItem) FilterValue() string { return i.model.Name }
func (i listItem) Title() string {
	if i.selected {
		return "[x] " + i.model.Name
	}
	return "[ ] " + i.model.Name
}
func (i listItem) Description() string { return i.model.Endpoint }

// historySessionItem wraps a store.Session for display in the history list.
type historySessionItem struct {
	session store.Session
	results []store.StoredResult
}

func (i historySessionItem) FilterValue() string {
	return i.session.Timestamp.Format(time.RFC3339) + " " + strings.Join(i.session.ModelsTested, " ")
}

func (i historySessionItem) Title() string {
	bestTTFT, bestTPS := bestSessionMetrics(i.results)
	return fmt.Sprintf("%s  %d models  TTFT:%s  TPS:%.1f",
		i.session.Timestamp.Format("2006-01-02"),
		len(i.session.ModelsTested),
		formatDuration(bestTTFT),
		bestTPS)
}
func (i historySessionItem) Description() string { return "" }

// historyModelItem wraps a store.StoredResult for display in the history model list.
type historyModelItem struct {
	result store.StoredResult
}
func (i historyModelItem) FilterValue() string { return i.result.ModelName + " " + i.result.Endpoint }

func (i historyModelItem) Title() string {
	return fmt.Sprintf("%s (%s)  TTFT:%s  tok/s:%.1f",
		i.result.ModelName,
		i.result.Endpoint,
		formatDuration(i.result.Aggregate.MeanTTFT),
		i.result.Aggregate.MeanTPS)
}

func (i historyModelItem) Description() string { return "" }

// runEvent is the internal channel payload used by the benchmark goroutine.
type runEvent struct {
	Name     string
	Endpoint string
	RunIndex int
	Result   model.RunResult
	Results  []benchmark.Result
	Err      error
}

// keyMap holds application-level key bindings.
type keyMap struct {
	Quit         key.Binding
	EndpointNext key.Binding
	EndpointPrev key.Binding
	Sort         key.Binding
	ToggleSelect key.Binding
	Start        key.Binding
	Rerun        key.Binding
	Back         key.Binding
	History      key.Binding
}

func defaultKeyMap() keyMap {
	return keyMap{
		Quit: key.NewBinding(
			key.WithKeys("q", "ctrl+c"),
			key.WithHelp("q", "quit"),
		),
		EndpointNext: key.NewBinding(
			key.WithKeys("tab"),
			key.WithHelp("tab", "next endpoint"),
		),
		EndpointPrev: key.NewBinding(
			key.WithKeys("shift+tab"),
			key.WithHelp("shift+tab", "prev endpoint"),
		),
		Sort: key.NewBinding(
			key.WithKeys("s"),
			key.WithHelp("s", "sort"),
		),
		ToggleSelect: key.NewBinding(
			key.WithKeys(" ", "space"),
			key.WithHelp("space", "toggle"),
		),
		Start: key.NewBinding(
			key.WithKeys("enter"),
			key.WithHelp("enter", "start"),
		),
		Rerun: key.NewBinding(
			key.WithKeys("r"),
			key.WithHelp("r", "re-run"),
		),
		Back: key.NewBinding(
			key.WithKeys("esc", "backspace"),
			key.WithHelp("esc", "back"),
		),
		History: key.NewBinding(
			key.WithKeys("h"),
			key.WithHelp("h", "history"),
		),
	}
}

// AppModel is the top-level bubbletea model for the omt TUI.
type AppModel struct {
	cfg         model.Config
	localClient model.Client
	cloudClient model.Client

	state    screen
	endpoint endpointMode
	width    int
	height   int

	list          list.Model
	table         table.Model
	viewport      viewport.Model
	spinner       spinner.Model
	progress      progress.Model
	historyList   list.Model

	keys keyMap

	results          []benchmark.Result
	selected         map[string]bool
	runProgress      map[string][]model.RunResult
	runLive          []benchmark.Result
	completed        int
	total            int
	sortKey          sortKey
	detailResult     benchmark.Result
	runCh            chan runEvent
	runCtx           context.Context
	runCancel        context.CancelFunc
	cancelling       bool
	loadingErr       error
	store            store.Store
	sessions         []store.Session
	sessionModels    []store.StoredResult
	historyResults   map[string][]store.StoredResult
	historyFiltering bool
}

// New creates an AppModel ready to be run by tea.NewProgram.
func New(cfg model.Config, local, cloud model.Client, st store.Store) *AppModel {
	width, height := 60, 12

	delegate := list.NewDefaultDelegate()
	delegate.ShowDescription = false
	delegate.SetHeight(1)
	delegate.SetSpacing(0)

	lm := list.New(nil, delegate, width, max(4, height-3))
	lm.SetShowTitle(false)
	lm.SetShowStatusBar(false)
	lm.SetFilteringEnabled(true)
	lm.DisableQuitKeybindings()
	lm.SetShowHelp(false)

	histDelegate := list.NewDefaultDelegate()
	histDelegate.ShowDescription = false
	histDelegate.SetHeight(1)
	histDelegate.SetSpacing(0)

	hm := list.New(nil, histDelegate, width, max(4, height-3))
	hm.SetShowTitle(false)
	hm.SetShowStatusBar(false)
	hm.SetFilteringEnabled(true)
	hm.DisableQuitKeybindings()
	hm.SetShowHelp(false)

	tm := table.New(
		table.WithColumns([]table.Column{
			{Title: "Model", Width: 18},
			{Title: "End", Width: 6},
			{Title: "TTFT", Width: 9},
			{Title: "tok/s", Width: 8},
			{Title: "Total", Width: 8},
			{Title: "Tkn", Width: 5},
			{Title: "OK", Width: 4},
		}),
		table.WithFocused(true),
		table.WithHeight(max(4, height-3)),
		table.WithWidth(width),
	)

	vp := viewport.New(
		viewport.WithWidth(width),
		viewport.WithHeight(max(4, height-2)),
	)

	sm := spinner.New(spinner.WithSpinner(spinner.Dot))
	pm := progress.New(
		progress.WithoutPercentage(),
		progress.WithWidth(width-12),
	)

	return &AppModel{
		cfg:              cfg,
		localClient:      local,
		cloudClient:      cloud,
		store:            st,
		state:            screenSelect,
		endpoint:         endpointBoth,
		width:            width,
		height:           height,
		list:             lm,
		table:            tm,
		viewport:         vp,
		spinner:          sm,
		progress:         pm,
		historyList:      hm,
		keys:             defaultKeyMap(),
		selected:         make(map[string]bool),
		historyResults:   make(map[string][]store.StoredResult),
	}
}

// Init starts by loading the available models.
func (m *AppModel) Init() tea.Cmd {
	return m.loadModelsCmd()
}

// Update handles messages and drives the state machine.
func (m *AppModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
		m.resize()
		return m, nil

	case ListLoadedMsg:
		m.loadingErr = msg.Err
		m.setModels(msg.Models)
		return m, nil

	case ProgressMsg:
		if m.state != screenRunning {
			return m, nil
		}
		k := modelKey(msg.Name, msg.Endpoint)
		m.runProgress[k] = append(m.runProgress[k], msg.Result)
		if msg.RunIndex == m.cfg.Runs-1 {
			m.completed++
			m.runLive = append(m.runLive, benchmark.Result{
				Model:     model.Model{Name: msg.Name, Endpoint: msg.Endpoint},
				Runs:      append([]model.RunResult(nil), m.runProgress[k]...),
				Aggregate: metrics.Aggregate(m.runProgress[k]),
			})
		}
		return m, tea.Batch(m.runCmd(), m.spinnerTickCmd())

	case RunDoneMsg:
		if m.state != screenRunning && !m.cancelling {
			return m, nil
		}
		m.results = msg.Results
		if m.store != nil {
			go func() {
				if err := saveSession(m.store, m.cfg, msg.Results); err != nil {
					// bubbletea has no logger; use stdlib log for warnings.
					log.Printf("warning: failed to save session: %v\n", err)
				}
			}()
		}
		m.cancelling = false
		m.state = screenResults
		m.sortResults()
		m.buildResultsTable()
		m.resize()
		return m, nil

	case SessionsLoadedMsg:
		m.loadingErr = msg.Err
		m.sessions = msg.Sessions
		m.historyResults = make(map[string][]store.StoredResult, len(msg.Sessions))
		for i, s := range msg.Sessions {
			if i < len(msg.Results) {
				m.historyResults[s.ID] = msg.Results[i]
			}
		}
		m.buildHistoryItems()
		m.sessionModels = nil
		m.state = screenHistory
		m.resize()
		return m, nil

	case SessionModelsLoadedMsg:
		m.loadingErr = msg.Err
		m.sessionModels = msg.Models
		m.buildHistoryModelItems()
		m.historyList.SetFilterText("")
		return m, nil

	case tea.KeyPressMsg:
		return m.handleKeyPress(msg)

	case spinner.TickMsg:
		s, cmd := m.spinner.Update(msg)
		m.spinner = s
		return m, cmd

	case progress.FrameMsg:
		p, cmd := m.progress.Update(msg)
		m.progress = p
		return m, cmd

	default:
		return m.delegateUpdate(msg)
	}
}

// View renders the current screen.
func (m *AppModel) View() tea.View {
	var content string
	switch m.state {
	case screenSelect:
		content = m.selectView()
	case screenRunning:
		content = m.runningView()
	case screenResults:
		content = m.resultsView()
	case screenDetail:
		content = m.detailView()
	case screenHistory:
		content = m.historyView()
	}
	view := tea.NewView(content)
	view.AltScreen = true
	return view
}

// --- key handling ---

func (m *AppModel) handleKeyPress(msg tea.KeyPressMsg) (tea.Model, tea.Cmd) {
	switch m.state {
	case screenSelect:
		return m.handleSelectKey(msg)
	case screenRunning:
		return m.handleRunningKey(msg)
	case screenResults:
		return m.handleResultsKey(msg)
	case screenDetail:
		return m.handleDetailKey(msg)
	case screenHistory:
		return m.handleHistoryKey(msg)
	}
	return m, nil
}

func (m *AppModel) handleSelectKey(msg tea.KeyPressMsg) (tea.Model, tea.Cmd) {
	// Let the list component handle filtering and cursor movement unmodified.
	if m.list.SettingFilter() {
		return m.delegateUpdate(msg)
	}

	switch {
	case key.Matches(msg, m.keys.Quit):
		return m, tea.Quit

	case key.Matches(msg, m.keys.EndpointNext):
		m.cycleEndpoint(1)
		return m, m.loadModelsCmd()

	case key.Matches(msg, m.keys.EndpointPrev):
		m.cycleEndpoint(-1)
		return m, m.loadModelsCmd()

	case key.Matches(msg, m.keys.ToggleSelect):
		m.toggleSelection()
		return m, nil

	case key.Matches(msg, m.keys.Start):
		models := m.selectedModels()
		if len(models) == 0 {
			return m, nil
		}
		m.startBenchmark(models)
		m.state = screenRunning
		m.resize()
		return m, tea.Batch(m.runCmd(), m.spinnerTickCmd())

	case key.Matches(msg, m.keys.History):
		if m.store == nil {
			return m, nil
		}
		return m, m.loadSessionsCmd()
	}

	return m.delegateUpdate(msg)
}

func (m *AppModel) handleRunningKey(msg tea.KeyPressMsg) (tea.Model, tea.Cmd) {
	if key.Matches(msg, m.keys.Back) {
		m.cancelling = true
		if m.runCancel != nil {
			m.runCancel()
		}
		return m, nil
	}
	return m, nil
}

func (m *AppModel) handleResultsKey(msg tea.KeyPressMsg) (tea.Model, tea.Cmd) {
	// Let the table handle navigation first.
	t, cmd := m.table.Update(msg)
	m.table = t

	switch {
	case key.Matches(msg, m.keys.Quit):
		return m, tea.Quit

	case key.Matches(msg, m.keys.Rerun):
		m.state = screenSelect
		m.results = nil
		m.selected = make(map[string]bool)
		m.setModels(m.allModels())
		m.resize()
		return m, m.loadModelsCmd()

	case key.Matches(msg, m.keys.Sort):
		m.cycleSort()
		return m, nil

	case key.Matches(msg, m.keys.Start):
		idx := m.table.Cursor()
		if idx < 0 || idx >= len(m.results) {
			return m, cmd
		}
		m.detailResult = m.results[idx]
		m.buildDetailView()
		m.state = screenDetail
		m.resize()
		return m, nil
	}

	return m, cmd
}

func (m *AppModel) handleDetailKey(msg tea.KeyPressMsg) (tea.Model, tea.Cmd) {
	v, cmd := m.viewport.Update(msg)
	m.viewport = v

	if key.Matches(msg, m.keys.Back) {
		m.state = screenResults
		m.resize()
		return m, nil
	}
	return m, cmd
}

func (m *AppModel) handleHistoryKey(msg tea.KeyPressMsg) (tea.Model, tea.Cmd) {
	if m.historyList.SettingFilter() {
		m.historyFiltering = true
		return m.delegateUpdate(msg)
	}

	switch {
	case key.Matches(msg, m.keys.Quit):
		return m, tea.Quit
	}

	if m.sessionModels != nil {
		switch {
		case key.Matches(msg, m.keys.Back):
			m.sessionModels = nil
			m.buildHistoryItems()
			m.historyList.SetFilterText("")
			m.historyFiltering = false
			return m, nil

		case key.Matches(msg, m.keys.Start):
			item := m.historyList.SelectedItem()
			if item == nil {
				return m, nil
			}
			mi, ok := item.(historyModelItem)
			if !ok {
				return m, nil
			}
			sr := mi.result
			m.detailResult = benchmark.Result{
				Model:     model.Model{Name: sr.ModelName, Endpoint: sr.Endpoint},
				Runs:      sr.Runs,
				Aggregate: sr.Aggregate,
			}
			m.buildDetailView()
			m.state = screenDetail
			m.resize()
			return m, nil
		}
		return m.delegateUpdate(msg)
	}

	switch {
	case key.Matches(msg, m.keys.Back):
		m.state = screenSelect
		m.resize()
		return m, nil

	case key.Matches(msg, m.keys.Start):
		item := m.historyList.SelectedItem()
		if item == nil {
			return m, nil
		}
		si, ok := item.(historySessionItem)
		if !ok {
			return m, nil
		}
		return m, m.loadSessionModelsCmd(si.session.ID)

	case msg.String() == "/":
		m.historyFiltering = true
		return m.delegateUpdate(msg)
	}

	return m.delegateUpdate(msg)
}

func (m *AppModel) delegateUpdate(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch m.state {
	case screenSelect:
		l, cmd := m.list.Update(msg)
		m.list = l
		return m, cmd
	case screenResults:
		t, cmd := m.table.Update(msg)
		m.table = t
		return m, cmd
	case screenDetail:
		v, cmd := m.viewport.Update(msg)
		m.viewport = v
		return m, cmd
	case screenHistory:
		l, cmd := m.historyList.Update(msg)
		m.historyList = l
		m.historyFiltering = m.historyList.SettingFilter()
		return m, cmd
	}
	return m, nil
}

// --- commands ---

func (m *AppModel) loadModelsCmd() tea.Cmd {
	return func() tea.Msg {
		ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		defer cancel()

		var mu sync.Mutex
		var out []model.Model
		var errs []error
		var wg sync.WaitGroup

		load := func(client model.Client) {
			defer wg.Done()
			if client == nil {
				return
			}
			models, err := client.ListModels(ctx)
			mu.Lock()
			defer mu.Unlock()
			if err != nil {
				errs = append(errs, err)
				return
			}
			out = append(out, models...)
		}

		switch m.endpoint {
		case endpointLocal:
			wg.Add(1)
			go load(m.localClient)
		case endpointCloud:
			wg.Add(1)
			go load(m.cloudClient)
		default:
			wg.Add(2)
			go load(m.localClient)
			go load(m.cloudClient)
		}
		wg.Wait()

		var err error
		if len(errs) > 0 {
			err = errs[0]
		}
		return ListLoadedMsg{Models: out, Err: err}
	}
}

func (m *AppModel) loadSessionsCmd() tea.Cmd {
	return func() tea.Msg {
		sessions, err := m.store.ListSessions(50, 0)
		if err != nil {
			return SessionsLoadedMsg{Err: err}
		}
		results := make([][]store.StoredResult, 0, len(sessions))
		for _, s := range sessions {
			_, res, err := m.store.GetSession(s.ID)
			if err != nil {
				res = nil
			}
			results = append(results, res)
		}
		return SessionsLoadedMsg{Sessions: sessions, Results: results, Err: nil}
	}
}

func (m *AppModel) loadSessionModelsCmd(id string) tea.Cmd {
	return func() tea.Msg {
		_, models, err := m.store.GetSession(id)
		return SessionModelsLoadedMsg{Models: models, Err: err}
	}
}

func (m *AppModel) startBenchmark(models []model.Model) {
	m.runCh = make(chan runEvent)
	m.runProgress = make(map[string][]model.RunResult)
	m.runLive = nil
	m.completed = 0
	m.total = len(models)
	m.results = nil
	m.cancelling = false
	m.runCtx, m.runCancel = context.WithCancel(context.Background())

	go func() {
		defer close(m.runCh)
		var all []benchmark.Result

		runGroup := func(client model.Client, endpoint string, group []model.Model) {
			if len(group) == 0 || client == nil {
				return
			}
			r := &benchmark.Runner{
				Client: client,
				Config: m.cfg,
				Progress: func(name string, runIndex int, result model.RunResult) {
					m.runCh <- runEvent{
						Name:     name,
						Endpoint: endpoint,
						RunIndex: runIndex,
						Result:   result,
					}
				},
			}
			res, _ := r.Run(m.runCtx, group)
			all = append(all, res...)
		}

		switch m.endpoint {
		case endpointLocal:
			runGroup(m.localClient, "local", models)
		case endpointCloud:
			runGroup(m.cloudClient, "cloud", models)
		default:
			var localModels, cloudModels []model.Model
			for _, mo := range models {
				if mo.Endpoint == "cloud" {
					cloudModels = append(cloudModels, mo)
				} else {
					localModels = append(localModels, mo)
				}
			}
			runGroup(m.localClient, "local", localModels)
			runGroup(m.cloudClient, "cloud", cloudModels)
		}

		m.runCh <- runEvent{Results: all}
	}()
}

func (m *AppModel) runCmd() tea.Cmd {
	if m.runCh == nil {
		return nil
	}
	return func() tea.Msg {
		evt, ok := <-m.runCh
		if !ok {
			return nil
		}
		if evt.Results != nil || evt.Err != nil {
			return RunDoneMsg{Results: evt.Results, Err: evt.Err}
		}
		return ProgressMsg{
			Name:     evt.Name,
			Endpoint: evt.Endpoint,
			RunIndex: evt.RunIndex,
			Result:   evt.Result,
		}
	}
}

func (m *AppModel) spinnerTickCmd() tea.Cmd {
	return func() tea.Msg {
		return m.spinner.Tick()
	}
}

// --- state helpers ---

func (m *AppModel) cycleEndpoint(delta int) {
	m.endpoint = endpointMode((int(m.endpoint) + delta + 3) % 3)
}

func (m *AppModel) toggleSelection() {
	item, ok := m.list.SelectedItem().(listItem)
	if !ok {
		return
	}
	item.selected = !item.selected
	m.selected[modelKey(item.model.Name, item.model.Endpoint)] = item.selected
	idx := m.list.GlobalIndex()
	if idx >= 0 {
		m.list.SetItem(idx, item)
	}
}

func (m *AppModel) selectedModels() []model.Model {
	var out []model.Model
	for _, it := range m.list.Items() {
		item, ok := it.(listItem)
		if !ok {
			continue
		}
		if m.selected[modelKey(item.model.Name, item.model.Endpoint)] {
			out = append(out, item.model)
		}
	}
	return out
}

func (m *AppModel) allModels() []model.Model {
	var out []model.Model
	for _, it := range m.list.Items() {
		item, ok := it.(listItem)
		if ok {
			out = append(out, item.model)
		}
	}
	return out
}

func (m *AppModel) setModels(models []model.Model) {
	items := make([]list.Item, 0, len(models))
	for _, mo := range models {
		items = append(items, listItem{model: mo})
	}
	m.list.SetItems(items)
	m.list.ResetSelected()
	m.list.ResetFilter()
	m.resize()
}

func (m *AppModel) cycleSort() {
	m.sortKey = sortKey((int(m.sortKey) + 1) % 4)
	m.sortResults()
	m.buildResultsTable()
}

func (m *AppModel) sortResults() {
	switch m.sortKey {
	case sortTTFT:
		sort.SliceStable(m.results, func(i, j int) bool {
			return m.results[i].Aggregate.MeanTTFT < m.results[j].Aggregate.MeanTTFT
		})
	case sortTPS:
		sort.SliceStable(m.results, func(i, j int) bool {
			return m.results[i].Aggregate.MeanTPS > m.results[j].Aggregate.MeanTPS
		})
	case sortTotal:
		sort.SliceStable(m.results, func(i, j int) bool {
			return m.results[i].Aggregate.MeanTotal < m.results[j].Aggregate.MeanTotal
		})
	case sortModel:
		sort.SliceStable(m.results, func(i, j int) bool {
			if m.results[i].Model.Name == m.results[j].Model.Name {
				return m.results[i].Model.Endpoint < m.results[j].Model.Endpoint
			}
			return m.results[i].Model.Name < m.results[j].Model.Name
		})
	}
}

// --- views ---

func (m *AppModel) selectView() string {
	b := strings.Builder{}
	headerStyle := lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("#7D56F4"))
	b.WriteString(headerStyle.Render(fmt.Sprintf("Select models  [%s]  tab=endpoint  space=toggle  enter=start", m.endpoint)))
	b.WriteByte('\n')
	b.WriteString(m.list.View())
	b.WriteByte('\n')
	b.WriteString(m.selectFooter())
	return b.String()
}

func (m *AppModel) selectFooter() string {
	selectedCount := 0
	for _, v := range m.selected {
		if v {
			selectedCount++
		}
	}
	errMsg := ""
	if m.loadingErr != nil {
		errMsg = "  err: " + m.loadingErr.Error()
	}
	return fmt.Sprintf("%d models  %d selected  %s%s", len(m.list.Items()), selectedCount, m.endpointSummary(), errMsg)
}

func (m *AppModel) endpointSummary() string {
	switch m.endpoint {
	case endpointLocal:
		return "local endpoint"
	case endpointCloud:
		return "cloud endpoint"
	default:
		return "both endpoints"
	}
}

func (m *AppModel) runningView() string {
	b := strings.Builder{}
	pct := 0.0
	if m.total > 0 {
		pct = float64(m.completed) / float64(m.total)
	}
	b.WriteString(fmt.Sprintf("%s Running %d models  (%d/%d)", m.spinner.View(), m.total, m.completed, m.total))
	b.WriteByte('\n')
	b.WriteString(m.progress.ViewAs(pct))
	b.WriteByte('\n')

	// Show per-run metrics for in-progress models, then completed aggregates.
	remaining := m.height - 4
	if remaining > 0 {
		// Collect in-progress model display lines.
		keys := make([]string, 0, len(m.runProgress))
		for k := range m.runProgress {
			keys = append(keys, k)
		}
		sort.Strings(keys)

		var inProgress []string
		for _, k := range keys {
			runs := m.runProgress[k]
			if len(runs) >= m.cfg.Runs {
				continue
			}
			name, endpoint, _ := strings.Cut(k, "|")
			inProgress = append(inProgress, fmt.Sprintf("%s/%s", name, endpoint))
			for i, r := range runs {
				inProgress = append(inProgress, fmt.Sprintf("  Run %d: TTFT:%s  tok/s:%.1f  Total:%s",
					i+1,
					formatDuration(r.TTFT),
					r.TokensPerSec,
					formatDuration(r.TotalTime)))
			}
		}

		for _, line := range inProgress {
			if remaining <= 0 {
				break
			}
			b.WriteString(line)
			b.WriteByte('\n')
			remaining--
		}

		if remaining > 0 {
			start := max(0, len(m.runLive)-remaining)
			for _, r := range m.runLive[start:] {
				b.WriteString(fmt.Sprintf("%s/%s  TTFT:%s  tok/s:%.1f  Total:%s\n",
					r.Model.Name, r.Model.Endpoint,
					formatDuration(r.Aggregate.MeanTTFT),
					r.Aggregate.MeanTPS,
					formatDuration(r.Aggregate.MeanTotal)))
			}
		}
	}
	b.WriteString("esc=cancel")
	return b.String()
}

func (m *AppModel) resultsView() string {
	b := strings.Builder{}
	b.WriteString(fmt.Sprintf("Results  sort:%s  s=sort enter=detail r=rerun q=quit", m.sortKey))
	b.WriteByte('\n')
	b.WriteString(m.table.View())
	b.WriteByte('\n')
	b.WriteString(fmt.Sprintf("%d models", len(m.results)))
	return b.String()
}

func (m *AppModel) detailView() string {
	b := strings.Builder{}
	b.WriteString(fmt.Sprintf("Detail: %s (%s)  esc/back=results", m.detailResult.Model.Name, m.detailResult.Model.Endpoint))
	b.WriteByte('\n')
	b.WriteString(m.viewport.View())
	return b.String()
}

func (m *AppModel) historyView() string {
	b := strings.Builder{}
	if m.sessionModels != nil {
		b.WriteString(fmt.Sprintf("Session models  esc=back  enter=detail  %d models\n", len(m.sessionModels)))
		b.WriteString(m.historyList.View())
		return b.String()
	}
	b.WriteString("History  esc=back  enter=models  /=filter\n")
	if len(m.sessions) == 0 {
		b.WriteString("No history yet\n")
	} else {
		b.WriteString(m.historyList.View())
		b.WriteByte('\n')
	}
	if m.loadingErr != nil {
		b.WriteString("err: " + m.loadingErr.Error())
	} else {
		b.WriteString(fmt.Sprintf("%d sessions", len(m.sessions)))
	}
	return b.String()
}

func (m *AppModel) buildResultsTable() {
	rows := make([]table.Row, 0, len(m.results))
	for _, r := range m.results {
		ok := "Y"
		if r.Aggregate.SuccessCount == 0 {
			ok = "FAIL"
		}
		tokens := 0
		if r.Aggregate.SuccessCount > 0 {
			tokens = meanTokenCount(r.Runs)
		}
		rows = append(rows, table.Row{
			r.Model.Name,
			r.Model.Endpoint,
			formatDuration(r.Aggregate.MeanTTFT),
			fmt.Sprintf("%.1f", r.Aggregate.MeanTPS),
			formatDuration(r.Aggregate.MeanTotal),
			fmt.Sprintf("%d", tokens),
			ok,
		})
	}
	m.table.SetRows(rows)
	if len(rows) > 0 {
		m.table.SetCursor(0)
	}
}

func (m *AppModel) buildDetailView() {
	r := m.detailResult
	b := strings.Builder{}
	b.WriteString("Aggregate stats\n")
	b.WriteString(fmt.Sprintf("  TTFT  mean:%s  median:%s  min:%s  max:%s\n",
		formatDuration(r.Aggregate.MeanTTFT), formatDuration(r.Aggregate.MedianTTFT),
		formatDuration(r.Aggregate.MinTTFT), formatDuration(r.Aggregate.MaxTTFT)))
	b.WriteString(fmt.Sprintf("  tok/s mean:%.1f  median:%.1f  min:%.1f  max:%.1f\n",
		r.Aggregate.MeanTPS, r.Aggregate.MedianTPS, r.Aggregate.MinTPS, r.Aggregate.MaxTPS))
	b.WriteString(fmt.Sprintf("  Total mean:%s  median:%s  min:%s  max:%s\n",
		formatDuration(r.Aggregate.MeanTotal), formatDuration(r.Aggregate.MedianTotal),
		formatDuration(r.Aggregate.MinTotal), formatDuration(r.Aggregate.MaxTotal)))
	b.WriteString(fmt.Sprintf("  OK: %d/%d runs\n\n", r.Aggregate.SuccessCount, r.Aggregate.SuccessCount+r.Aggregate.FailCount))

	b.WriteString("Per-run breakdown\n")
	for i, run := range r.Runs {
		status := "OK"
		if run.Error != "" {
			status = "ERR: " + run.Error
		}
		b.WriteString(fmt.Sprintf("  run %d  TTFT:%s  tok/s:%.1f  Total:%s  tokens:%d  %s\n",
			i+1, formatDuration(run.TTFT), run.TokensPerSec,
			formatDuration(run.TotalTime), run.TokenCount, status))
	}
	m.viewport.SetContent(b.String())
	m.viewport.GotoTop()
}

func (m *AppModel) buildHistoryItems() {
	items := make([]list.Item, 0, len(m.sessions))
	for _, s := range m.sessions {
		items = append(items, historySessionItem{session: s, results: m.historyResults[s.ID]})
	}
	m.historyList.SetItems(items)
}

func (m *AppModel) buildHistoryModelItems() {
	items := make([]list.Item, 0, len(m.sessionModels))
	for _, r := range m.sessionModels {
		items = append(items, historyModelItem{result: r})
	}
	m.historyList.SetItems(items)
}

// --- sizing ---

func (m *AppModel) resize() {
	w := max(60, m.width)
	h := max(12, m.height)

	switch m.state {
	case screenSelect:
		m.list.SetSize(w, max(4, h-3))
	case screenResults:
		m.table.SetWidth(w)
		m.table.SetHeight(max(4, h-3))
	case screenDetail:
		m.viewport.SetWidth(w)
		m.viewport.SetHeight(max(4, h-2))
	case screenRunning:
		m.progress.SetWidth(w - 12)
	case screenHistory:
		m.historyList.SetSize(w, max(4, h-3))
	}
}

// --- helpers ---

func modelKey(name, endpoint string) string {
	return name + "|" + endpoint
}

func formatDuration(d time.Duration) string {
	if d == 0 {
		return "0s"
	}
	if d < time.Millisecond {
		return d.Round(time.Microsecond).String()
	}
	return d.Round(time.Millisecond).String()
}

func meanTokenCount(runs []model.RunResult) int {
	var sum int
	var n int
	for _, r := range runs {
		if r.Error == "" {
			sum += r.TokenCount
			n++
		}
	}
	if n == 0 {
		return 0
	}
	return sum / n
}

func bestSessionMetrics(results []store.StoredResult) (time.Duration, float64) {
	var bestTTFT time.Duration
	var bestTPS float64
	for _, r := range results {
		if r.Aggregate.SuccessCount == 0 {
			continue
		}
		if bestTTFT == 0 || r.Aggregate.MeanTTFT < bestTTFT {
			bestTTFT = r.Aggregate.MeanTTFT
		}
		if r.Aggregate.MeanTPS > bestTPS {
			bestTPS = r.Aggregate.MeanTPS
		}
	}
	return bestTTFT, bestTPS
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// saveSession persists the completed benchmark results to the store.
func saveSession(st store.Store, cfg model.Config, results []benchmark.Result) error {
	session := store.Session{
		ID:            newSessionID(),
		Timestamp:     time.Now(),
		ModelsTested:  make([]string, 0, len(results)),
		Prompt:        cfg.Prompt,
		ConfigSummary: fmt.Sprintf("runs=%d,warmup=%d,max_tokens=%d,timeout=%s,parallel=%d", cfg.Runs, cfg.Warmup, cfg.MaxTokens, cfg.Timeout, cfg.Parallel),
	}
	for _, r := range results {
		session.ModelsTested = append(session.ModelsTested, r.Model.Name)
	}

	stored := make([]store.StoredResult, 0, len(results))
	for _, r := range results {
		stored = append(stored, store.StoredResult{
			ModelName: r.Model.Name,
			Endpoint:  r.Model.Endpoint,
			Aggregate: r.Aggregate,
			Runs:      r.Runs,
		})
	}

	return st.SaveSession(session, stored)
}

// newSessionID returns a UUID-like string using crypto/rand.
func newSessionID() string {
	b := make([]byte, 16)
	if _, err := io.ReadFull(rand.Reader, b); err != nil {
		// Fallback to a timestamp-based ID.
		return fmt.Sprintf("%d", time.Now().UnixNano())
	}
	b[6] = (b[6] & 0x0f) | 0x40 // version 4
	b[8] = (b[8] & 0x3f) | 0x80 // variant RFC 4122
	return fmt.Sprintf("%x-%x-%x-%x-%x", b[0:4], b[4:6], b[6:8], b[8:10], b[10:16])
}
