// Package cli provides the Cobra command tree for the omt binary.
package cli

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/signal"
	"sort"
	"strconv"
	"strings"
	"text/tabwriter"
	"time"

	tea "charm.land/bubbletea/v2"
	"github.com/spf13/cobra"
	"github.com/simonteague6/ollama-model-tester/internal/benchmark"
	"github.com/simonteague6/ollama-model-tester/internal/config"
	"github.com/simonteague6/ollama-model-tester/internal/model"
	"github.com/simonteague6/ollama-model-tester/internal/ollama"
	"github.com/simonteague6/ollama-model-tester/internal/tui"
)

// Dependencies are overridable for testing so the CLI can be exercised without
// real network calls.
var (
	newLocalClient = ollama.NewLocalClient
	newCloudClient = ollama.NewCloudClient
	newRunner      = func(c model.Client, cfg model.Config) runner {
		return &benchmark.Runner{Client: c, Config: cfg}
	}
	runTUIProgram = func(cmd *cobra.Command, cfg *model.Config) {
		local := newLocalClient(cfg.LocalURL)
		cloud := newCloudClient(cfg.CloudURL, cfg.APIKey)
		m := tui.New(*cfg, local, cloud)
		p := tea.NewProgram(m, tea.WithInput(cmd.InOrStdin()), tea.WithOutput(cmd.OutOrStdout()))
		if _, err := p.Run(); err != nil {
			fmt.Fprintf(cmd.ErrOrStderr(), "tui error: %v\n", err)
		}
	}
)

// runner is the subset of benchmark.Runner used by the CLI.
type runner interface {
	Run(ctx context.Context, models []model.Model) ([]benchmark.Result, error)
}

// BuildRootCmd constructs the root cobra command with persistent flags and the
// list, run and tui subcommands. cfg is mutated by Changed() flag guards during
// command execution so that unset flags never clobber values from env/file.
func BuildRootCmd(cfg *model.Config) *cobra.Command {
	root := &cobra.Command{
		Use:   "omt",
		Run: func(cmd *cobra.Command, args []string) {
			runTUI(cmd, cfg)
		},
	}

	root.PersistentFlags().Bool("local", false, "Use the local Ollama endpoint only")
	root.PersistentFlags().Bool("cloud", false, "Use the cloud Ollama endpoint only")
	root.PersistentFlags().Bool("both", false, "Use both local and cloud endpoints (default when neither --local nor --cloud is set)")
	root.PersistentFlags().String("config", "", "Path to a TOML config file")

	root.AddCommand(buildListCmd(cfg))
	root.AddCommand(buildRunCmd(cfg))
	root.AddCommand(buildTuiCmd(cfg))

	root.PersistentPreRunE = func(cmd *cobra.Command, args []string) error {
		return loadAndApplyConfig(cmd, cfg)
	}

	return root
}

func buildListCmd(cfg *model.Config) *cobra.Command {
	return &cobra.Command{
		Use:   "list",
		Short: "List available models from selected endpoints",
		RunE: func(cmd *cobra.Command, args []string) error {
			endpoints := selectedEndpoints(cmd)
			if err := checkCloudAPIKey(endpoints, cfg.APIKey); err != nil {
				return err
			}

			ctx, stop := signal.NotifyContext(cmd.Context(), os.Interrupt)
			defer stop()

			models, err := discoverModels(ctx, endpoints, cfg)
			if err != nil {
				return err
			}
			return writeModelList(cmd.OutOrStdout(), models)
		},
	}
}

func buildRunCmd(cfg *model.Config) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "run [models...]",
		Short: "Run benchmarks against the specified models",
		RunE: func(cmd *cobra.Command, args []string) error {
			if len(args) == 0 {
				return errors.New("at least one model name is required")
			}

			if err := applyRunFlags(cmd, cfg); err != nil {
				return err
			}

			endpoints := selectedEndpoints(cmd)
			if err := checkCloudAPIKey(endpoints, cfg.APIKey); err != nil {
				return err
			}

			models := selectedModels(args, endpoints)

			ctx, stop := signal.NotifyContext(cmd.Context(), os.Interrupt)
			defer stop()

			results, err := runBenchmarks(ctx, endpoints, models, *cfg)
			if err != nil && !errors.Is(err, context.Canceled) {
				return err
			}

			jsonOut, _ := cmd.Flags().GetBool("json")
			sortKey, _ := cmd.Flags().GetString("sort")
			if sortKey == "" {
				sortKey = cfg.SortKey
			}

			if jsonOut {
				return writeJSON(cmd.OutOrStdout(), results)
			}
			return writeResultsTable(cmd.OutOrStdout(), results, sortKey)
		},
	}

	cmd.Flags().Int("runs", cfg.Runs, "Number of measured runs per model")
	cmd.Flags().Int("warmup", cfg.Warmup, "Number of warmup runs to discard")
	cmd.Flags().String("prompt", cfg.Prompt, "Prompt text sent to each model")
	cmd.Flags().Int("max-tokens", cfg.MaxTokens, "Maximum tokens to generate")
	cmd.Flags().Duration("timeout", cfg.Timeout, "Per-generation timeout")
	cmd.Flags().Int("parallel", cfg.Parallel, "Maximum parallel cloud models")
	cmd.Flags().String("sort", cfg.SortKey, "Sort results by: ttft, tok/s, total, model")
	cmd.Flags().Bool("json", false, "Output results as JSON")

	return cmd
}

func buildTuiCmd(cfg *model.Config) *cobra.Command {
	return &cobra.Command{
		Use:   "tui",
		Short: "Launch the interactive TUI",
		Run: func(cmd *cobra.Command, args []string) {
			runTUI(cmd, cfg)
		},
	}
}

// runTUI creates clients from config and launches the bubbletea TUI.
func runTUI(cmd *cobra.Command, cfg *model.Config) {
	runTUIProgram(cmd, cfg)
}

// loadAndApplyConfig reloads configuration from disk/env when --config was set,
// then applies any changed CLI flags to cfg using Changed() guards.
func loadAndApplyConfig(cmd *cobra.Command, cfg *model.Config) error {
	if cmd.Flags().Changed("config") {
		path, err := cmd.Flags().GetString("config")
		if err != nil {
			return err
		}
		loaded, err := config.Load(config.Sources{FilePath: path})
		if err != nil {
			return err
		}
		*cfg = loaded
	}
	return nil
}

func applyRunFlags(cmd *cobra.Command, cfg *model.Config) error {
	fs := cmd.Flags()
	if fs.Changed("runs") {
		v, err := fs.GetInt("runs")
		if err != nil {
			return err
		}
		cfg.Runs = v
	}
	if fs.Changed("warmup") {
		v, err := fs.GetInt("warmup")
		if err != nil {
			return err
		}
		cfg.Warmup = v
	}
	if fs.Changed("prompt") {
		v, err := fs.GetString("prompt")
		if err != nil {
			return err
		}
		cfg.Prompt = v
	}
	if fs.Changed("max-tokens") {
		v, err := fs.GetInt("max-tokens")
		if err != nil {
			return err
		}
		cfg.MaxTokens = v
	}
	if fs.Changed("timeout") {
		v, err := fs.GetDuration("timeout")
		if err != nil {
			return err
		}
		cfg.Timeout = v
	}
	if fs.Changed("parallel") {
		v, err := fs.GetInt("parallel")
		if err != nil {
			return err
		}
		cfg.Parallel = v
	}
	if fs.Changed("sort") {
		v, err := fs.GetString("sort")
		if err != nil {
			return err
		}
		cfg.SortKey = v
	}
	return nil
}

func selectedEndpoints(cmd *cobra.Command) []string {
	local, _ := cmd.Flags().GetBool("local")
	cloud, _ := cmd.Flags().GetBool("cloud")
	both, _ := cmd.Flags().GetBool("both")

	if both || (local && cloud) || (!local && !cloud) {
		return []string{"local", "cloud"}
	}
	if local {
		return []string{"local"}
	}
	return []string{"cloud"}
}

func checkCloudAPIKey(endpoints []string, apiKey string) error {
	for _, e := range endpoints {
		if e == "cloud" && apiKey == "" {
			return errors.New("cloud endpoint selected but OLLAMA_API_KEY is not configured. Set it via:\n  export OLLAMA_API_KEY=\"sk-...\"\n  or add api_key = \"sk-...\" to ~/.omt/config.toml")
		}
	}
	return nil
}

func discoverModels(ctx context.Context, endpoints []string, cfg *model.Config) ([]model.Model, error) {
	var all []model.Model
	for _, endpoint := range endpoints {
		client := clientForEndpoint(endpoint, cfg)
		models, err := client.ListModels(ctx)
		if err != nil {
			return nil, fmt.Errorf("list %s models: %w", endpoint, err)
		}
		all = append(all, models...)
	}
	return all, nil
}

func clientForEndpoint(endpoint string, cfg *model.Config) model.Client {
	if endpoint == "cloud" {
		return newCloudClient(cfg.CloudURL, cfg.APIKey)
	}
	return newLocalClient(cfg.LocalURL)
}

func selectedModels(names []string, endpoints []string) []model.Model {
	models := make([]model.Model, 0, len(names)*len(endpoints))
	for _, endpoint := range endpoints {
		for _, name := range names {
			models = append(models, model.Model{
				Name:     name,
				Endpoint: endpoint,
				Provider: endpoint,
			})
		}
	}
	return models
}

func runBenchmarks(ctx context.Context, endpoints []string, models []model.Model, cfg model.Config) ([]benchmark.Result, error) {
	byEndpoint := make(map[string][]model.Model)
	for _, m := range models {
		byEndpoint[m.Endpoint] = append(byEndpoint[m.Endpoint], m)
	}

	var all []benchmark.Result
	for _, endpoint := range endpoints {
		group := byEndpoint[endpoint]
		if len(group) == 0 {
			continue
		}
		client := clientForEndpoint(endpoint, &cfg)
		r := newRunner(client, cfg)
		results, err := r.Run(ctx, group)
		if err != nil && !errors.Is(err, context.Canceled) {
			return all, fmt.Errorf("benchmark %s: %w", endpoint, err)
		}
		all = append(all, results...)
		if errors.Is(err, context.Canceled) {
			return all, err
		}
	}
	return all, nil
}

func writeModelList(w io.Writer, models []model.Model) error {
	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "MODEL\tENDPOINT")
	for _, m := range models {
		fmt.Fprintf(tw, "%s\t%s\n", m.Name, m.Endpoint)
	}
	return tw.Flush()
}

type jsonRow struct {
	Model         string  `json:"model"`
	Endpoint      string  `json:"endpoint"`
	TTFTMs        int64   `json:"ttft_ms"`
	TokensPerSec  float64 `json:"tokens_per_sec"`
	TotalMs       int64   `json:"total_ms"`
	Tokens        int     `json:"tokens"`
	OK            string  `json:"ok"`
	MeanTTFTMs    int64   `json:"mean_ttft_ms"`
	MedianTTFTMs  int64   `json:"median_ttft_ms"`
	MinTTFTMs     int64   `json:"min_ttft_ms"`
	MaxTTFTMs     int64   `json:"max_ttft_ms"`
	MeanTPS       float64 `json:"mean_tokens_per_sec"`
	MedianTPS     float64 `json:"median_tokens_per_sec"`
	MinTPS        float64 `json:"min_tokens_per_sec"`
	MaxTPS        float64 `json:"max_tokens_per_sec"`
	MeanTotalMs   int64   `json:"mean_total_ms"`
	MedianTotalMs int64   `json:"median_total_ms"`
	MinTotalMs    int64   `json:"min_total_ms"`
	MaxTotalMs    int64   `json:"max_total_ms"`
	SuccessCount  int     `json:"success_count"`
	FailCount     int     `json:"fail_count"`
}

func writeJSON(w io.Writer, results []benchmark.Result) error {
	rows := make([]jsonRow, 0, len(results))
	for _, r := range results {
		rows = append(rows, resultToJSON(r))
	}
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(rows)
}

func resultToJSON(r benchmark.Result) jsonRow {
	a := r.Aggregate
	totalRuns := a.SuccessCount + a.FailCount
	return jsonRow{
		Model:         r.Model.Name,
		Endpoint:      r.Model.Endpoint,
		TTFTMs:        a.MeanTTFT.Milliseconds(),
		TokensPerSec:  a.MeanTPS,
		TotalMs:       a.MeanTotal.Milliseconds(),
		Tokens:        aggregateTokens(r.Runs),
		OK:            fmt.Sprintf("%d/%d", a.SuccessCount, totalRuns),
		MeanTTFTMs:    a.MeanTTFT.Milliseconds(),
		MedianTTFTMs:  a.MedianTTFT.Milliseconds(),
		MinTTFTMs:     a.MinTTFT.Milliseconds(),
		MaxTTFTMs:     a.MaxTTFT.Milliseconds(),
		MeanTPS:       a.MeanTPS,
		MedianTPS:     a.MedianTPS,
		MinTPS:        a.MinTPS,
		MaxTPS:        a.MaxTPS,
		MeanTotalMs:   a.MeanTotal.Milliseconds(),
		MedianTotalMs: a.MedianTotal.Milliseconds(),
		MinTotalMs:    a.MinTotal.Milliseconds(),
		MaxTotalMs:    a.MaxTotal.Milliseconds(),
		SuccessCount:  a.SuccessCount,
		FailCount:     a.FailCount,
	}
}

func aggregateTokens(runs []model.RunResult) int {
	tokens := 0
	for _, run := range runs {
		if run.Error == "" {
			tokens += run.TokenCount
		}
	}
	return tokens
}

func writeResultsTable(w io.Writer, results []benchmark.Result, sortKey string) error {
	sortResults(results, sortKey)

	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "MODEL\tENDPOINT\tTTFT\tTOK/S\tTOTAL\tTOKENS\tOK")
	for _, r := range results {
		a := r.Aggregate
		totalRuns := a.SuccessCount + a.FailCount
		ok := fmt.Sprintf("%d/%d", a.SuccessCount, totalRuns)
		if a.SuccessCount == 0 {
			ok = "FAIL"
		}
		fmt.Fprintf(tw, "%s\t%s\t%s\t%.2f\t%s\t%d\t%s\n",
			r.Model.Name,
			r.Model.Endpoint,
			formatDuration(a.MedianTTFT),
			a.MedianTPS,
			formatDuration(a.MedianTotal),
			aggregateTokens(r.Runs),
			ok,
		)
	}
	return tw.Flush()
}

func sortResults(results []benchmark.Result, sortKey string) {
	less := func(i, j int) bool {
		return results[i].Model.Name < results[j].Model.Name
	}
	switch strings.ToLower(sortKey) {
	case "ttft":
		less = func(i, j int) bool {
			return results[i].Aggregate.MedianTTFT > results[j].Aggregate.MedianTTFT
		}
	case "tok/s", "tok_s", "toks":
		less = func(i, j int) bool {
			return results[i].Aggregate.MedianTPS > results[j].Aggregate.MedianTPS
		}
	case "total":
		less = func(i, j int) bool {
			return results[i].Aggregate.MedianTotal > results[j].Aggregate.MedianTotal
		}
	case "model":
		less = func(i, j int) bool {
			return results[i].Model.Name > results[j].Model.Name
		}
	}
	sort.Slice(results, less)
}

func formatDuration(d time.Duration) string {
	if d < time.Second {
		return strconv.FormatInt(d.Milliseconds(), 10) + "ms"
	}
	return fmt.Sprintf("%.2fs", d.Seconds())
}
