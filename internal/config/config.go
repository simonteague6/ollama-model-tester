// Package config loads and merges omt configuration from CLI flags, environment
// variables, and a TOML config file, using precedence:
//
//	CLI flags > environment variables > TOML file > built-in defaults
package config

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/BurntSushi/toml"
	"github.com/simonteague6/ollama-model-tester/internal/model"
)

// Default constants used when no higher-precedence source provides a value.
const (
	DefaultLocalURL  = "http://localhost:11434"
	DefaultCloudURL  = "https://ollama.com/api"
	DefaultRuns      = 5
	DefaultWarmup    = 1
	DefaultPrompt    = "Tell me a short story about a robot learning to paint."
	DefaultMaxTokens = 256
	DefaultTimeout   = 60 * time.Second
	DefaultParallel  = 1
	DefaultSortKey   = "ttft"
)

// Default returns the built-in default configuration.
func Default() model.Config {
	return model.Config{
		LocalURL:  DefaultLocalURL,
		CloudURL:  DefaultCloudURL,
		Runs:      DefaultRuns,
		Warmup:    DefaultWarmup,
		Prompt:    DefaultPrompt,
		MaxTokens: DefaultMaxTokens,
		Timeout:   DefaultTimeout,
		Parallel:  DefaultParallel,
		SortKey:   DefaultSortKey,
	}
}

// Sources collects all configuration sources for merging. A nil Env map is
// treated as the current process environment.
type Sources struct {
	// CLI holds values parsed from CLI flags.
	CLI model.Config
	// CLISet records which CLI flags were explicitly set. Keys are the
	// kebab-case flag names (api-key, local-url, cloud-url, runs, warmup,
	// prompt, max-tokens, timeout, parallel, sort-key).
	CLISet map[string]bool
	// Env holds environment variables. If nil, Load reads from the current
	// process environment.
	Env map[string]string
	// FilePath overrides the config file path. If empty, the OMT_CONFIG
	// environment variable or ~/.omt/config.toml is used.
	FilePath string
}

// fileConfig is the on-disk TOML representation. Duration values are stored
// as strings and parsed with time.ParseDuration so that missing or empty
// values fall back to defaults cleanly.
type fileConfig struct {
	APIKey    string `toml:"api_key"`
	LocalURL  string `toml:"local_url"`
	CloudURL  string `toml:"cloud_url"`
	Runs      int    `toml:"runs"`
	Warmup    int    `toml:"warmup"`
	Prompt    string `toml:"prompt"`
	MaxTokens int    `toml:"max_tokens"`
	Timeout   string `toml:"timeout"`
	Parallel  int    `toml:"parallel"`
	SortKey   string `toml:"sort_key"`
}

// Load merges the provided sources into a single Config using the precedence
// CLI > env > file > defaults. A missing config file is not an error; a
// malformed TOML file is.
func Load(sources Sources) (model.Config, error) {
	cfg := Default()

	path, err := resolvePath(sources)
	if err != nil {
		return model.Config{}, err
	}

	if path != "" {
		if err := loadFile(path, &cfg); err != nil {
			return model.Config{}, err
		}
	}

	env := sources.Env
	if env == nil {
		env = envMap()
	}
	if err := applyEnv(env, &cfg); err != nil {
		return model.Config{}, err
	}

	applyCLI(sources.CLI, sources.CLISet, &cfg)

	return cfg, nil
}

func resolvePath(sources Sources) (string, error) {
	if sources.FilePath != "" {
		return sources.FilePath, nil
	}
	env := sources.Env
	if env == nil {
		env = envMap()
	}
	if p := env["OMT_CONFIG"]; p != "" {
		return p, nil
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(home, ".omt", "config.toml"), nil
}

func loadFile(path string, cfg *model.Config) error {
	if _, err := os.Stat(path); os.IsNotExist(err) {
		return nil
	} else if err != nil {
		return fmt.Errorf("read config file %s: %w", path, err)
	}

	var fc fileConfig
	if _, err := toml.DecodeFile(path, &fc); err != nil {
		return fmt.Errorf("parse config file %s: %w", path, err)
	}

	if fc.APIKey != "" {
		cfg.APIKey = fc.APIKey
	}
	if fc.LocalURL != "" {
		cfg.LocalURL = fc.LocalURL
	}
	if fc.CloudURL != "" {
		cfg.CloudURL = fc.CloudURL
	}
	if fc.Runs != 0 {
		cfg.Runs = fc.Runs
	}
	if fc.Warmup != 0 {
		cfg.Warmup = fc.Warmup
	}
	if fc.Prompt != "" {
		cfg.Prompt = fc.Prompt
	}
	if fc.MaxTokens != 0 {
		cfg.MaxTokens = fc.MaxTokens
	}
	if fc.Timeout != "" {
		d, err := time.ParseDuration(fc.Timeout)
		if err != nil {
			return fmt.Errorf("parse config file %s: invalid timeout %q: %w", path, fc.Timeout, err)
		}
		cfg.Timeout = d
	}
	if fc.Parallel != 0 {
		cfg.Parallel = fc.Parallel
	}
	if fc.SortKey != "" {
		cfg.SortKey = fc.SortKey
	}
	return nil
}

func applyEnv(env map[string]string, cfg *model.Config) error {
	if v := env["OLLAMA_API_KEY"]; v != "" {
		cfg.APIKey = v
	}
	if v := env["OLLAMA_LOCAL_URL"]; v != "" {
		cfg.LocalURL = v
	}
	if v := env["OLLAMA_CLOUD_URL"]; v != "" {
		cfg.CloudURL = v
	}
	if v := env["OLLAMA_RUNS"]; v != "" {
		n, err := strconv.Atoi(v)
		if err != nil {
			return fmt.Errorf("invalid OLLAMA_RUNS value %q: %w", v, err)
		}
		cfg.Runs = n
	}
	if v := env["OLLAMA_TIMEOUT"]; v != "" {
		d, err := time.ParseDuration(v)
		if err != nil {
			return fmt.Errorf("invalid OLLAMA_TIMEOUT value %q: %w", v, err)
		}
		cfg.Timeout = d
	}
	return nil
}

func applyCLI(cli model.Config, set map[string]bool, cfg *model.Config) {
	if set == nil {
		return
	}
	if set["api-key"] {
		cfg.APIKey = cli.APIKey
	}
	if set["local-url"] {
		cfg.LocalURL = cli.LocalURL
	}
	if set["cloud-url"] {
		cfg.CloudURL = cli.CloudURL
	}
	if set["runs"] {
		cfg.Runs = cli.Runs
	}
	if set["warmup"] {
		cfg.Warmup = cli.Warmup
	}
	if set["prompt"] {
		cfg.Prompt = cli.Prompt
	}
	if set["max-tokens"] {
		cfg.MaxTokens = cli.MaxTokens
	}
	if set["timeout"] {
		cfg.Timeout = cli.Timeout
	}
	if set["parallel"] {
		cfg.Parallel = cli.Parallel
	}
	if set["sort-key"] {
		cfg.SortKey = cli.SortKey
	}
}

func envMap() map[string]string {
	env := os.Environ()
	m := make(map[string]string, len(env))
	for _, e := range env {
		if i := strings.Index(e, "="); i >= 0 {
			m[e[:i]] = e[i+1:]
		}
	}
	return m
}

// Redact returns a copy of cfg with the API key replaced by "[REDACTED]".
// Use this before logging or printing a configuration.
func Redact(cfg model.Config) model.Config {
	cfg.APIKey = "[REDACTED]"
	return cfg
}

// String returns a redacted string representation of cfg suitable for logging.
func String(cfg model.Config) string {
	r := Redact(cfg)
	return fmt.Sprintf("Config{APIKey:%s LocalURL:%s CloudURL:%s Runs:%d Warmup:%d Prompt:%q MaxTokens:%d Timeout:%v Parallel:%d SortKey:%s}",
		r.APIKey, r.LocalURL, r.CloudURL, r.Runs, r.Warmup, r.Prompt, r.MaxTokens, r.Timeout, r.Parallel, r.SortKey)
}
