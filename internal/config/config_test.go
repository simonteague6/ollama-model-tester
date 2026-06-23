package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/simonteague6/ollama-model-tester/internal/model"
)

func writeFile(t *testing.T, dir, name, content string) string {
	t.Helper()
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("write test config: %v", err)
	}
	return path
}

func TestLoad(t *testing.T) {
	fileContent := `
api_key = "file-key"
local_url = "http://file.local"
cloud_url = "http://file.cloud"
runs = 3
warmup = 2
prompt = "file prompt"
max_tokens = 100
timeout = "30s"
parallel = 4
sort_key = "tps"
`


	tests := []struct {
		name    string
		sources Sources
		want    model.Config
		wantErr bool
		errSub  string
	}{
		{
			name: "defaults only",
			sources: Sources{
				FilePath: filepath.Join(t.TempDir(), "does-not-exist.toml"),
				Env:      map[string]string{},
			},
			want: Default(),
		},
		{
			name: "config file overrides defaults",
			sources: Sources{
				FilePath: writeFile(t, t.TempDir(), "config.toml", fileContent),
				Env:      map[string]string{},
			},
			want: model.Config{
				APIKey:    "file-key",
				LocalURL:  "http://file.local",
				CloudURL:  "http://file.cloud",
				Runs:      3,
				Warmup:    2,
				Prompt:    "file prompt",
				MaxTokens: 100,
				Timeout:   30 * time.Second,
				Parallel:  4,
				SortKey:   "tps",
			},
		},
		{
			name: "env overrides file",
			sources: Sources{
				FilePath: writeFile(t, t.TempDir(), "config.toml", fileContent),
				Env: map[string]string{
					"OLLAMA_API_KEY":    "env-key",
					"OLLAMA_LOCAL_URL":  "http://env.local",
					"OLLAMA_CLOUD_URL":  "http://env.cloud",
					"OLLAMA_RUNS":       "7",
					"OLLAMA_TIMEOUT":    "45s",
				},
			},
			want: model.Config{
				APIKey:    "env-key",
				LocalURL:  "http://env.local",
				CloudURL:  "http://env.cloud",
				Runs:      7,
				Warmup:    2,
				Prompt:    "file prompt",
				MaxTokens: 100,
				Timeout:   45 * time.Second,
				Parallel:  4,
				SortKey:   "tps",
			},
		},
		{
			name: "cli overrides env",
			sources: Sources{
				FilePath: writeFile(t, t.TempDir(), "config.toml", fileContent),
				Env: map[string]string{
					"OLLAMA_API_KEY":    "env-key",
					"OLLAMA_LOCAL_URL":  "http://env.local",
					"OLLAMA_CLOUD_URL":  "http://env.cloud",
					"OLLAMA_RUNS":       "7",
					"OLLAMA_TIMEOUT":    "45s",
				},
				CLI: model.Config{
					APIKey:    "cli-key",
					LocalURL:  "http://cli.local",
					CloudURL:  "http://cli.cloud",
					Runs:      10,
					Warmup:    0,
					Prompt:    "cli prompt",
					MaxTokens: 200,
					Timeout:   90 * time.Second,
					Parallel:  8,
					SortKey:   "total",
				},
				CLISet: map[string]bool{
					"api-key":    true,
					"local-url":  true,
					"cloud-url":  true,
					"runs":       true,
					"warmup":     true,
					"prompt":     true,
					"max-tokens": true,
					"timeout":    true,
					"parallel":   true,
					"sort-key":   true,
				},
			},
			want: model.Config{
				APIKey:    "cli-key",
				LocalURL:  "http://cli.local",
				CloudURL:  "http://cli.cloud",
				Runs:      10,
				Warmup:    0,
				Prompt:    "cli prompt",
				MaxTokens: 200,
				Timeout:   90 * time.Second,
				Parallel:  8,
				SortKey:   "total",
			},
		},
		{
			name: "unset cli flags do not clobber file or env",
			sources: Sources{
				FilePath: writeFile(t, t.TempDir(), "config.toml", fileContent),
				Env: map[string]string{
					"OLLAMA_RUNS":    "7",
					"OLLAMA_TIMEOUT": "45s",
				},
				CLI: model.Config{
					// Simulate flags parsed by cobra with zero values because
					// they were not provided by the user.
					Runs:    0,
					Timeout: 0,
				},
				CLISet: map[string]bool{
					// Neither runs nor timeout was explicitly set.
				},
			},
			want: model.Config{
				APIKey:    "file-key",
				LocalURL:  "http://file.local",
				CloudURL:  "http://file.cloud",
				Runs:      7,
				Warmup:    2,
				Prompt:    "file prompt",
				MaxTokens: 100,
				Timeout:   45 * time.Second,
				Parallel:  4,
				SortKey:   "tps",
			},
		},
		{
			name: "missing config file falls back to defaults",
			sources: Sources{
				FilePath: filepath.Join(t.TempDir(), "missing.toml"),
				Env:      map[string]string{},
			},
			want: Default(),
		},
		{
			name: "missing config file with omt_config env falls back to defaults",
			sources: Sources{
				Env: map[string]string{
					"OMT_CONFIG": filepath.Join(t.TempDir(), "missing.toml"),
				},
			},
			want: Default(),
		},
		{
			name: "omt_config env var overrides default config path",
			sources: Sources{
				Env: map[string]string{
					"OMT_CONFIG": writeFile(t, t.TempDir(), "override.toml", `
api_key = "override-key"
runs = 9
`),
				},
			},
			want: model.Config{
				APIKey:    "override-key",
				LocalURL:  DefaultLocalURL,
				CloudURL:  DefaultCloudURL,
				Runs:      9,
				Warmup:    DefaultWarmup,
				Prompt:    DefaultPrompt,
				MaxTokens: DefaultMaxTokens,
				Timeout:   DefaultTimeout,
				Parallel:  DefaultParallel,
				SortKey:   DefaultSortKey,
			},
		},
		{
			name: "file path source overrides omt_config env var",
			sources: Sources{
				FilePath: writeFile(t, t.TempDir(), "explicit.toml", `api_key = "explicit-key"`),
				Env: map[string]string{
					"OMT_CONFIG": writeFile(t, t.TempDir(), "env-config.toml", `api_key = "env-config-key"`),
				},
			},
			want: model.Config{
				APIKey:    "explicit-key",
				LocalURL:  DefaultLocalURL,
				CloudURL:  DefaultCloudURL,
				Runs:      DefaultRuns,
				Warmup:    DefaultWarmup,
				Prompt:    DefaultPrompt,
				MaxTokens: DefaultMaxTokens,
				Timeout:   DefaultTimeout,
				Parallel:  DefaultParallel,
				SortKey:   DefaultSortKey,
			},
		},
		{
			name: "malformed toml returns clear error",
			sources: Sources{
				FilePath: writeFile(t, t.TempDir(), "bad.toml", `api_key = "unterminated`),
				Env:      map[string]string{},
			},
			wantErr: true,
			errSub:  "parse config file",
		},
		{
			name: "invalid timeout in file returns clear error",
			sources: Sources{
				FilePath: writeFile(t, t.TempDir(), "bad-timeout.toml", `timeout = "not-a-duration"`),
				Env:      map[string]string{},
			},
			wantErr: true,
			errSub:  "invalid timeout",
		},
		{
			name: "invalid runs env var returns clear error",
			sources: Sources{
				FilePath: writeFile(t, t.TempDir(), "empty.toml", ""),
				Env: map[string]string{
					"OLLAMA_RUNS": "not-a-number",
				},
			},
			wantErr: true,
			errSub:  "invalid OLLAMA_RUNS",
		},
		{
			name: "invalid timeout env var returns clear error",
			sources: Sources{
				FilePath: writeFile(t, t.TempDir(), "empty.toml", ""),
				Env: map[string]string{
					"OLLAMA_TIMEOUT": "not-a-duration",
				},
			},
			wantErr: true,
			errSub:  "invalid OLLAMA_TIMEOUT",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := Load(tt.sources)
			if tt.wantErr {
				if err == nil {
					t.Fatalf("Load() expected error, got nil")
				}
				if tt.errSub != "" && !strings.Contains(err.Error(), tt.errSub) {
					t.Fatalf("Load() error %q does not contain %q", err.Error(), tt.errSub)
				}
				return
			}
			if err != nil {
				t.Fatalf("Load() unexpected error: %v", err)
			}
			if got != tt.want {
				t.Fatalf("Load() = %+v, want %+v", got, tt.want)
			}
		})
	}
}

func TestRedact(t *testing.T) {
	cfg := model.Config{
		APIKey:    "super-secret-key",
		LocalURL:  "http://localhost:11434",
		CloudURL:  "https://ollama.com/api",
		Runs:      5,
		Timeout:   60 * time.Second,
		SortKey:   "ttft",
	}

	redacted := Redact(cfg)
	if redacted.APIKey != "[REDACTED]" {
		t.Errorf("Redact() APIKey = %q, want [REDACTED]", redacted.APIKey)
	}
	if redacted.LocalURL != cfg.LocalURL {
		t.Errorf("Redact() clobbered LocalURL: got %q, want %q", redacted.LocalURL, cfg.LocalURL)
	}

	str := String(cfg)
	if strings.Contains(str, "super-secret-key") {
		t.Errorf("String() leaked API key: %s", str)
	}
	if !strings.Contains(str, "[REDACTED]") {
		t.Errorf("String() missing redaction marker: %s", str)
	}
}
