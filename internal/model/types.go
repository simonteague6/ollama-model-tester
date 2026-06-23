package model

import (
	"context"
	"time"
)

// Model represents a discovered Ollama model and the endpoint it was found on.
type Model struct {
	Name     string
	Endpoint string // "local" or "cloud"
	Provider string
}

// RunResult holds the metrics for a single benchmark run.
type RunResult struct {
	TTFT         time.Duration
	TokensPerSec float64
	TotalTime    time.Duration
	TokenCount   int
	Error        string // empty if no error
}

// AggregateResult summarizes a collection of RunResults.
type AggregateResult struct {
	MeanTTFT, MedianTTFT, MinTTFT, MaxTTFT time.Duration
	MeanTPS, MedianTPS, MinTPS, MaxTPS     float64
	MeanTotal, MedianTotal, MinTotal, MaxTotal time.Duration
	SuccessCount, FailCount                int
}

// Config contains all user-configurable settings for a benchmark session.
type Config struct {
	APIKey    string
	LocalURL  string
	CloudURL  string
	Runs      int
	Warmup    int
	Prompt    string
	MaxTokens int
	Timeout   time.Duration
	Parallel  int
	SortKey   string
}

// GenerateRequest is the payload sent to a Client for generation.
type GenerateRequest struct {
	Model     string
	Prompt    string
	MaxTokens int
	Stream    bool
}

// GenerateResponse represents one chunk from a streaming generation response.
type GenerateResponse struct {
	Token              string
	Done               bool
	TotalDuration      time.Duration
	LoadDuration       time.Duration
	PromptEvalDuration time.Duration
	EvalDuration       time.Duration
	EvalCount          int
	PromptEvalCount    int
}

// GenerateStream is the streaming response returned by Client.Generate.
type GenerateStream interface {
	Next() (GenerateResponse, error)
	Close() error
}

// Client abstracts access to an Ollama endpoint.
type Client interface {
	ListModels(ctx context.Context) ([]Model, error)
	Generate(ctx context.Context, req GenerateRequest) (GenerateStream, error)
}
