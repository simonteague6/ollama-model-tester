package ollama

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/simonteague6/ollama-model-tester/internal/model"
)

// Typed errors returned by Client implementations.
var (
	ErrAuth     = errors.New("authentication failed")
	ErrNotFound = errors.New("model not found")
	ErrRateLimit = errors.New("rate limited")
	ErrTimeout  = errors.New("request timed out")
	ErrServer   = errors.New("server error")
)

// NewLocalClient creates a Client for a local Ollama endpoint.
func NewLocalClient(baseURL string) model.Client {
	return &client{
		baseURL:    strings.TrimRight(baseURL, "/"),
		endpoint:   "local",
		httpClient: &http.Client{},
	}
}

// NewCloudClient creates a Client for the Ollama Cloud endpoint.
func NewCloudClient(baseURL, apiKey string) model.Client {
	return &client{
		baseURL:    strings.TrimRight(baseURL, "/"),
		apiKey:     apiKey,
		endpoint:   "cloud",
		httpClient: &http.Client{},
	}
}

type client struct {
	baseURL    string
	apiKey     string
	endpoint   string
	httpClient *http.Client
}

func (c *client) ListModels(ctx context.Context) ([]model.Model, error) {
	url := c.baseURL + "/api/tags"
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	c.setAuth(req)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, mapRequestError(err)
	}
	defer resp.Body.Close()

	if err := checkStatus(resp); err != nil {
		return nil, err
	}

	var payload tagsResponse
	if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil {
		return nil, err
	}

	models := make([]model.Model, 0, len(payload.Models))
	for _, m := range payload.Models {
		models = append(models, model.Model{
			Name:     m.Name,
			Endpoint: c.endpoint,
		})
	}
	return models, nil
}

func (c *client) Generate(ctx context.Context, req model.GenerateRequest) (model.GenerateStream, error) {
	body := generateRequestBody{
		Model:  req.Model,
		Prompt: req.Prompt,
		Stream: true,
	}
	if req.MaxTokens > 0 {
		body.Options = &generateOptions{NumPredict: req.MaxTokens}
	}

	data, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}

	url := c.baseURL + "/api/generate"
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	c.setAuth(httpReq)

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, mapRequestError(err)
	}

	if err := checkStatus(resp); err != nil {
		resp.Body.Close()
		return nil, err
	}

	return &stream{
		ctx:    ctx,
		rc:     resp.Body,
		reader: bufio.NewReader(resp.Body),
	}, nil
}

func (c *client) setAuth(req *http.Request) {
	if c.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.apiKey)
	}
}

func mapRequestError(err error) error {
	if errors.Is(err, context.DeadlineExceeded) || errors.Is(err, context.Canceled) {
		return ErrTimeout
	}
	return err
}

func checkStatus(resp *http.Response) error {
	switch resp.StatusCode {
	case http.StatusOK:
		return nil
	case http.StatusUnauthorized:
		return ErrAuth
	case http.StatusNotFound:
		return ErrNotFound
	case http.StatusTooManyRequests:
		return ErrRateLimit
	}
	if resp.StatusCode >= 500 {
		return ErrServer
	}
	return fmt.Errorf("unexpected status %d", resp.StatusCode)
}

type tagsResponse struct {
	Models []struct {
		Name string `json:"name"`
	} `json:"models"`
}

type generateRequestBody struct {
	Model   string           `json:"model"`
	Prompt  string           `json:"prompt"`
	Stream  bool             `json:"stream"`
	Options *generateOptions `json:"options,omitempty"`
}

type generateOptions struct {
	NumPredict int `json:"num_predict"`
}

type stream struct {
	ctx    context.Context
	rc     io.ReadCloser
	reader *bufio.Reader
}

func (s *stream) Next() (model.GenerateResponse, error) {
	if err := s.ctx.Err(); err != nil {
		return model.GenerateResponse{}, ErrTimeout
	}

	line, err := s.reader.ReadString('\n')
	if err != nil {
		if errors.Is(err, context.DeadlineExceeded) || errors.Is(err, context.Canceled) || s.ctx.Err() != nil {
			return model.GenerateResponse{}, ErrTimeout
		}
		return model.GenerateResponse{}, err
	}

	line = strings.TrimSpace(line)
	if line == "" {
		return s.Next()
	}

	var chunk generateChunk
	if err := json.Unmarshal([]byte(line), &chunk); err != nil {
		return model.GenerateResponse{}, err
	}

	return model.GenerateResponse{
		Token:              chunk.Response,
		Done:               chunk.Done,
		TotalDuration:      time.Duration(chunk.TotalDuration),
		LoadDuration:       time.Duration(chunk.LoadDuration),
		PromptEvalDuration: time.Duration(chunk.PromptEvalDuration),
		EvalDuration:       time.Duration(chunk.EvalDuration),
		EvalCount:          chunk.EvalCount,
		PromptEvalCount:    chunk.PromptEvalCount,
	}, nil
}

func (s *stream) Close() error {
	return s.rc.Close()
}

type generateChunk struct {
	Response           string `json:"response"`
	Done               bool   `json:"done"`
	TotalDuration      int64  `json:"total_duration"`
	LoadDuration       int64  `json:"load_duration"`
	PromptEvalDuration int64  `json:"prompt_eval_duration"`
	EvalDuration       int64  `json:"eval_duration"`
	EvalCount          int    `json:"eval_count"`
	PromptEvalCount    int    `json:"prompt_eval_count"`
}
