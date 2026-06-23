package ollama_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/simonteague6/ollama-model-tester/internal/model"
	"github.com/simonteague6/ollama-model-tester/internal/ollama"
)

func TestLocalClient_ListModels(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/tags" {
			t.Errorf("expected path /api/tags, got %s", r.URL.Path)
		}
		if r.Header.Get("Authorization") != "" {
			t.Errorf("local client must not send Authorization header")
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"models": []map[string]any{
				{"name": "llama3.1"},
				{"name": "mistral"},
			},
		})
	}))
	defer server.Close()

	client := ollama.NewLocalClient(server.URL)
	models, err := client.ListModels(context.Background())
	if err != nil {
		t.Fatalf("ListModels error: %v", err)
	}

	if len(models) != 2 {
		t.Fatalf("expected 2 models, got %d", len(models))
	}
	if models[0].Name != "llama3.1" || models[0].Endpoint != "local" {
		t.Errorf("unexpected model[0]: %+v", models[0])
	}
	if models[1].Name != "mistral" || models[1].Endpoint != "local" {
		t.Errorf("unexpected model[1]: %+v", models[1])
	}
}

func TestCloudClient_ListModels_SendsAuthHeader(t *testing.T) {
	var gotAuth string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"models": []map[string]any{
				{"name": "llama3.1:8b"},
			},
		})
	}))
	defer server.Close()

	client := ollama.NewCloudClient(server.URL, "secret-key")
	models, err := client.ListModels(context.Background())
	if err != nil {
		t.Fatalf("ListModels error: %v", err)
	}

	if gotAuth != "Bearer secret-key" {
		t.Errorf("expected Authorization header 'Bearer secret-key', got %q", gotAuth)
	}
	if len(models) != 1 || models[0].Name != "llama3.1:8b" || models[0].Endpoint != "cloud" {
		t.Errorf("unexpected models: %+v", models)
	}
}

func TestGenerate_StreamsNDJSON(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/generate" {
			t.Errorf("expected path /api/generate, got %s", r.URL.Path)
		}
		if r.Header.Get("Content-Type") != "application/json" {
			t.Errorf("expected Content-Type application/json, got %q", r.Header.Get("Content-Type"))
		}

		body := decodeGenerateBody(t, r)
		if body.Model != "test-model" || body.Prompt != "hello" || !body.Stream {
			t.Errorf("unexpected request body: %+v", body)
		}
		if body.Options == nil || body.Options.NumPredict != 10 {
			t.Errorf("expected options.num_predict=10, got %+v", body.Options)
		}

		w.Header().Set("Content-Type", "application/x-ndjson")
		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Fatal("response writer does not support flushing")
		}

		chunks := []string{"Hi", " there", "!"}
		for _, token := range chunks {
			_, _ = fmt.Fprintf(w, `{"response":%q,"done":false}`+"\n", token)
			flusher.Flush()
		}
		_, _ = fmt.Fprint(w, `{"response":"","done":true,"total_duration":5000000000,"load_duration":100000000,"prompt_eval_duration":200000000,"eval_duration":4500000000,"eval_count":3,"prompt_eval_count":2}`+"\n")
		flusher.Flush()
	}))
	defer server.Close()

	client := ollama.NewLocalClient(server.URL)
	stream, err := client.Generate(context.Background(), model.GenerateRequest{
		Model:     "test-model",
		Prompt:    "hello",
		MaxTokens: 10,
		Stream:    true,
	})
	if err != nil {
		t.Fatalf("Generate error: %v", err)
	}
	defer stream.Close()

	var tokens []string
	var final model.GenerateResponse
	for {
		resp, err := stream.Next()
		if err != nil {
			t.Fatalf("Next error: %v", err)
		}
		if resp.Done {
			final = resp
			break
		}
		tokens = append(tokens, resp.Token)
	}

	if strings.Join(tokens, "") != "Hi there!" {
		t.Errorf("expected tokens 'Hi there!', got %q", strings.Join(tokens, ""))
	}
	if final.TotalDuration != 5*time.Second {
		t.Errorf("expected TotalDuration 5s, got %v", final.TotalDuration)
	}
	if final.LoadDuration != 100*time.Millisecond {
		t.Errorf("expected LoadDuration 100ms, got %v", final.LoadDuration)
	}
	if final.PromptEvalDuration != 200*time.Millisecond {
		t.Errorf("expected PromptEvalDuration 200ms, got %v", final.PromptEvalDuration)
	}
	if final.EvalDuration != 4500*time.Millisecond {
		t.Errorf("expected EvalDuration 4500ms, got %v", final.EvalDuration)
	}
	if final.EvalCount != 3 {
		t.Errorf("expected EvalCount 3, got %d", final.EvalCount)
	}
	if final.PromptEvalCount != 2 {
		t.Errorf("expected PromptEvalCount 2, got %d", final.PromptEvalCount)
	}
}

func TestCloudClient_Generate_SendsAuthHeader(t *testing.T) {
	var gotAuth string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "application/x-ndjson")
		_, _ = fmt.Fprintln(w, `{"response":"","done":true}`)
	}))
	defer server.Close()

	client := ollama.NewCloudClient(server.URL, "cloud-key")
	stream, err := client.Generate(context.Background(), model.GenerateRequest{
		Model:  "m",
		Prompt: "p",
	})
	if err != nil {
		t.Fatalf("Generate error: %v", err)
	}
	defer stream.Close()

	if gotAuth != "Bearer cloud-key" {
		t.Errorf("expected Authorization 'Bearer cloud-key', got %q", gotAuth)
	}
}

func TestStatusCodeErrors(t *testing.T) {
	cases := []struct {
		status int
		want   error
	}{
		{http.StatusUnauthorized, ollama.ErrAuth},
		{http.StatusNotFound, ollama.ErrNotFound},
		{http.StatusTooManyRequests, ollama.ErrRateLimit},
		{http.StatusInternalServerError, ollama.ErrServer},
		{http.StatusBadGateway, ollama.ErrServer},
	}

	for _, tc := range cases {
		t.Run(fmt.Sprintf("ListModels_%d", tc.status), func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(tc.status)
			}))
			defer server.Close()

			client := ollama.NewLocalClient(server.URL)
			_, err := client.ListModels(context.Background())
			if !errors.Is(err, tc.want) {
				t.Errorf("expected %v, got %v", tc.want, err)
			}
		})

		t.Run(fmt.Sprintf("Generate_%d", tc.status), func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(tc.status)
			}))
			defer server.Close()

			client := ollama.NewLocalClient(server.URL)
			_, err := client.Generate(context.Background(), model.GenerateRequest{Model: "m", Prompt: "p"})
			if !errors.Is(err, tc.want) {
				t.Errorf("expected %v, got %v", tc.want, err)
			}
		})
	}
}

func TestContextCancellation(t *testing.T) {
	t.Run("ListModels", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			// Block until the client disconnects.
			<-r.Context().Done()
		}))
		defer server.Close()

		client := ollama.NewLocalClient(server.URL)
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		_, err := client.ListModels(ctx)
		if !errors.Is(err, ollama.ErrTimeout) {
			t.Errorf("expected ErrTimeout, got %v", err)
		}
	})

	t.Run("Generate", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			// Hold the response long enough for the client timeout to fire.
			time.Sleep(100 * time.Millisecond)
		}))
		defer server.Close()

		client := ollama.NewLocalClient(server.URL)
		ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
		defer cancel()

		_, err := client.Generate(ctx, model.GenerateRequest{Model: "m", Prompt: "p"})
		if !errors.Is(err, ollama.ErrTimeout) {
			t.Errorf("expected ErrTimeout, got %v", err)
		}
	})

	t.Run("StreamRead", func(t *testing.T) {
		streamServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/x-ndjson")
			_, _ = fmt.Fprintln(w, `{"response":"first","done":false}`)
		}))
		defer streamServer.Close()

		client := ollama.NewLocalClient(streamServer.URL)
		ctx, cancel := context.WithCancel(context.Background())
		stream, err := client.Generate(ctx, model.GenerateRequest{Model: "m", Prompt: "p"})
		if err != nil {
			t.Fatalf("Generate error: %v", err)
		}
		defer stream.Close()

		resp, err := stream.Next()
		if err != nil {
			t.Fatalf("first Next error: %v", err)
		}
		if resp.Token != "first" {
			t.Errorf("unexpected first token: %q", resp.Token)
		}

		// Cancel context before the next read; Next() observes ctx.Err().
		cancel()

		_, err = stream.Next()
		if !errors.Is(err, ollama.ErrTimeout) {
			t.Errorf("expected ErrTimeout, got %v", err)
		}
	})
}

type generateBody struct {
	Model   string `json:"model"`
	Prompt  string `json:"prompt"`
	Stream  bool   `json:"stream"`
	Options *struct {
		NumPredict int `json:"num_predict"`
	} `json:"options"`
}

func decodeGenerateBody(t *testing.T, r *http.Request) generateBody {
	t.Helper()
	var body generateBody
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		t.Fatalf("decode request body: %v", err)
	}
	return body
}
