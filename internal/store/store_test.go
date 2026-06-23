package store

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/simonteague6/ollama-model-tester/internal/model"
)

func sampleSession(id string) Session {
	return Session{
		ID:            id,
		Timestamp:     time.Date(2026, 6, 23, 12, 0, 0, 0, time.UTC),
		ModelsTested:  []string{"llama3.1", "qwen2"},
		Prompt:        "Explain Go interfaces",
		ConfigSummary: "runs=5,warmup=1",
	}
}

func sampleResults() []StoredResult {
	return []StoredResult{
		{
			ModelName: "llama3.1",
			Endpoint:  "local",
			Aggregate: model.AggregateResult{
				MeanTTFT:    10 * time.Millisecond,
				MedianTTFT:  11 * time.Millisecond,
				MinTTFT:     9 * time.Millisecond,
				MaxTTFT:     12 * time.Millisecond,
				MeanTPS:     42.5,
				MedianTPS:   43.0,
				MinTPS:      40.0,
				MaxTPS:      45.0,
				MeanTotal:   100 * time.Millisecond,
				MedianTotal: 101 * time.Millisecond,
				MinTotal:    99 * time.Millisecond,
				MaxTotal:    102 * time.Millisecond,
				SuccessCount: 5,
				FailCount:    0,
			},
			Runs: []model.RunResult{
				{TTFT: 10 * time.Millisecond, TokensPerSec: 42.0, TotalTime: 100 * time.Millisecond, TokenCount: 100},
				{TTFT: 11 * time.Millisecond, TokensPerSec: 43.0, TotalTime: 101 * time.Millisecond, TokenCount: 101},
			},
		},
	}
}

func mustOpen(t *testing.T) Store {
	t.Helper()
	s, err := NewSQLiteStore(":memory:")
	if err != nil {
		t.Fatalf("NewSQLiteStore: %v", err)
	}
	return s
}

func TestStore(t *testing.T) {
	tests := []struct {
		name string
		fn   func(t *testing.T, s Store)
	}{
		{
			name: "save and list round-trip",
			fn: func(t *testing.T, s Store) {
				want := sampleSession("session-a")
				if err := s.SaveSession(want, sampleResults()); err != nil {
					t.Fatalf("SaveSession: %v", err)
				}

				sessions, err := s.ListSessions(10, 0)
				if err != nil {
					t.Fatalf("ListSessions: %v", err)
				}
				if len(sessions) != 1 {
					t.Fatalf("ListSessions returned %d sessions, want 1", len(sessions))
				}
				got := sessions[0]
				if got.ID != want.ID || !got.Timestamp.Equal(want.Timestamp) || got.Prompt != want.Prompt || got.ConfigSummary != want.ConfigSummary {
					t.Errorf("ListSessions session mismatch: got %+v, want %+v", got, want)
				}
				if len(got.ModelsTested) != len(want.ModelsTested) || got.ModelsTested[0] != want.ModelsTested[0] {
					t.Errorf("ListSessions ModelsTested mismatch: got %v, want %v", got.ModelsTested, want.ModelsTested)
				}
			},
		},
		{
			name: "save and get round-trip",
			fn: func(t *testing.T, s Store) {
				want := sampleSession("session-b")
				wantResults := sampleResults()
				if err := s.SaveSession(want, wantResults); err != nil {
					t.Fatalf("SaveSession: %v", err)
				}

				gotSession, gotResults, err := s.GetSession(want.ID)
				if err != nil {
					t.Fatalf("GetSession: %v", err)
				}
				if gotSession.ID != want.ID || !gotSession.Timestamp.Equal(want.Timestamp) || gotSession.Prompt != want.Prompt {
					t.Errorf("GetSession session mismatch: got %+v, want %+v", gotSession, want)
				}
				if len(gotResults) != len(wantResults) {
					t.Fatalf("GetSession returned %d results, want %d", len(gotResults), len(wantResults))
				}
				if gotResults[0].ModelName != wantResults[0].ModelName || gotResults[0].Endpoint != wantResults[0].Endpoint {
					t.Errorf("GetSession result mismatch: got %+v, want %+v", gotResults[0], wantResults[0])
				}
				if len(gotResults[0].Runs) != len(wantResults[0].Runs) {
					t.Errorf("GetSession runs mismatch: got %d runs, want %d", len(gotResults[0].Runs), len(wantResults[0].Runs))
				}
			},
		},
		{
			name: "empty list",
			fn: func(t *testing.T, s Store) {
				sessions, err := s.ListSessions(10, 0)
				if err != nil {
					t.Fatalf("ListSessions: %v", err)
				}
				if len(sessions) != 0 {
					t.Fatalf("ListSessions returned %d sessions, want 0", len(sessions))
				}
			},
		},
		{
			name: "list with limit and offset",
			fn: func(t *testing.T, s Store) {
				for i, id := range []string{"session-1", "session-2", "session-3"} {
					sess := sampleSession(id)
					sess.Timestamp = sess.Timestamp.Add(time.Duration(i) * time.Hour)
					if err := s.SaveSession(sess, sampleResults()); err != nil {
						t.Fatalf("SaveSession %s: %v", id, err)
					}
				}

				sessions, err := s.ListSessions(2, 0)
				if err != nil {
					t.Fatalf("ListSessions limit=2: %v", err)
				}
				if len(sessions) != 2 {
					t.Fatalf("ListSessions limit=2 returned %d sessions, want 2", len(sessions))
				}
				if sessions[0].ID != "session-3" || sessions[1].ID != "session-2" {
					t.Errorf("ListSessions limit=2 order mismatch: got %v, want [session-3 session-2]", []string{sessions[0].ID, sessions[1].ID})
				}

				sessions, err = s.ListSessions(1, 1)
				if err != nil {
					t.Fatalf("ListSessions offset=1: %v", err)
				}
				if len(sessions) != 1 || sessions[0].ID != "session-2" {
					t.Errorf("ListSessions offset=1 mismatch: got %v, want [session-2]", []string{sessions[0].ID})
				}
			},
		},
		{
			name: "get missing session returns error",
			fn: func(t *testing.T, s Store) {
				_, _, err := s.GetSession("does-not-exist")
				if !errors.Is(err, ErrSessionNotFound) {
					t.Errorf("GetSession missing error = %v, want ErrSessionNotFound", err)
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			store := mustOpen(t)
			tt.fn(t, store)
		})
	}
}
func TestFileStoreAutoCreates(t *testing.T) {
	originalHome, ok := os.LookupEnv("HOME")
	defer func() {
		if ok {
			os.Setenv("HOME", originalHome)
		} else {
			os.Unsetenv("HOME")
		}
	}()

	home := t.TempDir()
	if err := os.Setenv("HOME", home); err != nil {
		t.Fatalf("set HOME: %v", err)
	}

	store, err := NewSQLiteStore("~/.omt/history.db")
	if err != nil {
		t.Fatalf("NewSQLiteStore: %v", err)
	}

	if err := store.SaveSession(sampleSession("session-file"), sampleResults()); err != nil {
		t.Fatalf("SaveSession: %v", err)
	}

	dbPath := filepath.Join(home, ".omt", "history.db")
	if _, err := os.Stat(dbPath); err != nil {
		t.Fatalf("database not created at %s: %v", dbPath, err)
	}

	_, _, err = store.GetSession("session-file")
	if err != nil {
		t.Fatalf("GetSession: %v", err)
	}
}

