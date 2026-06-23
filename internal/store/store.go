package store

import (
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/simonteague6/ollama-model-tester/internal/model"
	_ "modernc.org/sqlite"
)

// Session records a single benchmark run.
type Session struct {
	ID            string
	Timestamp     time.Time
	ModelsTested  []string
	Prompt        string
	ConfigSummary string
}

// StoredResult records the outcome for one model within a session.
type StoredResult struct {
	ModelName string
	Endpoint  string
	Aggregate model.AggregateResult
	Runs      []model.RunResult
}

// Store persists benchmark sessions and their results.
type Store interface {
	SaveSession(session Session, results []StoredResult) error
	ListSessions(limit, offset int) ([]Session, error)
	GetSession(id string) (Session, []StoredResult, error)
}

// ErrSessionNotFound is returned when a requested session does not exist.
var ErrSessionNotFound = errors.New("session not found")

// SQLiteStore is a SQLite-backed implementation of Store.
type SQLiteStore struct {
	db *sql.DB
}

var _ Store = (*SQLiteStore)(nil)

// NewSQLiteStore opens a SQLite-backed store at the given DSN.
// Use ":memory:" for an in-memory database, or a file path such as
// "~/.omt/history.db" for persistent storage.
func NewSQLiteStore(dsn string) (Store, error) {
	dsn, err := expandDSN(dsn)
	if err != nil {
		return nil, err
	}

	db, err := sql.Open("sqlite", dsn)
	if err != nil {
		return nil, fmt.Errorf("open sqlite: %w", err)
	}

	if _, err := db.Exec("PRAGMA foreign_keys = ON"); err != nil {
		_ = db.Close()
		return nil, fmt.Errorf("enable foreign keys: %w", err)
	}

	if err := migrate(db); err != nil {
		_ = db.Close()
		return nil, fmt.Errorf("migrate: %w", err)
	}

	return &SQLiteStore{db: db}, nil
}

func expandDSN(dsn string) (string, error) {
	if dsn == ":memory:" || strings.HasPrefix(dsn, ":memory:") {
		return dsn, nil
	}

	if strings.HasPrefix(dsn, "~/") || strings.HasPrefix(dsn, "~\\") {
		home, err := os.UserHomeDir()
		if err != nil {
			return "", fmt.Errorf("resolve home dir: %w", err)
		}
		dsn = filepath.Join(home, dsn[2:])
	}

	dir := filepath.Dir(dsn)
	if dir != "" && dir != "." {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return "", fmt.Errorf("create database dir: %w", err)
		}
	}

	return dsn, nil
}

func migrate(db *sql.DB) error {
	schema := `
CREATE TABLE IF NOT EXISTS sessions (
	id TEXT PRIMARY KEY,
	timestamp TEXT NOT NULL,
	models_tested TEXT NOT NULL,
	prompt TEXT NOT NULL,
	config_summary TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS results (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
	model_name TEXT NOT NULL,
	endpoint TEXT NOT NULL,
	aggregate TEXT NOT NULL,
	runs TEXT NOT NULL
);
`
	_, err := db.Exec(schema)
	return err
}

// SaveSession persists a session and its results in a single transaction.
func (s *SQLiteStore) SaveSession(session Session, results []StoredResult) error {
	modelsJSON, err := json.Marshal(session.ModelsTested)
	if err != nil {
		return fmt.Errorf("marshal models: %w", err)
	}

	tx, err := s.db.Begin()
	if err != nil {
		return fmt.Errorf("begin transaction: %w", err)
	}
	defer func() { _ = tx.Rollback() }()

	_, err = tx.Exec(
		`INSERT INTO sessions (id, timestamp, models_tested, prompt, config_summary) VALUES (?, ?, ?, ?, ?)`,
		session.ID,
		session.Timestamp.Format(time.RFC3339),
		modelsJSON,
		session.Prompt,
		session.ConfigSummary,
	)
	if err != nil {
		return fmt.Errorf("insert session: %w", err)
	}

	stmt, err := tx.Prepare(`INSERT INTO results (session_id, model_name, endpoint, aggregate, runs) VALUES (?, ?, ?, ?, ?)`)
	if err != nil {
		return fmt.Errorf("prepare result insert: %w", err)
	}
	defer stmt.Close()

	for _, r := range results {
		aggJSON, err := json.Marshal(r.Aggregate)
		if err != nil {
			return fmt.Errorf("marshal aggregate: %w", err)
		}
		runsJSON, err := json.Marshal(r.Runs)
		if err != nil {
			return fmt.Errorf("marshal runs: %w", err)
		}
		if _, err := stmt.Exec(session.ID, r.ModelName, r.Endpoint, aggJSON, runsJSON); err != nil {
			return fmt.Errorf("insert result: %w", err)
		}
	}

	if err := tx.Commit(); err != nil {
		return fmt.Errorf("commit transaction: %w", err)
	}
	return nil
}

// ListSessions returns sessions ordered newest first, paginated by limit and offset.
func (s *SQLiteStore) ListSessions(limit, offset int) ([]Session, error) {
	rows, err := s.db.Query(
		`SELECT id, timestamp, models_tested, prompt, config_summary FROM sessions ORDER BY timestamp DESC LIMIT ? OFFSET ?`,
		limit, offset,
	)
	if err != nil {
		return nil, fmt.Errorf("list sessions: %w", err)
	}
	defer rows.Close()

	var sessions []Session
	for rows.Next() {
		var sess Session
		var ts string
		var modelsJSON string
		if err := rows.Scan(&sess.ID, &ts, &modelsJSON, &sess.Prompt, &sess.ConfigSummary); err != nil {
			return nil, fmt.Errorf("scan session: %w", err)
		}
		sess.Timestamp, err = time.Parse(time.RFC3339, ts)
		if err != nil {
			return nil, fmt.Errorf("parse timestamp: %w", err)
		}
		if err := json.Unmarshal([]byte(modelsJSON), &sess.ModelsTested); err != nil {
			return nil, fmt.Errorf("unmarshal models: %w", err)
		}
		sessions = append(sessions, sess)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate sessions: %w", err)
	}
	return sessions, nil
}

// GetSession returns a single session and all of its stored results.
func (s *SQLiteStore) GetSession(id string) (Session, []StoredResult, error) {
	var sess Session
	var ts string
	var modelsJSON string
	row := s.db.QueryRow(
		`SELECT id, timestamp, models_tested, prompt, config_summary FROM sessions WHERE id = ?`,
		id,
	)
	err := row.Scan(&sess.ID, &ts, &modelsJSON, &sess.Prompt, &sess.ConfigSummary)
	if errors.Is(err, sql.ErrNoRows) {
		return Session{}, nil, ErrSessionNotFound
	}
	if err != nil {
		return Session{}, nil, fmt.Errorf("get session: %w", err)
	}
	sess.Timestamp, err = time.Parse(time.RFC3339, ts)
	if err != nil {
		return Session{}, nil, fmt.Errorf("parse timestamp: %w", err)
	}
	if err := json.Unmarshal([]byte(modelsJSON), &sess.ModelsTested); err != nil {
		return Session{}, nil, fmt.Errorf("unmarshal models: %w", err)
	}

	rows, err := s.db.Query(
		`SELECT model_name, endpoint, aggregate, runs FROM results WHERE session_id = ? ORDER BY id`,
		id,
	)
	if err != nil {
		return Session{}, nil, fmt.Errorf("list results: %w", err)
	}
	defer rows.Close()

	var results []StoredResult
	for rows.Next() {
		var r StoredResult
		var aggJSON string
		var runsJSON string
		if err := rows.Scan(&r.ModelName, &r.Endpoint, &aggJSON, &runsJSON); err != nil {
			return Session{}, nil, fmt.Errorf("scan result: %w", err)
		}
		if err := json.Unmarshal([]byte(aggJSON), &r.Aggregate); err != nil {
			return Session{}, nil, fmt.Errorf("unmarshal aggregate: %w", err)
		}
		if err := json.Unmarshal([]byte(runsJSON), &r.Runs); err != nil {
			return Session{}, nil, fmt.Errorf("unmarshal runs: %w", err)
		}
		results = append(results, r)
	}
	if err := rows.Err(); err != nil {
		return Session{}, nil, fmt.Errorf("iterate results: %w", err)
	}
	return sess, results, nil
}
