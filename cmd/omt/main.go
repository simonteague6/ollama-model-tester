package main

import (
	"fmt"
	"os"

	"github.com/simonteague6/ollama-model-tester/internal/cli"
	"github.com/simonteague6/ollama-model-tester/internal/config"
	"github.com/simonteague6/ollama-model-tester/internal/store"
)

func main() {
	cfg, err := config.Load(config.Sources{})
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	var st store.Store
	st, err = store.NewSQLiteStore("~/.omt/history.db")
	if err != nil {
		fmt.Fprintf(os.Stderr, "warning: failed to open result store: %v\n", err)
		st = nil
	}

	root := cli.BuildRootCmd(&cfg, st)
	if err := root.Execute(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
