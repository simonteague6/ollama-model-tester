package main

import (
	"fmt"
	"os"

	"github.com/simonteague6/ollama-model-tester/internal/cli"
	"github.com/simonteague6/ollama-model-tester/internal/config"
)

func main() {
	cfg, err := config.Load(config.Sources{})
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	root := cli.BuildRootCmd(&cfg)
	if err := root.Execute(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
