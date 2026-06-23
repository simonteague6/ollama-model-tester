package metrics

import (
	"math"
	"testing"
	"time"

	"github.com/simonteague6/ollama-model-tester/internal/model"
)

func TestTokensPerSec(t *testing.T) {
	cases := []struct {
		name        string
		evalCount   int
		evalDuration time.Duration
		want        float64
	}{
		{
			name:        "typical run",
			evalCount:   100,
			evalDuration: time.Second,
			want:        100,
		},
		{
			name:        "zero duration guard",
			evalCount:   100,
			evalDuration: 0,
			want:        0,
		},
		{
			name:        "zero tokens",
			evalCount:   0,
			evalDuration: time.Second,
			want:        0,
		},
		{
			name:        "subsecond duration",
			evalCount:   50,
			evalDuration: 500 * time.Millisecond,
			want:        100,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := TokensPerSec(tc.evalCount, tc.evalDuration)
			if math.Abs(got-tc.want) > 1e-9 {
				t.Fatalf("TokensPerSec(%d, %v) = %v, want %v", tc.evalCount, tc.evalDuration, got, tc.want)
			}
		})
	}
}

func TestAggregate(t *testing.T) {
	run := func(ttft, total time.Duration, tps float64) model.RunResult {
		return model.RunResult{TTFT: ttft, TotalTime: total, TokensPerSec: tps}
	}
	fail := func(err string) model.RunResult {
		return model.RunResult{Error: err}
	}

	cases := []struct {
		name string
		runs []model.RunResult
		want model.AggregateResult
	}{
		{
			name: "empty input",
			runs: []model.RunResult{},
			want: model.AggregateResult{},
		},
		{
			name: "single run",
			runs: []model.RunResult{
				run(time.Second, 5*time.Second, 10),
			},
			want: model.AggregateResult{
				MeanTTFT: time.Second, MedianTTFT: time.Second, MinTTFT: time.Second, MaxTTFT: time.Second,
				MeanTPS: 10, MedianTPS: 10, MinTPS: 10, MaxTPS: 10,
				MeanTotal: 5 * time.Second, MedianTotal: 5 * time.Second, MinTotal: 5 * time.Second, MaxTotal: 5 * time.Second,
				SuccessCount: 1,
			},
		},
		{
			name: "all success odd count",
			runs: []model.RunResult{
				run(100*time.Millisecond, 1*time.Second, 10),
				run(200*time.Millisecond, 2*time.Second, 20),
				run(300*time.Millisecond, 3*time.Second, 30),
			},
			want: model.AggregateResult{
				MeanTTFT: 200 * time.Millisecond,
				MedianTTFT: 200 * time.Millisecond,
				MinTTFT: 100 * time.Millisecond,
				MaxTTFT: 300 * time.Millisecond,
				MeanTPS: 20,
				MedianTPS: 20,
				MinTPS: 10,
				MaxTPS: 30,
				MeanTotal: 2 * time.Second,
				MedianTotal: 2 * time.Second,
				MinTotal: 1 * time.Second,
				MaxTotal: 3 * time.Second,
				SuccessCount: 3,
			},
		},
		{
			name: "all success even count averages median",
			runs: []model.RunResult{
				run(100*time.Millisecond, 1*time.Second, 10),
				run(200*time.Millisecond, 2*time.Second, 20),
				run(300*time.Millisecond, 3*time.Second, 30),
				run(400*time.Millisecond, 4*time.Second, 40),
			},
			want: model.AggregateResult{
				MeanTTFT: 250 * time.Millisecond,
				MedianTTFT: 250 * time.Millisecond,
				MinTTFT: 100 * time.Millisecond,
				MaxTTFT: 400 * time.Millisecond,
				MeanTPS: 25,
				MedianTPS: 25,
				MinTPS: 10,
				MaxTPS: 40,
				MeanTotal: 2500 * time.Millisecond,
				MedianTotal: 2500 * time.Millisecond,
				MinTotal: 1 * time.Second,
				MaxTotal: 4 * time.Second,
				SuccessCount: 4,
			},
		},
		{
			name: "all fail",
			runs: []model.RunResult{
				fail("timeout"),
				fail("timeout"),
				fail("not found"),
			},
			want: model.AggregateResult{
				SuccessCount: 0,
				FailCount:    3,
			},
		},
		{
			name: "mixed success and fail",
			runs: []model.RunResult{
				run(100*time.Millisecond, 1*time.Second, 10),
				fail("timeout"),
				run(300*time.Millisecond, 3*time.Second, 30),
			},
			want: model.AggregateResult{
				MeanTTFT: 200 * time.Millisecond,
				MedianTTFT: 200 * time.Millisecond,
				MinTTFT: 100 * time.Millisecond,
				MaxTTFT: 300 * time.Millisecond,
				MeanTPS: 20,
				MedianTPS: 20,
				MinTPS: 10,
				MaxTPS: 30,
				MeanTotal: 2 * time.Second,
				MedianTotal: 2 * time.Second,
				MinTotal: 1 * time.Second,
				MaxTotal: 3 * time.Second,
				SuccessCount: 2,
				FailCount:    1,
			},
		},
		{
			name: "zero duration included",
			runs: []model.RunResult{
				run(0, 0, 0),
				run(100*time.Millisecond, 1*time.Second, 10),
			},
			want: model.AggregateResult{
				MeanTTFT: 50 * time.Millisecond,
				MedianTTFT: 50 * time.Millisecond,
				MinTTFT: 0,
				MaxTTFT: 100 * time.Millisecond,
				MeanTPS: 5,
				MedianTPS: 5,
				MinTPS: 0,
				MaxTPS: 10,
				MeanTotal: 500 * time.Millisecond,
				MedianTotal: 500 * time.Millisecond,
				MinTotal: 0,
				MaxTotal: 1 * time.Second,
				SuccessCount: 2,
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := Aggregate(tc.runs)
			if got != tc.want {
				t.Fatalf("Aggregate mismatch:\n got: %+v\n want: %+v", got, tc.want)
			}
		})
	}
}

func TestMedianDurationOdd(t *testing.T) {
	got := medianDuration([]time.Duration{10 * time.Millisecond, 20 * time.Millisecond, 30 * time.Millisecond})
	if got != 20*time.Millisecond {
		t.Fatalf("median odd = %v, want 20ms", got)
	}
}

func TestMedianDurationEven(t *testing.T) {
	got := medianDuration([]time.Duration{10 * time.Millisecond, 20 * time.Millisecond, 30 * time.Millisecond, 40 * time.Millisecond})
	if got != 25*time.Millisecond {
		t.Fatalf("median even = %v, want 25ms", got)
	}
}

func TestMedianFloat64Even(t *testing.T) {
	got := medianFloat64([]float64{1, 2, 3, 4})
	if math.Abs(got-2.5) > 1e-9 {
		t.Fatalf("median float even = %v, want 2.5", got)
	}
}
