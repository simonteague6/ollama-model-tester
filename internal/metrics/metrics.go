package metrics

import (
	"sort"
	"time"

	"github.com/simonteague6/ollama-model-tester/internal/model"
)

// TokensPerSec returns the generation throughput from the server's eval metrics.
// It returns 0 when evalDuration is zero to avoid division-by-zero panics.
func TokensPerSec(evalCount int, evalDuration time.Duration) float64 {
	if evalDuration == 0 {
		return 0
	}
	return float64(evalCount) / float64(evalDuration) * 1e9
}

// Aggregate computes mean, median, min, and max TTFT, tokens-per-second, and
// total time over the successful runs in results. Failed runs (Error != "")
// are skipped. When every run failed, SuccessCount is zero and FailCount is the
// total number of runs, indicating a failed aggregate.
func Aggregate(results []model.RunResult) model.AggregateResult {
	if len(results) == 0 {
		return model.AggregateResult{}
	}

	successes := make([]model.RunResult, 0, len(results))
	failCount := 0
	for _, r := range results {
		if r.Error == "" {
			successes = append(successes, r)
		} else {
			failCount++
		}
	}

	if len(successes) == 0 {
		return model.AggregateResult{
			SuccessCount: 0,
			FailCount:    failCount,
		}
	}

	ttfts := make([]time.Duration, len(successes))
	tps := make([]float64, len(successes))
	totals := make([]time.Duration, len(successes))

	minTTFT := successes[0].TTFT
	maxTTFT := successes[0].TTFT
	minTPS := successes[0].TokensPerSec
	maxTPS := successes[0].TokensPerSec
	minTotal := successes[0].TotalTime
	maxTotal := successes[0].TotalTime

	var sumTTFT, sumTotal time.Duration
	var sumTPS float64

	for i, r := range successes {
		ttfts[i] = r.TTFT
		tps[i] = r.TokensPerSec
		totals[i] = r.TotalTime

		sumTTFT += r.TTFT
		sumTPS += r.TokensPerSec
		sumTotal += r.TotalTime

		if r.TTFT < minTTFT {
			minTTFT = r.TTFT
		}
		if r.TTFT > maxTTFT {
			maxTTFT = r.TTFT
		}
		if r.TokensPerSec < minTPS {
			minTPS = r.TokensPerSec
		}
		if r.TokensPerSec > maxTPS {
			maxTPS = r.TokensPerSec
		}
		if r.TotalTime < minTotal {
			minTotal = r.TotalTime
		}
		if r.TotalTime > maxTotal {
			maxTotal = r.TotalTime
		}
	}

	n := float64(len(successes))

	sort.Slice(ttfts, func(i, j int) bool { return ttfts[i] < ttfts[j] })
	sort.Float64s(tps)
	sort.Slice(totals, func(i, j int) bool { return totals[i] < totals[j] })

	return model.AggregateResult{
		MeanTTFT:   time.Duration(float64(sumTTFT) / n),
		MedianTTFT: medianDuration(ttfts),
		MinTTFT:    minTTFT,
		MaxTTFT:    maxTTFT,
		MeanTPS:    sumTPS / n,
		MedianTPS:  medianFloat64(tps),
		MinTPS:     minTPS,
		MaxTPS:     maxTPS,
		MeanTotal:  time.Duration(float64(sumTotal) / n),
		MedianTotal: medianDuration(totals),
		MinTotal:   minTotal,
		MaxTotal:   maxTotal,
		SuccessCount: len(successes),
		FailCount:  failCount,
	}
}

func medianDuration(d []time.Duration) time.Duration {
	if len(d) == 0 {
		return 0
	}
	mid := len(d) / 2
	if len(d)%2 == 1 {
		return d[mid]
	}
	return (d[mid-1] + d[mid]) / 2
}

func medianFloat64(f []float64) float64 {
	if len(f) == 0 {
		return 0
	}
	mid := len(f) / 2
	if len(f)%2 == 1 {
		return f[mid]
	}
	return (f[mid-1] + f[mid]) / 2
}
