#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

# ============================================================================
# Оптимизированные dtype (экономия памяти)
# ============================================================================

DTYPES_COMBINED = {
    'mode': 'category',
    'workload': 'category',
    'threads': 'int16',
    'size_mb': 'int32',
    'p50': 'float32',
    'p90': 'float32',
    'p99': 'float32',
    'p99_9': 'float32',
    'p999': 'float32'
}

DTYPES_RAW = {
    'cycles': 'int32'
}

# ============================================================================
# Загрузка данных
# ============================================================================

def load_combined(results_dir: Path) -> pd.DataFrame:
    path = results_dir / 'combined.csv'
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")

    try:
        df = pd.read_csv(path, dtype=DTYPES_COMBINED)
    except Exception as e:
        raise RuntimeError(f"Failed to read combined.csv: {e}")

    required = {'mode', 'workload', 'threads', 'p50', 'p99', 'p99_9'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    return df


def iter_raw_files(results_dir: Path) -> Iterator[tuple[str, Iterator[pd.DataFrame]]]:
    """Ленивый генератор raw файлов (chunked)"""
    for f in results_dir.glob('*_raw.csv'):
        try:
            chunks = pd.read_csv(
                f,
                skiprows=1,
                names=['cycles'],
                dtype=DTYPES_RAW,
                chunksize=1_000_000
            )
            yield f.name, chunks
        except Exception as e:
            print(f"[WARN] {f.name}: {e}", file=sys.stderr)


# ============================================================================
# Векторизированные метрики
# ============================================================================

def calc_numa_penalty(df: pd.DataFrame, percentile='p99') -> list[dict]:
    pivot = df.pivot_table(
        index=['workload', 'threads', 'size_mb'],
        columns='mode',
        values=percentile
    ).dropna()

    if 'local' not in pivot or 'remote' not in pivot:
        return []

    pivot['abs_penalty_ns'] = pivot['remote'] - pivot['local']
    pivot['rel_penalty_pct'] = pivot['abs_penalty_ns'] / pivot['local'] * 100

    return pivot.reset_index().rename(columns={
        'local': 'local_ns',
        'remote': 'remote_ns'
    }).to_dict('records')


def calc_tail_ratio(df: pd.DataFrame):
    df['p99_p50_ratio'] = df['p99'] / df['p50']


# ============================================================================
# Streaming spike detection (фикс!)
# ============================================================================

def detect_spikes_streaming(results_dir: Path,
                            threshold_factor=3.0,
                            min_cycles=100):

    total = 0
    max_val = 0
    sample = []

    # 🔹 Первый проход — собираем sample для медианы
    for _, chunks in iter_raw_files(results_dir):
        for chunk in chunks:
            vals = chunk['cycles'].values

            total += len(vals)
            max_val = max(max_val, vals.max())

            if len(sample) < 100_000:
                sample.extend(vals[:1000])

    if not sample:
        return {}

    median = np.median(sample)
    threshold = max(median * threshold_factor, min_cycles)

    # 🔹 Второй проход — считаем spikes
    spike_count = 0

    for _, chunks in iter_raw_files(results_dir):  # важно: новый генератор
        for chunk in chunks:
            vals = chunk['cycles'].values
            spike_count += np.sum(vals > threshold)

    return {
        'total_samples': total,
        'count': int(spike_count),
        'spike_rate_pct': spike_count / total * 100 if total else 0,
        'median_cycles': float(median),
        'threshold_cycles': float(threshold),
        'max_cycles': int(max_val)
    }


# ============================================================================
# JSON экспорт
# ============================================================================

def export_json(df, penalties, spike_summary, out_path: Path):
    data = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "runs": df.to_dict("records"),
        "penalties": penalties,
        "spikes": spike_summary
    }

    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"[JSON] saved: {out_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="NUMA latency analysis (optimized)")
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()

    if not args.results_dir.exists():
        print("ERROR: results_dir not found", file=sys.stderr)
        sys.exit(1)

    if not args.quiet:
        print(f"[INFO] analyzing {args.results_dir}")

    try:
        df = load_combined(args.results_dir)

        # ⚡ Векторизация вместо циклов
        penalties = calc_numa_penalty(df)

        # ⚡ inplace операция
        calc_tail_ratio(df)

        # ⚡ streaming
        spike_summary = detect_spikes_streaming(args.results_dir)

        if not args.quiet:
            print(f"[INFO] runs: {len(df)}")
            print(f"[INFO] penalties: {len(penalties)}")
            print(f"[INFO] spikes: {spike_summary.get('count', 0)}")

        if args.json:
            export_json(df, penalties, spike_summary, args.json)

        if not args.quiet:
            print("\n=== SAMPLE ===")
            print(df.head(10).to_string(index=False))

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()