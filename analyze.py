#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Iterator

import numpy as np
import pandas as pd

# ============================================================================
# Конфигурация памяти
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
# Загрузка данных (chunking + dtype)
# ============================================================================

def load_combined(results_dir: Path) -> pd.DataFrame:
    path = results_dir / 'combined.csv'
    if not path.exists():
        raise FileNotFoundError(path)

    try:
        df = pd.read_csv(path, dtype=DTYPES_COMBINED)
    except Exception as e:
        raise RuntimeError(f"Failed to read combined.csv: {e}")

    required = {'mode', 'workload', 'threads', 'p50', 'p99', 'p99_9'}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns: {required - set(df.columns)}")

    return df


def iter_raw_files(results_dir: Path) -> Iterator[tuple[str, Iterator[pd.DataFrame]]]:
    """Ленивый генератор raw-файлов с chunking"""
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

def calc_numa_penalty_vectorized(df: pd.DataFrame, percentile='p99') -> list[dict]:
    """
    Полностью векторизированный расчёт NUMA penalty
    """
    pivot = df.pivot_table(
        index=['workload', 'threads', 'size_mb'],
        columns='mode',
        values=percentile
    ).dropna()

    if 'local' not in pivot or 'remote' not in pivot:
        return []

    pivot['abs_penalty'] = pivot['remote'] - pivot['local']
    pivot['rel_penalty'] = pivot['abs_penalty'] / pivot['local'] * 100

    return pivot.reset_index().rename(columns={
        'local': 'local_ns',
        'remote': 'remote_ns'
    }).to_dict('records')


def calc_tail_ratio(df: pd.DataFrame) -> pd.DataFrame:
    df['p99_p50_ratio'] = df['p99'] / df['p50']
    return df


# ============================================================================
# Streaming spike detection (без загрузки в память)
# ============================================================================

def detect_spikes_streaming(raw_iter, threshold_factor=3.0, min_cycles=100):
    total = 0
    spike_count = 0
    max_val = 0

    # Для медианы используем reservoir sampling (упрощённо)
    sample = []

    for _, chunks in raw_iter:
        for chunk in chunks:
            vals = chunk['cycles'].values

            total += len(vals)
            max_val = max(max_val, vals.max())

            # собираем sample для медианы
            if len(sample) < 100000:
                sample.extend(vals[:1000])

    if not sample:
        return {}

    median = np.median(sample)
    threshold = max(median * threshold_factor, min_cycles)

    # второй проход — считаем spikes
    for _, chunks in iter_raw_files(results_dir):
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
# Perf (оптимизированный pivot)
# ============================================================================

def calc_llc_miss_rate(perf_df: pd.DataFrame) -> pd.DataFrame:
    if perf_df is None or perf_df.empty:
        return pd.DataFrame()

    perf_df['event'] = perf_df['event'].astype('category')

    llc = perf_df[perf_df['event'].str.contains('LLC', na=False)]

    grouped = llc.groupby(
        ['mode', 'workload', 'event'],
        observed=True
    )['count'].sum().unstack(fill_value=0)

    if 'LLC-load-misses' in grouped and 'LLC-loads' in grouped:
        grouped['llc_miss_rate'] = grouped['LLC-load-misses'] / grouped['LLC-loads'].replace(0, np.nan)

    return grouped.reset_index()


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('results_dir', type=Path)
    args = parser.parse_args()

    results_dir = args.results_dir

    df = load_combined(results_dir)

    # ⚡ Векторизированный NUMA penalty
    penalties = calc_numa_penalty_vectorized(df)

    # ⚡ Streaming spikes
    raw_iter = iter_raw_files(results_dir)
    spike_summary = detect_spikes_streaming(raw_iter)

    # ⚡ Tail ratio без копии
    calc_tail_ratio(df)

    print("Penalties:", len(penalties))
    print("Spikes:", spike_summary)


if __name__ == '__main__':
    main()