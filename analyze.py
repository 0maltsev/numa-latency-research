#!/usr/bin/env python3
import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Опциональные зависимости
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False
    print("[WARN] matplotlib/seaborn not installed. Plots will be skipped.", file=sys.stderr)


# ============================================================================
# Утилиты
# ============================================================================

def find_latest_results(base_path: Path = Path('.')) -> Optional[Path]:
    """Находит самую свежую директорию results_YYYYMMDD_HHMMSS"""
    candidates = list(base_path.glob('results_*'))
    if not candidates:
        return None
    # Сортируем по имени (формат даты позволяет лексикографическую сортировку)
    return sorted(candidates)[-1]


def load_combined(results_dir: Path) -> pd.DataFrame:
    """Загружает combined.csv с базовой валидацией"""
    path = results_dir / 'combined.csv'
    if not path.exists():
        raise FileNotFoundError(f"combined.csv not found in {results_dir}")
    df = pd.read_csv(path)
    required_cols = {'mode', 'workload', 'threads', 'p50', 'p99', 'p99_9'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"combined.csv missing columns: {missing}")
    return df


def load_raw_files(results_dir: Path, pattern: str = '*_raw.csv') -> dict[str, pd.DataFrame]:
    """Загружает все сырые файлы в словарь {filename: DataFrame}"""
    raw_data = {}
    for f in results_dir.glob(pattern):
        # Пропускаем заголовок "cycle_latencies"
        try:
            df = pd.read_csv(f, skiprows=1, header=None, names=['cycles'])
            raw_data[f.name] = df
        except Exception as e:
            print(f"[WARN] Failed to load {f.name}: {e}", file=sys.stderr)
    return raw_data


def load_perf_data(results_dir: Path) -> Optional[pd.DataFrame]:
    """Загружает perf_combined.csv если существует"""
    path = results_dir / 'perf_combined.csv'
    if not path.exists():
        return None
    return pd.read_csv(path)


# ============================================================================
# Метрики
# ============================================================================

def calc_numa_penalty(
        df: pd.DataFrame,
        workload: str,
        threads: int,
        size_mb: int,
        percentile: str = 'p99'
) -> Optional[dict]:
    """
    Считает разницу remote vs local для заданной конфигурации.
    Возвращает дикт с абсолютным и относительным пенальти.
    """
    local = df[
        (df.mode == 'local') &
        (df.workload == workload) &
        (df.threads == threads) &
        (df.size_mb == size_mb)
        ]
    remote = df[
        (df.mode == 'remote') &
        (df.workload == workload) &
        (df.threads == threads) &
        (df.size_mb == size_mb)
        ]
    if local.empty or remote.empty:
        return None

    local_val = local[percentile].values[0]
    remote_val = remote[percentile].values[0]
    abs_penalty = remote_val - local_val
    rel_penalty = (abs_penalty / local_val) * 100 if local_val > 0 else float('inf')

    return {
        'workload': workload,
        'threads': threads,
        'size_mb': size_mb,
        'percentile': percentile,
        'local_ns': local_val,
        'remote_ns': remote_val,
        'abs_penalty_ns': abs_penalty,
        'rel_penalty_pct': rel_penalty
    }


def calc_tail_ratio(df: pd.DataFrame, p_high: str = 'p99', p_low: str = 'p50') -> pd.DataFrame:
    """Добавляет колонку ratio = p_high / p_low как индикатор стабильности"""
    df = df.copy()
    df[f'{p_high}_{p_low}_ratio'] = df[p_high] / df[p_low]
    return df


def detect_spikes(
        raw_df: pd.DataFrame,
        threshold_factor: float = 3.0,
        min_cycles: int = 100
) -> dict:
    """
    Находит выбросы в сырых данных.
    Спайк = значение > median * threshold_factor AND > min_cycles.
    """
    if raw_df.empty:
        return {'count': 0, 'max': None, 'samples': []}

    median = raw_df['cycles'].median()
    threshold = max(median * threshold_factor, min_cycles)
    spikes = raw_df[raw_df['cycles'] > threshold]

    return {
        'count': len(spikes),
        'total_samples': len(raw_df),
        'spike_rate_pct': len(spikes) / len(raw_df) * 100 if len(raw_df) > 0 else 0,
        'median_cycles': median,
        'threshold_cycles': threshold,
        'max_cycles': int(raw_df['cycles'].max()),
        'samples': spikes['cycles'].head(10).tolist()  # первые 10 для инспекции
    }


def calc_llc_miss_rate(perf_df: pd.DataFrame) -> pd.DataFrame:
    """Считает LLC miss rate = misses / loads по режимам"""
    if perf_df is None or perf_df.empty:
        return pd.DataFrame()

    # Фильтруем только LLC-события
    llc = perf_df[perf_df['event'].str.contains('LLC', case=False, na=False)].copy()
    if llc.empty:
        return pd.DataFrame()

    # Pivot: строки = [mode, workload, ...], колонки = события
    pivot = llc.pivot_table(
        index=[c for c in llc.columns if c not in ['event', 'count']],
        columns='event',
        values='count',
        aggfunc='sum',
        fill_value=0
    ).reset_index()

    if 'LLC-load-misses' in pivot.columns and 'LLC-loads' in pivot.columns:
        pivot['llc_miss_rate'] = pivot['LLC-load-misses'] / pivot['LLC-loads'].replace(0, np.nan)

    return pivot


# ============================================================================
# Визуализация
# ============================================================================

def plot_ecdf_comparison(
        raw_files: dict[str, pd.DataFrame],
        results_dir: Path,
        workload_filter: Optional[str] = None,
        size_filter: Optional[int] = None,
        freq_mhz: float = 3000.0,
        max_percentile: float = 99.99
) -> list[Path]:
    """
    Строит ECDF-графики для сравнения хвостов распределения.
    Возвращает список сохранённых файлов.
    """
    if not HAS_PLOT:
        return []

    saved = []
    sns.set_style('whitegrid')

    # Группируем файлы по ключу (workload_size)
    groups = {}
    for fname, df in raw_files.items():
        # Парсим имя: local_ptr_chase_t1_32MB_raw.csv
        match = re.match(r'(\w+)_(\w+)_t(\d+)_(\d+)MB_raw\.csv', fname)
        if not match:
            continue
        mode, wl, threads, size = match.groups()
        if workload_filter and wl != workload_filter:
            continue
        if size_filter and int(size) != size_filter:
            continue
        key = f"{wl}_{size}MB"
        groups.setdefault(key, []).append((fname, mode, df))

    for key, entries in groups.items():
        if len(entries) < 2:
            continue  # нет чего сравнивать

        plt.figure(figsize=(10, 6))
        for fname, mode, df in entries:
            # Конвертация циклы → наносекунды
            ns = df['cycles'] * 1000.0 / freq_mhz
            sorted_lat = np.sort(ns.values)
            ecdf = np.arange(1, len(sorted_lat) + 1) / len(sorted_lat) * 100

            # Обрезаем до заданного перцентиля для читаемости
            cutoff = int(max_percentile / 100 * len(sorted_lat))
            plt.plot(
                sorted_lat[:cutoff], ecdf[:cutoff],
                label=mode, linewidth=1.5, alpha=0.9
            )

        plt.xscale('log')
        plt.xlabel('Latency (ns, log scale)')
        plt.ylabel('Cumulative %')
        plt.title(f'Tail latency ECDF: {key}')
        plt.grid(alpha=0.3, which='both')
        plt.legend(title='Mode')
        plt.tight_layout()

        out_path = results_dir / f'ecdf_{key}.png'
        plt.savefig(out_path, dpi=150)
        plt.close()
        saved.append(out_path)
        print(f"[PLOT] Saved: {out_path}")

    return saved


def plot_penalty_barchart(
        penalties: list[dict],
        results_dir: Path
) -> Optional[Path]:
    """Барчарт с абсолютным NUMA penalty по конфигурациям"""
    if not HAS_PLOT or not penalties:
        return None

    df = pd.DataFrame(penalties)
    df['label'] = df.apply(
        lambda r: f"{r['workload']} (t={r['threads']}, {r['size_mb']}MB)", axis=1
    )

    plt.figure(figsize=(12, 6))
    bars = plt.barh(df['label'], df['abs_penalty_ns'], color='steelblue')

    # Подписи со значениями
    for bar, val in zip(bars, df['abs_penalty_ns']):
        plt.text(val + 1, bar.get_y() + bar.get_height()/2, f'{val:.1f}ns', va='center', fontsize=9)

    plt.xlabel('NUMA Penalty (ns)')
    plt.title(f"Remote vs Local latency penalty ({penalties[0]['percentile']})")
    plt.gca().invert_yaxis()  # сверху — самые большие
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()

    out_path = results_dir / f"penalty_{penalties[0]['percentile']}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[PLOT] Saved: {out_path}")
    return out_path


# ============================================================================
# Отчёт
# ============================================================================

def generate_markdown_report(
        results_dir: Path,
        df: pd.DataFrame,
        penalties: list[dict],
        spike_summary: dict,
        llc_summary: Optional[pd.DataFrame],
        plots: list[Path],
        output_name: str = 'report.md'
) -> Path:
    """Генерирует Markdown-отчёт с основными метриками"""
    out_path = results_dir / output_name

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(f"# Benchmark Report\n")
        f.write(f"**Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Results dir**: `{results_dir.name}`\n\n")

        # Конфигурация
        f.write("## Configuration\n")
        f.write(f"- Workloads tested: `{', '.join(df['workload'].unique())}`\n")
        f.write(f"- Thread counts: `{sorted(df['threads'].unique())}`\n")
        f.write(f"- Buffer sizes: `{sorted(df['size_mb'].unique())} MB`\n\n")

        # Сводная таблица
        f.write("## Results Summary\n")
        cols = ['mode', 'workload', 'threads', 'size_mb', 'p50', 'p90', 'p99', 'p99_9', 'p999']
        display_cols = [c for c in cols if c in df.columns]
        f.write(df[display_cols].round(2).to_markdown(index=False))
        f.write("\n\n")

        # Tail ratio
        df_with_ratio = calc_tail_ratio(df)
        f.write("## Stability: p99/p50 Ratio\n")
        f.write("*Lower = more predictable. >3.0x may indicate instability.*\n\n")
        ratio_cols = ['mode', 'workload', 'threads', 'size_mb', 'p99_p50_ratio']
        f.write(df_with_ratio[ratio_cols].round(3).to_markdown(index=False))
        f.write("\n\n")

        # NUMA penalty
        if penalties:
            f.write("## NUMA Penalty (Remote vs Local)\n")
            f.write("*Measured at p99 latency*\n\n")
            for p in penalties:
                f.write(f"- **{p['workload']}** (t={p['threads']}, {p['size_mb']}MB): ")
                f.write(f"+{p['abs_penalty_ns']:.1f} ns (+{p['rel_penalty_pct']:.1f}%)\n")
            f.write("\n")

        # Spikes
        f.write("## Spike Analysis\n")
        if spike_summary:
            f.write(f"- Total raw samples analyzed: `{spike_summary.get('total_samples', 0):,}`\n")
            f.write(f"- Spikes detected: `{spike_summary.get('count', 0)}` ")
            f.write(f"({spike_summary.get('spike_rate_pct', 0):.4f}%)\n")
            f.write(f"- Median latency: `{spike_summary.get('median_cycles', 0):.0f}` cycles\n")
            f.write(f"- Max observed: `{spike_summary.get('max_cycles', 0):,}` cycles\n")
            if spike_summary.get('samples'):
                f.write(f"- Sample spike values (cycles): `{spike_summary['samples']}`\n")
        else:
            f.write("*No raw data available for spike analysis*\n")
        f.write("\n")

        # LLC miss rate
        if llc_summary is not None and not llc_summary.empty and 'llc_miss_rate' in llc_summary.columns:
            f.write("## LLC Cache Analysis (perf)\n")
            f.write("*Miss rate = LLC-load-misses / LLC-loads*\n\n")
            display = llc_summary[['mode', 'workload', 'llc_miss_rate']].copy()
            display['llc_miss_rate'] = (display['llc_miss_rate'] * 100).round(2).astype(str) + '%'
            f.write(display.to_markdown(index=False))
            f.write("\n\n")

        # Графики
        if plots:
            f.write("## Plots\n")
            for p in plots:
                f.write(f"![{p.name}]({p.name})\n\n")

        # Чеклист интерпретации
        f.write("## Interpretation Checklist\n")
        f.write("| Metric | Normal | Warning |\n")
        f.write("|----------|--------|---------|\n")
        f.write("| p99/p50 ratio | < 2.0x | > 3.0x → instability |\n")
        f.write("| NUMA penalty (p99) | 30–80 ns (Intel UPI) | > 150 ns → check numa_balancing |\n")
        f.write("| LLC miss rate (sequential) | < 10% | > 30% → prefetch issue |\n")
        f.write("| LLC miss rate (ptr_chase) | > 85% | < 70% → unexpected locality |\n")
        f.write("| Spike rate | < 0.01% | > 0.1% → OS jitter / interrupts |\n")

    print(f"[REPORT] Saved: {out_path}")
    return out_path


def export_json_summary(
        df: pd.DataFrame,
        penalties: list[dict],
        spike_summary: dict,
        output_path: Path
) -> Path:
    """Экспорт ключевых метрик в JSON для интеграции (Grafana, CI)"""
    summary = {
        'generated_at': pd.Timestamp.now().isoformat(),
        'configurations': df.to_dict(orient='records'),
        'numa_penalties': penalties,
        'spike_analysis': spike_summary,
        'metadata': {
            'total_runs': len(df),
            'workloads': list(df['workload'].unique()),
            'modes': list(df['mode'].unique())
        }
    }
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"[JSON] Saved: {output_path}")
    return output_path


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Анализ результатов numa-latency-bench',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'results_dir', nargs='?', type=Path, default=None,
        help='Директория с результатами (по умолчанию — последний запуск)'
    )
    parser.add_argument(
        '--freq-mhz', type=float, default=3000.0,
        help='Частота TSC в MHz для конвертации циклов → наносекунды'
    )
    parser.add_argument(
        '--workload', type=str, default=None,
        help='Фильтр по workload для графиков (ptr_chase, sequential, ...)'
    )
    parser.add_argument(
        '--size-mb', type=int, default=None,
        help='Фильтр по размеру буфера для графиков'
    )
    parser.add_argument(
        '--spike-threshold', type=float, default=3.0,
        help='Множитель медианы для детекции спайков в сырых данных'
    )
    parser.add_argument(
        '--no-plots', action='store_true',
        help='Не строить графики (даже если matplotlib доступен)'
    )
    parser.add_argument(
        '--perf', action='store_true',
        help='Включить анализ perf-счетчиков (если данные есть)'
    )
    parser.add_argument(
        '--json', type=Path, default=None,
        help='Экспортировать сводку в JSON по указанному пути'
    )
    parser.add_argument(
        '--quiet', action='store_true',
        help='Минимальный вывод (только ошибки)'
    )

    args = parser.parse_args()

    # Найти директорию с результатами
    results_dir = args.results_dir or find_latest_results()
    if not results_dir:
        print("ERROR: No results directory found. Run the benchmark first.", file=sys.stderr)
        sys.exit(1)

    if not args.quiet:
        print(f"[INFO] Analyzing: {results_dir}")

    try:
        # Загрузка данных
        df = load_combined(results_dir)
        raw_files = load_raw_files(results_dir)
        perf_df = load_perf_data(results_dir) if args.perf else None

        if not args.quiet:
            print(f"[INFO] Loaded {len(df)} aggregated runs, {len(raw_files)} raw files")
            if perf_df is not None:
                print(f"[INFO] Loaded {len(perf_df)} perf counter records")

        # Расчёт метрик
        penalties = []
        for wl in df['workload'].unique():
            for t in df['threads'].unique():
                for sz in df['size_mb'].unique():
                    p = calc_numa_penalty(df, wl, t, sz)
                    if p:
                        penalties.append(p)

        # Анализ спайков (агрегируем по всем сырым файлам)
        all_raw = pd.concat(raw_files.values(), ignore_index=True) if raw_files else pd.DataFrame()
        spike_summary = detect_spikes(all_raw, args.spike_threshold) if not all_raw.empty else {}

        # LLC miss rate
        llc_summary = calc_llc_miss_rate(perf_df) if args.perf and perf_df is not None else None

        # Графики
        plots = []
        if HAS_PLOT and not args.no_plots:
            plots += plot_ecdf_comparison(
                raw_files, results_dir,
                workload_filter=args.workload,
                size_filter=args.size_mb,
                freq_mhz=args.freq_mhz
            )
            if penalties:
                plot_penalty_barchart(penalties, results_dir)

        # Отчёт
        generate_markdown_report(
            results_dir, df, penalties, spike_summary, llc_summary, plots
        )

        # JSON экспорт
        if args.json:
            export_json_summary(df, penalties, spike_summary, args.json)

        # Быстрый вывод в консоль
        if not args.quiet:
            print("\n=== Quick Summary ===")
            print(df[['mode', 'workload', 'threads', 'p50', 'p99', 'p99_9']].head(10).to_string(index=False))
            if penalties:
                print(f"\nNUMA penalty samples: {len(penalties)}")
                for p in penalties[:3]:
                    print(f"  {p['workload']}: +{p['abs_penalty_ns']:.1f}ns ({p['rel_penalty_pct']:.1f}%)")

        if not args.quiet:
            print(f"\n[OK] Analysis complete. Check {results_dir}/report.md")

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        if not args.quiet:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()