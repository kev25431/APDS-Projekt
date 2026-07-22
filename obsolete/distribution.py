from pathlib import Path
from collections import Counter
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import optimize, stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = Path(
    "/Users/kevinlin/Applied-Data-Science-Projekt/"
    "Data_project_app/downloads/wvv-pjs-2026/full_api_data"
)
OUTPUT_DIR = Path("distribution_results")

VALUE_COLUMN = "passenger_boarding_measured"
STOP_COLUMN = "station_short"
QUALITY_COLUMN = "quality_factor"
MAX_QUALITY_FACTOR = 150
TOP_N_STOPS = 8

def load_boarding_frequencies(data_dir: Path) -> tuple[pd.Series, pd.Series, dict]:
    parquet_files = sorted(data_dir.glob("*.parquet"))
    requested_columns = [VALUE_COLUMN, STOP_COLUMN, QUALITY_COLUMN]

    if not parquet_files:
        raise FileNotFoundError(f"Keine Parquet-Dateien gefunden in: {data_dir}")
    
    overall_counter = Counter()
    stop_counter = Counter()
    meta = {
        "files": 0,
        "rows_loaded": 0,
        "rows_used": 0,
        "negative_values_removed": 0,
        "example": None,
    }
    
    for file in parquet_files:
        available_columns = set(pq.read_schema(file).names)
        columns_to_read = [
            column for column in requested_columns if column in available_columns
        ]

        if VALUE_COLUMN not in available_columns or STOP_COLUMN not in available_columns:
            continue

        df = pd.read_parquet(file, columns=columns_to_read)
        meta["files"] += 1
        meta["rows_loaded"] += len(df)

        if QUALITY_COLUMN not in df.columns:
            df[QUALITY_COLUMN] = pd.NA

        if meta["example"] is None and not df.empty:
            meta["example"] = df.head()

        if QUALITY_COLUMN in df.columns:
            df = df[df[QUALITY_COLUMN].isna() | (df[QUALITY_COLUMN] <= MAX_QUALITY_FACTOR)]

        df = df[[STOP_COLUMN, VALUE_COLUMN]].dropna()
        df[VALUE_COLUMN] = pd.to_numeric(df[VALUE_COLUMN], errors="coerce")
        df = df.dropna(subset=[VALUE_COLUMN])

        negative_values = (df[VALUE_COLUMN] < 0).sum()
        meta["negative_values_removed"] += int(negative_values)
        df = df[df[VALUE_COLUMN] >= 0]

        df["boardings"] = df[VALUE_COLUMN].round().astype(int)
        df["stop"] = df[STOP_COLUMN].astype(str)
        meta["rows_used"] += len(df)

        overall_counter.update(df["boardings"].to_numpy())
        stop_counter.update(zip(df["stop"], df["boardings"]))

    if not overall_counter:
        raise ValueError("Keine gültigen Boarding-Daten gefunden.")

    overall_freq = pd.Series(overall_counter).sort_index().astype(int)
    stop_index = pd.MultiIndex.from_tuples(
        stop_counter.keys(),
        names=["stop", "boardings"],
    )
    stop_freq = pd.Series(stop_counter.values(), index=stop_index).sort_index().astype(int)

    return overall_freq, stop_freq, meta


def series_to_arrays(freq: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    values = freq.index.to_numpy(dtype=int)
    frequencies = freq.to_numpy(dtype=float)
    return values, frequencies


def weighted_mean(values: np.ndarray, frequencies: np.ndarray) -> float:
    return np.average(values, weights=frequencies)


def weighted_variance(values: np.ndarray, frequencies: np.ndarray) -> float:
    mean = weighted_mean(values, frequencies)
    return np.average((values - mean) ** 2, weights=frequencies)


def mle_poisson(values: np.ndarray, frequencies: np.ndarray) -> dict:
    lam = weighted_mean(values, frequencies)

    return {
        "distribution": "poisson",
        "params": {"lambda": lam},
        "log_likelihood": (stats.poisson.logpmf(values, lam) * frequencies).sum(),
        "pmf": lambda x: stats.poisson.pmf(x, lam),
        "n_params": 1,
    }


def mle_geometric(values: np.ndarray, frequencies: np.ndarray) -> dict:
    shifted_values = values + 1
    p = 1 / weighted_mean(shifted_values, frequencies)

    return {
        "distribution": "geometric",
        "params": {"p": p},
        "log_likelihood": (stats.geom.logpmf(shifted_values, p) * frequencies).sum(),
        "pmf": lambda x: stats.geom.pmf(x + 1, p),
        "n_params": 1,
    }


def mle_negative_binomial(values: np.ndarray, frequencies: np.ndarray) -> dict:
    mean = weighted_mean(values, frequencies)
    variance = weighted_variance(values, frequencies)

    if variance <= mean:
        r_start = 1_000.0
        p_start = r_start / (r_start + mean)
    else:
        r_start = mean**2 / (variance - mean)
        p_start = r_start / (r_start + mean)

    p_start = np.clip(p_start, 1e-6, 1 - 1e-6)

    def unpack(theta: np.ndarray) -> tuple[float, float]:
        r = np.exp(theta[0])
        p = 1 / (1 + np.exp(-theta[1]))
        return r, p

    def negative_log_likelihood(theta: np.ndarray) -> float:
        r, p = unpack(theta)
        return -(stats.nbinom.logpmf(values, r, p) * frequencies).sum()

    start = np.array([np.log(max(r_start, 1e-6)), np.log(p_start / (1 - p_start))])
    result = optimize.minimize(negative_log_likelihood, start, method="Nelder-Mead")
    r, p = unpack(result.x)

    return {
        "distribution": "negative_binomial",
        "params": {"r": r, "p": p},
        "log_likelihood": -result.fun,
        "pmf": lambda x: stats.nbinom.pmf(x, r, p),
        "n_params": 2,
    }


def chi_square_gof(values: np.ndarray, frequencies: np.ndarray, fit: dict) -> dict:
    max_value = int(values.max())
    support = np.arange(max_value + 1)
    observed_full = np.zeros(max_value + 1)
    observed_full[values] = frequencies
    expected_full = fit["pmf"](support) * frequencies.sum()

    observed_bins = []
    expected_bins = []
    running_observed = 0.0
    running_expected = 0.0

    for obs, exp in zip(observed_full, expected_full):
        running_observed += obs
        running_expected += exp

        if running_expected >= 5:
            observed_bins.append(running_observed)
            expected_bins.append(running_expected)
            running_observed = 0.0
            running_expected = 0.0

    tail_observed = running_observed
    tail_expected = running_expected + frequencies.sum() * (1 - fit["pmf"](support).sum())

    if tail_observed > 0 or tail_expected > 0:
        if expected_bins:
            observed_bins[-1] += tail_observed
            expected_bins[-1] += tail_expected
        else:
            observed_bins.append(tail_observed)
            expected_bins.append(tail_expected)

    observed_bins = np.array(observed_bins)
    expected_bins = np.array(expected_bins)
    expected_bins *= observed_bins.sum() / expected_bins.sum()

    chi2_stat = ((observed_bins - expected_bins) ** 2 / expected_bins).sum()
    dof = len(observed_bins) - 1 - fit["n_params"]
    p_value = stats.chi2.sf(chi2_stat, dof) if dof > 0 else np.nan

    return {
        "chi2": chi2_stat,
        "dof": dof,
        "p_value": p_value,
        "bins": len(observed_bins),
    }

def fit_distributions(freq: pd.Series) -> pd.DataFrame:
    values, frequencies = series_to_arrays(freq)
    sample_size = frequencies.sum()

    fits = [
        mle_poisson(values, frequencies),
        mle_geometric(values, frequencies),
        mle_negative_binomial(values, frequencies),
    ]

    rows = []

    for fit in fits:
        chi_square = chi_square_gof(values, frequencies, fit)
        aic = 2 * fit["n_params"] - 2 * fit["log_likelihood"]
        bic = np.log(sample_size) * fit["n_params"] - 2 * fit["log_likelihood"]

        rows.append({
            "distribution": fit["distribution"],
            "params": fit["params"],
            "log_likelihood": fit["log_likelihood"],
            "aic": aic,
            "bic": bic,
            "chi2": chi_square["chi2"],
            "dof": chi_square["dof"],
            "p_value": chi_square["p_value"],
            "chi2_bins": chi_square["bins"],
        })

    return pd.DataFrame(rows).sort_values(["aic", "chi2"])

def describe_frequency(freq: pd.Series) -> dict:
    values, frequencies = series_to_arrays(freq)
    sample_size = frequencies.sum()
    mean = weighted_mean(values, frequencies)
    std = np.sqrt(weighted_variance(values, frequencies))
    zero_share = freq.get(0, 0) / sample_size
    
    return {
        "sample_size": int(sample_size),
        "min": int(values.min()),
        "max": int(values.max()),
        "mean": mean,
        "std": std,
        "zero_share": zero_share,
    }

def pmf_for_result(x: np.ndarray, row: pd.Series) -> np.ndarray:
    params = row["params"]

    if row["distribution"] == "poisson":
        return stats.poisson.pmf(x, params["lambda"])
    if row["distribution"] == "geometric":
        return stats.geom.pmf(x + 1, params["p"])
    if row["distribution"] == "negative_binomial":
        return stats.nbinom.pmf(x, params["r"], params["p"])

    raise ValueError(f"Unbekannte Distribution: {row['distribution']}")


def plot_histogram_with_fits(freq: pd.Series, fit_results: pd.DataFrame, path: Path):
    values, frequencies = series_to_arrays(freq)
    cumulative = np.cumsum(frequencies) / frequencies.sum()
    max_plot_value = int(values[np.searchsorted(cumulative, 0.995)])
    x = np.arange(0, max_plot_value + 1)

    observed = np.zeros(max_plot_value + 1)
    mask = values <= max_plot_value
    observed[values[mask]] = frequencies[mask]
    observed = observed / frequencies.sum()

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.bar(x, observed, alpha=0.55, label="Observed")

    for _, row in fit_results.iterrows():
        ax.plot(
            x,
            pmf_for_result(x, row),
            marker="o",
            markersize=3,
            linewidth=1.4,
            label=row["distribution"],
        )

    ax.set_title("Distribution Fit: Personen pro Haltestellenereignis")
    ax.set_xlabel("Einsteigende Personen")
    ax.set_ylabel("Relative Häufigkeit / Wahrscheinlichkeit")
    ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def analyze_overall_distribution(overall_freq: pd.Series) -> pd.DataFrame:
    stats_summary = describe_frequency(overall_freq)
    fit_results = fit_distributions(overall_freq)

    print("\nGesamtverteilung: Personen pro Haltestellenereignis")
    print(
        f"Sample size: {stats_summary['sample_size']:,}"
        f" | min = {stats_summary['min']}"
        f" | max = {stats_summary['max']}"
        f" | mean = {stats_summary['mean']:.2f}"
        f" | std = {stats_summary['std']:.2f}"
        f" | zero_share = {stats_summary['zero_share']:.2%}"
    )
    print(fit_results.to_string(index=False))

    fit_results.to_csv(OUTPUT_DIR / "overall_fit_results.csv", index=False)
    overall_freq.to_csv(OUTPUT_DIR / "overall_boarding_frequencies.csv")
    plot_histogram_with_fits(
        overall_freq,
        fit_results,
        OUTPUT_DIR / "overall_histogram_fits.png",
    )

    return fit_results


def analyze_top_stops(stop_freq: pd.Series) -> pd.DataFrame:
    stop_sample_sizes = stop_freq.groupby(level="stop").sum().sort_values(ascending=False)
    top_stops = stop_sample_sizes.head(TOP_N_STOPS).index
    rows = []

    for stop in top_stops:
        freq = stop_freq.loc[stop].sort_index()
        fit_results = fit_distributions(freq)
        summary = describe_frequency(freq)
        best = fit_results.iloc[0]

        rows.append(
            {
                "stop": stop,
                "sample_size": summary["sample_size"],
                "mean": summary["mean"],
                "std": summary["std"],
                "zero_share": summary["zero_share"],
                "best_distribution": best["distribution"],
                "best_params": best["params"],
                "best_aic": best["aic"],
                "best_chi2": best["chi2"],
                "best_p_value": best["p_value"],
            }
        )

        safe_stop_name = str(stop).replace("/", "_").replace(" ", "_")
        plot_histogram_with_fits(
            freq,
            fit_results,
            OUTPUT_DIR / f"histogram_fits_stop_{safe_stop_name}.png",
        )

    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT_DIR / "top_stop_fit_summary.csv", index=False)

    print("\nBeste Fits für die häufigsten Haltestellen")
    print(result.to_string(index=False))

    return result


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    overall_freq, stop_freq, meta = load_boarding_frequencies(DATA_DIR)

    print(meta["example"])
    print(
        f"\nGeladene Dateien: {meta['files']}"
        f" | geladene Zeilen: {meta['rows_loaded']:,}"
        f" | genutzte Zeilen: {meta['rows_used']:,}"
        f" | entfernte negative Werte: {meta['negative_values_removed']:,}"
    )

    analyze_overall_distribution(overall_freq)
    analyze_top_stops(stop_freq)

    print(f"\nErgebnisse gespeichert in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()