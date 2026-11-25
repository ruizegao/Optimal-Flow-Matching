#!/usr/bin/env python3
"""Compare cluster training profiles and save CSV tables and PNG figures.

Pass a training output directory, individual run directories, or profile files.
By default, search amortized_tests, cc_experiments/runs, then cc_experiments/results.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS = HERE.parent / "amortized_tests"
METHODS = {
    "plain": ("Plain L-BFGS", "#3C4856"),
    "wolfe": ("Full Wolfe", "#1766A2"),
    "retry": ("Wolfe retry", "#B55323"),
    "mlp": ("MLP", "#08785D"),
    "mlp-retry": ("MLP + retry", "#724A9A"),
    "restricted": ("Restricted time", "#00818A"),
    "restricted-wolfe": ("Restricted + Wolfe", "#AC4269"),
    "cap1000": ("Plain cap 1,000", "#847332"),
}
DESCRIPTIONS = {
    "plain": "Uses capped L-BFGS without a line search.",
    "wolfe": "Uses Strong Wolfe line search on every batch.",
    "retry": "Retries extreme inverse outputs with Strong Wolfe.",
    "mlp": "Warm-starts plain L-BFGS with a learned MLP.",
    "mlp-retry": "Combines an MLP warm start with a Strong Wolfe retry.",
    "restricted": "Limits extreme t values and uses plain L-BFGS.",
    "restricted-wolfe": "Limits extreme t values and uses Strong Wolfe on every batch.",
    "cap1000": "Allows up to 1,000 outer steps of plain L-BFGS.",
}
NUMERIC_FIELDS = (
    "true_loss", "t_total", "t_inverse", "t_ihv", "t_amort", "inverse_steps",
    "inverse_func_evals", "inverse_max_residual", "inverse_max_displacement",
    "cap_hit", "wolfe_retried", "heldout_sliced_w2", "cuda_peak_reserved_mib",
)
SPIKE_FIELDS = [
    "source", "method", "seed", "iteration", "true_loss", "previous_median",
    "spike_ratio", "in_summary", "iteration_ms", "inverse_steps",
    "inverse_func_evals", "inverse_max_residual", "wolfe_retried",
]


@dataclass
class Run:
    path: Path
    method: str
    seed: str
    data: pd.DataFrame


def find_profiles(inputs):
    paths = set()
    if not inputs:
        for directory in (DEFAULT_RESULTS, HERE / "runs", HERE / "results"):
            paths.update(directory.rglob("profile.csv"))
            if paths:
                break
    for path in inputs:
        if path.is_file():
            paths.add(path)
        elif path.is_dir():
            profiles = list(path.rglob("profile.csv"))
            if not profiles:
                raise ValueError(f"No profile.csv files under {path}")
            paths.update(profiles)
        else:
            raise ValueError(f"Input does not exist: {path}")
    if not paths:
        raise ValueError("No profile.csv files found. Pass the directory containing your exported runs.")
    return sorted({path.resolve() for path in paths})


def load_run(path, spike_factor):
    data = pd.read_csv(path)
    required = {"iteration", "true_loss", "t_total", "inverse_steps", "inverse_func_evals"}
    if missing := required.difference(data.columns):
        raise ValueError(f"{path} is missing cluster columns {', '.join(sorted(missing))}")
    if data.empty:
        raise ValueError(f"{path} has no logged iterations")
    iteration = pd.to_numeric(data.pop("iteration"), errors="coerce")
    if (not np.isfinite(iteration).all() or (iteration < 0).any()
            or (iteration % 1 != 0).any() or iteration.duplicated().any()):
        raise ValueError(f"{path} needs unique, nonnegative integer iteration indices")
    data.index = pd.Index(iteration.astype(int), name="iteration")
    data = data.sort_index()

    for key in NUMERIC_FIELDS:
        if key in data:
            column = data[key]
            if key in {"cap_hit", "wolfe_retried"}:
                column = column.astype(str).str.lower().replace({"true": "1", "false": "0"})
            data[key] = pd.to_numeric(column, errors="coerce").replace([np.inf, -np.inf], np.nan)
        else:
            data[key] = np.nan
    if (data["t_total"] < 0).any():
        raise ValueError(f"{path} contains negative iteration times")

    root = path.parent
    seed = root.name.removeprefix("seed_") if root.name.startswith("seed_") else ""
    method = root.parent.name if seed else (root.name if path.name == "profile.csv" else path.stem)

    data["training_seconds"] = data["t_total"].cumsum(skipna=False)
    data["previous_median"] = data["true_loss"].shift(1).rolling(100, min_periods=20).median()
    data["spike_ratio"] = data["true_loss"] / data["previous_median"].where(data["previous_median"] > 0)
    data["spike"] = data["spike_ratio"] >= spike_factor
    return Run(path, method, seed, data)


def common_iterations(runs, count):
    shared = runs[0].data.index
    for run in runs[1:]:
        shared = shared.intersection(run.data.index)
    if count is not None:
        if count > len(shared):
            raise ValueError(f"Requested {count} shared iterations, but only {len(shared)} are available")
        shared = shared[:count]
    if shared.empty:
        raise ValueError("The input profiles have no shared iterations")
    return shared


def tables(runs, shared, window):
    summaries, events = [], []
    for run in runs:
        source = str(run.path.relative_to(HERE.parent) if run.path.is_relative_to(HERE.parent) else run.path)
        d = run.data
        cut = d if shared is None else d.loc[shared]
        evaluations = cut["heldout_sliced_w2"].dropna()
        at = evaluations.index[-1] if len(evaluations) else None
        spikes = d.loc[d["spike"]]
        available_vram = cut["cuda_peak_reserved_mib"].where(cut["cuda_peak_reserved_mib"] > 0)
        time = cut["t_total"].sum(min_count=len(cut))
        inv_time = cut["t_inverse"].sum(min_count=len(cut))
        summaries.append({
            "source": source, "method": run.method, "seed": run.seed,
            "logged_iterations": len(d), "summarized_iterations": len(cut),
            "first_iteration": cut.index[0], "last_iteration": cut.index[-1],
            "training_seconds": time, "mean_iteration_ms": cut["t_total"].mean() * 1000,
            "p95_iteration_ms": cut["t_total"].quantile(.95) * 1000,
            "mean_inverse_ms": cut["t_inverse"].mean() * 1000,
            "inverse_time_share": inv_time / time if time > 0 else np.nan,
            "mean_inverse_steps": cut["inverse_steps"].mean(),
            "mean_inverse_func_evals": cut["inverse_func_evals"].mean(),
            "p95_inverse_max_residual": cut["inverse_max_residual"].quantile(.95),
            "cap_hit_fraction": cut["cap_hit"].mean(),
            "wolfe_retry_fraction": cut["wolfe_retried"].mean(),
            "mean_amort_ms": cut["t_amort"].mean() * 1000,
            "final_true_loss": cut["true_loss"].iloc[-1],
            "tail_median_true_loss": cut["true_loss"].tail(window).median(),
            "peak_true_loss": cut["true_loss"].max(),
            "spikes": int(cut["spike"].sum()), "spikes_all_logged": len(spikes),
            "first_spike_iteration": spikes.index.min() if len(spikes) else np.nan,
            "heldout_eval_iteration": at if at is not None and pd.notna(d.at[at, "heldout_sliced_w2"]) else np.nan,
            "heldout_sliced_w2": d.at[at, "heldout_sliced_w2"] if at is not None else np.nan,
            "peak_cuda_reserved_mib": available_vram.max(),
        })
        for i, row in spikes.iterrows():
            events.append({
                "source": source, "method": run.method, "seed": run.seed, "iteration": i,
                "true_loss": row["true_loss"], "previous_median": row["previous_median"],
                "spike_ratio": row["spike_ratio"], "in_summary": shared is None or i in shared,
                "iteration_ms": row["t_total"] * 1000, "inverse_steps": row["inverse_steps"],
                "inverse_func_evals": row["inverse_func_evals"], "inverse_max_residual": row["inverse_max_residual"],
                "wolfe_retried": row["wolfe_retried"],
            })

    records = pd.DataFrame(summaries)
    grouped = records.groupby("method", sort=False)
    measures = records.loc[:, "training_seconds":].columns
    methods = grouped[measures].median().add_prefix("median_")
    methods.insert(0, "runs", grouped.size())
    methods.insert(1, "median_summarized_iterations", grouped["summarized_iterations"].median())
    methods.insert(1, "min_logged_iterations", grouped["logged_iterations"].min())
    methods.insert(2, "max_logged_iterations", grouped["logged_iterations"].max())
    return records, methods.reset_index(), pd.DataFrame(events, columns=SPIKE_FIELDS)


def presentation(groups):
    labels, colors = {}, {}
    for i, method in enumerate(groups):
        labels[method], colors[method] = METHODS.get(method, (method, plt.get_cmap("tab20")(i % 20)))
    return labels, colors


def training_figure(groups, labels, colors, window, spike_factor, output):
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5))
    metrics = (
        ("true_loss", "True loss", 1., "median", True),
        ("t_total", "Iteration time (ms)", 1000., "mean", True),
        ("inverse_steps", "L-BFGS outer steps", 1., "mean", False),
        ("inverse_max_residual", "Inverse max residual (rolling p95)", 1., "quantile", True),
    )
    for ax, (key, title, scale, stat, log) in zip(axes.flat, metrics):
        finite_curves = []
        for group, runs in groups.items():
            curves = []
            for run in runs:
                stream = run.data[key] * scale
                if not stream.notna().any():
                    continue
                roll = stream.rolling(window, min_periods=1)
                if stat == "quantile":
                    curve = roll.quantile(.95)
                else:
                    curve = getattr(roll, stat)()
                curves.append(curve)
                if key == "true_loss":
                    spikes = stream[run.data["spike"] & (stream > 0)]
                    ax.scatter(spikes.index, spikes, c=[colors[group]], marker="x",
                               s=29, linewidths=1.2, zorder=4)
            if curves:
                center = pd.concat(curves, axis=1).median(axis=1)
                if log:
                    center = center.where(center > 0)
                finite_curves.extend(center.dropna().tolist())
                ax.plot(center.index, center, color=colors[group], label=labels[group], lw=1.45)
        if not finite_curves:
            ax.text(.5, .52, "This metric was unavailable in the profiles", ha="center", va="center",
                    transform=ax.transAxes, color="#52646F")
        elif log:
            ax.set_yscale("log")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel("Training iteration")
        ax.grid(axis="y", alpha=.25)
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("OFM training and inverse stability", fontsize=19, fontweight="bold", y=.97)
    if names:
        fig.legend(handles, names, ncols=min(4, len(names)), loc="upper center",
                   bbox_to_anchor=(.5, .916), frameon=False, fontsize=10)
    fig.text(.095, .037, f"Full logged histories. Lines show {window}-batch rolling summaries and the median across runs. "
             f"Crosses mark losses ≥ {spike_factor:g}× the preceding median.", fontsize=9.2, color="#52646F")
    fig.subplots_adjust(left=.095, right=.97, bottom=.11, top=.80, wspace=.26, hspace=.36)
    fig.savefig(output, dpi=175, facecolor="white")
    plt.close(fig)


def distribution(ax, groups, labels, colors, shared, field, title):
    values, names, fills = [], [], []
    for method, runs in groups.items():
        observations = pd.concat([(run.data[field] if shared is None else run.data.loc[shared, field]) for run in runs]).dropna()
        if len(observations):
            values.append(observations)
            names.append(labels[method])
            fills.append(colors[method])
    if values:
        boxes = ax.boxplot(values, vert=False, patch_artist=True, showmeans=True,
                           meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 4},
                           medianprops={"color": "black"},
                           flierprops={"marker": ".", "markersize": 3, "alpha": .35})
        for box, color in zip(boxes["boxes"], fills):
            box.set_facecolor(color)
            box.set_alpha(.55)
        ax.set_yticks(range(1, len(names) + 1), names)
        ax.invert_yaxis()
        if min(v.min() for v in values) > 0:
            ax.set_xscale("log")
            ax.set_xlabel("Log scale")
    else:
        ax.text(.5, .5, "No logged evaluations", ha="center", transform=ax.transAxes)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(axis="x", alpha=.25)


def efficiency_figure(groups, methods, labels, colors, shared, output):
    fig = plt.figure(figsize=(14, max(19, 13 + .85 * len(groups))))
    grid = fig.add_gridspec(6, 2, height_ratios=[1.25, 1, 1, 1, 1.9, 1.9])
    axes = [fig.add_subplot(grid[r, c]) for r in range(1, 4) for c in range(2)]
    has_quality = any(r.data["heldout_sliced_w2"].gt(0).any() for runs in groups.values() for r in runs)
    ordered = methods.set_index("method").loc[list(groups)].copy()
    ordered["retry_percent"] = 100 * ordered["median_wolfe_retry_fraction"]
    bars = [
        ("median_mean_inverse_func_evals", "Mean inverse evaluations per batch", ".2f"),
        ("median_spikes", "Loss spikes in summarized iterations", ".1f"),
        ("retry_percent", "Batches retried with Wolfe (%)", ".2f"),
        ("median_mean_amort_ms", "Mean MLP overhead per batch (ms)", ".3g"),
    ]
    for ax, (column, title, fmt) in zip(axes, bars):
        observed = ordered[column].dropna()
        ax.barh(range(len(observed)), observed, color=[colors[key] for key in observed.index], height=.6)
        ax.set_yticks(range(len(observed)), [labels[key] for key in observed.index])
        ax.invert_yaxis()
        if len(observed):
            ax.set_xlim(0, max(1e-12, observed.max() * 1.23) if observed.max() else 1)
            for y, value in enumerate(observed):
                ax.annotate(format(value, fmt), (value, y), xytext=(4, 0),
                            textcoords="offset points", va="center", fontsize=9)
        else:
            ax.text(.5, .52, "This metric was unavailable in the profiles", ha="center", va="center",
                    transform=ax.transAxes, color="#52646F")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(axis="x", alpha=.25)
        ax.set_axisbelow(True)

    distribution(axes[4], groups, labels, colors, shared, "true_loss", "Training true loss by batch")
    distribution(axes[5], groups, labels, colors, shared, "heldout_sliced_w2", "Held-out sliced W₂ by checkpoint")

    summary_ax = fig.add_subplot(grid[0, :])
    summary_ax.axis("off")

    def number(value, spec):
        return format(value, spec) if pd.notna(value) else "—"

    table = summary_ax.table(
        cellText=[
            [labels[key], number(row["median_summarized_iterations"], ",.0f"),
             number(row["median_training_seconds"], ",.0f"),
             number(row["median_heldout_sliced_w2"], ".4g"),
             number(row["median_final_true_loss"], ".4g"), DESCRIPTIONS.get(key, "")]
            for key, row in ordered.iterrows()
        ],
        colLabels=["Method", "Iterations", "Train time (s)", "Final held-out W₂", "Final training loss", "Description"],
        colWidths=[.16, .09, .11, .14, .13, .37], cellLoc="left", colLoc="left",
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    method_keys = list(ordered.index)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        cell.set_facecolor("#E3EBEF" if r == 0 else ("#F1F5F7" if r % 2 else "white"))
        if r == 0:
            cell.set_text_props(weight="bold")
        elif c == 0:
            cell.set_text_props(color=colors[method_keys[r - 1]], weight="bold")

    runners_up = []
    for column in range(1, 5):
        values = {}
        for row in range(1, len(method_keys) + 1):
            cell = table[row, column]
            displayed = cell.get_text().get_text()
            if displayed != "—":
                values[cell.get_text()] = float(displayed.replace(",", ""))
        ranks = sorted(set(values.values()), reverse=column == 1)
        for label, value in values.items():
            if value == ranks[0]:
                label.set_weight("bold")
            elif len(ranks) > 1 and value == ranks[1]:
                runners_up.append(label)

    def quality_curve(ax, by_iteration):
        for key, runs in groups.items():
            label = labels[key]
            for run in runs:
                d = run.data.loc[run.data["heldout_sliced_w2"].gt(0)]
                if not by_iteration:
                    d = d.dropna(subset=["training_seconds"])
                if d.empty:
                    continue
                ax.plot(d.index if by_iteration else d["training_seconds"], d["heldout_sliced_w2"],
                        color=colors[key], marker=".", ms=3, lw=1.4, alpha=.85, label=label)
                label = None
        ax.set_title("Held-out map quality by iteration" if by_iteration else "Held-out map quality by training time", loc="left", fontweight="bold")
        ax.set_xlabel("Training iteration" if by_iteration else "Cumulative training time (s)")
        ax.set_yscale("log")
        ax.set_ylabel("Sliced W₂ (log scale)")
        ax.grid(axis="both", alpha=.25)
        if ax.lines:
            ax.legend(fontsize=9, frameon=False, ncols=4, loc="lower center", bbox_to_anchor=(.5, 1.03))

    if has_quality:
        quality_curve(fig.add_subplot(grid[4, :]), by_iteration=False)
        quality_curve(fig.add_subplot(grid[5, :]), by_iteration=True)

    fig.suptitle("OFM experiment comparison", fontsize=19, fontweight="bold", y=.97)
    caption = ("Summaries use each run’s full history." if shared is None else f"{len(shared):,} shared iterations.")
    caption += " Summary values show medians across runs. Bold = best, underline = second best."
    if has_quality:
        caption += " Quality curves show full histories."
    fig.text(.17, .047, "Boxes show the middle 50%. Line = median, diamond = mean, dots = outliers. Restricted-time losses average over a different time range.", fontsize=9, color="#52646F")
    fig.text(.17, .916, caption, fontsize=10, color="#52646F")
    fig.subplots_adjust(left=.17, right=.97, bottom=.105, top=.875, wspace=.50, hspace=.65)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for label in runners_up:
        bounds = label.get_window_extent(renderer)
        left, right = fig.transFigure.inverted().transform(
            [(bounds.x0, bounds.y0), (bounds.x1, bounds.y0)]
        )
        fig.add_artist(plt.Line2D(
            [left[0], right[0]], [left[1], right[1]],
            transform=fig.transFigure, color=label.get_color(), linewidth=.8,
        ))
    fig.savefig(output, dpi=175, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="*", help="Profile CSV files or directories to search")
    parser.add_argument("--output", type=Path,
                        help="Output directory (default: amortized_tests/comparison)")
    parser.add_argument("--iterations", type=int, help="Use the first N shared iterations in the comparison tables")
    parser.add_argument("--window", type=int, default=100, help="Rolling plot and final loss window in batches")
    parser.add_argument("--spike-factor", type=float, default=4.,
                        help="True loss relative to the preceding 100-batch median (minimum 20 prior batches)")
    args = parser.parse_args()
    if args.window < 1 or (args.iterations is not None and args.iterations < 1):
        parser.error("The window and comparison budget must be positive integers")
    if not 1 <= args.spike_factor < np.inf:
        parser.error("Use a finite spike factor of at least 1")
    try:
        runs = [load_run(p, args.spike_factor) for p in find_profiles(args.inputs)]
        shared = common_iterations(runs, args.iterations) if args.iterations is not None else None
    except (OSError, ValueError) as error:
        parser.error(str(error))

    result_root = DEFAULT_RESULTS
    if len(args.inputs) == 1 and args.inputs[0].is_dir() and args.inputs[0].name == "amortized_tests":
        result_root = args.inputs[0]
    output = args.output or result_root / "comparison"
    groups = {}
    for run in sorted(runs, key=lambda r: list(METHODS).index(r.method) if r.method in METHODS else len(METHODS)):
        groups.setdefault(run.method, []).append(run)
    labels, colors = presentation(groups)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
        "axes.spines.right": False, "axes.edgecolor": "#B2BEC6",
        "axes.labelcolor": "#253744", "text.color": "#253744",
    })
    output.mkdir(parents=True, exist_ok=True)
    runs_table, methods, events = tables(runs, shared, args.window)
    for name, table in (("run_summary", runs_table), ("method_summary", methods), ("spikes", events)):
        table.to_csv(output / f"{name}.csv", index=False, float_format="%.8g")
    training_figure(groups, labels, colors, args.window, args.spike_factor, output / "training_curves.png")
    efficiency_figure(groups, methods, labels, colors, shared, output / "comparison.png")
    for run in runs:
        if len(run.data) < 30_000:
            print(f"{run.method} seed {run.seed}: {len(run.data):,} logged iterations out of the default 30,000")
    print(f"Wrote tables and figures for {len(runs)} runs to {output}")


if __name__ == "__main__":
    main()
