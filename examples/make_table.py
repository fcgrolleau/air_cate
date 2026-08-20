"""Generate a LaTeX simulation-results table from simulation_results.json."""

import json
import math
import os
import sys

RESULTS_PATH = os.path.join(
    os.path.dirname(__file__), "figures", "simulation_results.json"
)
OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "latex", "simulation_table.tex"
)

DGPS = ["Weibull", "Mixture", "IST-3"]
METHODS = ["AIR-oracle", "TP-oracle", "AIR", "Tensor-prod", "RSF-T", "RSF-S", "CSF"]
COL_HEADERS = [
    "MGD",
    "TP",
    "MGD",
    "TP",
    "T-learner",
    "S-learner",
    "grf package",
]
# Spans must sum to len(METHODS); col_spec below derives its width from that too.
GROUP_HEADERS = [
    (r"AIR (oracle nuis.)", 2),
    (r"Cross-fitted AIR", 2),
    (r"RSF baselines", 2),
    (r"Causal surv. forest", 1),
]


def load_results(path):
    with open(path) as f:
        data = json.load(f)
    return data["results"], data["n_values"]


def _mean_se(values):
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    mean = sum(values) / n
    se = (sum((v - mean) ** 2 for v in values) / (n * (n - 1))) ** 0.5 if n > 1 else 0.0
    return mean, se


def compute_stats(entries):
    imses = [e["imse"] for e in entries if not math.isnan(e["imse"])]
    runtimes = [e["runtime_seconds"] for e in entries]
    if not imses:
        return float("nan"), float("nan"), 0.0, 0.0
    mean, se = _mean_se(imses)
    mean_rt, se_rt = _mean_se(runtimes)
    return mean, se, mean_rt, se_rt


# One fixed precision per quantity, applied to every cell in the table. Fixed
# beats per-cell precision here: a column where 0.1000 sits under 0.0503 reads as
# one scale the eye can scan down, whereas 0.1 under 0.0503 invites the reader to
# compare digit positions instead of values. Trailing zeros are the point, not
# noise. Runtimes span 0.3s to ~20000s, so they carry one decimal -- enough to
# separate the sub-second baselines without printing false precision on the
# five-digit CSF timings.
IMSE_DECIMALS = 4
RUNTIME_DECIMALS = 1


def _fmt_val(v, decimals):
    return f"{v:.{decimals}f}"


def _fmt_floor(v, decimals, bold=False):
    """Fixed-precision value, or a "<" bound when it would round to zero.

    A printed 0.0 makes two false claims -- an instantaneous fit, or a zero
    standard error -- so anything below the column's resolution is reported as a
    bound at that resolution instead ("<0.1", "<0.0001"). The zero test is on the
    formatted string rather than a threshold, so it stays correct if the decimal
    counts above ever change.

    bold matters because \\textbf does not reach inside math mode: wrapping $<0.1$
    in \\textbf leaves it roman while the digits around it turn bold. \\boldsymbol
    (amsmath) is the math-mode equivalent. The {<} braces make < an ordinary
    symbol rather than a relation, dropping the thin space TeX would otherwise
    insert after the opening parenthesis.
    """
    text = _fmt_val(v, decimals)
    if float(text) != 0:
        return text
    bound = rf"{{<}}{_fmt_val(10.0**-decimals, decimals)}"
    return rf"$\boldsymbol{{{bound}}}$" if bold else f"${bound}$"


def fmt_imse(mean, se, bold=False):
    if math.isnan(mean):
        return "---"
    mean_str = _fmt_floor(mean, IMSE_DECIMALS, bold=bold)
    se_str = "---" if math.isnan(se) else _fmt_floor(se, IMSE_DECIMALS, bold=bold)
    return f"{mean_str} ({se_str})"


def fmt_runtime(rt, se_rt, bold=False):
    """Mean runtime and its SE, both in seconds.

    The trailing "s" is repeated on the SE so neither number can be mistaken for
    the iMSE line directly above it in the same cell.
    """
    mean_str = _fmt_floor(rt, RUNTIME_DECIMALS, bold=bold)
    se_str = _fmt_floor(se_rt, RUNTIME_DECIMALS, bold=bold)
    return f"{mean_str}s ({se_str}s)"


def best_methods(values, decimals):
    """Methods tied for the lowest value *as printed* at `decimals` places.

    Bolding a strict argmin looks arbitrary whenever a competitor rounds to the
    same printed number -- routine among the sub-second RSF runtimes, where two
    methods can both land on "<0.1s". Comparing formatted strings marks every
    method the reader has no way to distinguish in the table.
    """
    if not values:
        return set()
    target = _fmt_val(min(values.values()), decimals)
    return {m for m, v in values.items() if _fmt_val(v, decimals) == target}


def _n_replications(results):
    """Modal number of replications per cell, for the caption.

    Cells can differ if some (DGP, method, n) shards failed, so report the mode
    rather than assuming the nominal seed count.
    """
    counts = {}
    for dgp in DGPS:
        for method in METHODS:
            for entries in results.get(dgp, {}).get(method, {}).values():
                counts[len(entries)] = counts.get(len(entries), 0) + 1
    if not counts:
        return 0
    return max(counts, key=counts.get)


def make_table(results, n_values):
    n_reps = _n_replications(results)
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{Accuracy and cost across {n_reps} replications for each combination of DGP, "
                 r"sample size, and method. Each cell reports the mean iMSE (SE) on the first line and the "
                 r"mean runtime in seconds (SE) on the second. Within each row, the lowest iMSE and the "
                 r"shortest runtime are shown in bold. Values below the displayed precision are reported as "
                 r"a bound rather than rounded to zero. Runtimes measured on a single core of an "
                 r"AMD EPYC 7502 CPU (2.5\,GHz base).}")
    lines.append(r"\label{tab:simulation}")
    lines.append(r"\resizebox{\textwidth}{!}{%")

    col_spec = "ll" + "c" * len(METHODS)  # DGP + n, then one per method
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\hline")

    # Group header row
    parts = ["", ""]
    for name, span in GROUP_HEADERS:
        parts.append(rf"\multicolumn{{{span}}}{{c}}{{\textbf{{{name}}}}}")
    lines.append(" & ".join(parts) + r" \\")

    # Horizontal rules under group headers
    col = 3  # 1-indexed, first method column
    cmidrules = []
    for _, span in GROUP_HEADERS:
        cmidrules.append(rf"\cline{{{col}-{col + span - 1}}}")
        col += span
    lines.append(" ".join(cmidrules))

    # Column header row
    parts = [r"\textbf{DGP}", r"\textbf{$n$}"] + [rf"\textbf{{{h}}}" for h in COL_HEADERS]
    lines.append(" & ".join(parts) + r" \\")
    lines.append(r"\hline")

    for dgp_idx, dgp in enumerate(DGPS):
        for n_idx, n in enumerate(n_values):
            n_str = str(n)

            # Compute stats for all methods
            stats = {}
            for method in METHODS:
                entries = results.get(dgp, {}).get(method, {}).get(n_str, [])
                stats[method] = compute_stats(entries)

            # Best iMSE and best runtime are marked independently. Both are
            # restricted to methods that actually ran at this n: a cell with no
            # replications has a 0.0 runtime from compute_stats, which would
            # otherwise win the runtime comparison outright.
            valid = {m: s[0] for m, s in stats.items() if not math.isnan(s[0])}
            best_imse = best_methods(valid, IMSE_DECIMALS)
            best_runtime = best_methods(
                {m: stats[m][2] for m in valid}, RUNTIME_DECIMALS
            )

            # Format n with thousands separator
            n_fmt = f"{n:,}".replace(",", r"{\,}")

            dgp_label = rf"\textbf{{{dgp}}}" if n_idx == 0 else ""

            cells = [dgp_label, f"${n_fmt}$"]
            for method in METHODS:
                mean, se, rt, se_rt = stats[method]
                # No replications means no runtime either, so the whole cell is a
                # dash rather than a runtime line under an em dash.
                if math.isnan(mean):
                    cells.append("---")
                    continue

                imse_bold = method in best_imse
                imse_str = fmt_imse(mean, se, bold=imse_bold)
                if imse_bold:
                    imse_str = rf"\textbf{{{imse_str}}}"

                rt_bold = method in best_runtime
                rt_str = fmt_runtime(rt, se_rt, bold=rt_bold)
                if rt_bold:
                    rt_str = rf"\textbf{{{rt_str}}}"

                cells.append(rf"\makecell{{{imse_str} \\ \scriptsize {rt_str}}}")

            lines.append(" & ".join(cells) + r" \\")

        # Separator between DGPs
        if dgp_idx < len(DGPS) - 1:
            lines.append(r"\hline")

    lines.append(r"\hline")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def main():
    results_path = sys.argv[1] if len(sys.argv) > 1 else RESULTS_PATH
    output_path = sys.argv[2] if len(sys.argv) > 2 else OUTPUT_PATH

    results, n_values = load_results(results_path)
    tex = make_table(results, n_values)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(tex + "\n")

    print(f"Table written to {output_path}")


if __name__ == "__main__":
    main()
