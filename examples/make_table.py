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
    "grf",
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


def _fmt_val(v, decimals):
    return f"{v:.{decimals}f}"


# Approximate an SE only when it is genuinely beyond the table's resolution.
# Everything above this gets printed as a number, at whatever precision that takes,
# so the reader never sees a bound in one cell and 0.0001 in the next.
SE_APPROX_BELOW = 1e-5
SE_MAX_DECIMALS = 5


def _se_decimals(se, decimals):
    """Decimals for the SE: the mean's precision, widened only if that shows zero.

    An SE is conventionally reported at the same precision as the estimate it
    belongs to, and that is what keeps a row readable. It breaks only when the SE
    is finer than the estimate's last digit: 0.0004 beside a mean of 0.145 would
    print as 0.000, which reads as a zero standard error. In that case, and only
    then, add decimals -- and stop at the first precision that shows something, so
    an SE of 9e-5 prints as 0.0001 rather than 0.00009 and stays flush with the
    rest of the column. Cells that already print a real number are left at the
    mean's precision instead of sprouting digits.

    The zero test is on the formatted string, not a threshold, so it stays correct
    if the decimal counts in fmt_imse ever change.
    """
    for d in range(decimals, SE_MAX_DECIMALS + 1):
        if float(_fmt_val(se, d)) != 0:
            return d
    return SE_MAX_DECIMALS


def fmt_se(se, decimals, bold=False):
    """SE text, or a bound when it is below the table's resolution entirely.

    bold matters because \\textbf does not reach inside math mode: wrapping
    $10^{-5}$ in \\textbf leaves it roman while the digits around it turn bold.
    \\boldsymbol (amsmath) is the math-mode equivalent. The {<} braces make < an
    ordinary symbol rather than a relation, dropping the thin space TeX would
    otherwise insert after the opening parenthesis.
    """
    if math.isnan(se):
        return "---"
    if se < SE_APPROX_BELOW:
        bound = rf"{{<}}10^{{-{SE_MAX_DECIMALS}}}"
        return rf"$\boldsymbol{{{bound}}}$" if bold else f"${bound}$"
    return _fmt_val(se, _se_decimals(se, decimals))


def fmt_imse(mean, se, bold=False):
    if math.isnan(mean):
        return "---"
    if mean >= 1:
        decimals = 2
    elif mean >= 0.1:
        decimals = 3
    else:
        decimals = 4
    return f"{_fmt_val(mean, decimals)} ({fmt_se(se, decimals, bold=bold)})"


def fmt_runtime(rt, se_rt):
    return f"{rt:.0f}s ({se_rt:.0f}s)"


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
    lines.append(rf"\caption{{Mean iMSE (SE) and average runtime in seconds (SE) across {n_reps} replications "
                 r"for each combination of DGP, sample size, and method. "
                 r"The lowest iMSE in each row is shown in bold. "
                 r"Runtimes measured on a single core of an AMD EPYC 7502 CPU (2.5\,GHz base).}")
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

            # Find best (lowest mean iMSE)
            valid = {m: s[0] for m, s in stats.items() if not math.isnan(s[0])}
            best_method = min(valid, key=valid.get) if valid else None

            # Format n with thousands separator
            n_fmt = f"{n:,}".replace(",", r"{\,}")

            dgp_label = rf"\textbf{{{dgp}}}" if n_idx == 0 else ""

            cells = [dgp_label, f"${n_fmt}$"]
            for method in METHODS:
                mean, se, rt, se_rt = stats[method]
                # A cell with no replications has no runtime either. compute_stats
                # returns 0.0 for it, which would print as "0s (0s)" and read as an
                # instantaneous fit rather than a method that was not run at this n.
                if math.isnan(mean):
                    cells.append("---")
                    continue
                is_best = method == best_method
                imse_str = fmt_imse(mean, se, bold=is_best)
                rt_str = fmt_runtime(rt, se_rt)
                cell = f"{imse_str}"
                if is_best:
                    cell = rf"\textbf{{{cell}}}"
                cell = rf"\makecell{{{cell} \\ \scriptsize {rt_str}}}"
                cells.append(cell)

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
