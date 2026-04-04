# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research paper: **"Learning Time-Varying Conditional Average Treatment Effects from Right-Censored Data"** by François Grolleau (Stanford) and Alex Fernandes (Université Paris Cité). The method is called **Surv-CATE**.

This is a LaTeX-only repository — no application code, tests, or build system.

## Repository Structure

```
latex/
├── main.tex                    # Main paper source
├── main.bib                    # Bibliography (~60+ references)
├── related_work_frazier.tex    # Related work section (included from main.tex)
├── figure1.pdf                 # Standalone figures
├── prelim_res.pdf
├── surv_sim.pdf
└── figures/                    # Generated figure outputs
    ├── km_curve_by_treatment.pdf
    ├── mse_violin_plots.pdf
    └── var_mse_vs_n.pdf
```

## Building the Paper

Compile with the standard multi-pass LaTeX + BibTeX workflow from the `latex/` directory:

```bash
cd latex
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## LaTeX Conventions

- Bibliography style: `unsrtnat` (numbered, unsorted)
- Custom math operators: `\EX` (expectation), `\PR` (probability), `\IDF` (indicator), `\eqdef` (definition equality), `\indep` (independence)
- Custom commands: `\tend` (end time), `\ms` (mathscr), `\abs`, `\norm`, `\R`, `\Rp`, `\Rpstar`
- Theorem environments: theorem, proposition, lemma, definition, corollary, assumption, remark
- Algorithm style: `algorithm` + `algpseudocode` with `\textbf{Input:}` / `\textbf{Output:}`

## Research Context

- **Problem**: Estimating patient-level time-varying treatment effect curves from right-censored survival data
- **Approach**: R-learner framework with augmented inverse-probability-of-censoring-weighted (AIPW) pseudo-outcomes; integrated squared loss over time
- **Key property**: Neyman orthogonality — nuisance estimation errors affect the treatment-effect loss only at second order
- **Parametrization**: Covariate-gated mixture of survival-function differences (universally approximating, fully differentiable)
- **Applications**: Precision medicine — e.g., cardiac surgery vs. PCI, stroke trials
