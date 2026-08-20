#!/bin/bash
# Compile a single table fragment into a standalone PDF and open it.
#
#   ./preview.sh                                  # simulation_table_flips50.tex
#   ./preview.sh simulation_table_metrics50.tex   # any other fragment
#
# Aux files (.aux/.log) go in .preview_build/; the PDF lands next to this script.
set -euo pipefail

cd "$(dirname "$0")"

# Table fragment to preview; the .tex extension is optional.
TABLE="${1:-simulation_table_flips50}"
TABLE="${TABLE%.tex}"

if [ ! -f "$TABLE.tex" ]; then
    echo "error: $TABLE.tex not found in $(pwd)" >&2
    exit 1
fi

BUILD=".preview_build"
OUT="preview_$TABLE.pdf"
mkdir -p "$BUILD"

# -jobname keeps one PDF per table; -halt-on-error stops at the first real error
# instead of dropping into TeX's interactive prompt.
/Library/TeX/texbin/pdflatex \
    -interaction=nonstopmode \
    -halt-on-error \
    -output-directory="$BUILD" \
    -jobname="preview_$TABLE" \
    "\\def\\tablefile{$TABLE}\\input{preview_table.tex}" \
    > /dev/null

cp "$BUILD/preview_$TABLE.pdf" "$OUT"
echo "wrote $(pwd)/$OUT"
open "$OUT"
