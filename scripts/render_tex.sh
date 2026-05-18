#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/render_tex.sh [--publish-parent] path/to/file.tex [pdf|xelatex|lualatex]

Renders a TeX document into a sibling build/<document-name>/ directory.
Examples:
  scripts/render_tex.sh docs/01/notes/02.valuation.tex
  scripts/render_tex.sh --publish-parent docs/01/notes/02.valuation.tex
  scripts/render_tex.sh docs/01/notes/02.valuation.tex xelatex
EOF
}

if ! command -v latexmk >/dev/null 2>&1; then
  echo "latexmk is not available in PATH." >&2
  exit 1
fi

publish_parent=0

while [ $# -gt 0 ]; do
  case "$1" in
    --publish-parent)
      publish_parent=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --publish-parnt)
      echo "Unknown option: $1" >&2
      echo "Did you mean --publish-parent?" >&2
      exit 1
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
    *)
      break
      ;;
  esac
done

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
  usage >&2
  exit 1
fi

input="$1"
engine="${2:-pdf}"

case "$engine" in
  pdf)
    engine_flag="-pdf"
    engine_binary="pdflatex"
    ;;
  xelatex)
    engine_flag="-xelatex"
    engine_binary="xelatex"
    ;;
  lualatex)
    engine_flag="-lualatex"
    engine_binary="lualatex"
    ;;
  *)
    echo "Unsupported engine: $engine" >&2
    usage >&2
    exit 1
    ;;
esac

if ! command -v "$engine_binary" >/dev/null 2>&1; then
  echo "Required TeX engine is not available in PATH: $engine_binary" >&2
  exit 1
fi

if [ ! -f "$input" ]; then
  echo "TeX source not found: $input" >&2
  exit 1
fi

case "$input" in
  *.tex)
    ;;
  *)
    echo "Expected a .tex file: $input" >&2
    exit 1
    ;;
esac

input_abs="$(realpath "$input")"
src_dir="$(dirname "$input_abs")"
src_file="$(basename "$input_abs")"
doc_name="${src_file%.tex}"
out_dir="$src_dir/build/$doc_name"
publish_dir="$(realpath "$src_dir/..")"
published_pdf="$publish_dir/$doc_name.pdf"

if [ ! -s "$input_abs" ]; then
  echo "Refusing to build an empty TeX file: $input_abs" >&2
  exit 1
fi

if ! grep -q '\\documentclass' "$input_abs"; then
  echo "TeX source does not contain \\documentclass: $input_abs" >&2
  echo "If this file is a fragment, build its root document instead." >&2
  exit 1
fi

if ! grep -q '\\begin{document}' "$input_abs"; then
  echo "TeX source does not contain \\begin{document}: $input_abs" >&2
  echo "If this file is a fragment, build its root document instead." >&2
  exit 1
fi

mkdir -p "$out_dir"

(
  cd "$src_dir"
  latexmk \
    "$engine_flag" \
    -interaction=nonstopmode \
    -file-line-error \
    -synctex=1 \
    -outdir="$out_dir" \
    "$src_file"
)

printf 'Built PDF: %s\n' "$out_dir/$doc_name.pdf"

if [ "$publish_parent" -eq 1 ]; then
  install -m 0644 "$out_dir/$doc_name.pdf" "$published_pdf"
  printf 'Published PDF: %s\n' "$published_pdf"
fi
