#!/usr/bin/env bash
# Builds the project, generates the dataset (first run only) and processes
# every image in data/input on the GPU. Extra arguments are passed through to
# bin/edge_detect, e.g.  ./run.sh --num_streams 8 --threshold 60
set -euo pipefail
cd "$(dirname "$0")"

make build

if [ -z "$(ls -A data/input 2>/dev/null)" ]; then
  echo "Generating dataset in data/input ..."
  make data
fi

mkdir -p logs output
./bin/edge_detect \
  --input_dir data/input \
  --output_dir output \
  --log_file logs/run_log.txt \
  --summary_file logs/summary.md \
  --num_streams 4 \
  --block_size 16 \
  --sigma 1.4 \
  --gallery 8 \
  --verify \
  "$@"

echo
echo "Done. Edge maps: output/edges | previews: output/gallery"
echo "Log: logs/run_log.txt | summary: logs/summary.md"
