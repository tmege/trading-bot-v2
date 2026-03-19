#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
fi

COIN="${1:-BTC}"
STRATEGY="${2:-trading_bot/strategies/btc_sniper_1h.py}"
OUTPUT_DIR="data/grid_results"
mkdir -p "$OUTPUT_DIR"

echo "Grid search: $COIN with $STRATEGY"

for TP in 1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0 6.0; do
    for SL in 0.5 1.0 1.5 2.0 2.5 3.0 4.0 5.0 6.0; do
        echo "Testing TP=${TP}% SL=${SL}%"
        python -m trading_bot.backtest.runner "$COIN" "$STRATEGY" \
            --tp "$TP" --sl "$SL" \
            --output "${OUTPUT_DIR}/${COIN}_tp${TP}_sl${SL}.json" \
            2>/dev/null
    done
done

echo "Grid search complete. Results in $OUTPUT_DIR/"
