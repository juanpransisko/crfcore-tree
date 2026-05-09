#!/usr/bin/env bash
#  Train and Evals
#   ./run.sh - train + evaluate (full pipeline)
#   ./run.sh train - train only
#   ./run.sh eval - evaluate only
#   ./run.sh serve - launch Flask web app
#   ./run.sh test - run pytest suite
#   ./run.sh clean - remove models and results
#
# Uses .crfenv Python environment by default. Override with:
#   PYTHON=/path/to/python ./run.sh

set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
CRFENV="$DIR/.crfenv/bin/python"

if [ -x "$CRFENV" ]; then
    PYTHON="${PYTHON:-$CRFENV}"
else
    PYTHON="${PYTHON:-python3}"
    echo "Warning: .crfenv not found, falling back to $PYTHON"
fi

check_corpus() {
    if [ ! -f "$DIR/corpus/trainHPOST.words" ] || [ ! -f "$DIR/corpus/trainHPOST.tags" ]; then
        echo "Error: corpus/ not found. Place or symlink .words and .tags files there."
        echo "  e.g., ln -s ../corpus $DIR/corpus"
        exit 1
    fi
}

train() {
    check_corpus
    echo "=== Training CRF model ==="
    "$PYTHON" "$DIR/train_crf.py"
    echo ""
}

eval_model() {
    if [ ! -f "$DIR/models/crfpost_model.crfsuite" ]; then
        echo "Error: No trained model found. Run './run.sh train' first."
        exit 1
    fi
    echo "=== Evaluating CRF model ==="
    "$PYTHON" "$DIR/eval_crf.py" --model crfsuite
    echo ""
}

serve() {
    if [ ! -f "$DIR/app/pos_models/crfpost_model.crfsuite" ]; then
        echo "Error: No model in app/pos_models/. Train first or copy model there."
        exit 1
    fi
    echo "=== Starting TreeCo Flask app ==="
    cd "$DIR/app"
    "$PYTHON" main.py
}

run_tests() {
    echo "=== Running tests ==="
    "$PYTHON" -m pytest "$DIR/tests/" -v
}

clean() {
    echo "Removing models/ and results/..."
    rm -rf "$DIR/models" "$DIR/results"
    echo "Done."
}

CMD="${1:-all}"

case "$CMD" in
    train)
        train
        ;;
    eval)
        eval_model
        ;;
    all)
        train
        eval_model
        echo "=== Pipeline complete ==="
        echo "Model:   $DIR/models/crfpost_model.crfsuite"
        echo "Results: $DIR/results/"
        ;;
    serve)
        serve
        ;;
    test)
        run_tests
        ;;
    clean)
        clean
        ;;
    *)
        echo "Usage: $0 {train|eval|all|serve|test|clean}"
        exit 1
        ;;
esac
