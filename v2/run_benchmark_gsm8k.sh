#!/bin/bash
#
# Run GSM8k benchmark using model.generate()
#
# Usage:
#   ./run_benchmark_gsm8k.sh                          # Base model, 20 samples
#   ./run_benchmark_gsm8k.sh --smoothquant            # SmoothQuant model
#   ./run_benchmark_gsm8k.sh --limit 100 --batch-size 8
#

# Default values
MODEL="base"
LIMIT=20
BATCH_SIZE=4
THRESHOLD=0.9
VERBOSE=""
OUTPUT=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --smoothquant|-sq)
            MODEL="smoothquant"
            shift
            ;;
        --base)
            MODEL="base"
            shift
            ;;
        --limit|-l)
            LIMIT="$2"
            shift 2
            ;;
        --batch-size|-b)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --threshold|-t)
            THRESHOLD="$2"
            shift 2
            ;;
        --verbose|-v)
            VERBOSE="--verbose"
            shift
            ;;
        --output|-o)
            OUTPUT="--output $2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --smoothquant, -sq      Use SmoothQuant model (default: base)"
            echo "  --base                  Use base model"
            echo "  --limit, -l NUM         Limit number of samples (default: 20)"
            echo "  --batch-size, -b NUM    Batch size (default: 4)"
            echo "  --threshold, -t NUM     Generation threshold (default: 0.9)"
            echo "  --verbose, -v           Print detailed output"
            echo "  --output, -o FILE       Save results to JSON file"
            echo "  --help, -h              Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --limit 20                    # Quick test with base model"
            echo "  $0 --smoothquant --limit 100    # Test SmoothQuant on 100 samples"
            echo "  $0 --smoothquant --limit 50 --verbose --output results.json"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Activate virtual environment if it exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Set environment variables
export CUDA_VISIBLE_DEVICES=0

echo "============================================================"
echo "GSM8k Benchmark (using model.generate())"
echo "============================================================"
echo "Model:      ${MODEL}"
echo "Limit:      ${LIMIT} samples"
echo "Batch size: ${BATCH_SIZE}"
echo "Threshold:  ${THRESHOLD}"
echo "============================================================"
echo ""

# Run the benchmark
python benchmark_gsm8k.py \
    --model "${MODEL}" \
    --limit "${LIMIT}" \
    --batch-size "${BATCH_SIZE}" \
    --threshold "${THRESHOLD}" \
    ${VERBOSE} \
    ${OUTPUT}
