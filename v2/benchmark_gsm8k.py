#!/usr/bin/env python3
"""
GSM8k Benchmark Script for Fast-dLLM

Uses model.generate() with proper chat template formatting.
Supports batch processing for efficient evaluation.

Usage:
    python benchmark_gsm8k.py --model base --limit 20
    python benchmark_gsm8k.py --model smoothquant --limit 100 --batch-size 8
"""

import argparse
import re
import json
import os
import time
from typing import List, Dict, Optional, Tuple

import torch
import numpy as np
import random
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================================
# Constants
# ============================================================================

MODEL_NAME = "Efficient-Large-Model/Fast_dLLM_v2_7B"
SMOOTHQUANT_PATH = "models/fast_dllm_smoothquant.pt"

# GSM8k answer extraction patterns
ANS_RE = re.compile(r"####\s*(\-?[\d,\.]+)")
BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")
NUMBER_RE = re.compile(r"(\-?[\d,\.]+)")


# ============================================================================
# Utility Functions
# ============================================================================

def fix_seed(seed: int = 42):
    """Fix random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def extract_answer(text: str) -> Optional[str]:
    """
    Extract the numerical answer from model output.
    Tries multiple patterns: #### format, \\boxed{}, and last number.
    """
    # Clean up the text
    text = text.strip()
    
    # Try #### format first (GSM8k standard)
    match = ANS_RE.search(text)
    if match:
        return match.group(1).replace(",", "")
    
    # Try \\boxed{} format
    match = BOXED_RE.search(text)
    if match:
        boxed_content = match.group(1)
        # Extract number from boxed content
        num_match = NUMBER_RE.search(boxed_content)
        if num_match:
            return num_match.group(1).replace(",", "")
    
    # Fallback: find the last number in the text
    numbers = NUMBER_RE.findall(text)
    if numbers:
        return numbers[-1].replace(",", "")
    
    return None


def extract_gold_answer(answer_text: str) -> Optional[str]:
    """Extract the gold answer from GSM8k answer field."""
    match = ANS_RE.search(answer_text)
    if match:
        return match.group(1).replace(",", "")
    return None


def normalize_answer(answer: Optional[str]) -> Optional[float]:
    """Normalize answer string to float for comparison."""
    if answer is None:
        return None
    try:
        return float(answer.replace(",", ""))
    except ValueError:
        return None


def answers_match(pred: Optional[str], gold: Optional[str]) -> bool:
    """Check if predicted and gold answers match."""
    pred_norm = normalize_answer(pred)
    gold_norm = normalize_answer(gold)
    
    if pred_norm is None or gold_norm is None:
        return False
    
    # Allow small floating point tolerance
    return abs(pred_norm - gold_norm) < 1e-5


# ============================================================================
# Model Loading
# ============================================================================

def load_model(model_type: str = "base", device: str = "cuda:0"):
    """
    Load the Fast-dLLM model.
    
    Args:
        model_type: "base" or "smoothquant"
        device: Device to load the model on
    
    Returns:
        model, tokenizer
    """
    print(f"\n{'='*60}")
    print(f"Loading Fast-dLLM model ({model_type})")
    print(f"{'='*60}")
    
    print(f"Loading base model: {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True
    )
    
    if model_type == "smoothquant":
        if os.path.exists(SMOOTHQUANT_PATH):
            print(f"Loading SmoothQuant weights from {SMOOTHQUANT_PATH}...")
            state_dict = torch.load(SMOOTHQUANT_PATH, map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
            print("✓ SmoothQuant weights loaded")
        else:
            print(f"WARNING: SmoothQuant weights not found at {SMOOTHQUANT_PATH}")
            print("Proceeding with base model weights.")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model.eval()
    print(f"✓ Model loaded on {device}")
    
    return model, tokenizer


# ============================================================================
# Data Loading
# ============================================================================

def load_gsm8k_data(split: str = "test", limit: Optional[int] = None) -> List[Dict]:
    """
    Load GSM8k dataset from HuggingFace.
    
    Args:
        split: "train" or "test"
        limit: Optional limit on number of samples
    
    Returns:
        List of question/answer dicts
    """
    print(f"\nLoading GSM8k dataset ({split} split)...")
    dataset = load_dataset("gsm8k", "main", split=split)
    
    data = []
    for i, item in enumerate(dataset):
        if limit and i >= limit:
            break
        data.append({
            "question": item["question"],
            "answer": item["answer"],
            "gold": extract_gold_answer(item["answer"])
        })
    
    print(f"✓ Loaded {len(data)} samples")
    return data


# ============================================================================
# Prompt Formatting
# ============================================================================

def format_prompt(question: str) -> List[Dict[str, str]]:
    """
    Format a GSM8k question as a chat message.
    Uses the same format as the chatbot for consistency.
    """
    prompt = f"""Question: {question}

Please solve this step by step. Show your reasoning, then give your final numerical answer after "####"."""

    messages = [
        {"role": "system", "content": "You are a helpful math assistant. Solve problems step by step and give the final numerical answer after ####."},
        {"role": "user", "content": prompt}
    ]
    return messages


# ============================================================================
# Batch Generation
# ============================================================================

def generate_batch(
    model,
    tokenizer,
    questions: List[str],
    max_new_tokens: int = 512,
    block_size: int = 32,
    small_block_size: int = 8,
    threshold: float = 0.9,
) -> List[str]:
    """
    Generate responses for a batch of questions using model.generate().
    
    Args:
        model: The Fast-dLLM model
        tokenizer: The tokenizer
        questions: List of question strings
        max_new_tokens: Maximum tokens to generate
        block_size: Block size for diffusion
        small_block_size: Small block size
        threshold: Generation threshold
    
    Returns:
        List of generated response strings
    """
    device = model.device
    
    # Format all prompts using chat template
    all_texts = []
    for q in questions:
        messages = format_prompt(q)
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        all_texts.append(text)
    
    # Tokenize with padding
    inputs = tokenizer(
        all_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    ).to(device)
    
    input_lengths = [
        (inputs["input_ids"][i] != tokenizer.pad_token_id).sum().item()
        for i in range(len(questions))
    ]
    
    # Generate using model.generate()
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            tokenizer=tokenizer,
            block_size=block_size,
            max_new_tokens=max_new_tokens,
            small_block_size=small_block_size,
            threshold=threshold,
        )
    
    # Decode responses (only the generated part)
    responses = []
    for i, output in enumerate(outputs):
        # Get only the newly generated tokens
        generated = output[input_lengths[i]:]
        response = tokenizer.decode(generated, skip_special_tokens=True)
        responses.append(response)
    
    return responses


# ============================================================================
# Evaluation
# ============================================================================

def evaluate(
    model,
    tokenizer,
    data: List[Dict],
    batch_size: int = 4,
    max_new_tokens: int = 512,
    threshold: float = 0.9,
    verbose: bool = False,
) -> Tuple[float, List[Dict]]:
    """
    Evaluate model on GSM8k dataset.
    
    Args:
        model: The model
        tokenizer: The tokenizer
        data: List of question/answer dicts
        batch_size: Batch size for generation
        max_new_tokens: Max tokens to generate
        threshold: Generation threshold
        verbose: Print detailed output
    
    Returns:
        (accuracy, results_list)
    """
    print(f"\n{'='*60}")
    print(f"Running GSM8k Evaluation")
    print(f"{'='*60}")
    print(f"Samples: {len(data)}")
    print(f"Batch size: {batch_size}")
    print(f"Max new tokens: {max_new_tokens}")
    print(f"Threshold: {threshold}")
    print()
    
    results = []
    correct = 0
    total = 0
    
    start_time = time.time()
    
    # Process in batches
    for i in tqdm(range(0, len(data), batch_size), desc="Evaluating"):
        batch = data[i:i + batch_size]
        questions = [item["question"] for item in batch]
        golds = [item["gold"] for item in batch]
        
        # Generate responses
        responses = generate_batch(
            model,
            tokenizer,
            questions,
            max_new_tokens=max_new_tokens,
            threshold=threshold,
        )
        
        # Evaluate each response
        for j, (response, gold) in enumerate(zip(responses, golds)):
            pred = extract_answer(response)
            is_correct = answers_match(pred, gold)
            
            if is_correct:
                correct += 1
            total += 1
            
            result = {
                "idx": i + j,
                "question": questions[j],
                "gold": gold,
                "predicted": pred,
                "correct": is_correct,
                "response": response[:500] + "..." if len(response) > 500 else response,
            }
            results.append(result)
            
            if verbose:
                status = "✓" if is_correct else "✗"
                print(f"\n[{i+j}] {status} Gold: {gold}, Pred: {pred}")
                print(f"Response: {response[:200]}...")
    
    elapsed = time.time() - start_time
    accuracy = correct / total if total > 0 else 0.0
    
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Correct: {correct}/{total}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Time: {elapsed:.1f}s ({elapsed/total:.2f}s per sample)")
    print(f"{'='*60}")
    
    return accuracy, results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Fast-dLLM on GSM8k using model.generate()",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark_gsm8k.py --model base --limit 20
  python benchmark_gsm8k.py --model smoothquant --limit 100 --batch-size 8
  python benchmark_gsm8k.py --model base --threshold 0.85 --verbose
        """
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        choices=["base", "smoothquant"],
        default="base",
        help="Model type to use (default: base)"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Limit number of samples (default: all)"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=4,
        help="Batch size for generation (default: 4)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Max new tokens to generate (default: 512)"
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.9,
        help="Generation threshold (default: 0.9)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use (default: cuda:0)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed output"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file for results (JSON)"
    )
    
    args = parser.parse_args()
    
    # Fix seed
    fix_seed(args.seed)
    
    # Load model
    model, tokenizer = load_model(args.model, args.device)
    
    # Load data
    data = load_gsm8k_data(split="test", limit=args.limit)
    
    # Evaluate
    accuracy, results = evaluate(
        model,
        tokenizer,
        data,
        batch_size=args.batch_size,
        max_new_tokens=args.max_tokens,
        threshold=args.threshold,
        verbose=args.verbose,
    )
    
    # Save results if requested
    if args.output:
        output_data = {
            "model": args.model,
            "accuracy": accuracy,
            "num_samples": len(data),
            "batch_size": args.batch_size,
            "threshold": args.threshold,
            "results": results,
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")
    
    return accuracy


if __name__ == "__main__":
    main()
