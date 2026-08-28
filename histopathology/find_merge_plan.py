import argparse
from pathlib import Path

import numpy as np


def solve_optimal_merge_plan_maxmin(similarity_matrix, threshold):
    """Minimise block count, then maximise the weakest block-boundary similarity."""
    n_layers = len(similarity_matrix)
    dp = [(float("inf"), -1.0, -1)] * (n_layers + 1)
    dp[n_layers] = (0, float("inf"), -1)

    for start in range(n_layers - 1, -1, -1):
        for end in range(start, n_layers):
            block_similarity = float(similarity_matrix[start, end])
            if block_similarity < threshold:
                continue

            remaining_blocks, remaining_worst, _ = dp[end + 1]
            if remaining_blocks == float("inf"):
                continue

            candidate_blocks = 1 + remaining_blocks
            candidate_worst = min(block_similarity, remaining_worst)
            best_blocks, best_worst, _ = dp[start]

            if candidate_blocks < best_blocks or (
                candidate_blocks == best_blocks and candidate_worst > best_worst
            ):
                dp[start] = (candidate_blocks, candidate_worst, end + 1)

    plan = []
    current = 0
    while current < n_layers:
        _, _, next_start = dp[current]
        if next_start == -1:
            raise ValueError(f"No valid partition from layer {current} at threshold {threshold}.")
        plan.append((current, next_start - 1))
        current = next_start
    return plan


def print_plan(plan, matrix):
    """Print a merge plan and the similarity score at each selected block boundary."""
    print(f"Merge plan ({len(plan)} blocks): {plan}")
    for index, (start, end) in enumerate(plan):
        print(
            f"  block {index}: layers {start:02d}-{end:02d} "
            f"similarity={matrix[start, end]:.4f}"
        )


def main():
    """Discover a max-min merge plan from a saved block-similarity matrix."""
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix", type=Path, help="Path to *_avgcos_patches.npy.")
    parser.add_argument("--threshold", type=float, default=0.4)
    args = parser.parse_args()

    matrix = np.load(args.matrix)
    plan = solve_optimal_merge_plan_maxmin(matrix, args.threshold)
    print_plan(plan, matrix)


if __name__ == "__main__":
    main()
