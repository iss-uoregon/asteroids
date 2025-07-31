import numpy as np
import time
from src.params_list import params_list
from src.magnitude import calculate_sigma
from src.error_functions import estimate_sigma_distribution

"""
Driver file for estimating period uncertainty through magnitude shuffling.

Steps:
1. Load normalization data for bin weighting (count_tbl)
2. Call `calculate_sigma` with shuffling enabled for one example
3. Run `estimate_sigma_distribution` to build a sigma distribution and estimate period error

Requires:
- `count_cache.npz` (generated in folding_module.py)
"""

start = time.time()

def main():
    # 1. Loaded count table information for calculate_sigma normalization. First generated in folding_module.py for reference.
    counts = np.load("count_cache.npz")
    count_tbl = counts['count_tbl']
    avg_counts = counts["avg_counts"]
    total_counts = counts["total_counts"]

    # 2. Example of a calculate_sigma call with shuffline enabled using the Sabine data.
    shuffled_sigma_example = calculate_sigma(4.2940, params_list, avg_counts, total_counts, return_table=True, shuffled_mags=True)
    # print(shuffled_sigma_example)

    # 3. Example sigma distribution using shuffled magnitudes of the Sabine dataset. Used for period error estimation for well-sampled data.
    mu, sigma, fwhm, sigmas, best_tbl = estimate_sigma_distribution(
        4.2940,
        calculate_sigma=calculate_sigma,
        params_list=params_list,
        avg_counts=avg_counts,
        total_counts=total_counts,
        trials=5000,
        return_best_table=True,
        plot=True,
        verbose=True
    )

if __name__ == "__main__":
    main()

print(f"Elapsed time: {time.time() - start:.2f} sec")